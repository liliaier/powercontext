/**
 * 浏览器端侧 embedding 封装（Transformers.js / ONNX Runtime Web）
 *
 * - 文档向量化与查询向量化均在浏览器本地完成，无任何云端推理调用
 * - 优先使用 WebGPU；不可用/初始化超时/失败时自动降级 WASM，并在界面提示
 * - 支持 INT8 量化（q8）与 FP16（fp16）及 FP32 基线，便于量化对比
 */

import { pipeline, env } from "@huggingface/transformers";

export type DType = "fp32" | "fp16" | "q8";
export type Device = "webgpu" | "wasm";

export interface LoadResult {
  device: Device;
  dtype: DType;
  modelId: string;
  dimension: number;
  loadMs: number;
  /** 是否发生过 WebGPU→WASM 降级 */
  fellBack: boolean;
  /** 降级原因（如有） */
  fallbackReason?: string;
}

const MODEL_ID = "Xenova/all-MiniLM-L6-v2";

// 允许从 HuggingFace Hub 拉取模型（首次会缓存到浏览器）
env.allowLocalModels = false;

/** WebGPU 初始化超时（避免 headless / 驱动异常时无限挂起） */
const WEBGPU_INIT_TIMEOUT_MS = 20000;

function withTimeout<T>(p: Promise<T>, ms: number, reason: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(reason)), ms);
    p.then(
      (v) => {
        clearTimeout(timer);
        resolve(v);
      },
      (e) => {
        clearTimeout(timer);
        reject(e);
      }
    );
  });
}

/** 是否通过 URL 参数强制 WASM（?force_wasm=1，便于排障与对照） */
export function forceWasmRequested(): boolean {
  try {
    return new URLSearchParams(location.search).get("force_wasm") === "1";
  } catch {
    return false;
  }
}

/** 检测当前环境是否支持 WebGPU */
export async function detectWebGPU(): Promise<boolean> {
  try {
    if (forceWasmRequested()) return false;
    if (typeof navigator === "undefined" || !(navigator as any).gpu) return false;
    const adapter = await (navigator as any).gpu.requestAdapter();
    return !!adapter;
  } catch {
    return false;
  }
}

export class Embedder {
  private extractor: any = null;
  device: Device = "wasm";
  dtype: DType = "q8";
  dimension = 0;
  modelId = MODEL_ID;
  fellBack = false;
  fallbackReason?: string;

  private async buildPipeline(dtype: DType, device: Device) {
    return await pipeline("feature-extraction", MODEL_ID, {
      dtype: dtype === "fp32" ? "fp32" : dtype === "fp16" ? "fp16" : "q8",
      device: device as any,
    } as any);
  }

  /** 加载模型（优先 WebGPU，失败/超时自动降级 WASM） */
  async load(opts: { dtype?: DType; forceDevice?: Device } = {}): Promise<LoadResult> {
    const dtype = opts.dtype ?? this.dtype;
    const canWebGPU = opts.forceDevice === "wasm" ? false : await detectWebGPU();
    const want: Device = opts.forceDevice ?? (canWebGPU ? "webgpu" : "wasm");

    this.dtype = dtype;
    this.fellBack = false;
    this.fallbackReason = undefined;

    const t0 = performance.now();
    let device: Device = want;

    if (want === "webgpu") {
      try {
        this.extractor = await withTimeout(
          this.buildPipeline(dtype, "webgpu"),
          WEBGPU_INIT_TIMEOUT_MS,
          "WebGPU 初始化超时"
        );
      } catch (e) {
        // 自动降级 WARSM
        device = "wasm";
        this.fellBack = true;
        this.fallbackReason = String((e as Error)?.message || e);
        this.extractor = await this.buildPipeline(dtype, "wasm");
      }
    } else {
      this.extractor = await this.buildPipeline(dtype, "wasm");
    }

    const loadMs = performance.now() - t0;
    this.device = device;

    // 推理一次以确定维度
    const probe = await this.embed(["维度探测"]);
    this.dimension = probe[0]?.length ?? 0;

    return {
      device,
      dtype,
      modelId: MODEL_ID,
      dimension: this.dimension,
      loadMs,
      fellBack: this.fellBack,
      fallbackReason: this.fallbackReason,
    };
  }

  isLoaded(): boolean {
    return this.extractor !== null;
  }

  /** 批量向量化（mean pooling + L2 归一化） */
  async embed(texts: string[]): Promise<number[][]> {
    if (!this.extractor) throw new Error("模型尚未加载");
    if (texts.length === 0) return [];
    const out = await this.extractor(texts, { pooling: "mean", normalize: true });
    const arr = out.tolist();
    // 单条时 tolist 可能返回 [dim] 或 [[dim]]，统一成 [[dim]]
    if (typeof arr[0] === "number") return [arr as number[]];
    return arr as number[][];
  }
}
