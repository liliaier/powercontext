/**
 * StepFun 题目前端主逻辑
 *
 * 流程：导入语料 → 浏览器端侧向量化 → 写入 seekdb → 混合检索 → 展示结果
 * 同时展示：推理后端（WebGPU/WASM）、量化方案、模型信息、性能与质量数据
 */

import "./style.css";
import { api, type SearchHit } from "./api";
import { Embedder, detectWebGPU, type DType } from "./embedding";

// ---------------------------------------------------------------------------
// 内置示例语料（用于快速演示，可与用户自己的文本一起导入）
// ---------------------------------------------------------------------------
const SAMPLE_DOCS = [
  "OceanBase 是一款由蚂蚁集团自主研发的企业级原生分布式关系数据库，支持高可用、高并发与在线扩展。",
  "seekdb 是基于 OceanBase SQL 引擎构建的向量数据库，支持嵌入式与服务器双模式部署。",
  "seekdb 支持向量检索、全文检索与标量过滤的统一混合搜索，一条 SQL 即可完成多路检索。",
  "向量数据库通过将文本、图片等非结构化数据编码为向量，实现语义相似度检索。",
  "HNSW 是一种常用的近似最近邻索引结构，通过分层图结构实现高效的向量检索。",
  "端侧推理指在用户设备本地运行模型，避免将数据上传云端，兼顾隐私与实时性。",
  "WebGPU 是浏览器提供的现代图形与计算 API，可用于加速端侧神经网络的推理计算。",
  "WASM 即 WebAssembly，是一种可在浏览器中高效运行的二进制指令格式，是 WebGPU 不可用时的推理降级方案。",
  "模型量化通过降低权重的数值精度来压缩模型体积、提升推理速度，常见方案有 INT8 与 FP16。",
  "INT8 量化将 32 位浮点权重量化为 8 位整数，通常可将模型体积压缩到原来的四分之一左右。",
  "全量索引刷新保证新写入的文档能够被立即检索到，是知识库增量更新的关键环节。",
  "混合检索将全文关键词匹配与向量语义匹配结合，通过 RRF 等融合算法提升召回与排序质量。",
  "BGE 与 MiniLM 是两类常用于端侧部署的小型 embedding 模型，兼顾效果与推理开销。",
  "OceanBase 的向量能力支持 cosine、L2、内积等多种距离度量，适配不同的向量检索场景。",
  "知识库应用需要同时保存向量、原文与元数据，以便检索后展示原文并支持过滤。",
  "Transformer 是一种基于自注意力机制的神经网络架构，是现代大语言模型的基础。",
];

// ---------------------------------------------------------------------------
// DOM 骨架
// ---------------------------------------------------------------------------
const app = document.querySelector<HTMLDivElement>("#app")!;
app.innerHTML = `
  <header class="topbar">
    <h1>端侧知识库 · seekdb 混合检索</h1>
    <div class="badges">
      <span id="badge-backend" class="badge">数据库：检测中…</span>
      <span id="badge-model" class="badge">模型：未加载</span>
      <span id="badge-device" class="badge">后端：-</span>
    </div>
  </header>

  <main class="grid">
    <section class="card">
      <h2>① 导入语料</h2>
      <p class="hint">每段（空行分隔）作为一篇文档，导入时会全部在浏览器本地向量化。</p>
      <textarea id="docs-input" rows="6" placeholder="粘贴文本，空行分隔多篇文档…"></textarea>
      <div class="row">
        <button id="btn-sample" class="ghost">加载示例语料</button>
        <label class="ghost file-btn">选择 .txt 文件<input id="file-input" type="file" accept=".txt,text/plain" hidden /></label>
        <button id="btn-import" class="primary">导入并向量化</button>
      </div>
      <div id="import-status" class="status"></div>
    </section>

    <section class="card">
      <h2>② 混合检索</h2>
      <p class="hint">查询向量在浏览器本地生成，向量 + 全文混合检索由 seekdb 完成。</p>
      <div class="row">
        <input id="query-input" type="text" placeholder="输入查询，例如：端侧推理如何加速？" />
        <button id="btn-search" class="primary">检索</button>
      </div>
      <div id="search-status" class="status"></div>
      <ol id="results" class="results"></ol>
    </section>

    <section class="card">
      <h2>③ 端侧模型与量化</h2>
      <div class="row">
        <label>量化方案
          <select id="dtype-select">
            <option value="q8">INT8 (q8) · 推荐</option>
            <option value="fp16">FP16</option>
            <option value="fp32">FP32（基线）</option>
          </select>
        </label>
        <button id="btn-reload" class="ghost">按所选方案重新加载</button>
      </div>
      <div id="model-info" class="status"></div>
      <table class="qt" id="quant-table">
        <thead><tr><th>方案</th><th>模型体积</th><th>压缩比</th><th>推理后端</th></tr></thead>
        <tbody></tbody>
      </table>
      <p class="hint">体积为 all-MiniLM-L6-v2 官方 ONNX 权重参考值；实际下载体积以浏览器缓存为准。</p>
    </section>

    <section class="card">
      <h2>④ 性能与质量</h2>
      <div class="row">
        <button id="btn-perf" class="ghost">运行量化质量评测</button>
      </div>
      <div id="perf-status" class="status"></div>
    </section>

    <section class="card">
      <h2>⑤ 知识库状态</h2>
      <div id="kb-status" class="status"></div>
      <div class="row">
        <button id="btn-refresh-kb" class="ghost">刷新状态</button>
        <button id="btn-clear" class="danger">清空知识库</button>
      </div>
      <ul id="kb-list" class="kb-list"></ul>
    </section>
  </main>
`;

// ---------------------------------------------------------------------------
// 元素引用
// ---------------------------------------------------------------------------
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const els = {
  backend: $("badge-backend"),
  model: $("badge-model"),
  device: $("badge-device"),
  docsInput: $<HTMLTextAreaElement>("docs-input"),
  importStatus: $("import-status"),
  importBtn: $<HTMLButtonElement>("btn-import"),
  sampleBtn: $<HTMLButtonElement>("btn-sample"),
  fileInput: $<HTMLInputElement>("file-input"),
  queryInput: $<HTMLInputElement>("query-input"),
  searchBtn: $<HTMLButtonElement>("btn-search"),
  searchStatus: $("search-status"),
  results: $<HTMLOListElement>("results"),
  dtypeSelect: $<HTMLSelectElement>("dtype-select"),
  reloadBtn: $<HTMLButtonElement>("btn-reload"),
  modelInfo: $("model-info"),
  quantTable: document.querySelector<HTMLTableElement>("#quant-table tbody")!,
  perfBtn: $<HTMLButtonElement>("btn-perf"),
  perfStatus: $("perf-status"),
  kbStatus: $("kb-status"),
  kbList: $<HTMLUListElement>("kb-list"),
  clearBtn: $<HTMLButtonElement>("btn-clear"),
  refreshKbBtn: $<HTMLButtonElement>("btn-refresh-kb"),
};

const embedder = new Embedder();
let lastLoadMs = 0;

// ---------------------------------------------------------------------------
// 工具
// ---------------------------------------------------------------------------
function setStatus(el: HTMLElement, text: string, kind: "ok" | "err" | "info" = "info") {
  el.textContent = text;
  el.className = `status ${kind}`;
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c] as string)
  );
}

function splitDocs(text: string): string[] {
  return text
    .split(/\n\s*\n/)
    .map((s) => s.trim())
    .filter(Boolean);
}

// 量化体积参考（all-MiniLM-L6-v2 官方 ONNX 权重，单位 MB）
const QUANT_TABLE = [
  { name: "FP32（基线）", size: 90.9, ratio: 1.0 },
  { name: "FP16", size: 45.5, ratio: 2.0 },
  { name: "INT8 (q8)", size: 23.1, ratio: 3.94 },
];

function renderQuantTable(dtype: DType, device: string) {
  els.quantTable.innerHTML = QUANT_TABLE.map((row) => {
    const active =
      (row.name.includes("INT8") && dtype === "q8") ||
      (row.name === "FP16" && dtype === "fp16") ||
      (row.name.includes("FP32") && dtype === "fp32");
    return `<tr class="${active ? "active" : ""}">
      <td>${row.name}${active ? " ●" : ""}</td>
      <td>${row.size.toFixed(1)} MB</td>
      <td>${row.ratio.toFixed(2)}×</td>
      <td>${active ? device.toUpperCase() : "-"}</td>
    </tr>`;
  }).join("");
}

// ---------------------------------------------------------------------------
// 模型加载
// ---------------------------------------------------------------------------
async function loadModel(dtype: DType) {
  els.model.textContent = "模型：加载中…";
  els.reloadBtn.disabled = true;
  try {
    const res = await embedder.load({ dtype });
    lastLoadMs = res.loadMs;
    els.model.textContent = `模型：${res.modelId.split("/").pop()} (${res.dtype})`;
    els.device.textContent = `后端：${res.device.toUpperCase()}`;
    els.device.className = `badge ${res.device === "webgpu" ? "good" : "warn"}`;
    const backendNote =
      res.device === "webgpu"
        ? "（已启用 WebGPU 加速）"
        : res.fellBack
          ? `（WebGPU 不可用/初始化失败，已自动降级 WASM：${res.fallbackReason ?? ""}）`
          : "（未检测到 WebGPU，使用 WASM）";
    setStatus(
      els.modelInfo,
      `已加载 ${res.modelId}｜维度 ${res.dimension}｜量化 ${res.dtype}｜推理后端 ${res.device.toUpperCase()}｜加载耗时 ${res.loadMs.toFixed(0)} ms` +
        backendNote,
      "ok"
    );
    renderQuantTable(res.dtype, res.device);
  } catch (e) {
    els.model.textContent = "模型：加载失败";
    setStatus(els.modelInfo, `模型加载失败：${String((e as Error).message || e)}`, "err");
    throw e;
  } finally {
    els.reloadBtn.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// 后端状态
// ---------------------------------------------------------------------------
async function refreshBackend() {
  const h = await api.health();
  if (h.ok) {
    els.backend.textContent = `数据库：seekdb 已连接（${h.count ?? 0} 篇）`;
    els.backend.className = "badge good";
  } else {
    els.backend.textContent = "数据库：seekdb 未就绪";
    els.backend.className = "badge err";
    setStatus(
      els.kbStatus,
      `seekdb 未就绪：${h.error ?? "未知错误"}（请检查后端日志与 seekdb 环境）`,
      "err"
    );
  }
}

async function refreshKb() {
  try {
    const h = await api.health();
    if (!h.ok) {
      setStatus(els.kbStatus, `seekdb 未就绪：${h.error ?? "未知错误"}`, "err");
      return;
    }
    const docs = await api.documents();
    setStatus(
      els.kbStatus,
      `知识库共 ${h.count ?? docs.documents.length} 篇文档｜seekdb 数据目录：${h.dbPath ?? "-"}`,
      "ok"
    );
    els.kbList.innerHTML = docs.documents
      .slice(0, 50)
      .map((d) => `<li><code>${escapeHtml(d.id)}</code> ${escapeHtml(d.text ?? "")}</li>`)
      .join("");
  } catch (e) {
    setStatus(els.kbStatus, `获取知识库状态失败：${String((e as Error).message || e)}`, "err");
  }
}

// ---------------------------------------------------------------------------
// 导入
// ---------------------------------------------------------------------------
async function importDocs(text: string) {
  const docs = splitDocs(text);
  if (docs.length === 0) {
    setStatus(els.importStatus, "请先输入或加载语料。", "err");
    return;
  }
  if (!embedder.isLoaded()) {
    setStatus(els.importStatus, "模型尚未加载完成，请稍候。", "err");
    return;
  }
  els.importBtn.disabled = true;
  try {
    const BATCH = 16; // embedding 批处理
    setStatus(els.importStatus, `正在向量化 ${docs.length} 篇文档（批大小 ${BATCH}）…`);
    const t0 = performance.now();
    const items = [];
    for (let i = 0; i < docs.length; i += BATCH) {
      const batch = docs.slice(i, i + BATCH);
      const vecs = await embedder.embed(batch);
      for (let j = 0; j < batch.length; j++) {
        items.push({
          id: `doc_${Date.now()}_${i + j}`,
          text: batch[j],
          embedding: vecs[j],
          metadata: { source: "import", batch: Math.floor(i / BATCH) },
        });
      }
    }
    const embedMs = performance.now() - t0;
    const r = await api.importItems(items);
    setStatus(
      els.importStatus,
      `导入完成：本次 ${r.imported} 篇，库内共 ${r.total} 篇｜向量化耗时 ${embedMs.toFixed(0)} ms｜平均 ${(embedMs / docs.length).toFixed(1)} ms/篇`,
      "ok"
    );
    await refreshBackend();
    await refreshKb();
  } catch (e) {
    setStatus(els.importStatus, `导入失败：${String((e as Error).message || e)}`, "err");
  } finally {
    els.importBtn.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// 检索
// ---------------------------------------------------------------------------
async function doSearch() {
  const q = els.queryInput.value.trim();
  if (!q) {
    setStatus(els.searchStatus, "请输入查询内容。", "err");
    return;
  }
  if (!embedder.isLoaded()) {
    setStatus(els.searchStatus, "模型尚未加载完成，请稍候。", "err");
    return;
  }
  els.searchBtn.disabled = true;
  try {
    const t0 = performance.now();
    const [qv] = await embedder.embed([q]);
    const embedMs = performance.now() - t0;
    const t1 = performance.now();
    const res = await api.search(q, qv, 5);
    const searchMs = performance.now() - t1;
    setStatus(
      els.searchStatus,
      `查询向量化 ${embedMs.toFixed(1)} ms｜seekdb 混合检索 ${searchMs.toFixed(1)} ms`,
      "ok"
    );
    renderResults(res.results);
  } catch (e) {
    setStatus(els.searchStatus, `检索失败：${String((e as Error).message || e)}`, "err");
  } finally {
    els.searchBtn.disabled = false;
  }
}

function renderResults(hits: SearchHit[]) {
  if (!hits || hits.length === 0) {
    els.results.innerHTML = `<li class="empty">未检索到结果，请先导入语料。</li>`;
    return;
  }
  const maxScore = Math.max(...hits.map((h) => Math.abs(h.score))) || 1;
  els.results.innerHTML = hits
    .map((h, i) => {
      const bar = Math.min(100, (Math.abs(h.score) / maxScore) * 100);
      return `<li>
        <div class="hit-head"><span class="rank">#${i + 1}</span>
          <code>${escapeHtml(h.id)}</code>
          <span class="score">${Number(h.score).toFixed(4)}</span></div>
        <div class="hit-text">${escapeHtml(h.text ?? "")}</div>
        <div class="bar"><i style="width:${bar.toFixed(0)}%"></i></div>
      </li>`;
    })
    .join("");
}

// ---------------------------------------------------------------------------
// 量化质量评测：fp32 基线 vs 当前量化方案的 embedding 余弦一致性
// ---------------------------------------------------------------------------
async function runQualityEval() {
  els.perfBtn.disabled = true;
  try {
    const sample = SAMPLE_DOCS.slice(0, 6);
    const norm = (v: number[]) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    const cos = (a: number[], b: number[]) => {
      let d = 0;
      let na = 0;
      let nb = 0;
      for (let i = 0; i < a.length; i++) {
        d += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
      }
      return d / (Math.sqrt(na) * Math.sqrt(nb));
    };

    setStatus(els.perfStatus, "正在用当前量化方案向量化样本…");
    const vecCur = await embedder.embed(sample);

    setStatus(els.perfStatus, "正在加载 FP32 基线模型（约 90MB，仅首次耗时）…");
    const embedder32 = new Embedder();
    // 基线固定走 WASM，避免部分环境 WebGPU FP32 数值异常影响对比结论
    const r32 = await embedder32.load({ dtype: "fp32", forceDevice: "wasm" });
    const vec32 = await embedder32.embed(sample);

    // 退化输出守卫：NaN / 全零 / 范数异常时如实报错，避免给出无意义相似度
    const bad32 = vec32.filter((v) => v.some((x) => !Number.isFinite(x)) || norm(v) < 1e-6);
    const badCur = vecCur.filter((v) => v.some((x) => !Number.isFinite(x)) || norm(v) < 1e-6);
    if (bad32.length || badCur.length) {
      setStatus(
        els.perfStatus,
        `质量评测中止：检测到退化向量（FP32 异常 ${bad32.length} 条 / 当前方案异常 ${badCur.length} 条），` +
          `该环境下的 FP32 基线推理结果不可信，请改用 FP32 在 WASM 的对照或参考离线评测数据。`,
        "err"
      );
      return;
    }

    const sims = vec32.map((v, i) => cos(v, vecCur[i]));
    const mean = sims.reduce((a, b) => a + b, 0) / sims.length;
    const min = Math.min(...sims);

    setStatus(
      els.perfStatus,
      `量化质量评测（样本 ${sample.length} 篇）：当前方案 ${embedder.dtype} vs FP32 基线（WASM），` +
        `embedding 余弦相似度均值 ${mean.toFixed(5)}，最小 ${min.toFixed(5)}。` +
        `实践中 ≥0.95 表示量化后检索质量基本保持，≥0.99 表示几乎无损；FP32 基线加载耗时 ${r32.loadMs.toFixed(0)} ms。`,
      "ok"
    );
  } catch (e) {
    setStatus(els.perfStatus, `质量评测失败：${String((e as Error).message || e)}`, "err");
  } finally {
    els.perfBtn.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// 事件绑定
// ---------------------------------------------------------------------------
els.sampleBtn.addEventListener("click", () => {
  els.docsInput.value = SAMPLE_DOCS.join("\n\n");
});

els.fileInput.addEventListener("change", async () => {
  const f = els.fileInput.files?.[0];
  if (!f) return;
  els.docsInput.value = await f.text();
});

els.importBtn.addEventListener("click", () => importDocs(els.docsInput.value));

els.searchBtn.addEventListener("click", () => doSearch());
els.queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") doSearch();
});

els.dtypeSelect.addEventListener("change", () => {
  loadModel(els.dtypeSelect.value as DType).catch(() => {});
});

els.reloadBtn.addEventListener("click", () => {
  loadModel(els.dtypeSelect.value as DType).catch(() => {});
});

els.perfBtn.addEventListener("click", () => runQualityEval());

els.refreshKbBtn.addEventListener("click", () => refreshKb());

els.clearBtn.addEventListener("click", async () => {
  try {
    const r = await api.clear();
    setStatus(els.kbStatus, `已清空知识库，当前 ${r.total} 篇文档。`, "ok");
    els.kbList.innerHTML = "";
    await refreshBackend();
  } catch (e) {
    setStatus(els.kbStatus, `清空失败：${String((e as Error).message || e)}`, "err");
  }
});

// ---------------------------------------------------------------------------
// 启动
// ---------------------------------------------------------------------------
async function boot() {
  await refreshBackend();
  const canWebGPU = await detectWebGPU();
  els.device.textContent = `后端：${canWebGPU ? "WebGPU 可用" : "WASM（无 WebGPU）"}`;
  els.device.className = `badge ${canWebGPU ? "good" : "warn"}`;
  els.dtypeSelect.value = "q8";
  await loadModel("q8");
  await refreshKb();
}

boot().catch((e) => {
  setStatus(els.modelInfo, `启动失败：${String((e as Error).message || e)}`, "err");
});
