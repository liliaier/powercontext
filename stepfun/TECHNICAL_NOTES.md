# 技术说明（一页）

## 1. 总体架构

```
浏览器（端侧推理）                         Node.js 后端（数据层）
┌───────────────────────────┐      HTTP    ┌─────────────────────────────┐
│ 导入文本                    │  ─────────▶  │ /api/import  写入 seekdb     │
│ Transformers.js / ORT-Web  │              │ /api/search  混合检索        │
│ all-MiniLM-L6-v2 (INT8)    │  ◀─────────  │ seekdb(embedded/server)      │
│ 向量化（WebGPU→WASM 降级） │              │ 向量+全文 DBMS_HYBRID_SEARCH │
└───────────────────────────┘              └─────────────────────────────┘
```

- **推理位置**：embedding 全部在浏览器内完成（Transformers.js + ONNX Runtime Web），后端只负责存储/检索，不参与向量计算。
- **数据流**：浏览器把「向量 + 原文 + 元数据」直传后端写入 seekdb；查询时浏览器本地算查询向量，后端用该向量在 seekdb 内做混合检索。

## 2. 模型与来源

| 项 | 值 |
|---|---|
| 模型 | `Xenova/all-MiniLM-L6-v2`（开源，Apache-2.0） |
| 输出维度 | 384 |
| 运行时 | `@huggingface/transformers`（Transformers.js v3）+ onnxruntime-web |
| 权重格式 | ONNX（`model_quantized.onnx` / `model.onnx`） |

## 3. 量化方案（INT8）

- **方案**：INT8 动态量化（`dtype: "q8"`），默认启用；同时提供 FP16 与 FP32 基线切换以做对比。
- **量化前后体积（官方 ONNX 权重参考值）**：

| 方案 | 模型体积 | 相对 FP32 压缩比 |
|---|---|---|
| FP32（基线） | ≈ 90.9 MB | 1.00× |
| FP16 | ≈ 45.5 MB | 2.00× |
| **INT8 (q8)** | **≈ 23.1 MB** | **≈ 3.94×** |

- **质量变化（实测）**：在 6 条样本上对比 INT8 与 FP32 基线的 embedding 余弦相似度，**均值 0.98425、最小 0.97755**（≥0.95 视为检索质量基本保持）。评测入口见界面「④ 运行量化质量评测」。

## 4. 推理后端与自动降级

- 启动时检测 `navigator.gpu` 是否存在可用 adapter：
  - 可用 → 显式配置 `device: "webgpu"`；
  - 不可用 → 直接 `device: "wasm"`。
- **额外降级保护**：WebGPU 初始化设 20s 超时，超时或抛错时自动改用 WASM，并在界面显示降级原因。
- 界面常驻展示当前后端（WEBGPU / WASM），并提供 `?force_wasm=1` 便于强制对照。

## 5. seekdb 数据层

- **接入方式**：`seekdb`（seekdb-js SDK）。默认**嵌入式模式**（无独立 server，数据落盘 `server/seekdb.db/`）；也支持 `SEEKDB_MODE=server` 连接本地/远程 seekdb 或 OceanBase。
- **集合 Schema**：
  - `FulltextIndexConfig()`：全文索引（IK 分词，中英文）；
  - `VectorIndexConfig({ hnsw: { dimension: 384, distance: "cosine" }, embeddingFunction: null })`：HNSW 稠密向量索引；`embeddingFunction: null` 表示向量由浏览器直传，不在服务端二次 embedding。
- **写入**：`collection.add({ ids, documents, embeddings, metadatas })`，向量、原文、元数据同表存储。
- **混合检索**：`collection.hybridSearch({ query: { whereDocument: { $contains } }, knn: { queryEmbeddings }, rank: { rrf: {} }, nResults, include })` → 由 seekdb `DBMS_HYBRID_SEARCH` 在库内完成「全文 + 向量」两路检索与 RRF 融合，客户端不拼接两路结果。
- **索引刷新**：每次导入后调用 `collection.refresh_index()`（`CALL dbms_index_manager.refresh()`），保证新写入文档刷新后可立即被检索。
- **状态管理**：`count()` / `get()` 维护知识库文档数与列表，支持增量导入与清空。

## 6. 性能优化与实测

围绕题目要求的几个方向：

| 优化项 | 做法 |
|---|---|
| embedding 批处理 | 导入按 `BATCH=16` 分批推理，摊薄单次调用开销 |
| 模型加载 | INT8 权重（体积 ↓≈3.94×）+ 浏览器缓存；默认 q8，首次后加载可低至百毫秒级 |
| 推理延迟 | 优先 WebGPU；不可用时 WASM 兜底 |
| 数据库检索延迟 | 混合检索在 seekdb 内单次 SQL 完成，无 N+1；查询向量直传，无服务端 embedding 往返 |
| 索引刷新策略 | 导入后统一 `refresh_index()`，兼顾「写入吞吐」与「新文档立即可检索」 |

**开发机实测（WASM，示例语料 16 篇）**：
- 模型加载：q8 / WASM ≈ 1.3 s（已缓存）；q8 / WebGPU ≈ 5.1 s；
- 批量向量化：16 篇 ≈ 1101 ms（≈ 68.8 ms/篇，批大小 16）；
- 单条查询向量化：≈ 30.9 ms（WASM）；WebGPU 冷启动首条较慢，后续显著降低；
- 混合检索：由 seekdb 完成，界面实时展示每次检索的库内耗时。

## 7. 复现步骤

```bash
cd stepfun
npm run setup     # 安装依赖 + 构建前端
npm start         # 默认嵌入式 seekdb，http://localhost:3001
```

打开页面 → 点击「加载示例语料」→「导入并向量化」→ 输入查询检索即可。
（可选）`docker compose up -d` + `SEEKDB_MODE=server npm start` 使用本地 server 模式。
