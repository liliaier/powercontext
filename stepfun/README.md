# 端侧 AI 推理性能优化与 seekdb 混合检索

> OceanBase × 阶跃星辰（StepFun）赛题提交物
> 在浏览器本地完成向量化，用 seekdb 完成向量 + 全文混合检索的迷你个人知识库。

## 一、项目简介

一个可运行的 Web 应用，完整实现「导入语料 → 浏览器本地向量化 → 写入 seekdb → 混合检索 → 展示结果」：

- **端侧推理**：用 `@huggingface/transformers`（Transformers.js / ONNX Runtime Web）在**用户浏览器内**执行 embedding 推理，文档与查询向量化均在本地完成，**不调用任何云端推理 API**。
- **模型量化**：默认使用 `all-MiniLM-L6-v2` 的 **INT8 (q8)** 量化权重，并提供 FP16 / FP32 基线切换，便于量化前后对比。
- **推理后端**：优先 **WebGPU**，显式配置；WebGPU 不可用或初始化失败/超时时**自动降级 WASM**，界面明确提示当前后端。
- **seekdb 数据层**：通过 `seekdb`（seekdb-js SDK）连接 seekdb，保存向量、原文与元数据；检索由 seekdb 的 `DBMS_HYBRID_SEARCH` 完成**向量 + 全文混合检索**（RRF 融合），不在客户端拼接两路结果。
- **状态与刷新**：支持增量导入、写入后 `refresh_index()` 保证新文档立即可检索、知识库状态查询与清空。

## 二、目录结构

```
stepfun/
├── README.md                 # 本文件：启动与验收说明
├── TECHNICAL_NOTES.md        # 一页技术说明（模型/量化/后端/seekdb/优化/实测）
├── package.json              # 根脚本（setup / start）
├── docker-compose.yml        # 可选：以「本地 server 方式」部署 seekdb
├── server/                   # Node.js 后端
│   ├── package.json
│   └── src/
│       ├── index.mjs         # HTTP 服务：导入 / 混合检索 / 文档 / 状态 / 清空 + 托管前端
│       └── db.mjs            # seekdb 数据层封装（embedded / server / mock 三模式）
└── web/                      # 前端（Vite + TypeScript + Transformers.js）
    ├── package.json
    ├── vite.config.ts
    ├── index.html
    └── src/
        ├── main.ts           # UI 与业务流程
        ├── embedding.ts      # 端侧 embedding（WebGPU/WASM、量化、降级）
        ├── api.ts            # 后端 API 客户端
        └── style.css
```

## 三、启动方式

### 方式 A（推荐，默认）：seekdb 嵌入式模式

无需单独部署数据库，seekdb 以嵌入式方式在后端进程内运行，数据落盘到 `server/seekdb.db/`。

> 环境要求：Node.js ≥ 20（推荐 **Node 20 LTS**，与 seekdb-js 官方 CI 构建版本一致）。
> seekdb 原生绑定会在首次使用时自动下载并缓存（`@seekdb/js-bindings`）。

```bash
cd stepfun
npm run setup     # 安装 server/web 依赖并构建前端（首次较慢）
npm start         # 启动后端，默认 http://localhost:3001
```

浏览器打开 <http://localhost:3001> 即可使用。

### 方式 B（可选）：seekdb 本地 server 模式

若评测环境不便使用嵌入式原生绑定，可用 Docker 启动 seekdb server，再让后端以 server 模式连接：

```bash
cd stepfun
docker compose up -d          # 启动 seekdb（MySQL 协议，端口 2881）
npm run setup
SEEKDB_MODE=server npm start  # 以上线 seekdb server
```

### 方式 C（开发预览）：mock 模式

仅用于前端流程联调（内存模拟存储/检索，**无真实 seekdb**，不可用于评测）：

```bash
SEEKDB_MODE=mock npm start
```

### 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `PORT` | `3001` | 后端端口 |
| `SEEKDB_MODE` | `embedded` | `embedded` / `server` / `mock` |
| `SEEKDB_PATH` | `./seekdb.db` | 嵌入式模式数据目录 |
| `SEEKDB_HOST` / `SEEKDB_PORT` / `SEEKDB_USER` / `SEEKDB_PASSWORD` | `127.0.0.1` / `2881` / `root` / 空 | server 模式连接参数 |
| `SEEKDB_DATABASE` | `stepfun_kb` | 数据库名 |

## 四、使用流程

1. 页面顶部状态栏确认：数据库已连接、模型已加载、当前推理后端（WebGPU / WASM）。
2. 在「① 导入语料」中点击「加载示例语料」或粘贴/选择自己的文本（空行分隔多篇文档），点击「导入并向量化」。
3. 在「② 混合检索」输入查询，点击「检索」，查看按混合检索排序的结果与对应原文。
4. 在「③ 端侧模型与量化」切换 INT8 / FP16 / FP32 并重新加载，查看模型体积与压缩比。
5. 在「④ 性能与质量」运行量化质量评测（FP32 基线 vs 当前量化方案）。
6. 在「⑤ 知识库状态」查看/刷新/清空知识库。

> 排障：URL 加 `?force_wasm=1` 可强制 WASM 后端（用于对照 WebGPU 与 WASM）。

## 五、验收对照（对应题目 3.1）

| 验收项 | 本项目的实现 |
|---|---|
| 完整流程 | 导入文本 → 浏览器向量化 → 写入 seekdb → 查询 → 展示排序结果与原文 |
| 本地推理 | embedding 全部在浏览器内由 Transformers.js/ONNX Runtime Web 计算，无云端推理调用 |
| 模型量化 | 默认 INT8(q8)；支持 FP16/FP32；见 `TECHNICAL_NOTES.md` 的体积/压缩比/质量数据 |
| 后端与降级 | 优先 WebGPU；不可用/失败/超时自动降级 WASM，界面明确提示 |
| seekdb 数据层 | 向量、原文、元数据存于 seekdb；混合检索由 seekdb `DBMS_HYBRID_SEARCH` 完成 |
| 状态与刷新 | 增量导入；写入后 `refresh_index()`；知识库状态查询/清空 |
| 可复现性 | 上述启动方式 + 环境变量说明，可一键启动应用与数据库 |

## 六、技术说明

见 [TECHNICAL_NOTES.md](./TECHNICAL_NOTES.md)。
