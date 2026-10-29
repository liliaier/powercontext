/**
 * StepFun 题目后端入口
 *
 * - 提供 HTTP API（导入 / 混合检索 / 文档列表 / 统计 / 清空）
 * - 托管前端构建产物（web/dist）
 * - seekdb 初始化失败时不崩溃：仍托管前端，API 返回明确错误，便于定位环境问题
 *
 * 启动：node src/index.mjs
 * 环境变量：
 *   PORT        服务端口（默认 3001）
 *   SEEKDB_MODE embedded / server / mock（默认 embedded）
 *   SEEKDB_PATH seekdb 数据目录（默认 ./seekdb.db）
 */

import express from "express";
import cors from "cors";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";
import {
  initDb,
  importDocuments,
  hybridSearch,
  count,
  listDocuments,
  clearCollection,
  currentMode,
} from "./db.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT || 3001);
const DB_PATH = process.env.SEEKDB_PATH || path.join(__dirname, "..", "seekdb.db");
const WEB_DIST = path.join(__dirname, "..", "..", "web", "dist");

// ---- 初始化 seekdb（带重试；失败不致命） ----
let dbError = null;

async function initDbWithRetry(attempts = 3) {
  for (let i = 1; i <= attempts; i++) {
    try {
      await initDb({ path: DB_PATH });
      dbError = null;
      console.log(`[stepfun] seekdb 已就绪（mode=${currentMode()}）`);
      return;
    } catch (e) {
      dbError = String(e?.message || e);
      console.error(`[stepfun] seekdb 初始化失败（第 ${i}/${attempts} 次）：${dbError}`);
      if (i < attempts) await new Promise((r) => setTimeout(r, 800 * i));
    }
  }
}

await initDbWithRetry();

if (!fs.existsSync(path.join(WEB_DIST, "index.html"))) {
  console.warn(
    `[stepfun] 警告：未找到前端构建产物 ${WEB_DIST}/index.html，请先执行 \`npm run build\`（或根目录 \`npm run setup\`）。`
  );
}

const app = express();
app.use(cors());
app.use(express.json({ limit: "100mb" }));

/** 若 seekdb 未就绪，返回 503 */
function requireDb(_req, res, next) {
  if (dbError) {
    return res.status(503).json({
      error: `seekdb 未就绪：${dbError}。请检查 SEEKDB_MODE / 原生绑定是否可下载（或改用 server 模式）。`,
    });
  }
  next();
}

// ---- API ----
app.get("/api/health", async (_req, res) => {
  if (dbError) {
    return res.status(503).json({ ok: false, mode: currentMode(), error: dbError, dbPath: DB_PATH });
  }
  try {
    res.json({ ok: true, mode: currentMode(), count: await count(), dbPath: DB_PATH });
  } catch (e) {
    res.status(503).json({ ok: false, mode: currentMode(), error: String(e?.message || e), dbPath: DB_PATH });
  }
});

app.post("/api/import", requireDb, async (req, res) => {
  try {
    const items = req.body?.items;
    if (!Array.isArray(items) || items.length === 0) {
      return res.status(400).json({ error: "items 必须是非空数组" });
    }
    const total = await importDocuments(items);
    res.json({ ok: true, imported: items.length, total });
  } catch (e) {
    res.status(500).json({ error: String(e?.message || e) });
  }
});

app.post("/api/search", requireDb, async (req, res) => {
  try {
    const { queryText, queryEmbedding, topK } = req.body ?? {};
    if (!Array.isArray(queryEmbedding) || queryEmbedding.length === 0) {
      return res.status(400).json({ error: "queryEmbedding 必填" });
    }
    const results = await hybridSearch({
      queryText: queryText ?? "",
      queryEmbedding,
      topK: Number(topK) || 5,
    });
    res.json({ ok: true, results });
  } catch (e) {
    res.status(500).json({ error: String(e?.message || e) });
  }
});

app.get("/api/documents", requireDb, async (_req, res) => {
  res.json({ ok: true, documents: await listDocuments() });
});

app.post("/api/clear", requireDb, async (_req, res) => {
  res.json({ ok: true, total: await clearCollection() });
});

// ---- 前端静态资源 ----
app.use(express.static(WEB_DIST));
app.get("*", (_req, res) => {
  res.sendFile(path.join(WEB_DIST, "index.html"));
});

app.listen(PORT, () => {
  console.log(`[stepfun] 后端已启动: http://localhost:${PORT}`);
  console.log(`[stepfun] 模式: ${currentMode()}，seekdb 数据目录: ${DB_PATH}`);
});
