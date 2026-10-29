/**
 * StepFun 题目后端入口
 *
 * - 提供 HTTP API（导入 / 混合检索 / 文档列表 / 统计 / 清空）
 * - 托管前端构建产物（web/dist）
 *
 * 启动：node src/index.mjs
 * 环境变量：
 *   PORT        服务端口（默认 3001）
 *   SEEKDB_PATH seekdb 数据目录（默认 ./seekdb.db）
 */

import express from "express";
import cors from "cors";
import path from "path";
import { fileURLToPath } from "url";
import {
  initDb,
  importDocuments,
  hybridSearch,
  count,
  listDocuments,
  clearCollection,
} from "./db.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT || 3001);
const DB_PATH = process.env.SEEKDB_PATH || path.join(__dirname, "..", "seekdb.db");
const WEB_DIST = path.join(__dirname, "..", "..", "web", "dist");

await initDb({ path: DB_PATH });

const app = express();
app.use(cors());
app.use(express.json({ limit: "100mb" }));

// ---- API ----
app.get("/api/health", async (_req, res) => {
  res.json({ ok: true, count: await count(), dbPath: DB_PATH });
});

app.post("/api/import", async (req, res) => {
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

app.post("/api/search", async (req, res) => {
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

app.get("/api/documents", async (_req, res) => {
  res.json({ ok: true, documents: await listDocuments() });
});

app.post("/api/clear", async (_req, res) => {
  res.json({ ok: true, total: await clearCollection() });
});

// ---- 前端静态资源 ----
app.use(express.static(WEB_DIST));
app.get("*", (_req, res) => {
  res.sendFile(path.join(WEB_DIST, "index.html"));
});

app.listen(PORT, () => {
  console.log(`[stepfun] seekdb 后端已启动: http://localhost:${PORT}`);
  console.log(`[stepfun] seekdb 数据目录: ${DB_PATH}`);
});
