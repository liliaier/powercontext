#!/usr/bin/env node
/**
 * 后端数据层自测脚本（不依赖浏览器）
 *
 * 用途：在评测/复现环境里快速验证「seekdb 建集合 → 写入 → 混合检索 → 计数 → 清空」整条链路是否正常。
 * 由于浏览器端侧 embedding 无法在 Node 中复用，这里用随机向量占位（只验证 seekdb 数据通路与 SQL，
 * 不验证语义质量）。
 *
 * 用法：
 *   1) 先启动后端：npm start（或 SEEKDB_MODE=server / mock）
 *   2) 另开终端：node scripts/selftest.mjs
 *   可用环境变量 SELFTEST_URL（默认 http://localhost:3001）指定地址。
 */

const BASE = (process.env.SELFTEST_URL || "http://localhost:3001").replace(/\/$/, "");
const DIM = 384;

const randVec = () => Array.from({ length: DIM }, () => Math.random());

async function jsonFetch(path, init) {
  const res = await fetch(BASE + path, init);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(`HTTP ${res.status} ${path}: ${body?.error ?? ""}`);
  return body;
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

async function main() {
  console.log(`[selftest] 目标: ${BASE}`);

  // 1) 健康检查（应已连接 seekdb）
  const health = await jsonFetch("/api/health");
  console.log("[selftest] health:", JSON.stringify(health));
  assert(health.ok, `seekdb 未就绪: ${health.error ?? "unknown"}`);

  // 2) 清空，保证从干净状态开始
  await jsonFetch("/api/clear", { method: "POST" });

  // 3) 导入 3 篇文档（随机向量占位）
  const docs = [
    "OceanBase 是一款分布式关系数据库，支持高可用与在线扩展。",
    "seekdb 支持向量检索与全文检索的统一混合搜索。",
    "端侧推理指在用户浏览器本地运行 embedding 模型，保护隐私。",
  ];
  const items = docs.map((text, i) => ({
    id: `selftest_${i}`,
    text,
    embedding: randVec(),
    metadata: { source: "selftest" },
  }));
  const imported = await jsonFetch("/api/import", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items }),
  });
  console.log("[selftest] import:", JSON.stringify(imported));
  assert(imported.total >= items.length, "写入数量不符");

  // 4) 混合检索（全文命中 "seekdb" + 随机查询向量）
  const searched = await jsonFetch("/api/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      queryText: "seekdb 混合检索",
      queryEmbedding: randVec(),
      topK: 5,
    }),
  });
  console.log(
    "[selftest] search:",
    JSON.stringify(searched.results?.map((r) => ({ id: r.id, score: r.score })) ?? [])
  );
  assert(Array.isArray(searched.results) && searched.results.length > 0, "混合检索未返回结果");

  // 5) 文档列表 + 计数
  const listed = await jsonFetch("/api/documents");
  console.log("[selftest] documents:", listed.documents?.length ?? 0);
  assert((listed.documents?.length ?? 0) >= items.length, "文档列表数量不符");

  // 6) 清空收尾
  const cleared = await jsonFetch("/api/clear", { method: "POST" });
  assert(cleared.total === 0, "清空后计数不为 0");

  console.log(`\n[selftest] ✅ PASSED —— seekdb 建集合/写入/混合检索/计数/清空 全链路正常`);
}

main().catch((e) => {
  console.error(`\n[selftest] ❌ FAILED —— ${String(e?.message || e)}`);
  process.exit(1);
});
