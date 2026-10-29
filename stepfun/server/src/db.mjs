/**
 * seekdb 数据层封装（StepFun 题目）
 *
 * 支持三种运行模式（通过 SEEKDB_MODE 环境变量选择）：
 * - embedded（默认）：嵌入式 seekdb，数据落盘本地，无独立 server（Linux/CI 上开箱即用）
 * - server：连接本地/远程 seekdb 或 OceanBase（MySQL 协议）
 * - mock：内存模拟，仅用于本地联调前端流程（不做真实存储/检索，禁止用于评测）
 *
 * 职责：
 * - 建集合：全文索引（IK 分词）+ 稠密向量索引（HNSW, cosine）
 * - 导入：浏览器算好的向量 + 原文 + 元数据 直接写入 seekdb
 * - 检索：向量 + 全文「混合检索」，由 seekdb 的 DBMS_HYBRID_SEARCH 完成（RRF 融合）
 * - 索引刷新：写入后 refresh_index，保证新文档立即可检索
 */

import {
  SeekdbClient,
  Schema,
  FulltextIndexConfig,
  VectorIndexConfig,
} from "seekdb";

const DIMENSION = 384; // all-MiniLM-L6-v2 输出维度
const COLLECTION_NAME = "documents";

const MODE = (process.env.SEEKDB_MODE || "embedded").toLowerCase();
const HOST = process.env.SEEKDB_HOST || "127.0.0.1";
const PORT = Number(process.env.SEEKDB_PORT || 2881);
const USER = process.env.SEEKDB_USER || "root";
const PASSWORD = process.env.SEEKDB_PASSWORD || "";
const DATABASE = process.env.SEEKDB_DATABASE || "stepfun_kb";
const DB_PATH = process.env.SEEKDB_PATH || "./seekdb.db";

let client = null;
let collection = null;

// ---- 内存 mock（仅本地联调） ----
const mockStore = [];

function cosineSimilarity(a, b) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

/** 简单关键词命中（模拟全文检索分支） */
function keywordHit(queryText, docText) {
  if (!queryText) return 0;
  const terms = queryText.toLowerCase().split(/[\s,.;:?!，。；：？！、]+/).filter(Boolean);
  if (terms.length === 0) return 0;
  const lower = (docText || "").toLowerCase();
  return terms.filter((t) => lower.includes(t)).length / terms.length;
}

export function currentMode() {
  return MODE;
}

/** 初始化（幂等） */
export async function initDb(options = {}) {
  if (MODE === "mock") {
    mockStore.length = 0;
    return { client: null, collection: null };
  }

  if (client && collection) return { client, collection };

  const database = options.database || DATABASE;
  if (MODE === "server") {
    client = new SeekdbClient({
      host: options.host || HOST,
      port: options.port || PORT,
      user: options.user || USER,
      password: options.password || PASSWORD,
      database,
    });
  } else {
    client = new SeekdbClient({ path: options.path || DB_PATH, database });
  }

  const schema = new Schema()
    .createIndex(new FulltextIndexConfig())
    .createIndex(
      new VectorIndexConfig({
        hnsw: { dimension: DIMENSION, distance: "cosine" },
        embeddingFunction: null,
      })
    );

  collection = await client.getOrCreateCollection({
    name: COLLECTION_NAME,
    schema,
  });

  return { client, collection };
}

async function refreshIndex() {
  if (MODE === "mock") return;
  try {
    await collection.refresh_index();
  } catch {
    // 版本不支持时静默跳过
  }
}

/** 批量导入文档 */
export async function importDocuments(items) {
  if (!items || items.length === 0) return await count();

  if (MODE === "mock") {
    for (const it of items) {
      mockStore.push({
        id: it.id,
        text: it.text,
        embedding: it.embedding,
        metadata: it.metadata ?? {},
      });
    }
    return mockStore.length;
  }

  const ids = items.map((i) => i.id);
  const documents = items.map((i) => i.text);
  const embeddings = items.map((i) => i.embedding);
  const metadatas = items.map((i) => i.metadata ?? {});

  await collection.add({ ids, documents, embeddings, metadatas });
  await refreshIndex();
  return await count();
}

/** 混合检索（全文 + 向量） */
export async function hybridSearch({ queryText = "", queryEmbedding, topK = 5 }) {
  if (MODE === "mock") {
    const qt = (queryText ?? "").trim();
    const scored = mockStore.map((d) => ({
      id: d.id,
      text: d.text,
      metadata: d.metadata,
      score: 0.5 * keywordHit(qt, d.text) + 0.5 * cosineSimilarity(queryEmbedding, d.embedding),
    }));
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK);
  }

  const options = {
    knn: { queryEmbeddings: [queryEmbedding], nResults: topK },
    rank: { rrf: {} },
    nResults: topK,
    include: ["documents", "metadatas"],
  };
  const qt = (queryText ?? "").trim();
  if (qt) {
    options.query = { whereDocument: { $contains: qt }, nResults: topK };
  }

  const res = await collection.hybridSearch(options);
  const ids = res.ids?.[0] ?? [];
  const documents = res.documents?.[0] ?? [];
  const metadatas = res.metadatas?.[0] ?? [];
  const distances = res.distances?.[0] ?? [];

  return ids.map((id, i) => ({
    id,
    text: documents[i] ?? null,
    metadata: metadatas[i] ?? null,
    score: distances[i] ?? 0,
  }));
}

export async function count() {
  if (MODE === "mock") return mockStore.length;
  return await collection.count();
}

export async function listDocuments(limit = 1000) {
  if (MODE === "mock") {
    return mockStore.slice(0, limit).map((d) => ({ id: d.id, text: d.text, metadata: d.metadata }));
  }
  const res = await collection.get({ limit, include: ["documents", "metadatas"] });
  const ids = res.ids ?? [];
  const documents = res.documents ?? [];
  const metadatas = res.metadatas ?? [];
  return ids.map((id, i) => ({ id, text: documents[i] ?? null, metadata: metadatas[i] ?? null }));
}

export async function clearCollection() {
  if (MODE === "mock") {
    mockStore.length = 0;
    return 0;
  }
  const all = await collection.get({ limit: 100000 });
  if (all.ids && all.ids.length > 0) {
    await collection.delete({ ids: all.ids });
  }
  await refreshIndex();
  return await count();
}

export async function close() {
  if (client) {
    try {
      await client.close();
    } catch {
      // ignore
    }
    client = null;
    collection = null;
  }
}
