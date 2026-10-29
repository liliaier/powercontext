/** 后端 API 客户端 */

export interface SearchHit {
  id: string;
  text: string | null;
  metadata: Record<string, unknown> | null;
  score: number;
}

export interface ImportItem {
  id: string;
  text: string;
  embedding: number[];
  metadata?: Record<string, unknown>;
}

async function post<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error((data as any)?.error || `HTTP ${res.status}`);
  return data as T;
}

async function get<T>(url: string): Promise<T> {
  const res = await fetch(url);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error((data as any)?.error || `HTTP ${res.status}`);
  return data as T;
}

export const api = {
  health: () => get<{ ok: boolean; count: number; dbPath: string }>("/api/health"),

  importItems: (items: ImportItem[]) =>
    post<{ ok: boolean; imported: number; total: number }>("/api/import", { items }),

  search: (queryText: string, queryEmbedding: number[], topK: number) =>
    post<{ ok: boolean; results: SearchHit[] }>("/api/search", {
      queryText,
      queryEmbedding,
      topK,
    }),

  documents: () =>
    get<{ ok: boolean; documents: { id: string; text: string | null; metadata: unknown }[] }>(
      "/api/documents"
    ),

  clear: () => post<{ ok: boolean; total: number }>("/api/clear", {}),
};
