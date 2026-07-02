// Typed fetchers for every engine endpoint (see types.ts — the frozen contract).

import type {
  AskResponse,
  DreamRunDetail,
  DreamRunSummary,
  EngineConfig,
  IngestResponse,
  MemoryDetail,
  MemorySummary,
  RecallTrace,
  SelfTestResponse,
  VerificationEventDTO,
  Vitals,
} from "./types";

export const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined) ??
  "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // non-JSON error body — keep statusText
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

function post<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

// ------------------------------------------------------------------ config

export function getConfig(): Promise<EngineConfig> {
  return request<EngineConfig>("/config");
}

// ------------------------------------------------------------------ memory

export interface ListMemoriesParams {
  status?: string;
  mtype?: string;
  q?: string;
  limit?: number;
}

export function listMemories(
  params: ListMemoriesParams = {},
): Promise<MemorySummary[]> {
  const qs = new URLSearchParams();
  if (params.status) qs.set("status", params.status);
  if (params.mtype) qs.set("mtype", params.mtype);
  if (params.q) qs.set("q", params.q);
  if (params.limit !== undefined) qs.set("limit", String(params.limit));
  const suffix = qs.toString() ? `?${qs.toString()}` : "";
  return request<MemorySummary[]>(`/memory${suffix}`);
}

export function getMemory(id: string): Promise<MemoryDetail> {
  return request<MemoryDetail>(`/memory/${encodeURIComponent(id)}`);
}

export function pinMemory(
  id: string,
  pinned: boolean,
): Promise<{ id: string; pinned: boolean }> {
  return post(`/memory/${encodeURIComponent(id)}/pin`, { pinned });
}

export function forceSelfTest(id: string): Promise<SelfTestResponse> {
  return post(`/memory/${encodeURIComponent(id)}/recall`);
}

export function verifyNow(id: string): Promise<VerificationEventDTO> {
  return post(`/memory/${encodeURIComponent(id)}/verify`);
}

// -------------------------------------------------------------- ingest/ask

export interface IngestBody {
  text: string;
  source_ref?: string | null;
  source_kind?: string | null;
}

export function ingest(body: IngestBody): Promise<IngestResponse> {
  return post("/ingest", body);
}

export interface AskBody {
  query: string;
  token_budget?: number;
}

export function ask(body: AskBody): Promise<AskResponse> {
  return post("/ask", body);
}

// ------------------------------------------------------------------- dream

export function runDream(): Promise<{
  run_id: string;
  stats: Record<string, number>;
}> {
  return post("/dream");
}

export function getDreamLog(): Promise<DreamRunSummary[]> {
  return request<DreamRunSummary[]>("/dream/log");
}

export function getDreamRun(id: string): Promise<DreamRunDetail> {
  return request<DreamRunDetail>(`/dream/${encodeURIComponent(id)}`);
}

// ------------------------------------------------------------------ traces

export function getLatestTrace(): Promise<RecallTrace> {
  return request<RecallTrace>("/recall/trace/latest");
}

export function getTrace(id: string): Promise<RecallTrace> {
  return request<RecallTrace>(`/recall/trace/${encodeURIComponent(id)}`);
}

// ------------------------------------------------------------------ vitals

export function getVitals(): Promise<Vitals> {
  return request<Vitals>("/vitals");
}
