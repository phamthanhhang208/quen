// FROZEN API CONTRACT — mirrors src/quen/api.py DTOs.
// Change here only together with api.py and tests/test_api.py.

export type MType = "episodic" | "semantic";
export type MemStatus = "active" | "deprecated" | "superseded";
export type VerificationOutcome = "confirmed" | "refuted" | "unverifiable";

// GET /config
export interface EngineConfig {
  fsrs_version: string; // "FSRS-4.5"
  decay: number; // -0.5
  factor: number; // 19/81
  eviction_r_threshold: number; // theta
  eviction_ttl_days: number;
  trust_threshold: number;
  freshness_half_life_days: number;
  default_token_budget: number;
}

// GET /memory — list rows
export interface MemorySummary {
  id: string;
  snippet: string; // first ~120 chars of content
  mtype: MType;
  status: MemStatus;
  importance: number;
  salience: number;
  difficulty: number;
  stability: number;
  last_review_at: string; // ISO — client computes live R from this + stability
  review_count: number;
  confidence: number;
  freshness_days: number;
  last_verified_at: string | null;
  pinned: boolean;
  valid_from: string;
  valid_to: string | null;
  superseded_by: string | null;
  source_ref: string | null;
  created_at: string;
  triple: [string, string, string] | null;
}

export interface ReviewRecord {
  at: string;
  kind: "use_judged" | "self_test" | "verification" | "manual";
  grade: 1 | 2 | 3 | 4;
  elapsed_days: number;
  r_before: number;
  d_before: number;
  s_before: number;
  d_after: number;
  s_after: number;
}

export interface AuditRecord {
  seq: number;
  at: string;
  actor: string;
  action: string;
  memory_id: string | null;
  run_id: string | null;
  detail: Record<string, unknown> | null;
}

export interface VerificationRecord {
  at: string;
  verifier: string;
  outcome: VerificationOutcome;
  evidence: string | null;
}

// GET /memory/{id}
export interface MemoryDetail extends MemorySummary {
  content: string;
  content_verbatim: string;
  provenance: { id: string; snippet: string }[];
  reviews: ReviewRecord[];
  verifications: VerificationRecord[];
  audit: AuditRecord[];
}

// POST /ask -> AskResponse; GET /recall/trace/... -> RecallTrace
export interface AskResponse {
  trace_id: string;
  answer: string;
  answer_confidence: number;
  abstained: boolean;
}

export interface UsedMemory {
  memory_id: string;
  snippet: string;
  relevance: number;
  retrievability: number;
  importance_norm: number;
  score: number;
  tokens: number;
  trust: number;
  confidence: number;
  freshness_days: number;
}

export interface ExcludedMemory {
  memory_id: string;
  snippet: string;
  relevance: number;
  reason: string; // e.g. "superseded by abc123 on 2026-06-10"
  status: MemStatus;
  superseded_by: string | null;
}

export interface VerificationEventDTO {
  memory_id: string;
  verifier: string;
  outcome: VerificationOutcome;
  evidence: string | null;
  at: string;
}

export interface RecallTrace {
  trace_id: string;
  at: string;
  query: string;
  answer: string | null;
  answer_confidence: number | null;
  abstained: boolean;
  token_budget: number;
  tokens_used: number; // memory-content tokens (budget accounting)
  prompt_tokens: number | null; // incl. trust tags — what the reader saw
  used: UsedMemory[];
  excluded: ExcludedMemory[];
  counterfactual: { memory_id: string; snippet: string; status: MemStatus }[];
  verifications: VerificationEventDTO[];
}

// GET /dream/log + /dream/{run_id}
export interface DreamRunSummary {
  run_id: string;
  started_at: string;
  finished_at: string | null;
  journal: string | null;
  stats: Record<string, number> | null;
}

export interface DreamAction {
  at: string;
  phase: "reabstract" | "selftest" | "supersede" | "evict" | "compress" | "error";
  detail: Record<string, unknown>;
}

export interface DreamRunDetail extends DreamRunSummary {
  actions: DreamAction[];
}

// GET /vitals
export interface HistogramBucket {
  bucket: string; // e.g. "0.0-0.1"
  count: number;
}

export interface CalibrationBin {
  bin: string; // predicted bucket label, e.g. "0.4-0.5"
  predicted_mean: number;
  empirical: number;
  count: number;
}

export interface ConfidenceCalibrationStratum {
  freshness_bucket: string; // e.g. "<7d", "7-30d", ">30d"
  bins: CalibrationBin[];
  ece: number | null;
  count: number;
}

export interface Vitals {
  counts_by_status: Record<MemStatus, number>;
  r_histogram: HistogramBucket[];
  s_histogram: HistogramBucket[];
  kpis: {
    fama: number | null;
    forgetting_precision: number | null;
    forgetting_recall: number | null;
    avg_tokens_per_query: number | null;
  };
  retention_calibration: CalibrationBin[];
  confidence_by_freshness_bucket: ConfidenceCalibrationStratum[];
}

// POST /ingest
export interface IngestResponse {
  stored: { id: string; snippet: string }[];
  reinforced: string[];
  skipped: { content: string; reason: string }[];
}

// POST /memory/{id}/recall (force self-test)
export interface SelfTestResponse {
  memory_id: string;
  probe: string;
  expected: string;
  answer: string;
  passed: boolean;
  ds: number; // stability delta
}
