// SWR wrappers over the typed API layer — all polling at 3s.

import useSWR, { type SWRResponse } from "swr";
import {
  getConfig,
  getDreamLog,
  getDreamRun,
  getLatestTrace,
  getMemory,
  getTrace,
  getVitals,
  listMemories,
  type ListMemoriesParams,
  ApiError,
} from "@/lib/api";
import type {
  DreamRunDetail,
  DreamRunSummary,
  EngineConfig,
  MemoryDetail,
  MemorySummary,
  RecallTrace,
  Vitals,
} from "@/lib/types";
import {
  FALLBACK_DECAY,
  FALLBACK_EVICTION_R_THRESHOLD,
  FALLBACK_FACTOR,
} from "@/lib/fsrs";

const POLL = { refreshInterval: 3000 };

export function useConfig(): SWRResponse<EngineConfig> {
  return useSWR<EngineConfig>("config", () => getConfig(), {
    refreshInterval: 30_000,
  });
}

/** Curve constants from /config with hardcoded fallbacks while loading/offline. */
export function useCurveConstants(): {
  decay: number;
  factor: number;
  theta: number;
  config: EngineConfig | undefined;
} {
  const { data } = useConfig();
  return {
    decay: data?.decay ?? FALLBACK_DECAY,
    factor: data?.factor ?? FALLBACK_FACTOR,
    theta: data?.eviction_r_threshold ?? FALLBACK_EVICTION_R_THRESHOLD,
    config: data,
  };
}

export function useMemories(
  params: ListMemoriesParams = {},
): SWRResponse<MemorySummary[]> {
  return useSWR<MemorySummary[]>(
    ["memories", params.status ?? "", params.mtype ?? "", params.q ?? ""],
    () => listMemories(params),
    POLL,
  );
}

export function useMemory(id: string | null): SWRResponse<MemoryDetail> {
  return useSWR<MemoryDetail>(
    id ? ["memory", id] : null,
    () => getMemory(id as string),
    POLL,
  );
}

export function useDreamLog(): SWRResponse<DreamRunSummary[]> {
  return useSWR<DreamRunSummary[]>("dream-log", () => getDreamLog(), POLL);
}

export function useDreamRun(id: string | null): SWRResponse<DreamRunDetail> {
  return useSWR<DreamRunDetail>(
    id ? ["dream-run", id] : null,
    () => getDreamRun(id as string),
    POLL,
  );
}

/** Latest trace, or a specific trace when an id is given. 404 (no traces yet)
 *  is surfaced as `data === undefined` with a `notFound` flag, not an error. */
export function useTrace(id: string | null): SWRResponse<RecallTrace> & {
  notFound: boolean;
} {
  const swr = useSWR<RecallTrace>(
    id ? ["trace", id] : "trace-latest",
    () => (id ? getTrace(id) : getLatestTrace()),
    { ...POLL, shouldRetryOnError: true },
  );
  const notFound = swr.error instanceof ApiError && swr.error.status === 404;
  return { ...swr, notFound };
}

export function useVitals(): SWRResponse<Vitals> {
  return useSWR<Vitals>("vitals", () => getVitals(), POLL);
}
