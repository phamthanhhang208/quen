// Client-side mirror of the FSRS-4.5 forgetting curve ONLY.
// Constants come from GET /config; these are the hardcoded fallbacks.

export const FALLBACK_DECAY = -0.5;
export const FALLBACK_FACTOR = 19 / 81;
export const FALLBACK_EVICTION_R_THRESHOLD = 0.3;

/** R(t) = (1 + factor * t / S) ^ decay — FSRS-4.5 forgetting curve. */
export function retrievability(
  elapsedDays: number,
  stability: number,
  factor: number = FALLBACK_FACTOR,
  decay: number = FALLBACK_DECAY,
): number {
  if (stability <= 0) return 0;
  const t = Math.max(0, elapsedDays);
  return Math.pow(1 + (factor * t) / stability, decay);
}

/** Days until R(t) first drops to `target` (inverse of the curve). */
export function daysUntilRetrievability(
  target: number,
  stability: number,
  factor: number = FALLBACK_FACTOR,
  decay: number = FALLBACK_DECAY,
): number {
  if (stability <= 0) return 0;
  if (target >= 1) return 0;
  if (target <= 0) return Infinity;
  return ((Math.pow(target, 1 / decay) - 1) * stability) / factor;
}

/** Fractional days elapsed since an ISO timestamp, at `nowMs`. */
export function elapsedDaysSince(iso: string, nowMs: number): number {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return 0;
  return Math.max(0, (nowMs - then) / 86_400_000);
}
