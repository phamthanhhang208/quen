import { useEffect, useState } from "react";

/** Current time in ms, re-rendering every `intervalMs` (default 1s) so
 *  live-retrievability cells visibly decay. */
export function useNow(intervalMs = 1000): number {
  const [now, setNow] = useState<number>(() => Date.now());
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now()), intervalMs);
    return () => window.clearInterval(t);
  }, [intervalMs]);
  return now;
}
