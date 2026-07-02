import { WifiOff } from "lucide-react";
import { API_BASE } from "@/lib/api";

/** Subtle banner shown when SWR reports a fetch error (engine down). */
export function OfflineBanner({ error }: { error: unknown }) {
  if (!error) return null;
  return (
    <div className="mb-3 flex items-center gap-2 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
      <WifiOff className="h-3.5 w-3.5 shrink-0" aria-hidden />
      <span>
        engine offline — cannot reach <span className="font-mono">{API_BASE}</span>
        ; retrying…
      </span>
    </div>
  );
}
