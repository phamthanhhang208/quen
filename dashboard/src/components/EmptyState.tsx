import type { ReactNode } from "react";

/** Friendly muted empty/loading placeholder. */
export function EmptyState({ children }: { children: ReactNode }) {
  return (
    <div className="flex items-center justify-center rounded-md border border-dashed border-zinc-200 px-4 py-8 text-sm text-zinc-400">
      {children}
    </div>
  );
}
