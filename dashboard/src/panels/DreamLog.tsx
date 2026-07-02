import { useState } from "react";
import { ChevronDown, ChevronRight, MoonStar } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { EmptyState } from "@/components/EmptyState";
import { OfflineBanner } from "@/components/OfflineBanner";
import { useDreamLog, useDreamRun } from "@/hooks/useApi";
import { runDream } from "@/lib/api";
import type { DreamAction, DreamRunSummary } from "@/lib/types";
import { fmtDuration, fmtWhen, shortId } from "@/lib/utils";

const PHASE_ORDER: DreamAction["phase"][] = [
  "reabstract",
  "selftest",
  "supersede",
  "evict",
  "compress",
  "error",
];

const STAT_KEYS = [
  "reabstracted",
  "self_tests",
  "superseded",
  "evicted",
  "compressed",
] as const;

function str(v: unknown): string {
  return typeof v === "string" ? v : v == null ? "" : String(v);
}

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

// ------------------------------------------------------------ action rows

function ActionRow({ action }: { action: DreamAction }) {
  const d = action.detail;
  switch (action.phase) {
    case "supersede": {
      const rule = str(d.rule);
      const deterministic = rule.toLowerCase().startsWith("det");
      return (
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="font-mono text-zinc-600">
            {shortId(str(d.old_id))}
          </span>
          <span className="text-zinc-400">→</span>
          <span className="font-mono text-zinc-600">
            {shortId(str(d.new_id))}
          </span>
          <Badge variant={deterministic ? "secondary" : "outline"}>
            {deterministic ? "deterministic" : "NLI"}
          </Badge>
          {str(d.label) && <span className="text-zinc-500">{str(d.label)}</span>}
          {num(d.nli_confidence) !== null && (
            <span className="text-zinc-400">
              conf {(num(d.nli_confidence) as number).toFixed(2)}
            </span>
          )}
        </div>
      );
    }
    case "selftest": {
      const passed = d.passed === true;
      const ds = num(d.ds) ?? 0;
      return (
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <Badge variant={passed ? "success" : "destructive"}>
            {passed ? "pass" : "fail"}
          </Badge>
          <span
            className={
              ds >= 0 ? "tabular-nums text-green-700" : "tabular-nums text-red-600"
            }
          >
            ΔS {ds >= 0 ? "+" : ""}
            {ds.toFixed(2)}
          </span>
          <span className="font-mono text-zinc-500">
            {shortId(str(d.memory_id))}
          </span>
          <span className="min-w-0 flex-1 truncate text-zinc-600" title={str(d.probe)}>
            {str(d.probe)}
          </span>
        </div>
      );
    }
    case "evict": {
      const r = num(d.r);
      return (
        <div className="flex items-center gap-2 text-xs">
          <span className="font-mono text-zinc-600">
            {shortId(str(d.memory_id))}
          </span>
          <span className="text-zinc-500">
            final R {r !== null ? r.toFixed(3) : "?"}
          </span>
        </div>
      );
    }
    case "reabstract": {
      const prov = Array.isArray(d.provenance) ? d.provenance.length : 0;
      return (
        <div className="text-xs">
          <p className="text-zinc-700">{str(d.generalization)}</p>
          <p className="mt-0.5 text-zinc-400">
            <span className="font-mono">{shortId(str(d.memory_id))}</span> · from{" "}
            {prov} episodic{prov === 1 ? "" : "s"}
          </p>
        </div>
      );
    }
    case "compress":
      return (
        <div className="flex items-center gap-2 text-xs text-zinc-600">
          <span className="font-mono">{shortId(str(d.memory_id))}</span>
          <span className="text-zinc-400">→ {str(d.chars)} chars</span>
        </div>
      );
    case "error":
      return (
        <div className="text-xs text-red-600">
          {str(d.phase)}: {str(d.error)}
        </div>
      );
  }
}

// --------------------------------------------------------------- run card

function RunActions({ runId }: { runId: string }) {
  const { data, error, isLoading } = useDreamRun(runId);
  if (isLoading && !data)
    return <EmptyState>loading actions…</EmptyState>;
  if (error && !data)
    return <p className="text-xs text-red-600">failed to load run actions</p>;
  const actions = data?.actions ?? [];
  if (actions.length === 0)
    return <p className="text-xs text-zinc-400">no actions recorded</p>;

  const grouped = PHASE_ORDER.map((phase) => ({
    phase,
    items: actions.filter((a) => a.phase === phase),
  })).filter((g) => g.items.length > 0);

  return (
    <div className="space-y-3">
      {grouped.map((g) => (
        <div key={g.phase}>
          <div className="mb-1 text-xs font-medium uppercase tracking-wide text-zinc-500">
            {g.phase} ({g.items.length})
          </div>
          <ul className="space-y-1.5">
            {g.items.map((a, i) => (
              <li key={i} className="rounded-md bg-zinc-50 px-2 py-1.5">
                <ActionRow action={a} />
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}

function RunCard({ run }: { run: DreamRunSummary }) {
  const [open, setOpen] = useState(false);
  return (
    <Card>
      <CardHeader className="pb-2">
        <button
          type="button"
          className="flex w-full flex-wrap items-center gap-2 text-left"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
        >
          {open ? (
            <ChevronDown className="h-4 w-4 text-zinc-400" aria-hidden />
          ) : (
            <ChevronRight className="h-4 w-4 text-zinc-400" aria-hidden />
          )}
          <span className="font-mono text-sm">{shortId(run.run_id)}</span>
          <span className="text-xs text-zinc-500">
            {fmtWhen(run.started_at)} · {fmtDuration(run.started_at, run.finished_at)}
          </span>
          <span className="ml-auto flex flex-wrap gap-1">
            {STAT_KEYS.map((k) =>
              run.stats && run.stats[k] !== undefined ? (
                <Badge key={k} variant="secondary">
                  {k} {run.stats[k]}
                </Badge>
              ) : null,
            )}
          </span>
        </button>
      </CardHeader>
      <CardContent className="space-y-3">
        {run.journal && (
          <p className="text-sm italic text-zinc-600">{run.journal}</p>
        )}
        {open && <RunActions runId={run.run_id} />}
      </CardContent>
    </Card>
  );
}

// ------------------------------------------------------------------ panel

export function DreamLog() {
  const { data, error, isLoading, mutate } = useDreamLog();
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);

  const onRun = async () => {
    setRunning(true);
    setRunError(null);
    try {
      await runDream();
      await mutate();
    } catch (e) {
      setRunError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="space-y-3">
      <OfflineBanner error={error} />
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium text-zinc-700">
          Dream runs {data ? `(${data.length})` : ""}
        </h2>
        <Button size="sm" disabled={running} onClick={onRun}>
          <MoonStar className="h-3.5 w-3.5" aria-hidden />
          {running ? "dreaming…" : "Run dream now"}
        </Button>
      </div>
      {runError && <p className="text-xs text-red-600">dream failed: {runError}</p>}
      {isLoading && !data ? (
        <EmptyState>loading dream log…</EmptyState>
      ) : !data || data.length === 0 ? (
        <EmptyState>no dream runs yet — press "Run dream now"</EmptyState>
      ) : (
        data.map((run) => <RunCard key={run.run_id} run={run} />)
      )}
    </div>
  );
}
