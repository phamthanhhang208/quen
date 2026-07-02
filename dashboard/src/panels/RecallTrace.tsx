import { useState } from "react";
import { AlertTriangle, Search } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { EmptyState } from "@/components/EmptyState";
import { OfflineBanner } from "@/components/OfflineBanner";
import { useConfig, useTrace } from "@/hooks/useApi";
import { ask } from "@/lib/api";
import { CHART } from "@/lib/chart";
import type { RecallTrace as RecallTraceDTO, UsedMemory, VerificationOutcome } from "@/lib/types";
import { fmtWhen, humanDays, shortId } from "@/lib/utils";

// Retrieval score weights — mirror src/quen/config.py (not exposed by /config).
const W_REL = 0.65;
const W_R = 0.2;
const W_IMP = 0.15;

function outcomeVariant(
  outcome: VerificationOutcome,
): "success" | "destructive" | "warning" {
  if (outcome === "confirmed") return "success";
  if (outcome === "refuted") return "destructive";
  return "warning";
}

/** Tiny horizontal stacked bar of the three weighted score components. */
function ScoreBar({ m }: { m: UsedMemory }) {
  const parts = [
    { label: "relevance", value: W_REL * Math.max(0, m.relevance), color: CHART.series1 },
    { label: "R", value: W_R * Math.max(0, m.retrievability), color: CHART.series2 },
    { label: "importance", value: W_IMP * Math.max(0, m.importance_norm), color: "#a1a1aa" },
  ];
  const total = parts.reduce((s, p) => s + p.value, 0) || 1;
  return (
    <div className="flex items-center gap-2">
      <span className="w-10 text-right tabular-nums">{m.score.toFixed(2)}</span>
      <div
        className="flex h-2 w-24 overflow-hidden rounded-sm bg-zinc-100"
        title={parts
          .map((p) => `${p.label} ${(p.value / total * 100).toFixed(0)}%`)
          .join(" · ")}
      >
        {parts.map((p) => (
          <div
            key={p.label}
            style={{
              width: `${(p.value / total) * 100}%`,
              backgroundColor: p.color,
            }}
            className="border-r border-white last:border-r-0"
          />
        ))}
      </div>
    </div>
  );
}

function TrustChip({ m }: { m: UsedMemory }) {
  return (
    <span className="whitespace-nowrap rounded-full bg-zinc-100 px-2 py-0.5 text-[11px] text-zinc-600">
      trust {m.trust.toFixed(2)} · conf {m.confidence.toFixed(2)} ·{" "}
      {humanDays(m.freshness_days)} old
    </span>
  );
}

function TraceView({
  trace,
  onSelectMemory,
}: {
  trace: RecallTraceDTO;
  onSelectMemory: (id: string) => void;
}) {
  const [counterfactual, setCounterfactual] = useState(false);
  const usedIds = new Set(trace.used.map((u) => u.memory_id));
  const budgetPct =
    trace.token_budget > 0 ? (trace.tokens_used / trace.token_budget) * 100 : 0;

  return (
    <div className="space-y-4">
      {/* query + answer */}
      <div>
        <div className="text-xs font-medium uppercase tracking-wide text-zinc-500">
          query · {fmtWhen(trace.at)}
        </div>
        <p className="mt-1 text-sm font-medium">{trace.query}</p>
      </div>
      <div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-medium uppercase tracking-wide text-zinc-500">
            answer
          </span>
          {trace.answer_confidence !== null && (
            <>
              <Badge variant="outline">
                confidence {trace.answer_confidence.toFixed(2)}
              </Badge>
              <Progress
                value={trace.answer_confidence * 100}
                className="w-28"
              />
            </>
          )}
          {trace.abstained && (
            <Badge variant="warning">
              <AlertTriangle className="mr-1 h-3 w-3" aria-hidden />
              abstained
            </Badge>
          )}
        </div>
        <p className="mt-1 whitespace-pre-wrap rounded-md bg-zinc-50 p-3 text-sm">
          {trace.answer ?? "(no answer recorded)"}
        </p>
      </div>

      {/* token meter */}
      <div>
        <div className="mb-1 flex justify-between text-xs text-zinc-500">
          <span>tokens</span>
          <span className="tabular-nums">
            {trace.tokens_used} / {trace.token_budget}
          </span>
        </div>
        <Progress value={budgetPct} />
      </div>

      <Separator />

      {/* used memories */}
      <div>
        <div className="mb-1 text-xs font-medium uppercase tracking-wide text-zinc-500">
          used memories ({trace.used.length})
        </div>
        {trace.used.length === 0 ? (
          <p className="text-xs text-zinc-400">nothing injected</p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>memory</TableHead>
                <TableHead>tokens</TableHead>
                <TableHead>
                  score{" "}
                  <span className="normal-case text-zinc-400">
                    (rel / R / imp)
                  </span>
                </TableHead>
                <TableHead>trust</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {trace.used.map((m) => (
                <TableRow key={m.memory_id}>
                  <TableCell className="max-w-[320px]">
                    <button
                      type="button"
                      className="block w-full truncate text-left hover:underline"
                      title={m.snippet}
                      onClick={() => onSelectMemory(m.memory_id)}
                    >
                      <span className="font-mono text-xs text-zinc-400">
                        {shortId(m.memory_id)}
                      </span>{" "}
                      {m.snippet}
                    </button>
                  </TableCell>
                  <TableCell className="tabular-nums">{m.tokens}</TableCell>
                  <TableCell>
                    <ScoreBar m={m} />
                  </TableCell>
                  <TableCell>
                    <TrustChip m={m} />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </div>

      {/* excluded-relevant */}
      <div>
        <div className="mb-1 text-xs font-medium uppercase tracking-wide text-zinc-500">
          excluded though relevant ({trace.excluded.length})
        </div>
        {trace.excluded.length === 0 ? (
          <p className="text-xs text-zinc-400">nothing excluded</p>
        ) : (
          <ul className="space-y-1">
            {trace.excluded.map((m) => (
              <li key={m.memory_id} className="text-xs text-zinc-500">
                <button
                  type="button"
                  className="font-mono text-zinc-400 hover:underline"
                  onClick={() => onSelectMemory(m.memory_id)}
                >
                  {shortId(m.memory_id)}
                </button>{" "}
                <span className="text-zinc-500">{m.snippet}</span>{" "}
                <span className="italic text-zinc-400">— {m.reason}</span>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* counterfactual */}
      <div>
        <label className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-zinc-500">
          counterfactual — what append-only RAG would inject
          <Switch checked={counterfactual} onCheckedChange={setCounterfactual} />
        </label>
        {counterfactual &&
          (trace.counterfactual.length === 0 ? (
            <p className="mt-1 text-xs text-zinc-400">no counterfactual recorded</p>
          ) : (
            <ul className="mt-2 space-y-1">
              {trace.counterfactual.map((m) => {
                const wrongly = !usedIds.has(m.memory_id);
                return (
                  <li
                    key={m.memory_id}
                    className={
                      wrongly
                        ? "rounded-md bg-red-50 px-2 py-1 text-xs text-red-800"
                        : "rounded-md px-2 py-1 text-xs text-zinc-500"
                    }
                  >
                    <button
                      type="button"
                      className="font-mono hover:underline"
                      onClick={() => onSelectMemory(m.memory_id)}
                    >
                      {shortId(m.memory_id)}
                    </button>{" "}
                    {m.snippet}{" "}
                    <span className="opacity-70">[{m.status}]</span>
                    {wrongly && (
                      <span className="ml-1 font-medium">
                        — plain RAG would have injected this
                      </span>
                    )}
                  </li>
                );
              })}
            </ul>
          ))}
      </div>

      {/* verification events */}
      <div>
        <div className="mb-1 text-xs font-medium uppercase tracking-wide text-zinc-500">
          verification events ({trace.verifications.length})
        </div>
        {trace.verifications.length === 0 ? (
          <p className="text-xs text-zinc-400">no verification fired</p>
        ) : (
          <ul className="space-y-1.5">
            {trace.verifications.map((v, i) => (
              <li key={i} className="flex items-start gap-2 text-xs">
                <Badge variant={outcomeVariant(v.outcome)}>{v.outcome}</Badge>
                <div className="min-w-0">
                  <div className="text-zinc-500">
                    {v.verifier} ·{" "}
                    <button
                      type="button"
                      className="font-mono hover:underline"
                      onClick={() => onSelectMemory(v.memory_id)}
                    >
                      {shortId(v.memory_id)}
                    </button>{" "}
                    · {fmtWhen(v.at)}
                  </div>
                  {v.evidence && (
                    <div className="truncate font-mono text-[11px] text-zinc-600">
                      {v.evidence}
                    </div>
                  )}
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ panel

interface Props {
  onSelectMemory: (id: string) => void;
}

export function RecallTrace({ onSelectMemory }: Props) {
  const { data: config } = useConfig();
  const [query, setQuery] = useState("");
  const [budget, setBudget] = useState<string>("");
  const [traceId, setTraceId] = useState<string | null>(null);
  const [asking, setAsking] = useState(false);
  const [askError, setAskError] = useState<string | null>(null);

  const { data: trace, error, isLoading, notFound, mutate } = useTrace(traceId);

  const effectiveBudget =
    budget !== "" ? Number(budget) : config?.default_token_budget;

  const onAsk = async () => {
    if (!query.trim()) return;
    setAsking(true);
    setAskError(null);
    try {
      const res = await ask({
        query: query.trim(),
        token_budget: effectiveBudget,
      });
      setTraceId(res.trace_id);
      await mutate();
    } catch (e) {
      setAskError(e instanceof Error ? e.message : String(e));
    } finally {
      setAsking(false);
    }
  };

  return (
    <div className="space-y-3">
      <OfflineBanner error={notFound ? null : error} />
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Ask</CardTitle>
        </CardHeader>
        <CardContent>
          <form
            className="flex flex-wrap gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              void onAsk();
            }}
          >
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="ask the memory engine…"
              className="min-w-[240px] flex-1"
            />
            <Input
              type="number"
              min={1}
              value={budget !== "" ? budget : (config?.default_token_budget ?? "")}
              onChange={(e) => setBudget(e.target.value)}
              className="w-28"
              aria-label="token budget"
            />
            <Button type="submit" disabled={asking || !query.trim()}>
              <Search className="h-3.5 w-3.5" aria-hidden />
              {asking ? "asking…" : "Ask"}
            </Button>
          </form>
          {askError && (
            <p className="mt-2 text-xs text-red-600">ask failed: {askError}</p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">
            {traceId ? "Trace" : "Latest trace"}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading && !trace ? (
            <EmptyState>loading trace…</EmptyState>
          ) : notFound ? (
            <EmptyState>no recall traces yet — ask something above</EmptyState>
          ) : !trace ? (
            <EmptyState>trace unavailable</EmptyState>
          ) : (
            <TraceView trace={trace} onSelectMemory={onSelectMemory} />
          )}
        </CardContent>
      </Card>
    </div>
  );
}
