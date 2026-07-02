import { useMemo, useState } from "react";
import { FlaskConical, Pin, PinOff, ShieldCheck } from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { EmptyState } from "@/components/EmptyState";
import { OfflineBanner } from "@/components/OfflineBanner";
import { useCurveConstants, useMemory } from "@/hooks/useApi";
import { useNow } from "@/hooks/useNow";
import { forceSelfTest, pinMemory, verifyNow } from "@/lib/api";
import { CHART, gradeColor, gradeLabel } from "@/lib/chart";
import {
  daysUntilRetrievability,
  elapsedDaysSince,
  retrievability,
} from "@/lib/fsrs";
import type {
  MemoryDetail as MemoryDetailDTO,
  ReviewRecord,
  SelfTestResponse,
  VerificationEventDTO,
  VerificationOutcome,
} from "@/lib/types";
import { fmtWhen, humanDays, shortId } from "@/lib/utils";
import { statusBadgeVariant } from "./MemoriesTable";

// ------------------------------------------------------------------ curve

interface CurvePoint {
  t: number; // days since first anchor
  r: number;
}

interface CurveMarker {
  t: number;
  grade: 1 | 2 | 3 | 4;
  kind: ReviewRecord["kind"];
}

function buildCurve(
  mem: MemoryDetailDTO,
  nowMs: number,
  factor: number,
  decay: number,
  theta: number,
): { points: CurvePoint[]; markers: CurveMarker[]; nowT: number } {
  const anchors: { atMs: number; s: number; review: ReviewRecord | null }[] =
    mem.reviews.length > 0
      ? mem.reviews.map((rv) => ({
          atMs: new Date(rv.at).getTime(),
          s: rv.s_after,
          review: rv,
        }))
      : [
          {
            atMs: new Date(mem.last_review_at).getTime(),
            s: mem.stability,
            review: null,
          },
        ];
  anchors.sort((a, b) => a.atMs - b.atMs);

  const t0 = anchors[0].atMs;
  const toDays = (ms: number) => (ms - t0) / 86_400_000;
  const points: CurvePoint[] = [];
  const markers: CurveMarker[] = [];
  const SAMPLES = 24;

  for (let i = 0; i < anchors.length; i++) {
    const a = anchors[i];
    const next = anchors[i + 1];
    let endMs: number;
    if (next) {
      endMs = next.atMs;
    } else {
      // Extend past "now" until the curve crosses theta (plus a margin).
      const horizonDays =
        Math.max(daysUntilRetrievability(theta, a.s, factor, decay), 1) * 1.25;
      endMs = Math.max(nowMs, a.atMs + horizonDays * 86_400_000);
    }
    if (a.review) {
      markers.push({
        t: toDays(a.atMs),
        grade: a.review.grade,
        kind: a.review.kind,
      });
    }
    const spanMs = Math.max(endMs - a.atMs, 1);
    for (let j = 0; j <= SAMPLES; j++) {
      const ms = a.atMs + (spanMs * j) / SAMPLES;
      const elapsed = (ms - a.atMs) / 86_400_000;
      points.push({
        t: toDays(ms),
        r: retrievability(elapsed, a.s, factor, decay),
      });
    }
  }
  return { points, markers, nowT: toDays(nowMs) };
}

function ForgettingCurve({ mem }: { mem: MemoryDetailDTO }) {
  const { decay, factor, theta } = useCurveConstants();
  const nowMs = useNow(5000);
  const { points, markers, nowT } = useMemo(
    () => buildCurve(mem, nowMs, factor, decay, theta),
    [mem, nowMs, factor, decay, theta],
  );

  const elapsed = elapsedDaysSince(mem.last_review_at, nowMs);
  const total = daysUntilRetrievability(theta, mem.stability, factor, decay);
  const remaining = total - elapsed;

  return (
    <div>
      <div className="h-52 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={points} margin={{ top: 8, right: 12, bottom: 4, left: -18 }}>
            <CartesianGrid stroke={CHART.grid} strokeDasharray="2 4" />
            <XAxis
              dataKey="t"
              type="number"
              domain={["dataMin", "dataMax"]}
              tick={{ fontSize: 10, fill: CHART.axis }}
              tickFormatter={(v: number) => `${Math.round(v)}d`}
              stroke={CHART.grid}
            />
            <YAxis
              domain={[0, 1]}
              tick={{ fontSize: 10, fill: CHART.axis }}
              tickFormatter={(v: number) => v.toFixed(1)}
              stroke={CHART.grid}
            />
            <Tooltip
              isAnimationActive={false}
              formatter={(value) => [Number(value).toFixed(3), "R"]}
              labelFormatter={(t) => `day ${Number(t).toFixed(1)}`}
              contentStyle={{ fontSize: 11, borderColor: CHART.grid }}
            />
            <ReferenceLine
              y={theta}
              stroke={CHART.threshold}
              strokeDasharray="4 4"
              label={{
                value: "θ",
                position: "right",
                fill: CHART.threshold,
                fontSize: 11,
              }}
            />
            <ReferenceLine
              x={nowT}
              stroke={CHART.axis}
              strokeDasharray="2 2"
              label={{
                value: "now",
                position: "top",
                fill: CHART.axis,
                fontSize: 10,
              }}
            />
            <Line
              type="monotone"
              dataKey="r"
              stroke={CHART.series1}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
            {markers.map((m, i) => (
              <ReferenceDot
                key={i}
                x={m.t}
                y={1}
                r={4}
                fill={gradeColor(m.grade)}
                stroke="#ffffff"
                strokeWidth={1.5}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-zinc-500">
        <span>
          {remaining > 0 ? (
            <>
              forgotten in ~
              <span className="font-medium text-zinc-900">
                {humanDays(remaining)}
              </span>
            </>
          ) : (
            <span className="text-red-600">below θ now</span>
          )}
        </span>
        <span className="text-zinc-300">·</span>
        {([1, 2, 3, 4] as const).map((g) => (
          <span key={g} className="inline-flex items-center gap-1">
            <span
              className="inline-block h-2 w-2 rounded-full"
              style={{ backgroundColor: gradeColor(g) }}
            />
            {gradeLabel(g)}
          </span>
        ))}
      </div>
    </div>
  );
}

// -------------------------------------------------------------- validity

function ValidityTimeline({ mem }: { mem: MemoryDetailDTO }) {
  const nowMs = useNow(60_000);
  const from = new Date(mem.valid_from).getTime();
  const to = mem.valid_to ? new Date(mem.valid_to).getTime() : nowMs;
  const created = new Date(mem.created_at).getTime();
  const start = Math.min(from, created);
  const end = Math.max(to, nowMs);
  const span = Math.max(end - start, 1);
  const left = ((from - start) / span) * 100;
  const width = Math.max(((to - from) / span) * 100, 2);

  return (
    <div>
      <div className="relative h-3 w-full rounded-full bg-zinc-100">
        <div
          className="absolute top-0 h-3 rounded-full bg-zinc-400"
          style={{ left: `${left}%`, width: `${width}%` }}
          title={`${fmtWhen(mem.valid_from)} → ${mem.valid_to ? fmtWhen(mem.valid_to) : "now"}`}
        />
      </div>
      <div className="mt-1 flex justify-between text-[11px] text-zinc-500">
        <span>{fmtWhen(mem.valid_from)}</span>
        <span>{mem.valid_to ? fmtWhen(mem.valid_to) : "now"}</span>
      </div>
    </div>
  );
}

// ------------------------------------------------------------- helpers

function outcomeVariant(
  outcome: VerificationOutcome,
): "success" | "destructive" | "warning" {
  if (outcome === "confirmed") return "success";
  if (outcome === "refuted") return "destructive";
  return "warning";
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <div className="mb-1.5 text-xs font-medium uppercase tracking-wide text-zinc-500">
      {children}
    </div>
  );
}

// ---------------------------------------------------------------- panel

interface Props {
  id: string;
  onSelect: (id: string) => void;
}

export function MemoryDetail({ id, onSelect }: Props) {
  const { data: mem, error, isLoading, mutate } = useMemory(id);
  const [verbatim, setVerbatim] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [selfTest, setSelfTest] = useState<SelfTestResponse | null>(null);
  const [verifyResult, setVerifyResult] = useState<VerificationEventDTO | null>(
    null,
  );
  const [actionError, setActionError] = useState<string | null>(null);

  const act = async (name: string, fn: () => Promise<void>) => {
    setBusy(name);
    setActionError(null);
    try {
      await fn();
      await mutate();
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  if (isLoading && !mem) {
    return (
      <Card>
        <CardContent className="pt-4">
          <EmptyState>loading memory…</EmptyState>
        </CardContent>
      </Card>
    );
  }
  if (!mem) {
    return (
      <Card>
        <CardContent className="pt-4">
          <OfflineBanner error={error} />
          <EmptyState>memory {shortId(id)} unavailable</EmptyState>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center gap-2">
          <CardTitle className="text-base">
            memory <span className="font-mono text-sm">{shortId(mem.id)}</span>
          </CardTitle>
          <Badge variant={statusBadgeVariant(mem.status)}>{mem.status}</Badge>
          <Badge variant="outline">{mem.mtype}</Badge>
          {mem.pinned && (
            <Badge variant="secondary">
              <Pin className="mr-1 h-3 w-3" aria-hidden /> pinned
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <OfflineBanner error={error} />

        {/* content + verbatim toggle */}
        <div>
          <div className="mb-1.5 flex items-center justify-between">
            <SectionTitle>content</SectionTitle>
            <label className="flex items-center gap-2 text-xs text-zinc-500">
              verbatim
              <Switch checked={verbatim} onCheckedChange={setVerbatim} />
            </label>
          </div>
          <p className="whitespace-pre-wrap rounded-md bg-zinc-50 p-3 text-sm">
            {verbatim ? mem.content_verbatim : mem.content}
          </p>
          {mem.triple && (
            <div className="mt-2 flex flex-wrap items-center gap-1 text-xs">
              <Badge variant="secondary">{mem.triple[0]}</Badge>
              <span className="text-zinc-400">→</span>
              <Badge variant="outline">{mem.triple[1]}</Badge>
              <span className="text-zinc-400">→</span>
              <Badge variant="secondary">{mem.triple[2]}</Badge>
            </div>
          )}
        </div>

        {/* actions */}
        <div className="flex flex-wrap gap-2">
          <Button
            size="sm"
            variant="outline"
            disabled={busy !== null}
            onClick={() =>
              act("pin", async () => {
                await pinMemory(mem.id, !mem.pinned);
              })
            }
          >
            {mem.pinned ? (
              <>
                <PinOff className="h-3.5 w-3.5" aria-hidden /> Unpin
              </>
            ) : (
              <>
                <Pin className="h-3.5 w-3.5" aria-hidden /> Pin
              </>
            )}
          </Button>
          <Button
            size="sm"
            variant="outline"
            disabled={busy !== null}
            onClick={() =>
              act("selftest", async () => {
                setSelfTest(await forceSelfTest(mem.id));
              })
            }
          >
            <FlaskConical className="h-3.5 w-3.5" aria-hidden />
            {busy === "selftest" ? "testing…" : "Force self-test"}
          </Button>
          <Button
            size="sm"
            variant="outline"
            disabled={busy !== null}
            onClick={() =>
              act("verify", async () => {
                setVerifyResult(await verifyNow(mem.id));
              })
            }
          >
            <ShieldCheck className="h-3.5 w-3.5" aria-hidden />
            {busy === "verify" ? "verifying…" : "Verify now"}
          </Button>
        </div>
        {actionError && (
          <p className="text-xs text-red-600">action failed: {actionError}</p>
        )}
        {selfTest && (
          <div className="rounded-md border border-zinc-200 bg-zinc-50 p-2 text-xs">
            <div className="flex items-center gap-2">
              <Badge variant={selfTest.passed ? "success" : "destructive"}>
                {selfTest.passed ? "pass" : "fail"}
              </Badge>
              <span
                className={
                  selfTest.ds >= 0 ? "text-green-700" : "text-red-600"
                }
              >
                ΔS {selfTest.ds >= 0 ? "+" : ""}
                {selfTest.ds.toFixed(2)}
              </span>
            </div>
            <p className="mt-1 text-zinc-600">probe: {selfTest.probe}</p>
            <p className="text-zinc-600">answer: {selfTest.answer}</p>
          </div>
        )}
        {verifyResult && (
          <div className="rounded-md border border-zinc-200 bg-zinc-50 p-2 text-xs">
            <div className="flex items-center gap-2">
              <Badge variant={outcomeVariant(verifyResult.outcome)}>
                {verifyResult.outcome}
              </Badge>
              <span className="text-zinc-500">{verifyResult.verifier}</span>
            </div>
            {verifyResult.evidence && (
              <p className="mt-1 font-mono text-[11px] text-zinc-600">
                {verifyResult.evidence}
              </p>
            )}
          </div>
        )}

        <Separator />

        {/* forgetting curve */}
        <div>
          <SectionTitle>forgetting curve</SectionTitle>
          <ForgettingCurve mem={mem} />
        </div>

        {/* validity */}
        <div>
          <SectionTitle>validity</SectionTitle>
          <ValidityTimeline mem={mem} />
          {mem.superseded_by && (
            <button
              type="button"
              className="mt-1 text-xs text-zinc-600 underline decoration-dotted hover:text-zinc-900"
              onClick={() => onSelect(mem.superseded_by as string)}
            >
              superseded by {shortId(mem.superseded_by)}
            </button>
          )}
        </div>

        {/* provenance */}
        <div>
          <SectionTitle>provenance ({mem.provenance.length})</SectionTitle>
          {mem.provenance.length === 0 ? (
            <p className="text-xs text-zinc-400">no source memories</p>
          ) : (
            <ul className="space-y-1">
              {mem.provenance.map((p) => (
                <li key={p.id}>
                  <button
                    type="button"
                    className="w-full truncate rounded px-1 py-0.5 text-left text-xs hover:bg-zinc-100"
                    onClick={() => onSelect(p.id)}
                    title={p.snippet}
                  >
                    <span className="font-mono text-zinc-500">
                      {shortId(p.id)}
                    </span>{" "}
                    {p.snippet}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* verification history */}
        <div>
          <SectionTitle>verifications ({mem.verifications.length})</SectionTitle>
          {mem.verifications.length === 0 ? (
            <p className="text-xs text-zinc-400">never verified</p>
          ) : (
            <ul className="space-y-1.5">
              {mem.verifications.map((v, i) => (
                <li key={i} className="flex items-start gap-2 text-xs">
                  <Badge variant={outcomeVariant(v.outcome)}>{v.outcome}</Badge>
                  <div className="min-w-0">
                    <div className="text-zinc-500">
                      {v.verifier} · {fmtWhen(v.at)}
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

        {/* audit tail */}
        <div>
          <SectionTitle>audit tail ({mem.audit.length})</SectionTitle>
          {mem.audit.length === 0 ? (
            <p className="text-xs text-zinc-400">no audit rows</p>
          ) : (
            <ul className="max-h-48 space-y-1 overflow-auto">
              {mem.audit.map((a) => (
                <li key={a.seq} className="text-xs text-zinc-600">
                  <span className="text-zinc-400">{fmtWhen(a.at)}</span>{" "}
                  <span className="font-medium">{a.actor}</span>{" "}
                  <span className="font-mono">{a.action}</span>
                  {a.detail && (
                    <span className="ml-1 break-all text-[11px] text-zinc-400">
                      {JSON.stringify(a.detail)}
                    </span>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
