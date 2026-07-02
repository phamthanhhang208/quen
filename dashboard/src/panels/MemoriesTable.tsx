import { useMemo, useState } from "react";
import { ArrowDown, ArrowUp, ArrowUpDown, Pin } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
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
import { useCurveConstants, useMemories } from "@/hooks/useApi";
import { useNow } from "@/hooks/useNow";
import { elapsedDaysSince, retrievability } from "@/lib/fsrs";
import type { MemStatus, MemorySummary } from "@/lib/types";
import { cn, fmtDate, humanDays, shortId } from "@/lib/utils";

type SortKey =
  | "snippet"
  | "mtype"
  | "status"
  | "importance"
  | "difficulty"
  | "stability"
  | "r"
  | "confidence"
  | "freshness_days"
  | "pinned"
  | "valid_from";

const STATUS_FILTERS: (MemStatus | "all")[] = [
  "all",
  "active",
  "deprecated",
  "superseded",
];

export function statusBadgeVariant(
  status: MemStatus,
): "default" | "secondary" | "outline" {
  if (status === "active") return "default";
  if (status === "deprecated") return "secondary";
  return "outline";
}

interface Props {
  selectedId: string | null;
  onSelect: (id: string) => void;
}

export function MemoriesTable({ selectedId, onSelect }: Props) {
  const { data, error, isLoading } = useMemories();
  const { decay, factor, theta } = useCurveConstants();
  const nowMs = useNow(1000);

  const [filter, setFilter] = useState("");
  const [status, setStatus] = useState<MemStatus | "all">("all");
  const [sortKey, setSortKey] = useState<SortKey>("valid_from");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const liveR = (m: MemorySummary): number =>
    retrievability(
      elapsedDaysSince(m.last_review_at, nowMs),
      m.stability,
      factor,
      decay,
    );

  const rows = useMemo(() => {
    let out = data ?? [];
    if (status !== "all") out = out.filter((m) => m.status === status);
    const needle = filter.trim().toLowerCase();
    if (needle) {
      out = out.filter(
        (m) =>
          m.snippet.toLowerCase().includes(needle) ||
          m.id.toLowerCase().includes(needle) ||
          (m.triple?.join(" ").toLowerCase().includes(needle) ?? false),
      );
    }
    const val = (m: MemorySummary): string | number | boolean => {
      if (sortKey === "r") return liveR(m);
      return m[sortKey];
    };
    const sorted = [...out].sort((a, b) => {
      const va = val(a);
      const vb = val(b);
      let cmp: number;
      if (typeof va === "string" && typeof vb === "string") {
        cmp = va.localeCompare(vb);
      } else {
        cmp = Number(va) - Number(vb);
      }
      return sortDir === "asc" ? cmp : -cmp;
    });
    return sorted;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, status, filter, sortKey, sortDir, nowMs, factor, decay]);

  const toggleSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  const SortableHead = ({
    k,
    children,
    className,
  }: {
    k: SortKey;
    children: React.ReactNode;
    className?: string;
  }) => (
    <TableHead className={className}>
      <button
        type="button"
        onClick={() => toggleSort(k)}
        className="inline-flex items-center gap-1 hover:text-zinc-900"
      >
        {children}
        {sortKey === k ? (
          sortDir === "asc" ? (
            <ArrowUp className="h-3 w-3" aria-hidden />
          ) : (
            <ArrowDown className="h-3 w-3" aria-hidden />
          )
        ) : (
          <ArrowUpDown className="h-3 w-3 opacity-40" aria-hidden />
        )}
      </button>
    </TableHead>
  );

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex flex-wrap items-center gap-2">
          <CardTitle className="mr-auto text-base">Memories</CardTitle>
          <Input
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="filter content / id / triple…"
            className="h-8 w-56"
          />
          <div className="flex gap-1">
            {STATUS_FILTERS.map((s) => (
              <Button
                key={s}
                size="sm"
                variant={status === s ? "default" : "outline"}
                onClick={() => setStatus(s)}
              >
                {s}
              </Button>
            ))}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <OfflineBanner error={error} />
        {isLoading && !data ? (
          <EmptyState>loading memories…</EmptyState>
        ) : rows.length === 0 ? (
          <EmptyState>no memories match</EmptyState>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <SortableHead k="snippet" className="min-w-[180px]">
                  content
                </SortableHead>
                <SortableHead k="mtype">type</SortableHead>
                <SortableHead k="status">status</SortableHead>
                <SortableHead k="importance">imp</SortableHead>
                <SortableHead k="difficulty">D</SortableHead>
                <SortableHead k="stability">S</SortableHead>
                <SortableHead k="r">R</SortableHead>
                <SortableHead k="confidence">conf</SortableHead>
                <SortableHead k="freshness_days">fresh</SortableHead>
                <SortableHead k="pinned">pin</SortableHead>
                <SortableHead k="valid_from">validity</SortableHead>
                <TableHead>sup. by</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((m) => {
                const r = liveR(m);
                return (
                  <TableRow
                    key={m.id}
                    data-state={m.id === selectedId ? "selected" : undefined}
                    className="cursor-pointer"
                    onClick={() => onSelect(m.id)}
                  >
                    <TableCell className="max-w-[280px] truncate" title={m.snippet}>
                      {m.snippet}
                    </TableCell>
                    <TableCell className="text-zinc-500">{m.mtype}</TableCell>
                    <TableCell>
                      <Badge variant={statusBadgeVariant(m.status)}>
                        {m.status}
                      </Badge>
                    </TableCell>
                    <TableCell className="tabular-nums">
                      {m.importance.toFixed(1)}
                    </TableCell>
                    <TableCell className="tabular-nums">
                      {m.difficulty.toFixed(1)}
                    </TableCell>
                    <TableCell className="tabular-nums">
                      {m.stability.toFixed(1)}
                    </TableCell>
                    <TableCell
                      className={cn(
                        "tabular-nums",
                        r < theta && "font-medium text-red-600",
                      )}
                    >
                      {r.toFixed(2)}
                    </TableCell>
                    <TableCell className="tabular-nums">
                      {m.confidence.toFixed(2)}
                    </TableCell>
                    <TableCell className="tabular-nums">
                      {humanDays(m.freshness_days)}
                    </TableCell>
                    <TableCell>
                      {m.pinned ? (
                        <Pin
                          className="h-3.5 w-3.5 text-zinc-700"
                          aria-label="pinned"
                        />
                      ) : null}
                    </TableCell>
                    <TableCell className="whitespace-nowrap text-xs text-zinc-500">
                      {fmtDate(m.valid_from)} → {m.valid_to ? fmtDate(m.valid_to) : "now"}
                    </TableCell>
                    <TableCell>
                      {m.superseded_by ? (
                        <button
                          type="button"
                          className="font-mono text-xs text-zinc-600 underline decoration-dotted hover:text-zinc-900"
                          onClick={(e) => {
                            e.stopPropagation();
                            onSelect(m.superseded_by as string);
                          }}
                        >
                          {shortId(m.superseded_by)}
                        </button>
                      ) : null}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}
