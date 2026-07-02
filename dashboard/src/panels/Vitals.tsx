import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  BarChart,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { EmptyState } from "@/components/EmptyState";
import { OfflineBanner } from "@/components/OfflineBanner";
import { useVitals } from "@/hooks/useApi";
import { CHART } from "@/lib/chart";
import type {
  CalibrationBin,
  ConfidenceCalibrationStratum,
  HistogramBucket,
  MemStatus,
} from "@/lib/types";

const STATUSES: MemStatus[] = ["active", "deprecated", "superseded"];

function StatTile({
  label,
  value,
}: {
  label: string;
  value: string | number;
}) {
  return (
    <Card>
      <CardContent className="p-4">
        <div className="text-xs text-zinc-500">{label}</div>
        <div className="mt-1 text-2xl font-semibold tabular-nums">{value}</div>
      </CardContent>
    </Card>
  );
}

function fmtKpi(v: number | null, digits = 3): string {
  return v === null ? "—" : v.toFixed(digits);
}

function Histogram({
  title,
  buckets,
}: {
  title: string;
  buckets: HistogramBucket[];
}) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        {buckets.length === 0 ? (
          <EmptyState>no data yet</EmptyState>
        ) : (
          <div className="h-44">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={buckets} margin={{ top: 4, right: 8, bottom: 0, left: -22 }}>
                <CartesianGrid stroke={CHART.grid} strokeDasharray="2 4" vertical={false} />
                <XAxis
                  dataKey="bucket"
                  tick={{ fontSize: 9, fill: CHART.axis }}
                  stroke={CHART.grid}
                  interval={0}
                  angle={-30}
                  height={34}
                  textAnchor="end"
                />
                <YAxis
                  allowDecimals={false}
                  tick={{ fontSize: 10, fill: CHART.axis }}
                  stroke={CHART.grid}
                />
                <Tooltip
                  isAnimationActive={false}
                  contentStyle={{ fontSize: 11, borderColor: CHART.grid }}
                />
                <Bar
                  dataKey="count"
                  fill={CHART.series1}
                  radius={[4, 4, 0, 0]}
                  isAnimationActive={false}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/** Empirical (bars) vs predicted (line = perfect-calibration diagonal). */
function CalibrationChart({ bins }: { bins: CalibrationBin[] }) {
  if (bins.length === 0) {
    return <EmptyState>no calibration events yet</EmptyState>;
  }
  return (
    <div className="h-52">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={bins} margin={{ top: 4, right: 8, bottom: 0, left: -22 }}>
          <CartesianGrid stroke={CHART.grid} strokeDasharray="2 4" vertical={false} />
          <XAxis
            dataKey="bin"
            tick={{ fontSize: 9, fill: CHART.axis }}
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
            formatter={(value, name) => [Number(value).toFixed(3), String(name)]}
            contentStyle={{ fontSize: 11, borderColor: CHART.grid }}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Bar
            dataKey="empirical"
            name="empirical"
            fill={CHART.series1}
            radius={[4, 4, 0, 0]}
            isAnimationActive={false}
          />
          <Line
            dataKey="predicted_mean"
            name="predicted (diagonal)"
            stroke={CHART.series2}
            strokeWidth={2}
            strokeDasharray="4 3"
            dot={{ r: 3, fill: CHART.series2 }}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function FreshnessStratum({
  stratum,
}: {
  stratum: ConfidenceCalibrationStratum;
}) {
  return (
    <div>
      <div className="mb-1 flex items-baseline gap-2 text-xs">
        <span className="font-medium text-zinc-700">
          {stratum.freshness_bucket}
        </span>
        <span className="text-zinc-500">
          ECE {stratum.ece === null ? "—" : stratum.ece.toFixed(3)}
        </span>
        <span className="text-zinc-400">n={stratum.count}</span>
      </div>
      {stratum.bins.length === 0 ? (
        <p className="text-xs text-zinc-400">no events in this stratum</p>
      ) : (
        <div className="h-36">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={stratum.bins}
              margin={{ top: 4, right: 8, bottom: 0, left: -22 }}
            >
              <CartesianGrid stroke={CHART.grid} strokeDasharray="2 4" vertical={false} />
              <XAxis
                dataKey="bin"
                tick={{ fontSize: 9, fill: CHART.axis }}
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
                formatter={(value, name) => [Number(value).toFixed(3), String(name)]}
                contentStyle={{ fontSize: 11, borderColor: CHART.grid }}
              />
              <Bar
                dataKey="predicted_mean"
                name="stated confidence"
                fill={CHART.series2}
                radius={[4, 4, 0, 0]}
                isAnimationActive={false}
              />
              <Bar
                dataKey="empirical"
                name="correctness"
                fill={CHART.series1}
                radius={[4, 4, 0, 0]}
                isAnimationActive={false}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

export function Vitals() {
  const { data, error, isLoading } = useVitals();

  if (isLoading && !data) {
    return <EmptyState>loading vitals…</EmptyState>;
  }
  if (!data) {
    return (
      <div className="space-y-3">
        <OfflineBanner error={error} />
        <EmptyState>vitals unavailable</EmptyState>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <OfflineBanner error={error} />

      {/* status counts */}
      <div className="grid grid-cols-3 gap-3">
        {STATUSES.map((s) => (
          <StatTile key={s} label={s} value={data.counts_by_status[s] ?? 0} />
        ))}
      </div>

      {/* histograms */}
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <Histogram title="Retrievability distribution" buckets={data.r_histogram} />
        <Histogram title="Stability distribution" buckets={data.s_histogram} />
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatTile label="FAMA" value={fmtKpi(data.kpis.fama)} />
        <StatTile
          label="forgetting precision"
          value={fmtKpi(data.kpis.forgetting_precision)}
        />
        <StatTile
          label="forgetting recall"
          value={fmtKpi(data.kpis.forgetting_recall)}
        />
        <StatTile
          label="avg tokens / query"
          value={
            data.kpis.avg_tokens_per_query === null
              ? "—"
              : Math.round(data.kpis.avg_tokens_per_query)
          }
        />
      </div>

      {/* retention calibration */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">
            Retention calibration — predicted R vs empirical recall
          </CardTitle>
        </CardHeader>
        <CardContent>
          <CalibrationChart bins={data.retention_calibration} />
        </CardContent>
      </Card>

      {/* confidence by freshness */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">
            Confidence calibration by freshness
          </CardTitle>
        </CardHeader>
        <CardContent>
          {data.confidence_by_freshness_bucket.length === 0 ? (
            <EmptyState>
              no judged answers yet — confidence calibration appears after
              traces are judged
            </EmptyState>
          ) : (
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
              {data.confidence_by_freshness_bucket.map((s) => (
                <FreshnessStratum key={s.freshness_bucket} stratum={s} />
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
