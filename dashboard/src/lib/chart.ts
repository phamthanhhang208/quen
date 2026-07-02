// Shared chart color roles (validated categorical palette; light surface).
// Series colors carry identity; status colors are reserved for pass/fail.

export const CHART = {
  series1: "#2a78d6", // blue — primary series (histograms, empirical)
  series2: "#1baf7a", // aqua — secondary series (predicted)
  grid: "#e4e4e7", // zinc-200
  axis: "#71717a", // zinc-500
  threshold: "#e34948", // status: serious — eviction theta line
  good: "#008300", // status: good — pass / confirmed / easy
  warn: "#eda100", // status: warning — hard
  bad: "#e34948", // status: serious — fail / refuted / again
} as const;

/** Marker color per FSRS grade (1=again .. 4=easy). */
export function gradeColor(grade: 1 | 2 | 3 | 4): string {
  switch (grade) {
    case 1:
      return CHART.bad;
    case 2:
      return CHART.warn;
    case 3:
      return CHART.series1;
    case 4:
      return CHART.good;
  }
}

export function gradeLabel(grade: 1 | 2 | 3 | 4): string {
  return ["again", "hard", "good", "easy"][grade - 1] as string;
}
