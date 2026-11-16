import type { Cell } from "./types";

export function numbers(values: Cell[]): number[] {
  const out: number[] = [];
  for (const v of values) if (typeof v === "number" && Number.isFinite(v)) out.push(v);
  return out;
}

export function mean(xs: number[]): number {
  if (!xs.length) return NaN;
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

export function std(xs: number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  let s = 0;
  for (const x of xs) s += (x - m) ** 2;
  return Math.sqrt(s / (xs.length - 1));
}

export function quantile(sorted: number[], q: number): number {
  if (!sorted.length) return NaN;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export interface NumericSummary {
  count: number;
  mean: number;
  std: number;
  variance: number;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  skew: number;
}

export function summarize(values: Cell[]): NumericSummary | null {
  const xs = numbers(values);
  if (!xs.length) return null;
  const sorted = [...xs].sort((a, b) => a - b);
  const m = mean(xs);
  const sd = std(xs);
  let s3 = 0;
  for (const x of xs) s3 += ((x - m) / (sd || 1)) ** 3;
  return {
    count: xs.length,
    mean: m,
    std: sd,
    variance: sd * sd,
    min: sorted[0],
    q1: quantile(sorted, 0.25),
    median: quantile(sorted, 0.5),
    q3: quantile(sorted, 0.75),
    max: sorted[sorted.length - 1],
    skew: xs.length > 2 ? s3 / xs.length : 0,
  };
}

export function pearson(a: Cell[], b: Cell[]): number {
  let n = 0;
  let sa = 0;
  let sb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i];
    const y = b[i];
    if (typeof x === "number" && typeof y === "number") {
      n++;
      sa += x;
      sb += y;
    }
  }
  if (n < 3) return NaN;
  const ma = sa / n;
  const mb = sb / n;
  let cov = 0;
  let va = 0;
  let vb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i];
    const y = b[i];
    if (typeof x === "number" && typeof y === "number") {
      cov += (x - ma) * (y - mb);
      va += (x - ma) ** 2;
      vb += (y - mb) ** 2;
    }
  }
  return va && vb ? cov / Math.sqrt(va * vb) : NaN;
}

/** Autocorrelation for lags 0..maxLag on a gap-free numeric series. */
export function acf(xs: number[], maxLag: number): number[] {
  const n = xs.length;
  const m = mean(xs);
  let denom = 0;
  for (const x of xs) denom += (x - m) ** 2;
  const out: number[] = [];
  for (let k = 0; k <= Math.min(maxLag, n - 1); k++) {
    let s = 0;
    for (let t = k; t < n; t++) s += (xs[t] - m) * (xs[t - k] - m);
    out.push(denom ? s / denom : 0);
  }
  return out;
}

export function rollingMean(values: (number | null)[], window: number): (number | null)[] {
  const out: (number | null)[] = new Array(values.length).fill(null);
  let sum = 0;
  let count = 0;
  const q: (number | null)[] = [];
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    q.push(v);
    if (typeof v === "number") {
      sum += v;
      count++;
    }
    if (q.length > window) {
      const old = q.shift();
      if (typeof old === "number") {
        sum -= old;
        count--;
      }
    }
    if (q.length === window && count > 0) out[i] = sum / count;
  }
  return out;
}

/** Least-squares slope of the series against its position, per step. */
export function trendSlope(xs: number[]): number {
  const n = xs.length;
  if (n < 3) return 0;
  const mx = (n - 1) / 2;
  const my = mean(xs);
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    num += (i - mx) * (xs[i] - my);
    den += (i - mx) ** 2;
  }
  return den ? num / den : 0;
}

/** The lag (>=2) with the strongest positive autocorrelation peak, if it is meaningful. */
export function seasonality(xs: number[]): { period: number; strength: number } | null {
  if (xs.length < 12) return null;
  // detrend with first differences so a trend doesn't dominate the ACF
  const diff = xs.slice(1).map((x, i) => x - xs[i]);
  const r = acf(diff, Math.min(Math.floor(diff.length / 2), 400));
  const band = 1.96 / Math.sqrt(diff.length);
  let best: { period: number; strength: number } | null = null;
  for (let k = 2; k < r.length - 1; k++) {
    if (r[k] > r[k - 1] && r[k] >= r[k + 1] && r[k] > band * 1.5) {
      if (!best || r[k] > best.strength) best = { period: k, strength: r[k] };
    }
  }
  return best;
}

export function histogramBins(xs: number[], bins = 30) {
  if (!xs.length) return { edges: [], counts: [] };
  let min = Infinity;
  let max = -Infinity;
  for (const x of xs) {
    if (x < min) min = x;
    if (x > max) max = x;
  }
  if (min === max) return { edges: [min, max], counts: [xs.length] };
  const w = (max - min) / bins;
  const counts = new Array(bins).fill(0);
  for (const x of xs) counts[Math.min(bins - 1, Math.floor((x - min) / w))]++;
  return { edges: Array.from({ length: bins + 1 }, (_, i) => min + i * w), counts };
}

export function fmt(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e6 || a < 1e-3)) return v.toExponential(2);
  return Number(v.toFixed(digits)).toLocaleString(undefined, { maximumFractionDigits: digits });
}
