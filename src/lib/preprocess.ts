import { parseDateValue, toCsv } from "./csv";
import { mean, numbers, quantile, std } from "./stats";
import type { Cell, Dataset, PrepOptions, ProcessedData, TaskType } from "./types";

const MAX_ONEHOT = 30;

export function defaultPrep(ds: Dataset, dateCol: string | null): PrepOptions {
  const numeric = ds.columns.filter((c) => c.kind === "numeric" && c.name !== dateCol);
  const last = ds.columns[ds.columns.length - 1];
  // a low-cardinality text column at the end is almost always the label to predict
  const labelLast = last && last.name !== dateCol && last.kind === "categorical" && last.unique <= 20;
  const target = labelLast
    ? last.name
    : numeric.length
      ? numeric[numeric.length - 1].name
      : (ds.columns.find((c) => c.name !== dateCol)?.name ?? null);
  return {
    target,
    dateCol,
    task: inferTask(ds, target),
    features: numeric.filter((c) => c.name !== target).map((c) => c.name),
    missing: "interpolate",
    outliers: "none",
    iqrK: 1.5,
    zMax: 3,
    clipTarget: false,
    encoding: "onehot",
    scaling: "none",
    testFraction: 0.2,
  };
}

export function inferTask(ds: Dataset, target: string | null): TaskType {
  if (!target) return "regression";
  const meta = ds.columns.find((c) => c.name === target);
  if (!meta) return "regression";
  if (meta.kind !== "numeric") return "classification";
  const nonMissing = ds.nRows - meta.missing;
  const values = numbers(ds.data[target]);
  const allInts = values.every((v) => Number.isInteger(v));
  return allInts && meta.unique <= Math.min(10, Math.max(2, nonMissing / 20)) ? "classification" : "regression";
}

function isMissing(v: Cell) {
  return v === null || v === "" || (typeof v === "number" && Number.isNaN(v));
}

function fillNumeric(values: Cell[], how: PrepOptions["missing"]): Cell[] {
  const out = [...values];
  const xs = numbers(values);
  if (how === "mean" || how === "median" || how === "zero") {
    const sorted = [...xs].sort((a, b) => a - b);
    const fillv = how === "zero" ? 0 : how === "mean" ? mean(xs) : quantile(sorted, 0.5);
    return out.map((v) => (isMissing(v) ? fillv : v));
  }
  if (how === "ffill" || how === "interpolate") {
    if (how === "interpolate") {
      let last = -1;
      for (let i = 0; i < out.length; i++) {
        if (typeof out[i] === "number") {
          if (last >= 0 && i - last > 1) {
            const a = out[last] as number;
            const b = out[i] as number;
            for (let j = last + 1; j < i; j++) out[j] = a + ((b - a) * (j - last)) / (i - last);
          }
          last = i;
        }
      }
    }
    for (let i = 1; i < out.length; i++) if (isMissing(out[i])) out[i] = out[i - 1];
    for (let i = out.length - 2; i >= 0; i--) if (isMissing(out[i])) out[i] = out[i + 1];
    return out;
  }
  if (how === "bfill") {
    for (let i = out.length - 2; i >= 0; i--) if (isMissing(out[i])) out[i] = out[i + 1];
    for (let i = 1; i < out.length; i++) if (isMissing(out[i])) out[i] = out[i - 1];
    return out;
  }
  return out;
}

function mode(values: Cell[]): Cell {
  const counts = new Map<Cell, number>();
  let best: Cell = null;
  let bestN = 0;
  for (const v of values) {
    if (isMissing(v)) continue;
    const n = (counts.get(v) ?? 0) + 1;
    counts.set(v, n);
    if (n > bestN) {
      best = v;
      bestN = n;
    }
  }
  return best;
}

function clip(values: Cell[], opts: PrepOptions, trainRows: number): { values: Cell[]; clipped: number } {
  if (opts.outliers === "none") return { values, clipped: 0 };
  const xs = numbers(values.slice(0, trainRows));
  if (xs.length < 4) return { values, clipped: 0 };
  const sorted = [...xs].sort((a, b) => a - b);
  let lo: number;
  let hi: number;
  if (opts.outliers === "iqr") {
    const q1 = quantile(sorted, 0.25);
    const q3 = quantile(sorted, 0.75);
    lo = q1 - opts.iqrK * (q3 - q1);
    hi = q3 + opts.iqrK * (q3 - q1);
  } else if (opts.outliers === "zscore") {
    const m = mean(xs);
    const s = std(xs);
    lo = m - opts.zMax * s;
    hi = m + opts.zMax * s;
  } else {
    lo = quantile(sorted, 0.01);
    hi = quantile(sorted, 0.99);
  }
  let clipped = 0;
  const out = values.map((v) => {
    if (typeof v !== "number") return v;
    if (v < lo) {
      clipped++;
      return lo;
    }
    if (v > hi) {
      clipped++;
      return hi;
    }
    return v;
  });
  return { values: out, clipped };
}

function isoDate(v: Cell, kind: string): string {
  if (kind === "datetime") {
    const t = parseDateValue(v);
    return t === null ? String(v) : new Date(t).toISOString().slice(0, 19).replace("T00:00:00", "");
  }
  return String(v);
}

export function preprocess(ds: Dataset, opts: PrepOptions): ProcessedData {
  const report: string[] = [];
  if (!opts.target) throw new Error("Pick a target column first.");
  const target = opts.target;
  const metaOf = (n: string) => ds.columns.find((c) => c.name === n);
  const features = opts.features.filter((f) => f !== target && f !== opts.dateCol && metaOf(f));

  // 1. chronological order
  let order = Array.from({ length: ds.nRows }, (_, i) => i);
  const dateMeta = opts.dateCol ? metaOf(opts.dateCol) : undefined;
  if (dateMeta) {
    const keyOf = (v: Cell) => (dateMeta.kind === "datetime" ? parseDateValue(v) : typeof v === "number" ? v : null);
    const keys = ds.data[dateMeta.name].map(keyOf);
    const before = order.length;
    order = order.filter((i) => keys[i] !== null);
    if (before !== order.length) report.push(`Dropped ${before - order.length} rows with an unreadable ${dateMeta.name}.`);
    const sorted = order.every((v, i, arr) => i === 0 || (keys[arr[i - 1]] as number) <= (keys[v] as number));
    if (!sorted) {
      order.sort((a, b) => (keys[a] as number) - (keys[b] as number));
      report.push(`Sorted rows chronologically by ${dateMeta.name}.`);
    }
  }

  // 2. rows with no target can't be used for classification
  const isClass = opts.task === "classification";
  if (isClass) {
    const before = order.length;
    order = order.filter((i) => !isMissing(ds.data[target][i]));
    if (before !== order.length) report.push(`Dropped ${before - order.length} rows with no ${target} label.`);
  }

  const pick = (col: string) => order.map((i) => ds.data[col][i]);
  let cols: Record<string, Cell[]> = {};
  const used = [...features, target];
  for (const c of used) cols[c] = pick(c);
  let dates = dateMeta ? pick(dateMeta.name).map((v) => isoDate(v, dateMeta.kind)) : null;

  // 3. missing values
  const numericCols = used.filter((c) => metaOf(c)?.kind === "numeric" && !(isClass && c === target));
  const catCols = used.filter((c) => !numericCols.includes(c) && c !== target);
  const missingBefore = used.reduce((s, c) => s + cols[c].filter(isMissing).length, 0);
  if (opts.missing === "drop") {
    const keep: number[] = [];
    for (let r = 0; r < order.length; r++) if (used.every((c) => !isMissing(cols[c][r]))) keep.push(r);
    if (keep.length !== order.length) {
      report.push(`Dropped ${order.length - keep.length} rows that had missing values.`);
      for (const c of used) cols[c] = keep.map((r) => cols[c][r]);
      if (dates) dates = keep.map((r) => dates![r]);
    }
  } else if (missingBefore > 0) {
    for (const c of numericCols) cols[c] = fillNumeric(cols[c], opts.missing);
    for (const c of catCols) {
      const m = mode(cols[c]);
      cols[c] = cols[c].map((v) => (isMissing(v) ? m : v));
    }
    const label = { ffill: "forward fill", bfill: "backward fill", interpolate: "linear interpolation", mean: "the column mean", median: "the column median", zero: "zero", drop: "" }[opts.missing];
    report.push(`Filled ${missingBefore} missing values using ${label} (categories use the most common value).`);
  }
  // anything still missing (e.g. an all-empty column) is dropped
  const rowsNow = cols[target].length;
  const stillOk: number[] = [];
  for (let r = 0; r < rowsNow; r++) if (used.every((c) => !isMissing(cols[c][r]))) stillOk.push(r);
  if (stillOk.length !== rowsNow) {
    report.push(`Dropped ${rowsNow - stillOk.length} rows that still had gaps.`);
    for (const c of used) cols[c] = stillOk.map((r) => cols[c][r]);
    if (dates) dates = stillOk.map((r) => dates![r]);
  }
  const rows = cols[target].length;
  if (rows < 10) throw new Error(`Only ${rows} usable rows are left after cleaning. Try a different missing-value option.`);
  const trainRows = Math.max(2, Math.floor(rows * (1 - opts.testFraction)));

  // 4. outliers (limits come from the training window only)
  if (opts.outliers !== "none") {
    let total = 0;
    for (const c of numericCols) {
      if (c === target && !opts.clipTarget) continue;
      const { values, clipped } = clip(cols[c], opts, trainRows);
      cols[c] = values;
      total += clipped;
    }
    const how = { iqr: `IQR × ${opts.iqrK}`, zscore: `z-score > ${opts.zMax}`, percentile: "1st–99th percentile", none: "" }[opts.outliers];
    report.push(`Clipped ${total} outlier values (${how}).`);
  }

  // 5. categorical features
  const featureColumns: string[] = [];
  const out: Record<string, Cell[]> = {};
  for (const c of features) {
    if (numericCols.includes(c)) {
      out[c] = cols[c];
      featureColumns.push(c);
      continue;
    }
    const levels = Array.from(new Set(cols[c].map(String))).sort();
    if (opts.encoding === "drop") {
      report.push(`Dropped text column ${c}.`);
      continue;
    }
    if (opts.encoding === "onehot" && levels.length <= MAX_ONEHOT) {
      for (const lv of levels.slice(1)) {
        const name = `${c}_${lv}`;
        out[name] = cols[c].map((v) => (String(v) === lv ? 1 : 0));
        featureColumns.push(name);
      }
      report.push(`One-hot encoded ${c} into ${levels.length - 1} columns.`);
    } else {
      const map = new Map(levels.map((lv, i) => [lv, i]));
      out[c] = cols[c].map((v) => map.get(String(v)) ?? 0);
      featureColumns.push(c);
      report.push(
        opts.encoding === "onehot"
          ? `Label encoded ${c} (${levels.length} categories is too many for one-hot).`
          : `Label encoded ${c} (${levels.length} categories).`,
      );
    }
  }

  // 6. scaling (fit on the training window, applied everywhere)
  if (opts.scaling !== "none") {
    const scaled: string[] = [];
    for (const c of features.filter((f) => numericCols.includes(f))) {
      const xs = numbers(out[c].slice(0, trainRows));
      if (opts.scaling === "standard") {
        const m = mean(xs);
        const s = std(xs) || 1;
        out[c] = out[c].map((v) => (typeof v === "number" ? (v - m) / s : v));
      } else {
        const lo = Math.min(...xs);
        const span = Math.max(...xs) - lo || 1;
        out[c] = out[c].map((v) => (typeof v === "number" ? (v - lo) / span : v));
      }
      scaled.push(c);
    }
    if (scaled.length) report.push(`${opts.scaling === "standard" ? "Standardised" : "Min-max scaled"} ${scaled.length} numeric features using the training window.`);
  }

  out[target] = cols[target];
  const columns = [...featureColumns, target];
  const classes = isClass ? Array.from(new Set(cols[target].map(String))).sort() : null;
  if (isClass) report.push(`Classification target with ${classes!.length} classes: ${classes!.slice(0, 8).join(", ")}${classes!.length > 8 ? "…" : ""}.`);
  if (!report.length) report.push("No cleaning was needed.");
  cols = out;

  return {
    columns,
    rows,
    data: cols,
    target,
    featureColumns,
    dates,
    csv: toCsv(columns, cols, rows),
    report,
    droppedRows: ds.nRows - rows,
    classes,
  };
}
