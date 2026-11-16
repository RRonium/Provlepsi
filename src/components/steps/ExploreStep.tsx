"use client";
import { Activity, ArrowRight, BarChart3, BoxSelect, CalendarClock, Grid3x3, LineChart, Waves } from "lucide-react";
import { useMemo, useState } from "react";
import type { Data } from "plotly.js";
import { parseDateValue } from "@/lib/csv";
import { inferTask } from "@/lib/preprocess";
import { acf, fmt, histogramBins, numbers, pearson, rollingMean, seasonality, summarize, trendSlope } from "@/lib/stats";
import type { Cell, Dataset } from "@/lib/types";
import { useStudio } from "@/store/studio";
import { Plot } from "../Plot";
import { Badge, Button, Card, CardHeader, Label, SectionTitle, Segmented, Select, Stat, Toggle, cx } from "../ui";

type Tab = "line" | "corr" | "hist" | "box" | "acf";

function useOrder(ds: Dataset, dateCol: string | null) {
  return useMemo(() => {
    const idx = Array.from({ length: ds.nRows }, (_, i) => i);
    if (!dateCol) return { order: idx, x: idx as (string | number)[], isDate: false };
    const meta = ds.columns.find((c) => c.name === dateCol);
    const raw = ds.data[dateCol];
    const keys = raw.map((v) => (meta?.kind === "datetime" ? parseDateValue(v) : typeof v === "number" ? v : null));
    const order = idx.filter((i) => keys[i] !== null).sort((a, b) => (keys[a] as number) - (keys[b] as number));
    const isDate = meta?.kind === "datetime";
    const x = order.map((i) => (isDate ? new Date(keys[i] as number).toISOString() : (keys[i] as number)));
    return { order, x, isDate };
  }, [ds, dateCol]);
}

function inferFrequency(x: (string | number)[], isDate: boolean): string | null {
  if (!isDate || x.length < 3) return null;
  const diffs: number[] = [];
  for (let i = 1; i < Math.min(x.length, 500); i++) diffs.push(Date.parse(String(x[i])) - Date.parse(String(x[i - 1])));
  diffs.sort((a, b) => a - b);
  const d = diffs[Math.floor(diffs.length / 2)] / 864e5;
  if (d < 1 / 24 + 1e-6) return "sub-hourly";
  if (d < 0.99) return "hourly";
  if (d < 1.5) return "daily";
  if (d < 8) return "weekly";
  if (d < 32) return "monthly";
  if (d < 95) return "quarterly";
  return "yearly";
}

function MiniHist({ values }: { values: Cell[] }) {
  const { counts } = histogramBins(numbers(values), 16);
  const max = Math.max(1, ...counts);
  return (
    <svg viewBox="0 0 64 18" className="h-[18px] w-16" aria-hidden>
      {counts.map((c, i) => (
        <rect key={i} x={i * 4} y={18 - (c / max) * 18} width={3.2} height={(c / max) * 18} rx={0.6} className="fill-accent/70" />
      ))}
    </svg>
  );
}

export function ExploreStep() {
  const ds = useStudio((s) => s.dataset)!;
  const prep = useStudio((s) => s.prep)!;
  const setPrep = useStudio((s) => s.setPrep);
  const go = useStudio((s) => s.go);
  const [tab, setTab] = useState<Tab>("line");
  const [rollWin, setRollWin] = useState(7);
  const [showRoll, setShowRoll] = useState(true);
  const [histCol, setHistCol] = useState<string | null>(null);
  const [boxStandard, setBoxStandard] = useState(true);

  const target = prep.target;
  const numericCols = ds.columns.filter((c) => c.kind === "numeric");
  const { order, x, isDate } = useOrder(ds, prep.dateCol);
  const freq = inferFrequency(x, isDate);
  const totalMissing = ds.columns.reduce((s, c) => s + c.missing, 0);

  const targetSeries = useMemo(() => (target ? order.map((i) => ds.data[target][i]) : []), [ds, order, target]);
  const targetNums = useMemo(() => targetSeries.map((v) => (typeof v === "number" ? v : null)), [targetSeries]);
  const targetMeta = ds.columns.find((c) => c.name === target);
  const targetIsNumeric = targetMeta?.kind === "numeric";

  const insights = useMemo(() => {
    if (!targetIsNumeric) return null;
    const gapless = targetNums.filter((v): v is number => v !== null);
    const s = summarize(targetSeries);
    return { s, slope: trendSlope(gapless), season: seasonality(gapless), acf: acf(gapless, Math.min(60, Math.floor(gapless.length / 3))), n: gapless.length };
  }, [targetIsNumeric, targetNums, targetSeries]);

  const charts = useMemo((): { data: Data[]; layout?: object } => {
    if (tab === "line") {
      if (!target) return { data: [] };
      const traces: Data[] = [
        { type: "scattergl", mode: "lines", x, y: targetIsNumeric ? targetNums : targetSeries, name: target, line: { width: 1.4, color: "#5cc8ef" } },
      ];
      if (showRoll && targetIsNumeric)
        traces.push({ type: "scattergl", mode: "lines", x, y: rollingMean(targetNums, rollWin), name: `${rollWin}-step rolling mean`, line: { width: 2.2, color: "#ff7a45" } });
      return { data: traces, layout: { xaxis: { title: { text: prep.dateCol ?? "row" } }, yaxis: { title: { text: target } } } };
    }
    if (tab === "corr") {
      const cols = numericCols.slice(0, 30).map((c) => c.name);
      const z = cols.map((a) => cols.map((b) => (a === b ? 1 : pearson(ds.data[a], ds.data[b]))));
      return {
        data: [
          {
            type: "heatmap",
            x: cols,
            y: cols,
            z,
            zmin: -1,
            zmax: 1,
            colorscale: [
              [0, "#2563eb"],
              [0.5, "rgba(128,128,128,0.08)"],
              [1, "#ff7a45"],
            ],
            text: z.map((r) => r.map((v) => (Number.isFinite(v) ? v.toFixed(2) : ""))) as unknown as string[],
            texttemplate: cols.length <= 12 ? "%{text}" : "",
            hovertemplate: "%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
          } as Data,
        ],
        layout: { hovermode: "closest", margin: { l: 110, r: 16, t: 10, b: 100 }, yaxis: { autorange: "reversed" } },
      };
    }
    if (tab === "hist") {
      const col = histCol ?? target ?? numericCols[0]?.name;
      if (!col) return { data: [] };
      return {
        data: [{ type: "histogram", x: numbers(ds.data[col]), nbinsx: 40, marker: { color: "#ff7a45", line: { width: 0 } }, opacity: 0.9, name: col } as Data],
        layout: { bargap: 0.04, hovermode: "closest", xaxis: { title: { text: col } }, yaxis: { title: { text: "count" } } },
      };
    }
    if (tab === "box") {
      const cols = numericCols.slice(0, 16);
      return {
        data: cols.map((c) => {
          const xs = numbers(ds.data[c.name]);
          const s = summarize(ds.data[c.name]);
          const y = boxStandard && s && s.std ? xs.map((v) => (v - s.mean) / s.std) : xs;
          return { type: "box", y, name: c.name, boxpoints: "suspectedoutliers", marker: { size: 3 }, line: { width: 1.2 } } as Data;
        }),
        layout: { showlegend: false, hovermode: "closest", yaxis: { title: { text: boxStandard ? "z-score" : "value" } } },
      };
    }
    // acf
    if (!insights) return { data: [] };
    const band = 1.96 / Math.sqrt(insights.n);
    const lags = insights.acf.map((_, i) => i);
    return {
      data: [
        { type: "bar", x: lags, y: insights.acf, marker: { color: insights.acf.map((v) => (Math.abs(v) > band ? "#ff7a45" : "#64748b")) }, name: "ACF", hovertemplate: "lag %{x}: %{y:.3f}<extra></extra>" },
        { type: "scatter", mode: "lines", x: [0, lags.length - 1], y: [band, band], line: { dash: "dot", color: "#5cc8ef", width: 1 }, name: "95% band", hoverinfo: "skip" },
        { type: "scatter", mode: "lines", x: [0, lags.length - 1], y: [-band, -band], line: { dash: "dot", color: "#5cc8ef", width: 1 }, showlegend: false, hoverinfo: "skip" },
      ],
      layout: { bargap: 0.35, hovermode: "closest", xaxis: { title: { text: "lag" } }, yaxis: { title: { text: "autocorrelation" }, range: [-1, 1] } },
    };
  }, [tab, target, x, targetIsNumeric, targetNums, targetSeries, showRoll, rollWin, prep.dateCol, numericCols, ds, histCol, boxStandard, insights]);

  const span = isDate && x.length ? `${String(x[0]).slice(0, 10)} → ${String(x[x.length - 1]).slice(0, 10)}` : null;

  return (
    <div>
      <SectionTitle eyebrow="Step 2 · Exploratory data analysis" title={ds.name.replace(/\.(csv|tsv|txt)$/i, "")}>
        <Button variant="primary" size="lg" onClick={() => go(2)} disabled={!target} icon={<ArrowRight className="size-4" />}>
          Choose preprocessing
        </Button>
      </SectionTitle>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
        <Stat label="Rows" value={ds.nRows.toLocaleString()} hint={freq ? `${freq} observations` : undefined} />
        <Stat label="Columns" value={ds.columns.length} hint={`${numericCols.length} numeric · ${ds.columns.length - numericCols.length} other`} />
        <Stat label="Missing cells" value={`${((totalMissing / (ds.nRows * ds.columns.length)) * 100).toFixed(1)}%`} hint={`${totalMissing.toLocaleString()} empty values`} />
        <Stat label="Time span" value={span ? <span className="text-base">{span}</span> : "—"} hint={prep.dateCol ? `from ${prep.dateCol}` : "no time column picked"} />
        <Stat label="Task" value={<span className="capitalize">{prep.task}</span>} hint={target ? `predicting ${target}` : "pick a target"} tone="accent" />
      </div>

      <div className="mt-6 grid gap-6 xl:grid-cols-[320px_1fr]">
        <div className="space-y-6">
          <Card className="p-5">
            <div className="space-y-4">
              <div>
                <Label help="The column you want to forecast or classify.">Target column</Label>
                <Select
                  ariaLabel="Target column"
                  value={target}
                  onChange={(v) =>
                    setPrep({
                      target: v,
                      task: inferTask(ds, v),
                      features: ds.columns.filter((c) => c.kind === "numeric" && c.name !== v && c.name !== prep.dateCol).map((c) => c.name),
                    })
                  }
                  options={ds.columns.filter((c) => c.name !== prep.dateCol).map((c) => ({ value: c.name, label: `${c.name} · ${c.kind}` }))}
                />
              </div>
              <div>
                <Label help="Used to order rows in time and label the charts. It is not fed to the models.">Time column</Label>
                <Select
                  ariaLabel="Time column"
                  value={(prep.dateCol ?? "__none") as string}
                  onChange={(v) => {
                    const dateCol = v === "__none" ? null : v;
                    setPrep({ dateCol, features: prep.features.filter((f) => f !== dateCol) });
                  }}
                  options={[
                    { value: "__none", label: "None (use row order)" },
                    ...ds.columns.filter((c) => c.name !== target && c.kind !== "categorical").map((c) => ({ value: c.name, label: `${c.name} · ${c.kind}` })),
                  ]}
                />
              </div>
            </div>
          </Card>

          {insights?.s && (
            <Card>
              <CardHeader title="Target profile" subtitle={target ?? undefined} icon={<Activity className="size-4" />} />
              <dl className="grid grid-cols-2 gap-x-4 gap-y-3 px-5 pb-5 text-[13px]">
                {[
                  ["Mean", fmt(insights.s.mean)],
                  ["Variance", fmt(insights.s.variance)],
                  ["Std. dev.", fmt(insights.s.std)],
                  ["Median", fmt(insights.s.median)],
                  ["Min", fmt(insights.s.min)],
                  ["Max", fmt(insights.s.max)],
                  ["Skew", fmt(insights.s.skew, 2)],
                  ["Count", insights.s.count.toLocaleString()],
                ].map(([k, v]) => (
                  <div key={k}>
                    <dt className="text-[11px] tracking-wide text-faint uppercase">{k}</dt>
                    <dd className="tabular mt-0.5 font-medium">{v}</dd>
                  </div>
                ))}
              </dl>
              <div className="space-y-2 border-t border-line px-5 py-4 text-[13px]">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-muted">Trend</span>
                  <Badge tone={Math.abs(insights.slope) * insights.n > 0.5 * insights.s.std ? "accent" : "neutral"}>
                    {Math.abs(insights.slope) * insights.n > 0.5 * insights.s.std ? (insights.slope > 0 ? "Rising" : "Falling") : "Flat"} · {fmt(insights.slope, 3)}/step
                  </Badge>
                </div>
                <div className="flex items-center justify-between gap-2">
                  <span className="text-muted">Seasonality</span>
                  <Badge tone={insights.season ? "cyan" : "neutral"}>
                    {insights.season ? `Every ${insights.season.period} steps · r=${insights.season.strength.toFixed(2)}` : "None detected"}
                  </Badge>
                </div>
              </div>
            </Card>
          )}
        </div>

        <Card className="min-w-0">
          <div className="flex flex-wrap items-center justify-between gap-3 px-5 pt-5">
            <Segmented
              value={tab}
              onChange={setTab}
              options={[
                { value: "line", label: "Line plot", icon: <LineChart className="size-3.5" /> },
                { value: "corr", label: "Correlation", icon: <Grid3x3 className="size-3.5" /> },
                { value: "hist", label: "Histogram", icon: <BarChart3 className="size-3.5" /> },
                { value: "box", label: "Box plot", icon: <BoxSelect className="size-3.5" /> },
                { value: "acf", label: "Autocorrelation", icon: <Waves className="size-3.5" /> },
              ]}
            />
            <div className="flex items-center gap-3">
              {tab === "line" && targetIsNumeric && (
                <>
                  <Toggle checked={showRoll} onChange={setShowRoll} label="Rolling mean" />
                  {showRoll && (
                    <input type="range" min={2} max={Math.max(3, Math.min(120, Math.floor(ds.nRows / 4)))} value={rollWin} onChange={(e) => setRollWin(Number(e.target.value))} className="w-24" aria-label="Rolling window" />
                  )}
                </>
              )}
              {tab === "hist" && (
                <Select ariaLabel="Histogram column" className="h-8 w-44 text-[13px]" value={histCol ?? target ?? numericCols[0]?.name ?? null} onChange={setHistCol} options={numericCols.map((c) => ({ value: c.name, label: c.name }))} />
              )}
              {tab === "box" && <Toggle checked={boxStandard} onChange={setBoxStandard} label="Standardise" />}
            </div>
          </div>
          <div className="px-3 pt-2 pb-3">
            {charts.data.length ? (
              <Plot data={charts.data} layout={charts.layout} height={430} filename={`provlepsi-${tab}`} />
            ) : (
              <div className="grid h-[430px] place-items-center text-sm text-muted">
                {tab === "acf" ? "Autocorrelation needs a numeric target." : "Pick a target column to see this chart."}
              </div>
            )}
          </div>
          {tab === "acf" && insights && (
            <p className="px-5 pb-5 text-[12.5px] text-muted">
              Bars outside the dotted band are significant. Slowly decaying bars mean a trend; regular spikes mean a seasonal cycle, which is a good hint for the
              look-back window and ARIMA orders.
            </p>
          )}
        </Card>
      </div>

      <Card className="mt-6 overflow-hidden">
        <CardHeader title="Columns" subtitle="Detected type, completeness and distribution of every column" icon={<CalendarClock className="size-4" />} />
        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] text-[13px]">
            <thead>
              <tr className="border-y border-line bg-panel-2/60 text-left text-[11px] tracking-wide text-faint uppercase">
                {["Column", "Type", "Missing", "Unique", "Mean", "Std", "Min", "Max", "Distribution"].map((h) => (
                  <th key={h} className="px-4 py-2.5 font-medium">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ds.columns.map((c) => {
                const s = c.kind === "numeric" ? summarize(ds.data[c.name]) : null;
                return (
                  <tr key={c.name} className={cx("border-b border-line last:border-0", c.name === target && "bg-accent-soft/60")}>
                    <td className="px-4 py-2.5 font-medium">
                      {c.name}
                      {c.name === target && <Badge tone="accent" className="ml-2">target</Badge>}
                      {c.name === prep.dateCol && <Badge tone="cyan" className="ml-2">time</Badge>}
                    </td>
                    <td className="px-4 py-2.5">
                      <Badge>{c.kind}</Badge>
                    </td>
                    <td className={cx("tabular px-4 py-2.5", c.missing ? "text-warn" : "text-muted")}>{c.missing ? `${c.missing} (${((c.missing / ds.nRows) * 100).toFixed(1)}%)` : "0"}</td>
                    <td className="tabular px-4 py-2.5 text-muted">{c.unique > 10000 ? "10k+" : c.unique.toLocaleString()}</td>
                    <td className="tabular px-4 py-2.5">{s ? fmt(s.mean, 3) : "—"}</td>
                    <td className="tabular px-4 py-2.5">{s ? fmt(s.std, 3) : "—"}</td>
                    <td className="tabular px-4 py-2.5">{s ? fmt(s.min, 3) : "—"}</td>
                    <td className="tabular px-4 py-2.5">{s ? fmt(s.max, 3) : "—"}</td>
                    <td className="px-4 py-2.5">{s ? <MiniHist values={ds.data[c.name]} /> : <span className="text-faint">—</span>}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
