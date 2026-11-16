"use client";
import { ArrowLeft, ArrowRight, Check, Eraser, ListChecks, Scale, Scissors, SplitSquareHorizontal, Tags } from "lucide-react";
import { useMemo, type ReactNode } from "react";
import type { Data } from "plotly.js";
import { preprocess } from "@/lib/preprocess";
import { fmt } from "@/lib/stats";
import type { EncodingStrategy, MissingStrategy, OutlierStrategy, ScalingStrategy, TaskType } from "@/lib/types";
import { useStudio } from "@/store/studio";
import { Plot } from "../Plot";
import { Badge, Button, Callout, Card, CardHeader, Label, NumberField, SectionTitle, Segmented, Toggle, cx } from "../ui";

function Choice<T extends string>({ value, onChange, options }: { value: T; onChange: (v: T) => void; options: { value: T; label: string; hint: string }[] }) {
  return (
    <div className="grid gap-2 sm:grid-cols-2">
      {options.map((o) => (
        <button
          key={o.value}
          onClick={() => onChange(o.value)}
          className={cx(
            "flex items-start gap-2.5 rounded-xl border p-3 text-left transition",
            value === o.value ? "border-accent bg-accent-soft" : "border-line hover:border-line-strong",
          )}
        >
          <span className={cx("mt-0.5 grid size-4 shrink-0 place-items-center rounded-full border", value === o.value ? "border-accent bg-accent text-accent-fg" : "border-line-strong")}>
            {value === o.value && <Check className="size-3" strokeWidth={3} />}
          </span>
          <span>
            <span className="block text-[13px] font-medium">{o.label}</span>
            <span className="block text-[12px] leading-snug text-muted">{o.hint}</span>
          </span>
        </button>
      ))}
    </div>
  );
}

function Section({ icon, title, subtitle, children }: { icon: ReactNode; title: string; subtitle: string; children: ReactNode }) {
  return (
    <Card>
      <CardHeader title={title} subtitle={subtitle} icon={icon} />
      <div className="px-5 pb-5">{children}</div>
    </Card>
  );
}

export function PrepareStep() {
  const ds = useStudio((s) => s.dataset)!;
  const prep = useStudio((s) => s.prep)!;
  const setPrep = useStudio((s) => s.setPrep);
  const commitPrep = useStudio((s) => s.commitPrep);
  const go = useStudio((s) => s.go);

  const preview = useMemo(() => {
    try {
      return { data: preprocess(ds, prep), error: null };
    } catch (e) {
      return { data: null, error: e instanceof Error ? e.message : String(e) };
    }
  }, [ds, prep]);

  const candidates = ds.columns.filter((c) => c.name !== prep.target && c.name !== prep.dateCol);
  const hasCategorical = prep.features.some((f) => ds.columns.find((c) => c.name === f)?.kind === "categorical");
  const p = preview.data;

  const chart = useMemo((): Data[] => {
    if (!p || prep.task !== "regression") return [];
    const x = p.dates ?? Array.from({ length: p.rows }, (_, i) => i);
    const split = Math.floor(p.rows * (1 - prep.testFraction));
    const y = p.data[p.target] as number[];
    return [
      { type: "scattergl", mode: "lines", x: x.slice(0, split + 1), y: y.slice(0, split + 1), name: "Training window", line: { color: "#5cc8ef", width: 1.4 } },
      { type: "scattergl", mode: "lines", x: x.slice(split), y: y.slice(split), name: "Test window", line: { color: "#ff7a45", width: 1.4 } },
    ];
  }, [p, prep.task, prep.testFraction]);

  return (
    <div>
      <SectionTitle eyebrow="Step 3 · Preprocessing" title="Choose how to clean the data">
        <div className="flex gap-2">
          <Button variant="ghost" size="lg" onClick={() => go(1)} icon={<ArrowLeft className="size-4" />}>
            Back
          </Button>
          <Button
            variant="primary"
            size="lg"
            disabled={!p}
            onClick={() => {
              commitPrep();
              go(3);
            }}
            icon={<ArrowRight className="size-4" />}
          >
            Pick models
          </Button>
        </div>
      </SectionTitle>

      <div className="grid gap-6 xl:grid-cols-[1fr_420px]">
        <div className="space-y-6">
          <Section icon={<ListChecks className="size-4" />} title="Task and features" subtitle={`Predicting ${prep.target}. Choose which other columns the models may use.`}>
            <div className="mb-4 flex flex-wrap items-center gap-3">
              <Segmented<TaskType>
                value={prep.task}
                onChange={(task) => setPrep({ task })}
                options={[
                  { value: "regression", label: "Forecast a number" },
                  { value: "classification", label: "Classify a label" },
                ]}
              />
              <span className="text-[12px] text-muted">Detected automatically from the target. Switch it if the guess is wrong.</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {candidates.map((c) => {
                const on = prep.features.includes(c.name);
                return (
                  <button
                    key={c.name}
                    onClick={() => setPrep({ features: on ? prep.features.filter((f) => f !== c.name) : [...prep.features, c.name] })}
                    className={cx(
                      "inline-flex h-8 items-center gap-1.5 rounded-full border px-3 text-[12.5px] transition",
                      on ? "border-accent bg-accent-soft text-fg" : "border-line text-muted hover:border-line-strong",
                    )}
                  >
                    {on && <Check className="size-3.5 text-accent" />}
                    {c.name}
                    <span className="text-faint">{c.kind === "numeric" ? "123" : c.kind === "datetime" ? "date" : "abc"}</span>
                  </button>
                );
              })}
              {!candidates.length && <span className="text-[13px] text-muted">No other columns. Univariate models (ARIMA, random forest, deep learning) still work.</span>}
            </div>
          </Section>

          <Section icon={<Eraser className="size-4" />} title="Missing values" subtitle="How gaps in the data are handled before training.">
            <Choice<MissingStrategy>
              value={prep.missing}
              onChange={(missing) => setPrep({ missing })}
              options={[
                { value: "interpolate", label: "Interpolate", hint: "Draw a straight line across each gap. Best for smooth series." },
                { value: "ffill", label: "Forward fill", hint: "Carry the last known value forward." },
                { value: "bfill", label: "Backward fill", hint: "Use the next known value." },
                { value: "mean", label: "Column mean", hint: "Replace gaps with the average." },
                { value: "median", label: "Column median", hint: "Like mean, but robust to outliers." },
                { value: "drop", label: "Drop rows", hint: "Remove any row with a gap." },
              ]}
            />
          </Section>

          <Section icon={<Scissors className="size-4" />} title="Outliers" subtitle="Clip extreme values to a limit learned from the training window.">
            <Choice<OutlierStrategy>
              value={prep.outliers}
              onChange={(outliers) => setPrep({ outliers })}
              options={[
                { value: "none", label: "Keep as is", hint: "No clipping." },
                { value: "iqr", label: "IQR fences", hint: "Clip beyond Q1 − k·IQR and Q3 + k·IQR." },
                { value: "zscore", label: "Z-score", hint: "Clip values more than z standard deviations out." },
                { value: "percentile", label: "1st–99th percentile", hint: "Winsorise the extreme 1% on each side." },
              ]}
            />
            {prep.outliers !== "none" && (
              <div className="mt-4 flex flex-wrap items-end gap-5">
                {prep.outliers === "iqr" && (
                  <div className="w-32">
                    <Label>k</Label>
                    <NumberField value={prep.iqrK} min={0.5} max={5} step={0.1} onChange={(v) => setPrep({ iqrK: v ?? 1.5 })} />
                  </div>
                )}
                {prep.outliers === "zscore" && (
                  <div className="w-32">
                    <Label>z</Label>
                    <NumberField value={prep.zMax} min={1} max={6} step={0.1} onChange={(v) => setPrep({ zMax: v ?? 3 })} />
                  </div>
                )}
                {prep.task === "regression" && <Toggle checked={prep.clipTarget} onChange={(clipTarget) => setPrep({ clipTarget })} label="Also clip the target" />}
              </div>
            )}
          </Section>

          <div className="grid gap-6 lg:grid-cols-2">
            <Section icon={<Tags className="size-4" />} title="Text columns" subtitle={hasCategorical ? "How categorical features become numbers." : "None of the chosen features are text."}>
              <Segmented<EncodingStrategy>
                value={prep.encoding}
                onChange={(encoding) => setPrep({ encoding })}
                options={[
                  { value: "onehot", label: "One-hot" },
                  { value: "label", label: "Label" },
                  { value: "drop", label: "Drop" },
                ]}
              />
            </Section>
            <Section icon={<Scale className="size-4" />} title="Feature scaling" subtitle="Helps Ridge, Lasso and classifiers. The target is never scaled.">
              <Segmented<ScalingStrategy>
                value={prep.scaling}
                onChange={(scaling) => setPrep({ scaling })}
                options={[
                  { value: "none", label: "None" },
                  { value: "standard", label: "Standard" },
                  { value: "minmax", label: "Min-max" },
                ]}
              />
            </Section>
          </div>

          <Section icon={<SplitSquareHorizontal className="size-4" />} title="Chronological split" subtitle="The last part of the series is held out for testing. Rows are never shuffled.">
            <div className="flex items-center gap-4">
              <input
                type="range"
                min={0.05}
                max={0.5}
                step={0.05}
                value={prep.testFraction}
                onChange={(e) => setPrep({ testFraction: Number(e.target.value) })}
                className="flex-1"
                aria-label="Test fraction"
              />
              <div className="tabular w-40 text-right text-[13px]">
                <span className="font-semibold text-fg">{Math.round((1 - prep.testFraction) * 100)}%</span> train ·{" "}
                <span className="font-semibold text-accent">{Math.round(prep.testFraction * 100)}%</span> test
              </div>
            </div>
          </Section>
        </div>

        <div className="space-y-6 xl:sticky xl:top-24 xl:self-start">
          <Card>
            <CardHeader title="Preview" subtitle="Updates as you change options" actions={p && <Badge tone="good">{p.rows.toLocaleString()} rows ready</Badge>} />
            <div className="space-y-4 px-5 pb-5">
              {preview.error && <Callout tone="bad">{preview.error}</Callout>}
              {p && (
                <>
                  <div className="grid grid-cols-3 gap-2">
                    <div className="rounded-lg bg-panel-2 p-2.5">
                      <div className="text-[11px] text-faint uppercase">Features</div>
                      <div className="tabular text-lg font-semibold">{p.featureColumns.length}</div>
                    </div>
                    <div className="rounded-lg bg-panel-2 p-2.5">
                      <div className="text-[11px] text-faint uppercase">Train</div>
                      <div className="tabular text-lg font-semibold">{Math.floor(p.rows * (1 - prep.testFraction)).toLocaleString()}</div>
                    </div>
                    <div className="rounded-lg bg-panel-2 p-2.5">
                      <div className="text-[11px] text-faint uppercase">Test</div>
                      <div className="tabular text-lg font-semibold text-accent">{(p.rows - Math.floor(p.rows * (1 - prep.testFraction))).toLocaleString()}</div>
                    </div>
                  </div>
                  <ul className="space-y-1.5 text-[12.5px] text-muted">
                    {p.report.map((r) => (
                      <li key={r} className="flex gap-2">
                        <Check className="mt-0.5 size-3.5 shrink-0 text-good" /> {r}
                      </li>
                    ))}
                  </ul>
                  {chart.length > 0 && <Plot data={chart} height={220} layout={{ margin: { l: 44, r: 8, t: 8, b: 30 }, legend: { orientation: "h", y: -0.2 } }} />}
                  <div className="overflow-x-auto rounded-lg border border-line">
                    <table className="w-full text-[12px]">
                      <thead>
                        <tr className="bg-panel-2 text-left text-faint">
                          {p.dates && <th className="px-2.5 py-1.5 font-medium">{prep.dateCol}</th>}
                          {p.columns.slice(-5).map((c) => (
                            <th key={c} className={cx("px-2.5 py-1.5 font-medium whitespace-nowrap", c === p.target && "text-accent")}>
                              {c}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {Array.from({ length: Math.min(6, p.rows) }, (_, r) => (
                          <tr key={r} className="border-t border-line">
                            {p.dates && <td className="px-2.5 py-1.5 whitespace-nowrap text-muted">{p.dates[r]}</td>}
                            {p.columns.slice(-5).map((c) => {
                              const v = p.data[c][r];
                              return (
                                <td key={c} className="tabular px-2.5 py-1.5 whitespace-nowrap">
                                  {typeof v === "number" ? fmt(v, 3) : String(v)}
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
