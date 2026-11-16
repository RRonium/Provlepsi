"use client";
import { ArrowLeft, Download, FlaskConical, Medal, Plus, Sparkles, Trophy, Wand2 } from "lucide-react";
import { useMemo, useState } from "react";
import type { Data } from "plotly.js";
import { downloadText } from "@/lib/csv";
import { modelById } from "@/lib/models";
import { fmt } from "@/lib/stats";
import type { ModelResult, ProcessedData } from "@/lib/types";
import { useStudio } from "@/store/studio";
import { Plot, SERIES_COLORS } from "../Plot";
import { Badge, Button, Callout, Card, CardHeader, Label, NumberField, SectionTitle, Segmented, Stat, cx } from "../ui";

const MAX_CONTEXT = 300;

function xAxisFor(p: ProcessedData) {
  const at = (i: number) => (p.dates ? p.dates[i] : i);
  const future = (k: number) => {
    if (!p.dates) return Array.from({ length: k }, (_, i) => p.rows + i);
    const t = p.dates.slice(-6).map((d) => Date.parse(d));
    const diffs = t.slice(1).map((v, i) => v - t[i]).filter((d) => Number.isFinite(d) && d > 0).sort((a, b) => a - b);
    if (!diffs.length) return Array.from({ length: k }, (_, i) => `+${i + 1}`);
    const step = diffs[Math.floor(diffs.length / 2)];
    const last = t[t.length - 1];
    const daily = step >= 864e5 * 0.99;
    return Array.from({ length: k }, (_, i) => {
      const iso = new Date(last + step * (i + 1)).toISOString();
      return daily ? iso.slice(0, 10) : iso.slice(0, 19);
    });
  };
  return { at, future };
}

function labelOf(result: ModelResult, v: number | null) {
  const map = result.extras.class_key_map as Record<string, string> | undefined;
  return v === null ? "" : (map?.[String(v)] ?? String(v));
}

function forecastCsv(p: ProcessedData, result: ModelResult, isClass: boolean) {
  const { at, future } = xAxisFor(p);
  const lines = [`${p.dates ? "date" : "row"},actual,predicted,kind`];
  const test = result.test;
  const pred = result.testPredictions;
  if (test && pred) {
    test.index.forEach((ix, i) => {
      const a = isClass ? labelOf(result, test.values[i]) : (test.values[i] ?? "");
      const b = isClass ? labelOf(result, pred.values[i]) : (pred.values[i] ?? "");
      lines.push(`${at(ix)},${a},${b},test`);
    });
  }
  if (result.forecast?.length) {
    const xs = future(result.forecast.length);
    result.forecast.forEach((v, i) => lines.push(`${xs[i]},,${v},forecast`));
  }
  return lines.join("\n");
}

function Playground({ id, result }: { id: string; result: ModelResult }) {
  const spec = modelById(id);
  const predict = useStudio((s) => s.predict);
  const reforecast = useStudio((s) => s.reforecast);
  const [entry, setEntry] = useState<Record<string, number | null>>(() =>
    Object.fromEntries(result.featureColumns.map((c) => [c, typeof result.defaultEntry[c] === "number" ? (result.defaultEntry[c] as number) : null])),
  );
  const [steps, setSteps] = useState<number>(result.forecast?.length ?? 10);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const run = async (fn: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  if (spec.engine === "tf" || spec.id === "arima") {
    return (
      <div className="space-y-4">
        <p className="text-[13px] text-muted">Forecast any number of steps past the end of your data. The model feeds each prediction back in as input for the next step.</p>
        <div className="flex items-end gap-3">
          <div className="w-36">
            <Label>Steps ahead</Label>
            <NumberField value={steps} min={1} max={1000} step={1} onChange={(v) => setSteps(Math.max(1, Math.round(v ?? 1)))} />
          </div>
          <Button
            variant="primary"
            loading={busy}
            icon={<Wand2 className="size-4" />}
            onClick={() => void run(() => (spec.engine === "tf" ? reforecast(id, steps) : predict(id, { steps })))}
          >
            Forecast
          </Button>
        </div>
        {result.forecast && (
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {result.forecast.slice(0, 8).map((v, i) => (
              <div key={i} className="rounded-lg bg-panel-2 px-3 py-2">
                <div className="text-[11px] text-faint">t + {i + 1}</div>
                <div className="tabular text-[14px] font-semibold">{fmt(v)}</div>
              </div>
            ))}
          </div>
        )}
        {error && <Callout tone="bad">{error}</Callout>}
      </div>
    );
  }

  const probs = result.prediction.predicted_probabilities as Record<string, number> | undefined;
  return (
    <div className="space-y-4">
      <p className="text-[13px] text-muted">
        Edit any input and predict. Values start at the most recent row{result.featureColumns.some((c) => /lag/.test(c)) ? ", including the lag columns the script builds" : ""}.
      </p>
      <div className="grid max-h-[340px] gap-3 overflow-y-auto pr-1 sm:grid-cols-2 lg:grid-cols-3">
        {result.featureColumns.map((c) => (
          <div key={c}>
            <Label>{c}</Label>
            <NumberField value={entry[c] ?? null} step={0.01} nullable onChange={(v) => setEntry((e) => ({ ...e, [c]: v }))} />
          </div>
        ))}
      </div>
      <Button variant="primary" loading={busy} icon={<Wand2 className="size-4" />} onClick={() => void run(() => predict(id, entry))}>
        Predict
      </Button>
      {error && <Callout tone="bad">{error}</Callout>}
      {"single_prediction" in result.prediction && (
        <div className="rounded-xl border border-accent/40 bg-accent-soft px-4 py-3">
          <div className="text-[11px] tracking-wide text-faint uppercase">Predicted value</div>
          <div className="tabular text-2xl font-semibold text-accent">{fmt(result.prediction.single_prediction as number)}</div>
        </div>
      )}
      {"predicted_class_label" in result.prediction && (
        <div className="rounded-xl border border-accent/40 bg-accent-soft px-4 py-3">
          <div className="text-[11px] tracking-wide text-faint uppercase">Predicted class</div>
          <div className="text-2xl font-semibold text-accent">{String(result.prediction.predicted_class_label)}</div>
          {probs && (
            <div className="mt-3 space-y-1.5">
              {Object.entries(probs)
                .sort((a, b) => b[1] - a[1])
                .map(([k, v]) => (
                  <div key={k} className="flex items-center gap-3 text-[12.5px]">
                    <span className="w-24 truncate">{k}</span>
                    <div className="h-2 flex-1 overflow-hidden rounded-full bg-panel-2">
                      <div className="h-full rounded-full bg-accent" style={{ width: `${v * 100}%` }} />
                    </div>
                    <span className="tabular w-12 text-right text-muted">{(v * 100).toFixed(1)}%</span>
                  </div>
                ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ModelDetail({ id, result, p, isClass }: { id: string; result: ModelResult; p: ProcessedData; isClass: boolean }) {
  const spec = modelById(id);
  const { at, future } = useMemo(() => xAxisFor(p), [p]);

  const main = useMemo((): Data[] => {
    if (isClass) return [];
    const traces: Data[] = [];
    if (result.train) {
      const n = result.train.index.length;
      const from = Math.max(0, n - MAX_CONTEXT);
      traces.push({
        type: "scatter",
        mode: "lines",
        x: result.train.index.slice(from).map(at),
        y: result.train.values.slice(from),
        name: "History",
        line: { color: "#94a3b8", width: 1.2 },
      });
    }
    if (result.test) traces.push({ type: "scatter", mode: "lines", x: result.test.index.map(at), y: result.test.values, name: "Actual", line: { color: "#5cc8ef", width: 1.8 } });
    if (result.testPredictions)
      traces.push({ type: "scatter", mode: "lines", x: result.testPredictions.index.map(at), y: result.testPredictions.values, name: "Predicted", line: { color: "#ff7a45", width: 2 } });
    if (result.forecast?.length) {
      const lastX = result.test?.index.length ? at(result.test.index[result.test.index.length - 1]) : at(p.rows - 1);
      const lastY = result.test?.values.length ? result.test.values[result.test.values.length - 1] : null;
      traces.push({
        type: "scatter",
        mode: "lines+markers",
        x: [lastX, ...future(result.forecast.length)],
        y: [lastY, ...result.forecast],
        name: `Forecast (${result.forecast.length} steps)`,
        line: { color: "#ff7a45", width: 2, dash: "dot" },
        marker: { size: 4 },
      });
    }
    return traces;
  }, [result, isClass, at, future, p.rows]);

  const weights = result.extras.feature_weights as Record<string, number> | undefined;
  const weightEntries = weights ? Object.entries(weights).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 20) : [];
  const conf = result.extras.conf_matrix as number[][] | undefined;
  const classMap = result.extras.class_key_map as Record<string, string> | undefined;
  const classLabels = classMap ? Object.values(classMap) : [];
  const m = result.metrics;

  return (
    <div className="space-y-6">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {isClass ? (
          <>
            <Stat label="Accuracy" value={`${((m.accuracy ?? 0) * 100).toFixed(1)}%`} tone="accent" />
            <Stat label="Precision (weighted)" value={fmt(m.precision)} />
            <Stat label="Recall (weighted)" value={fmt(m.recall)} />
            <Stat label="Training time" value={`${result.seconds.toFixed(1)}s`} />
          </>
        ) : (
          <>
            <Stat label="RMSE" value={fmt(m.RMSE)} tone="accent" hint="Penalises large misses" />
            <Stat label="MAE" value={fmt(m.MAE)} hint="Average absolute miss" />
            <Stat label="R²" value={fmt(m.R2, 3)} hint={`MSE ${fmt(m.MSE)}`} />
            <Stat label="Training time" value={`${result.seconds.toFixed(1)}s`} hint={m.AIC !== undefined ? `AIC ${fmt(m.AIC, 1)} · BIC ${fmt(m.BIC, 1)}` : result.backend} />
          </>
        )}
      </div>

      {!isClass && (
        <Card>
          <CardHeader
            title="Forecast vs actual"
            subtitle="History (grey), the held-out test window, the model's predictions over it and any forecast beyond the data"
            actions={
              <Button size="sm" icon={<Download className="size-3.5" />} onClick={() => downloadText(`provlepsi-${id}-forecast.csv`, forecastCsv(p, result, false))}>
                CSV
              </Button>
            }
          />
          <div className="px-3 pb-3">
            <Plot data={main} height={400} filename={`provlepsi-${id}-forecast`} layout={{ xaxis: { title: { text: p.dates ? "time" : "row" } }, yaxis: { title: { text: p.target } } }} />
          </div>
        </Card>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader title="Try a prediction" subtitle="Uses the trained model kept in memory" icon={<FlaskConical className="size-4" />} />
          <div className="px-5 pb-5">
            <Playground key={result.sessionId ?? id} id={id} result={result} />
          </div>
        </Card>

        {isClass && conf && (
          <Card>
            <CardHeader
              title="Confusion matrix"
              subtitle="Rows are the true class, columns the predicted class, on the test window"
              actions={
                <Button size="sm" icon={<Download className="size-3.5" />} onClick={() => downloadText(`provlepsi-${id}-predictions.csv`, forecastCsv(p, result, true))}>
                  CSV
                </Button>
              }
            />
            <div className="px-3 pb-3">
              <Plot
                height={340}
                layout={{ hovermode: "closest", margin: { l: 90, r: 16, t: 8, b: 70 }, xaxis: { title: { text: "predicted" } }, yaxis: { title: { text: "actual" }, autorange: "reversed" } }}
                data={[
                  {
                    type: "heatmap",
                    z: conf,
                    x: classLabels.slice(0, conf.length),
                    y: classLabels.slice(0, conf.length),
                    colorscale: [
                      [0, "rgba(128,128,128,0.06)"],
                      [1, "#ff7a45"],
                    ],
                    texttemplate: "%{z}",
                    textfont: { color: "#9ca3af", size: 14 },
                    showscale: false,
                    hovertemplate: "actual %{y} → predicted %{x}: %{z}<extra></extra>",
                  } as Data,
                ]}
              />
            </div>
          </Card>
        )}

        {weightEntries.length > 0 && (
          <Card>
            <CardHeader
              title={spec.id === "gradient_boosting" ? "Feature importance" : "Coefficients"}
              subtitle={spec.id === "lasso" ? "Lasso sets weak lags to exactly zero" : spec.id === "ridge" ? "Ridge shrinks every weight toward zero" : "Share of the ensemble's splits"}
            />
            <div className="px-3 pb-3">
              <Plot
                height={Math.max(220, weightEntries.length * 22 + 60)}
                layout={{ hovermode: "closest", margin: { l: 130, r: 16, t: 8, b: 36 }, yaxis: { autorange: "reversed" } }}
                data={[
                  {
                    type: "bar",
                    orientation: "h",
                    y: weightEntries.map(([k]) => k),
                    x: weightEntries.map(([, v]) => v),
                    marker: { color: weightEntries.map(([, v]) => (v === 0 ? "#64748b" : v > 0 ? "#ff7a45" : "#5cc8ef")) },
                    hovertemplate: "%{y}: %{x:.4f}<extra></extra>",
                  },
                ]}
              />
              {spec.id === "lasso" && Array.isArray(result.extras.zeroed_out_features) && (
                <p className="px-2 pb-2 text-[12.5px] text-muted">
                  Kept {(result.extras.selected_features as string[]).length} features, zeroed out {(result.extras.zeroed_out_features as string[]).length}.
                </p>
              )}
            </div>
          </Card>
        )}

        {result.history && result.history.loss.length > 0 && (
          <Card>
            <CardHeader title="Training history" subtitle="Mean squared error on the scaled series, per epoch" />
            <div className="px-3 pb-3">
              <Plot
                height={280}
                layout={{ yaxis: { type: "log", title: { text: "loss" } }, xaxis: { title: { text: "epoch" } } }}
                data={[
                  { type: "scatter", mode: "lines", y: result.history.loss, name: "train", line: { color: "#5cc8ef" } },
                  { type: "scatter", mode: "lines", y: result.history.valLoss, name: "test", line: { color: "#ff7a45" } },
                ]}
              />
            </div>
          </Card>
        )}
      </div>

      {result.notes.length > 0 && (
        <Callout>
          <ul className="list-disc space-y-1 pl-4">
            {result.notes.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        </Callout>
      )}
    </div>
  );
}

export function ResultsStep() {
  const runs = useStudio((s) => s.runs);
  const processed = useStudio((s) => s.processed);
  const prep = useStudio((s) => s.prep)!;
  const go = useStudio((s) => s.go);
  const newProject = useStudio((s) => s.newProject);
  const isClass = prep.task === "classification";
  const primary = isClass ? "accuracy" : "RMSE";

  const done = Object.entries(runs)
    .filter(([, r]) => r.status === "done" && r.result)
    .map(([id, r]) => ({ id, spec: modelById(id), result: r.result! }))
    .sort((a, b) => {
      const va = a.result.metrics[primary] ?? (isClass ? -Infinity : Infinity);
      const vb = b.result.metrics[primary] ?? (isClass ? -Infinity : Infinity);
      return isClass ? vb - va : va - vb;
    });
  const [picked, setActive] = useState<string>(isClass ? "" : "__compare");
  const active = picked || done[0]?.id || "";
  const best = done[0];

  const compare = useMemo((): Data[] => {
    if (isClass || !processed || !done.length) return [];
    const { at } = xAxisFor(processed);
    const ref = done.find((d) => d.result.test)?.result.test;
    const traces: Data[] = [];
    if (ref) traces.push({ type: "scatter", mode: "lines", x: ref.index.map(at), y: ref.values, name: "Actual", line: { color: "#e2e8f0", width: 2.4 } });
    done.forEach((d, i) => {
      const tp = d.result.testPredictions;
      if (tp) traces.push({ type: "scatter", mode: "lines", x: tp.index.map(at), y: tp.values, name: d.spec.short, line: { color: SERIES_COLORS[i % SERIES_COLORS.length], width: 1.5 } });
    });
    return traces;
  }, [done, isClass, processed]);

  if (!processed) return <Callout tone="warn">Train some models first.</Callout>;

  const leaderboardCsv = () => {
    const cols = isClass ? ["accuracy", "precision", "recall"] : ["RMSE", "MAE", "MSE", "R2"];
    const lines = [["rank", "model", ...cols, "seconds"].join(",")];
    done.forEach((d, i) => lines.push([i + 1, d.spec.name, ...cols.map((c) => d.result.metrics[c] ?? ""), d.result.seconds.toFixed(2)].join(",")));
    downloadText("provlepsi-leaderboard.csv", lines.join("\n"));
  };

  return (
    <div>
      <SectionTitle eyebrow="Step 6 · Results" title={best ? `${best.spec.name} came out on top` : "No model finished"}>
        <div className="flex gap-2">
          <Button variant="ghost" size="lg" onClick={() => go(3)} icon={<ArrowLeft className="size-4" />}>
            Tune models
          </Button>
          <Button variant="secondary" size="lg" onClick={newProject} icon={<Plus className="size-4" />}>
            New project
          </Button>
        </div>
      </SectionTitle>

      {done.length > 0 && (
        <Card className="mb-6 overflow-hidden">
          <CardHeader
            title="Leaderboard"
            subtitle={isClass ? "Ranked by accuracy on the held-out window" : "Ranked by RMSE on the held-out window (lower is better)"}
            icon={<Trophy className="size-4" />}
            actions={
              <Button size="sm" icon={<Download className="size-3.5" />} onClick={leaderboardCsv}>
                CSV
              </Button>
            }
          />
          <div className="overflow-x-auto">
            <table className="w-full min-w-[640px] text-[13px]">
              <thead>
                <tr className="border-y border-line bg-panel-2/60 text-left text-[11px] tracking-wide text-faint uppercase">
                  <th className="w-12 px-5 py-2.5 font-medium">#</th>
                  <th className="px-3 py-2.5 font-medium">Model</th>
                  {(isClass ? ["Accuracy", "Precision", "Recall"] : ["RMSE", "MAE", "MSE", "R²"]).map((h) => (
                    <th key={h} className="px-3 py-2.5 text-right font-medium">
                      {h}
                    </th>
                  ))}
                  <th className="px-5 py-2.5 text-right font-medium">Time</th>
                </tr>
              </thead>
              <tbody>
                {done.map((d, i) => (
                  <tr
                    key={d.id}
                    onClick={() => setActive(d.id)}
                    className={cx("cursor-pointer border-b border-line transition last:border-0 hover:bg-panel-2/60", i === 0 && "bg-accent-soft/50")}
                  >
                    <td className="px-5 py-3">{i === 0 ? <Medal className="size-4 text-accent" /> : <span className="tabular text-faint">{i + 1}</span>}</td>
                    <td className="px-3 py-3 font-medium">
                      <span className="mr-2 inline-block size-2 rounded-full" style={{ background: SERIES_COLORS[i % SERIES_COLORS.length] }} />
                      {d.spec.name}
                    </td>
                    {(isClass ? ["accuracy", "precision", "recall"] : ["RMSE", "MAE", "MSE", "R2"]).map((k) => (
                      <td key={k} className={cx("tabular px-3 py-3 text-right", k === primary && "font-semibold")}>
                        {k === "accuracy" && d.result.metrics[k] != null ? `${((d.result.metrics[k] as number) * 100).toFixed(1)}%` : fmt(d.result.metrics[k], k === "R2" ? 3 : 4)}
                      </td>
                    ))}
                    <td className="tabular px-5 py-3 text-right text-muted">{d.result.seconds.toFixed(1)}s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {Object.entries(runs).some(([, r]) => r.status === "error") && (
        <div className="mb-6">
          <Callout tone="warn">
            {Object.entries(runs)
              .filter(([, r]) => r.status === "error")
              .map(([id, r]) => `${modelById(id).name}: ${r.error}`)
              .join(" · ")}
          </Callout>
        </div>
      )}

      {done.length > 0 && (
        <>
          <div className="mb-5 overflow-x-auto">
            <Segmented
              value={active}
              onChange={setActive}
              options={[
                ...(isClass ? [] : [{ value: "__compare", label: "Compare all", icon: <Sparkles className="size-3.5" /> }]),
                ...done.map((d) => ({ value: d.id, label: d.spec.short })),
              ]}
            />
          </div>
          {active === "__compare" && !isClass ? (
            <div className="grid gap-6 lg:grid-cols-[1.6fr_1fr]">
              <Card>
                <CardHeader title="Every model on the test window" subtitle="Click a legend entry to hide a model" />
                <div className="px-3 pb-3">
                  <Plot data={compare} height={420} filename="provlepsi-compare" layout={{ yaxis: { title: { text: processed.target } } }} />
                </div>
              </Card>
              <Card>
                <CardHeader title="Error comparison" subtitle="RMSE and MAE per model" />
                <div className="px-3 pb-3">
                  <Plot
                    height={420}
                    layout={{ barmode: "group", hovermode: "closest", margin: { l: 50, r: 10, t: 10, b: 90 } }}
                    data={[
                      { type: "bar", name: "RMSE", x: done.map((d) => d.spec.short), y: done.map((d) => d.result.metrics.RMSE ?? null), marker: { color: "#ff7a45" } },
                      { type: "bar", name: "MAE", x: done.map((d) => d.spec.short), y: done.map((d) => d.result.metrics.MAE ?? null), marker: { color: "#5cc8ef" } },
                    ]}
                  />
                </div>
              </Card>
            </div>
          ) : (
            (() => {
              const d = done.find((x) => x.id === active) ?? done[0];
              return <ModelDetail key={d.id} id={d.id} result={d.result} p={processed} isClass={isClass} />;
            })()
          )}
        </>
      )}
      {!done.length && <Badge tone="bad">Every model failed. Check the errors on the training step.</Badge>}
    </div>
  );
}
