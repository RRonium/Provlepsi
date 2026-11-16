"use client";
import { AlertTriangle, ArrowLeft, ArrowRight, CheckCircle2, Clock, Loader2, RotateCcw, Square } from "lucide-react";
import { useEffect, useState } from "react";
import { modelById } from "@/lib/models";
import { fmt } from "@/lib/stats";
import type { RunState } from "@/lib/types";
import { useStudio } from "@/store/studio";
import { Plot } from "../Plot";
import { Badge, Button, Callout, Card, SectionTitle, cx } from "../ui";

function useTick(active: boolean) {
  const [, setT] = useState(0);
  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => setT((t) => t + 1), 500);
    return () => clearInterval(id);
  }, [active]);
}

function elapsed(run: RunState) {
  if (!run.startedAt) return null;
  const s = ((run.finishedAt ?? Date.now()) - run.startedAt) / 1000;
  return s < 60 ? `${s.toFixed(1)}s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}

function RunRow({ id, run }: { id: string; run: RunState }) {
  const spec = modelById(id);
  const [open, setOpen] = useState(false);
  const primary = spec.task === "classification" ? "accuracy" : "RMSE";
  const metric = run.result?.metrics[primary];
  return (
    <div className="px-5 py-4">
      <div className="flex items-center gap-4">
        <div
          className={cx(
            "grid size-9 shrink-0 place-items-center rounded-xl",
            run.status === "done" && "bg-good-soft text-good",
            run.status === "error" && "bg-bad-soft text-bad",
            run.status === "running" && "pulse-ring bg-accent-soft text-accent",
            run.status === "queued" && "bg-panel-2 text-faint",
          )}
        >
          {run.status === "done" ? (
            <CheckCircle2 className="size-4.5" />
          ) : run.status === "error" ? (
            <AlertTriangle className="size-4.5" />
          ) : run.status === "running" ? (
            <Loader2 className="size-4.5 animate-spin" />
          ) : (
            <Clock className="size-4.5" />
          )}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[14px] font-semibold">{spec.name}</span>
            <Badge>{spec.engine === "tf" ? "TensorFlow.js" : "Python"}</Badge>
            {run.result?.backend && <Badge tone="cyan">{run.result.backend}</Badge>}
          </div>
          <div className={cx("mt-0.5 truncate text-[12.5px]", run.status === "error" ? "text-bad" : "text-muted")}>
            {run.status === "error" ? run.error : run.status === "done" ? "Finished" : (run.message ?? "Queued")}
          </div>
        </div>
        <div className="text-right">
          {run.status === "done" && metric !== undefined && (
            <div className="tabular text-[15px] font-semibold">
              {fmt(metric)} <span className="text-[11px] font-normal text-faint">{primary}</span>
            </div>
          )}
          <div className="tabular text-[12px] text-faint">{elapsed(run)}</div>
        </div>
      </div>
      {run.status === "running" && run.progress !== undefined && (
        <div className="mt-3 ml-13 h-1.5 overflow-hidden rounded-full bg-panel-2">
          <div className="h-full rounded-full bg-accent transition-all" style={{ width: `${Math.max(3, run.progress * 100)}%` }} />
        </div>
      )}
      {run.liveLoss && run.liveLoss.loss.length > 1 && (
        <div className="mt-2 ml-13">
          <Plot
            height={140}
            layout={{ margin: { l: 44, r: 8, t: 6, b: 24 }, showlegend: true, legend: { orientation: "h", x: 1, xanchor: "right", y: 1.25 }, yaxis: { type: "log" } }}
            data={[
              { type: "scatter", mode: "lines", y: run.liveLoss.loss, name: "train loss", line: { color: "#5cc8ef", width: 1.6 } },
              { type: "scatter", mode: "lines", y: run.liveLoss.valLoss, name: "test loss", line: { color: "#ff7a45", width: 1.6 } },
            ]}
          />
        </div>
      )}
      {run.status === "error" && run.traceback && (
        <div className="mt-2 ml-13">
          <button onClick={() => setOpen((o) => !o)} className="text-[12px] text-muted underline-offset-2 hover:underline">
            {open ? "Hide" : "Show"} Python traceback
          </button>
          {open && <pre className="mt-2 max-h-64 overflow-auto rounded-lg bg-panel-2 p-3 font-mono text-[11px] leading-relaxed text-muted">{run.traceback}</pre>}
        </div>
      )}
    </div>
  );
}

export function TrainStep() {
  const runs = useStudio((s) => s.runs);
  const running = useStudio((s) => s.running);
  const python = useStudio((s) => s.python);
  const go = useStudio((s) => s.go);
  const stop = useStudio((s) => s.stop);
  const runAll = useStudio((s) => s.runAll);
  useTick(running);

  const ids = Object.keys(runs);
  const done = ids.filter((id) => runs[id].status === "done").length;
  const failed = ids.filter((id) => runs[id].status === "error").length;
  const needsPython = ids.some((id) => modelById(id).engine === "python");

  return (
    <div>
      <SectionTitle eyebrow="Step 5 · Training" title={running ? "Training your models…" : `${done} of ${ids.length} models trained`}>
        <div className="flex gap-2">
          {running ? (
            <Button variant="danger" size="lg" onClick={stop} icon={<Square className="size-4" />}>
              Stop
            </Button>
          ) : (
            <>
              <Button variant="ghost" size="lg" onClick={() => go(3)} icon={<ArrowLeft className="size-4" />}>
                Models
              </Button>
              <Button variant="secondary" size="lg" onClick={() => void runAll()} icon={<RotateCcw className="size-4" />}>
                Train again
              </Button>
            </>
          )}
          <Button variant="primary" size="lg" disabled={!done} onClick={() => go(5)} icon={<ArrowRight className="size-4" />}>
            View results
          </Button>
        </div>
      </SectionTitle>

      <div className="mb-5 h-2 overflow-hidden rounded-full bg-panel-2">
        <div className="h-full rounded-full bg-gradient-to-r from-accent to-[#ffb38a] transition-all duration-500" style={{ width: `${ids.length ? ((done + failed) / ids.length) * 100 : 0}%` }} />
      </div>

      {needsPython && python.status === "loading" && (
        <div className="mb-5">
          <Callout icon={<Loader2 className="size-4 animate-spin" />}>
            <span className="font-medium text-fg">Preparing Python in your browser.</span> {python.message} The first load downloads about 30 MB and is cached
            afterwards.
          </Callout>
        </div>
      )}
      {needsPython && python.status === "error" && (
        <div className="mb-5">
          <Callout tone="bad" icon={<AlertTriangle className="size-4" />}>
            Python could not start: {python.message} You can switch engines from the button in the top bar.
          </Callout>
        </div>
      )}

      <Card className="divide-y divide-[var(--border)]">
        {ids.map((id) => (
          <RunRow key={id} id={id} run={runs[id]} />
        ))}
      </Card>
    </div>
  );
}
