"use client";
import { ArrowRight, Brain, Database, FileSpreadsheet, HardDrive, Lock, Sparkles, Trash2, UploadCloud } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { parseCsvText } from "@/lib/csv";
import { SAMPLES } from "@/lib/samples";
import { deleteDataset, listDatasets, loadDatasetText, saveDataset, type StoredDatasetMeta } from "@/lib/storage";
import { useStudio } from "@/store/studio";
import { Badge, Callout, Card, cx } from "../ui";

const MAX_BYTES = 200 * 1024 * 1024;

function bytes(n: number) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 ** 2).toFixed(1)} MB`;
}

function ago(t: number) {
  const s = (Date.now() - t) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return new Date(t).toLocaleDateString();
}

export function UploadStep() {
  const loadDataset = useStudio((s) => s.loadDataset);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [recent, setRecent] = useState<StoredDatasetMeta[]>([]);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    void listDatasets().then(setRecent);
  }, []);

  const ingest = useCallback(
    async (text: string, name: string, size: number, id?: string, persist = true) => {
      setError(null);
      setBusy(name);
      try {
        const ds = await parseCsvText(text, name, size, id);
        if (persist) await saveDataset({ id: ds.id, name, bytes: size, rows: ds.nRows, cols: ds.columns.length, addedAt: Date.now() }, text);
        loadDataset(ds);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(null);
      }
    },
    [loadDataset],
  );

  const onFile = async (file: File | undefined) => {
    if (!file) return;
    if (!/\.(csv|tsv|txt)$/i.test(file.name)) return setError("Please choose a .csv file.");
    if (file.size > MAX_BYTES) return setError("That file is over 200 MB. Try a sample of it, or use the local Python engine.");
    await ingest(await file.text(), file.name, file.size);
  };

  return (
    <div className="grid gap-10 lg:grid-cols-[1.05fr_1fr] lg:gap-14">
      <section className="pt-2 lg:pt-8">
        <Badge tone="accent" className="mb-5">
          <Lock className="size-3" /> Private by design
        </Badge>
        <h1 className="text-[34px] leading-[1.08] font-semibold tracking-tight sm:text-[46px]">
          Forecast your time series <span className="text-accent">without sending it anywhere.</span>
        </h1>
        <p className="mt-5 max-w-xl text-[15px] leading-relaxed text-muted">
          Drop in a CSV and Prόvlepsi profiles it, lets you choose how to clean it, and trains classical, machine-learning and
          deep-learning models side by side. Everything runs in this tab with WebAssembly and WebGL.
        </p>
        <div className="mt-8 grid max-w-xl gap-3 sm:grid-cols-3">
          {[
            { icon: <HardDrive className="size-4" />, t: "Stays local", d: "Stored in your browser only" },
            { icon: <Sparkles className="size-4" />, t: "14 models", d: "ARIMA to Transformers" },
            { icon: <Brain className="size-4" />, t: "Your GPU", d: "TensorFlow.js + Pyodide" },
          ].map((f) => (
            <div key={f.t} className="rounded-xl border border-line bg-panel/70 p-3.5">
              <div className="text-accent">{f.icon}</div>
              <div className="mt-2 text-[13px] font-medium">{f.t}</div>
              <div className="text-[12px] text-muted">{f.d}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="space-y-5">
        <Card className="p-2">
          <div
            role="button"
            tabIndex={0}
            onClick={() => input.current?.click()}
            onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              void onFile(e.dataTransfer.files[0]);
            }}
            className={cx(
              "flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-14 text-center transition",
              dragging ? "border-accent bg-accent-soft" : "border-line hover:border-line-strong hover:bg-panel-2/50",
            )}
          >
            <div className={cx("grid size-14 place-items-center rounded-2xl bg-accent-soft text-accent transition", dragging && "scale-110")}>
              <UploadCloud className="size-7" />
            </div>
            <div className="mt-4 text-[15px] font-medium">{busy ? `Reading ${busy}…` : "Drop a CSV here, or click to browse"}</div>
            <div className="mt-1 text-[13px] text-muted">One row per time step, one column per variable. Up to 200 MB.</div>
            <input ref={input} type="file" accept=".csv,.tsv,.txt,text/csv" className="hidden" onChange={(e) => void onFile(e.target.files?.[0])} />
          </div>
        </Card>
        {error && <Callout tone="bad">{error}</Callout>}

        <div>
          <div className="mb-2.5 flex items-center gap-2 text-[12px] font-medium tracking-wide text-faint uppercase">
            <FileSpreadsheet className="size-3.5" /> Try a sample
          </div>
          <div className="grid gap-2.5 sm:grid-cols-3">
            {SAMPLES.map((s) => (
              <button
                key={s.id}
                onClick={() => {
                  const text = s.build();
                  void ingest(text, `${s.name} (synthetic).csv`, text.length, `sample_${s.id}`, false);
                }}
                className="group rounded-xl border border-line bg-panel p-3.5 text-left transition hover:-translate-y-0.5 hover:border-accent"
              >
                <div className="flex items-center justify-between">
                  <span className="text-[13px] font-medium">{s.name}</span>
                  <ArrowRight className="size-3.5 text-faint transition group-hover:translate-x-0.5 group-hover:text-accent" />
                </div>
                <div className="mt-1 text-[12px] leading-snug text-muted">{s.description}</div>
                <Badge tone={s.task === "classification" ? "cyan" : "neutral"} className="mt-2.5">
                  {s.task}
                </Badge>
              </button>
            ))}
          </div>
        </div>

        {recent.length > 0 && (
          <div>
            <div className="mb-2.5 flex items-center gap-2 text-[12px] font-medium tracking-wide text-faint uppercase">
              <Database className="size-3.5" /> Saved in this browser
            </div>
            <Card className="divide-y divide-[var(--border)] overflow-hidden">
              {recent.map((d) => (
                <div key={d.id} className="flex items-center gap-3 px-4 py-3 transition hover:bg-panel-2/60">
                  <FileSpreadsheet className="size-4 shrink-0 text-faint" />
                  <button
                    className="min-w-0 flex-1 text-left"
                    onClick={async () => {
                      const text = await loadDatasetText(d.id);
                      if (text) await ingest(text, d.name, d.bytes, d.id, false);
                      else setError("That dataset is no longer stored in this browser.");
                    }}
                  >
                    <div className="truncate text-[13px] font-medium">{d.name}</div>
                    <div className="text-[12px] text-muted">
                      {d.rows.toLocaleString()} rows · {d.cols} columns · {bytes(d.bytes)} · {ago(d.addedAt)}
                    </div>
                  </button>
                  <button
                    aria-label={`Remove ${d.name}`}
                    onClick={async () => {
                      await deleteDataset(d.id);
                      setRecent(await listDatasets());
                    }}
                    className="grid size-8 place-items-center rounded-lg text-faint transition hover:bg-bad-soft hover:text-bad"
                  >
                    <Trash2 className="size-4" />
                  </button>
                </div>
              ))}
            </Card>
          </div>
        )}
      </section>
    </div>
  );
}
