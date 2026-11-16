"use client";
import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import { guessDateColumn } from "@/lib/csv";
import { MODELS, defaultParams, modelById } from "@/lib/models";
import { defaultPrep, preprocess } from "@/lib/preprocess";
import type { Dataset, ModelResult, PrepOptions, ProcessedData, RunState } from "@/lib/types";
import { localEngine, pyodideEngine, PythonError, type EngineKind, type PythonEngine, type RawRunResult } from "@/engines/python";
import { Comlink, deepEngine } from "@/engines/deep";
import type { BackendPref, DeepArch, DeepParams, DeepProgress } from "@/engines/tf.worker";

export const STEPS = ["Upload", "Explore", "Prepare", "Models", "Train", "Results"] as const;
export type Step = 0 | 1 | 2 | 3 | 4 | 5;
type ParamValue = number | string | null;
type EngineStatus = "idle" | "loading" | "ready" | "error";

interface StudioState {
  step: Step;
  maxStep: Step;
  dataset: Dataset | null;
  prep: PrepOptions | null;
  processed: ProcessedData | null;
  selected: Record<string, Record<string, ParamValue>>;
  runs: Record<string, RunState>;
  running: boolean;
  engineKind: EngineKind;
  localUrl: string;
  dlBackend: BackendPref;
  python: { status: EngineStatus; message: string; versions?: string };
  theme: "dark" | "light";

  go(step: Step): void;
  loadDataset(ds: Dataset): void;
  setPrep(patch: Partial<PrepOptions>): void;
  commitPrep(): ProcessedData;
  toggleModel(id: string): void;
  setParam(id: string, key: string, value: ParamValue): void;
  setEngine(kind: EngineKind, localUrl?: string): void;
  setDlBackend(b: BackendPref): void;
  ensurePython(): Promise<PythonEngine>;
  runAll(): Promise<void>;
  stop(): void;
  predict(id: string, entry: Record<string, unknown>): Promise<void>;
  reforecast(id: string, steps: number): Promise<void>;
  newProject(): void;
  toggleTheme(): void;
}

let pythonReady: Promise<PythonEngine> | null = null;
let stopFlag = false;

function engineFor(kind: EngineKind, url: string): PythonEngine {
  return kind === "local" ? localEngine(url) : pyodideEngine;
}

function r2(actual: (number | null)[], pred: (number | null)[]): number | null {
  const pairs = actual.map((a, i) => [a, pred[i]]).filter(([a, p]) => typeof a === "number" && typeof p === "number") as number[][];
  if (pairs.length < 2) return null;
  const m = pairs.reduce((s, [a]) => s + a, 0) / pairs.length;
  let ssr = 0;
  let sst = 0;
  for (const [a, p] of pairs) {
    ssr += (a - p) ** 2;
    sst += (a - m) ** 2;
  }
  return sst ? 1 - ssr / sst : null;
}

function fromPython(raw: RawRunResult, seconds: number, regression: boolean): ModelResult {
  const metrics = { ...raw.metrics };
  if (regression && raw.test && raw.test_predictions) metrics.R2 = r2(raw.test.values, raw.test_predictions.values);
  const forecast = Array.isArray(raw.prediction?.forecast) ? (raw.prediction.forecast as number[]) : undefined;
  return {
    sessionId: raw.session_id,
    metrics,
    extras: raw.extras,
    prediction: raw.prediction,
    featureColumns: raw.feature_columns,
    defaultEntry: raw.default_entry,
    train: raw.train,
    test: raw.test,
    testPredictions: raw.test_predictions,
    forecast,
    notes: raw.notes,
    seconds,
  };
}

export const useStudio = create<StudioState>()(
  persist(
    (set, get) => ({
      step: 0,
      maxStep: 0,
      dataset: null,
      prep: null,
      processed: null,
      selected: {},
      runs: {},
      running: false,
      engineKind: "pyodide",
      localUrl: "http://127.0.0.1:8765",
      dlBackend: "auto",
      python: { status: "idle", message: "" },
      theme: "dark",

      go(step) {
        set((s) => ({ step, maxStep: (Math.max(s.maxStep, step) as Step) }));
        if (typeof window !== "undefined") window.scrollTo({ top: 0, behavior: "smooth" });
      },

      loadDataset(ds) {
        const dateCol = guessDateColumn(ds.columns);
        const prep = defaultPrep(ds, dateCol);
        set({ dataset: ds, prep, processed: null, runs: {}, step: 1, maxStep: 1 });
        // warm Python up in the background while the user explores
        void get().ensurePython().catch(() => undefined);
      },

      setPrep(patch) {
        const prep = get().prep;
        if (!prep) return;
        set({ prep: { ...prep, ...patch }, processed: null, maxStep: Math.min(get().maxStep, 2) as Step });
      },

      commitPrep() {
        const { dataset, prep } = get();
        if (!dataset || !prep) throw new Error("Load a dataset first.");
        const processed = preprocess(dataset, prep);
        // drop selections that no longer fit the task
        const selected = Object.fromEntries(
          Object.entries(get().selected).filter(([id]) => modelById(id).task === prep.task),
        );
        if (!Object.keys(selected).length) {
          const picks = prep.task === "classification" ? ["logistic", "lda"] : processed.featureColumns.length ? ["random_forest", "ridge", "lstm"] : ["arima", "random_forest", "lstm"];
          for (const id of picks) selected[id] = defaultParams(modelById(id));
        }
        set({ processed, selected, runs: {} });
        return processed;
      },

      toggleModel(id) {
        const selected = { ...get().selected };
        if (selected[id]) delete selected[id];
        else selected[id] = defaultParams(modelById(id));
        set({ selected });
      },

      setParam(id, key, value) {
        const selected = get().selected;
        if (!selected[id]) return;
        set({ selected: { ...selected, [id]: { ...selected[id], [key]: value } } });
      },

      setEngine(kind, localUrl) {
        pythonReady = null;
        set({ engineKind: kind, localUrl: localUrl ?? get().localUrl, python: { status: "idle", message: "" } });
      },

      setDlBackend(dlBackend) {
        set({ dlBackend });
      },

      ensurePython() {
        if (pythonReady) return pythonReady;
        const { engineKind, localUrl } = get();
        const engine = engineFor(engineKind, localUrl);
        set({ python: { status: "loading", message: "Starting Python…" } });
        pythonReady = engine
          .init((message) => set((s) => ({ python: { ...s.python, message } })))
          .then((versions) => {
            set({ python: { status: "ready", message: "Python ready", versions } });
            return engine;
          })
          .catch((e: unknown) => {
            pythonReady = null;
            set({ python: { status: "error", message: e instanceof Error ? e.message : String(e) } });
            throw e;
          });
        return pythonReady;
      },

      async runAll() {
        const { processed, selected, prep } = get();
        if (!processed || !prep) return;
        stopFlag = false;
        const ids = Object.keys(selected);
        const runs: Record<string, RunState> = {};
        for (const id of ids) runs[id] = { status: "queued" };
        set({ runs, running: true });

        const update = (id: string, patch: Partial<RunState>) => set((s) => ({ runs: { ...s.runs, [id]: { ...s.runs[id], ...patch } } }));
        const fail = (id: string, e: unknown) =>
          update(id, {
            status: "error",
            finishedAt: Date.now(),
            error: e instanceof Error ? e.message : String(e),
            traceback: e instanceof PythonError ? e.traceback : undefined,
          });

        const pythonQueue = async () => {
          const pyIds = ids.filter((id) => modelById(id).engine === "python");
          if (!pyIds.length) return;
          let engine: PythonEngine;
          try {
            for (const id of pyIds) update(id, { message: "Waiting for Python…" });
            engine = await get().ensurePython();
          } catch (e) {
            for (const id of pyIds) fail(id, e);
            return;
          }
          for (const id of pyIds) {
            if (stopFlag) {
              update(id, { status: "error", error: "Stopped before it started." });
              continue;
            }
            const spec = modelById(id);
            const started = Date.now();
            update(id, { status: "running", startedAt: started, message: `Fitting with ${engine.kind === "local" ? "local Python" : "Pyodide"}…` });
            try {
              const params = { ...selected[id] };
              const config = { response_variable: processed.target, split_fraction: prep.testFraction, ...params, ...(spec.fixed ?? {}) };
              const raw = await engine.run(spec.file!, processed.csv, config);
              const seconds = (Date.now() - started) / 1000;
              update(id, { status: "done", finishedAt: Date.now(), result: fromPython(raw, seconds, spec.task === "regression"), message: undefined });
            } catch (e) {
              fail(id, e);
            }
          }
        };

        const deepQueue = async () => {
          const dlIds = ids.filter((id) => modelById(id).engine === "tf");
          if (!dlIds.length) return;
          const series = processed.data[processed.target].map(Number);
          const tfApi = deepEngine();
          for (const id of dlIds) {
            if (stopFlag) {
              update(id, { status: "error", error: "Stopped before it started." });
              continue;
            }
            const started = Date.now();
            update(id, { status: "running", startedAt: started, progress: 0, liveLoss: { loss: [], valLoss: [] }, message: "Building network…" });
            try {
              const params = selected[id] as unknown as DeepParams;
              const res = await tfApi.train(
                id,
                id as DeepArch,
                series,
                prep.testFraction,
                params,
                Comlink.proxy((p: DeepProgress) => {
                  const live = get().runs[id]?.liveLoss ?? { loss: [], valLoss: [] };
                  update(id, {
                    progress: p.epoch / p.epochs,
                    message: `Epoch ${p.epoch}/${p.epochs} · loss ${p.loss.toExponential(2)}`,
                    liveLoss: { loss: [...live.loss, p.loss], valLoss: p.valLoss === null ? live.valLoss : [...live.valLoss, p.valLoss] },
                  });
                }),
                get().dlBackend,
              );
              const seconds = (Date.now() - started) / 1000;
              const metrics: Record<string, number | null> = { ...res.metrics, R2: r2(res.test.values, res.testPredictions.values) };
              update(id, {
                status: "done",
                finishedAt: Date.now(),
                progress: 1,
                message: undefined,
                result: {
                  metrics,
                  extras: {},
                  prediction: {},
                  featureColumns: [],
                  defaultEntry: {},
                  train: res.train,
                  test: res.test,
                  testPredictions: res.testPredictions,
                  forecast: res.forecast,
                  history: res.history,
                  notes: res.stopped ? ["Training was stopped early."] : [],
                  seconds,
                  backend: res.backend,
                },
              });
            } catch (e) {
              fail(id, e);
            }
          }
        };

        await Promise.all([pythonQueue(), deepQueue()]);
        set({ running: false });
      },

      stop() {
        stopFlag = true;
        void deepEngine().stop();
      },

      async predict(id, entry) {
        const run = get().runs[id];
        if (!run?.result?.sessionId) return;
        const engine = await get().ensurePython();
        const prediction = await engine.predict(run.result.sessionId, entry);
        const forecast = Array.isArray(prediction.forecast) ? (prediction.forecast as number[]) : run.result.forecast;
        set((s) => ({ runs: { ...s.runs, [id]: { ...run, result: { ...run.result!, prediction, forecast } } } }));
      },

      async reforecast(id, steps) {
        const run = get().runs[id];
        if (!run?.result) return;
        const forecast = await deepEngine().forecast(id, steps);
        set((s) => ({ runs: { ...s.runs, [id]: { ...run, result: { ...run.result!, forecast } } } }));
      },

      newProject() {
        void deepEngine().reset();
        set({ dataset: null, prep: null, processed: null, runs: {}, step: 0, maxStep: 0 });
      },

      toggleTheme() {
        set({ theme: get().theme === "dark" ? "light" : "dark" });
      },
    }),
    {
      name: "provlepsi-settings",
      storage: createJSONStorage(() => localStorage),
      // only preferences persist; datasets live in IndexedDB
      partialize: (s) => ({ engineKind: s.engineKind, localUrl: s.localUrl, theme: s.theme, dlBackend: s.dlBackend }),
    },
  ),
);

export const ALL_MODELS = MODELS;
