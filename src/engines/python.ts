"use client";
import * as Comlink from "comlink";
import type { PyodideApi } from "./pyodide.worker";

export const PYODIDE_URL = process.env.NEXT_PUBLIC_PYODIDE_URL ?? "https://cdn.jsdelivr.net/pyodide/v314.0.8/full/";
export const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

export type EngineKind = "pyodide" | "local";

export interface RawRunResult {
  session_id: string;
  metrics: Record<string, number | null>;
  extras: Record<string, unknown>;
  prediction: Record<string, unknown>;
  feature_columns: string[];
  default_entry: Record<string, unknown>;
  train: { index: number[]; values: (number | null)[] } | null;
  test: { index: number[]; values: (number | null)[] } | null;
  test_predictions: { index: number[]; values: (number | null)[] } | null;
  notes: string[];
}

export class PythonError extends Error {
  traceback?: string;
  constructor(message: string, traceback?: string) {
    super(message);
    this.traceback = traceback;
  }
}

function unwrap<T>(json: string): T {
  const out = JSON.parse(json) as { ok: boolean; data?: T; error?: string; traceback?: string };
  if (!out.ok) throw new PythonError(out.error ?? "The model failed.", out.traceback);
  return out.data as T;
}

let worker: Worker | null = null;
let remote: Comlink.Remote<PyodideApi> | null = null;

function pyodideRemote() {
  if (!remote) {
    worker = new Worker(new URL("./pyodide.worker.ts", import.meta.url), { type: "module", name: "provlepsi-python" });
    remote = Comlink.wrap<PyodideApi>(worker);
  }
  return remote;
}

export interface PythonEngine {
  kind: EngineKind;
  init(onStatus: (msg: string) => void): Promise<string>;
  run(modelFile: string, csv: string, config: Record<string, unknown>): Promise<RawRunResult>;
  predict(sessionId: string, entry: Record<string, unknown>): Promise<Record<string, unknown>>;
}

export const pyodideEngine: PythonEngine = {
  kind: "pyodide",
  init: async (onStatus) => {
    try {
      return await pyodideRemote().init(PYODIDE_URL, BASE_PATH, Comlink.proxy(onStatus));
    } catch (e) {
      const why = e instanceof Error ? e.message : String(e);
      throw new Error(
        /fetch|import|network/i.test(why)
          ? `Couldn't download the Python runtime from ${new URL(PYODIDE_URL).host}. Check your connection, or switch to Local Python. (${why})`
          : why,
      );
    }
  },
  run: async (file, csv, config) => unwrap<RawRunResult>(await pyodideRemote().run(file, csv, config)),
  predict: async (id, entry) => unwrap<Record<string, unknown>>(await pyodideRemote().predict(id, entry)),
};

export function localEngine(baseUrl: string): PythonEngine {
  const url = baseUrl.replace(/\/$/, "");
  const post = async (path: string, body: unknown) => {
    const res = await fetch(`${url}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return res.text();
  };
  return {
    kind: "local",
    async init(onStatus) {
      onStatus(`Connecting to ${url}…`);
      let res: Response;
      try {
        res = await fetch(`${url}/health`);
      } catch {
        throw new Error(`No local engine answered at ${url}. Start it with: python python/local_server.py`);
      }
      const data = unwrap<{ versions: Record<string, string> }>(await res.text());
      const v = data.versions;
      return `Local Python ${v.python} · pandas ${v.pandas} · scikit-learn ${v.sklearn}`;
    },
    run: async (file, csv, config) => unwrap<RawRunResult>(await post("/run", { model_file: file, csv, config })),
    predict: async (id, entry) => unwrap<Record<string, unknown>>(await post("/predict", { session_id: id, entry })),
  };
}
