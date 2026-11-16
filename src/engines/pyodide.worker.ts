/// <reference lib="webworker" />
// Runs Python (pandas, scikit-learn, statsmodels) inside the browser with Pyodide,
// off the main thread. The dataset is written to Pyodide's in-memory filesystem,
// so it never leaves the device.
import * as Comlink from "comlink";

type Status = (message: string) => void;

interface PyodideLike {
  loadPackage(names: string[], opts?: { messageCallback?: (m: string) => void }): Promise<void>;
  runPython(code: string): unknown;
  pyimport(name: string): Record<string, (...args: unknown[]) => unknown>;
  FS: { writeFile(path: string, data: string): void; mkdirTree(path: string): void };
  version: string;
}

let pyodide: PyodideLike | null = null;
let booting: Promise<string> | null = null;
let assetBase = "";
const sourceCache = new Map<string, string>();

async function fetchText(path: string): Promise<string> {
  const cached = sourceCache.get(path);
  if (cached) return cached;
  const res = await fetch(`${assetBase}${path}`);
  if (!res.ok) throw new Error(`Could not load ${path} (${res.status})`);
  const text = await res.text();
  sourceCache.set(path, text);
  return text;
}

async function boot(indexURL: string, base: string, onStatus?: Status): Promise<string> {
  assetBase = base;
  onStatus?.("Downloading the Python runtime…");
  const mod = await import(/* webpackIgnore: true */ /* turbopackIgnore: true */ `${indexURL}pyodide.mjs`);
  const py = (await mod.loadPyodide({ indexURL })) as PyodideLike;
  onStatus?.("Loading pandas, scikit-learn and statsmodels…");
  await py.loadPackage(["numpy", "pandas", "scikit-learn", "statsmodels"], {
    messageCallback: (m) => {
      if (/^Loading /.test(m)) onStatus?.(m.length > 80 ? `${m.slice(0, 77)}…` : m);
    },
  });
  py.FS.mkdirTree("/provlepsi/data");
  py.FS.writeFile("/provlepsi/provlepsi_runner.py", await fetchText("/py/provlepsi_runner.py"));
  py.runPython("import sys\nsys.path.insert(0, '/provlepsi')\nimport provlepsi_runner");
  pyodide = py;
  const versions = py.runPython(
    "import pandas, sklearn, sys; f'Python {sys.version.split()[0]} · pandas {pandas.__version__} · scikit-learn {sklearn.__version__}'",
  ) as string;
  onStatus?.("Python is ready");
  return versions;
}

const api = {
  async init(indexURL: string, base: string, onStatus?: Status) {
    if (!booting) booting = boot(indexURL, base, onStatus).catch((e) => {
      booting = null;
      throw e;
    });
    return booting;
  },

  async run(modelFile: string, csv: string, config: Record<string, unknown>): Promise<string> {
    if (!pyodide) throw new Error("Python is still loading.");
    const path = "/provlepsi/data/dataset.csv";
    pyodide.FS.writeFile(path, csv);
    const source = await fetchText(`/py/models/${modelFile}`);
    const runner = pyodide.pyimport("provlepsi_runner");
    return runner.run_model_json(source, JSON.stringify({ ...config, csv_path: path }), modelFile) as string;
  },

  async predict(sessionId: string, entry: Record<string, unknown>): Promise<string> {
    if (!pyodide) throw new Error("Python is still loading.");
    return pyodide.pyimport("provlepsi_runner").predict_json(sessionId, JSON.stringify(entry)) as string;
  },
};

export type PyodideApi = typeof api;
Comlink.expose(api);
