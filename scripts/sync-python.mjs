// Copies the Python runner and model scripts into public/py so the browser
// (Pyodide) can fetch them. python/ stays the single source of truth.
import { cpSync, mkdirSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(root, "python");
const dest = join(root, "public", "py");

rmSync(dest, { recursive: true, force: true });
mkdirSync(join(dest, "models"), { recursive: true });
cpSync(join(src, "provlepsi_runner.py"), join(dest, "provlepsi_runner.py"));
const models = readdirSync(join(src, "models")).filter((f) => f.endsWith("_model.py"));
for (const f of models) cpSync(join(src, "models", f), join(dest, "models", f));
writeFileSync(join(dest, "manifest.json"), JSON.stringify({ models }, null, 2));
console.log(`synced ${models.length} model scripts to public/py`);
