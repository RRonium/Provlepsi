import Papa from "papaparse";
import type { Cell, ColumnKind, ColumnMeta, Dataset } from "./types";

const DATE_LIKE = /^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}([ T]\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?)?$|^[A-Za-z]{3,9}[ -]\d{1,2},?[ -]\d{2,4}$|^\d{1,2}[ -][A-Za-z]{3,9}[ -]\d{2,4}$/;
const DATE_NAME = /date|time|year|month|day|period|timestamp/i;

export function parseDateValue(v: Cell): number | null {
  if (v === null || v === "") return null;
  if (typeof v === "number") return null;
  const s = v.trim();
  // dd-mm-yyyy / dd/mm/yyyy -> yyyy-mm-dd (Date.parse treats these as US month-first)
  const dmy = /^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})(.*)$/.exec(s);
  if (dmy) {
    const [, a, b, y, rest] = dmy;
    const day = Number(a) > 12 ? a : Number(b) > 12 ? b : a;
    const month = day === a ? b : a;
    const t = Date.parse(`${y}-${month.padStart(2, "0")}-${day.padStart(2, "0")}${rest.replace(" ", "T")}`);
    return Number.isNaN(t) ? null : t;
  }
  const t = Date.parse(s);
  return Number.isNaN(t) ? null : t;
}

function detectKind(values: Cell[]): ColumnKind {
  let nums = 0;
  let dates = 0;
  let seen = 0;
  for (const v of values) {
    if (v === null || v === "") continue;
    seen++;
    if (typeof v === "number") nums++;
    else {
      if (DATE_LIKE.test(v.trim()) && parseDateValue(v) !== null) dates++;
    }
    if (seen > 2000) break;
  }
  if (seen === 0) return "numeric";
  if (dates / seen > 0.9) return "datetime";
  // integer years in a column called "year" stay numeric, but guessDateColumn still offers them as a time axis
  if (nums / seen > 0.95) return "numeric";
  return "categorical";
}

export function describeColumns(data: Record<string, Cell[]>, names: string[]): ColumnMeta[] {
  return names.map((name) => {
    const values = data[name];
    let missing = 0;
    const uniq = new Set<Cell>();
    for (const v of values) {
      if (v === null || v === "" || (typeof v === "number" && Number.isNaN(v))) missing++;
      else if (uniq.size < 10001) uniq.add(v);
    }
    const kind = detectKind(values);
    return { name, kind, missing, unique: uniq.size };
  });
}

export function guessDateColumn(cols: ColumnMeta[]): string | null {
  const dt = cols.find((c) => c.kind === "datetime");
  if (dt) return dt.name;
  const named = cols.find((c) => DATE_NAME.test(c.name) && c.kind === "numeric");
  return named ? named.name : null;
}

export async function parseCsvText(text: string, name: string, bytes: number, id?: string): Promise<Dataset> {
  const parsed = Papa.parse<Record<string, unknown>>(text, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: "greedy",
    transformHeader: (h, i) => (h.trim() === "" ? `column_${i + 1}` : h.trim()),
  });
  const fields = (parsed.meta.fields ?? []).filter((f, i, arr) => arr.indexOf(f) === i);
  if (!fields.length || !parsed.data.length) throw new Error("This file has no rows or no header row.");
  const data: Record<string, Cell[]> = {};
  for (const f of fields) data[f] = new Array(parsed.data.length);
  parsed.data.forEach((row, r) => {
    for (const f of fields) {
      const v = row[f];
      let cell: Cell;
      if (v === null || v === undefined || v === "") cell = null;
      else if (typeof v === "number") cell = Number.isFinite(v) ? v : null;
      else if (typeof v === "boolean") cell = v ? 1 : 0;
      else if (v instanceof Date) cell = v.toISOString();
      else {
        const s = String(v).trim();
        const lowered = s.toLowerCase();
        if (lowered === "na" || lowered === "nan" || lowered === "null" || lowered === "none" || s === "?") cell = null;
        else {
          // numbers with thousands separators, e.g. "1,234.5"
          const n = /^-?[\d,]+(\.\d+)?$/.test(s) ? Number(s.replace(/,/g, "")) : NaN;
          cell = Number.isFinite(n) ? n : s;
        }
      }
      data[f][r] = cell;
    }
  });
  const columns = describeColumns(data, fields);
  // a column of numeric-looking strings mixed with numbers: coerce
  for (const c of columns) {
    if (c.kind === "numeric") data[c.name] = data[c.name].map((v) => (typeof v === "number" ? v : null));
  }
  return {
    id: id ?? `ds_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`,
    name,
    bytes,
    addedAt: Date.now(),
    nRows: parsed.data.length,
    columns: describeColumns(data, fields),
    data,
  };
}

export function toCsv(columns: string[], data: Record<string, Cell[]>, rows: number): string {
  const esc = (v: Cell) => {
    if (v === null || v === undefined) return "";
    if (typeof v === "number") return Number.isFinite(v) ? String(v) : "";
    return /[",\n\r]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v;
  };
  const out: string[] = [columns.map((c) => esc(c)).join(",")];
  for (let r = 0; r < rows; r++) out.push(columns.map((c) => esc(data[c][r])).join(","));
  return out.join("\n");
}

export function downloadText(filename: string, text: string, type = "text/csv") {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
