"use client";
// Datasets are kept in this browser's IndexedDB only. Nothing is uploaded.
import { createStore, del, get, set } from "idb-keyval";

export interface StoredDatasetMeta {
  id: string;
  name: string;
  bytes: number;
  rows: number;
  cols: number;
  addedAt: number;
}

let store: ReturnType<typeof createStore> | null = null;
const db = () => (store ??= createStore("provlepsi", "datasets"));

export async function listDatasets(): Promise<StoredDatasetMeta[]> {
  try {
    return ((await get<StoredDatasetMeta[]>("index", db())) ?? []).sort((a, b) => b.addedAt - a.addedAt);
  } catch {
    return [];
  }
}

export async function saveDataset(meta: StoredDatasetMeta, text: string) {
  try {
    await set(`raw:${meta.id}`, text, db());
    const index = (await listDatasets()).filter((d) => d.id !== meta.id);
    await set("index", [meta, ...index].slice(0, 12), db());
  } catch {
    // storage can be unavailable (private mode); the session still works in memory
  }
}

export async function loadDatasetText(id: string): Promise<string | undefined> {
  try {
    return await get<string>(`raw:${id}`, db());
  } catch {
    return undefined;
  }
}

export async function deleteDataset(id: string) {
  try {
    await del(`raw:${id}`, db());
    await set("index", (await listDatasets()).filter((d) => d.id !== id), db());
  } catch {
    // ignore
  }
}
