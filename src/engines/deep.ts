"use client";
import * as Comlink from "comlink";
import type { DeepApi } from "./tf.worker";

let remote: Comlink.Remote<DeepApi> | null = null;

export function deepEngine() {
  if (!remote) {
    const worker = new Worker(new URL("./tf.worker.ts", import.meta.url), { type: "module", name: "provlepsi-tensorflow" });
    remote = Comlink.wrap<DeepApi>(worker);
  }
  return remote;
}

export { Comlink };
