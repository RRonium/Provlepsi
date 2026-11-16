"use client";
import { AlertTriangle, CheckCircle2, Globe, Laptop, Loader2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useStudio } from "@/store/studio";
import { Button, Segmented, cx } from "./ui";

export function EngineMenu({ onClose }: { onClose: () => void }) {
  const { engineKind, localUrl, python, setEngine, ensurePython, dlBackend, setDlBackend } = useStudio();
  const [url, setUrl] = useState(localUrl);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [onClose]);

  const option = (kind: "pyodide" | "local", icon: React.ReactNode, title: string, body: string) => (
    <button
      onClick={() => setEngine(kind, url)}
      className={cx(
        "flex w-full gap-3 rounded-xl border p-3 text-left transition",
        engineKind === kind ? "border-accent bg-accent-soft" : "border-line hover:border-line-strong",
      )}
    >
      <div className="mt-0.5 text-accent">{icon}</div>
      <div>
        <div className="text-[13px] font-medium">{title}</div>
        <div className="mt-0.5 text-[12px] leading-snug text-muted">{body}</div>
      </div>
    </button>
  );

  return (
    <div ref={ref} className="absolute top-11 right-0 z-40 w-[340px] max-w-[calc(100vw-2rem)] rounded-2xl border border-line bg-panel p-4 shadow-2xl">
      <div className="mb-3 text-[13px] font-semibold">Python engine</div>
      <div className="space-y-2">
        {option("pyodide", <Globe className="size-4" />, "In your browser (Pyodide)", "Nothing to install. The Python runtime downloads once, then everything runs in this tab.")}
        {option("local", <Laptop className="size-4" />, "Local Python on this computer", "Faster for big files. Run python python/local_server.py, which listens on localhost only.")}
      </div>
      {engineKind === "local" && (
        <div className="mt-3 flex gap-2">
          <input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onBlur={() => setEngine("local", url)}
            className="h-9 min-w-0 flex-1 rounded-lg border border-line bg-panel-2 px-3 font-mono text-[12px] outline-none focus:border-accent"
          />
        </div>
      )}
      <div className="mt-4 flex items-start gap-2 rounded-xl bg-panel-2 p-3 text-[12px] text-muted">
        {python.status === "ready" ? (
          <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-good" />
        ) : python.status === "loading" ? (
          <Loader2 className="mt-0.5 size-4 shrink-0 animate-spin text-warn" />
        ) : python.status === "error" ? (
          <AlertTriangle className="mt-0.5 size-4 shrink-0 text-bad" />
        ) : (
          <div className="mt-1.5 size-2 shrink-0 rounded-full bg-faint" />
        )}
        <div className="min-w-0 break-words">
          {python.status === "ready" ? python.versions : python.status === "idle" ? "Not started yet. It starts when you load a dataset." : python.message}
        </div>
      </div>
      {python.status !== "ready" && python.status !== "loading" && (
        <Button size="sm" className="mt-3 w-full" onClick={() => void ensurePython().catch(() => undefined)}>
          {python.status === "error" ? "Try again" : "Start now"}
        </Button>
      )}
      <div className="mt-5 mb-2 text-[13px] font-semibold">Neural network backend</div>
      <Segmented
        size="sm"
        value={dlBackend}
        onChange={setDlBackend}
        options={[
          { value: "auto", label: "Auto" },
          { value: "webgl", label: "GPU (WebGL)" },
          { value: "cpu", label: "CPU" },
        ]}
      />
      <p className="mt-2 text-[12px] leading-snug text-muted">Auto uses your GPU, and falls back to the CPU when the browser only has software graphics.</p>
    </div>
  );
}
