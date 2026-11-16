"use client";
import { Check, Cpu, Moon, ShieldCheck, Sun } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { STEPS, useStudio, type Step } from "@/store/studio";
import { cx } from "./ui";
import { EngineMenu } from "./EngineMenu";

export function Logo() {
  return (
    <div className="flex items-center gap-2.5">
      <svg viewBox="0 0 32 32" className="size-8" aria-hidden>
        <rect width="32" height="32" rx="9" className="fill-accent" />
        <path d="M6 21.5 11 16l4 3.5 5-7" stroke="var(--accent-fg)" strokeWidth="2.4" fill="none" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M20 12.5 26 8" stroke="var(--accent-fg)" strokeWidth="2.4" fill="none" strokeLinecap="round" strokeDasharray="1.5 3.2" />
      </svg>
      <div className="leading-none">
        <div className="text-[15px] font-semibold tracking-tight">
          Prόvlepsi <span className="text-accent">AI</span>
        </div>
        <div className="mt-1 hidden font-mono text-[10px] tracking-[0.14em] text-faint uppercase sm:block">Time-series AutoML studio</div>
      </div>
    </div>
  );
}

function Stepper() {
  const step = useStudio((s) => s.step);
  const maxStep = useStudio((s) => s.maxStep);
  const go = useStudio((s) => s.go);
  return (
    <nav aria-label="Workflow" className="flex items-center gap-1 overflow-x-auto">
      {STEPS.map((label, i) => {
        const done = i < step || (i <= maxStep && i !== step);
        const active = i === step;
        const reachable = i <= maxStep;
        return (
          <button
            key={label}
            disabled={!reachable}
            onClick={() => go(i as Step)}
            aria-current={active ? "step" : undefined}
            className={cx(
              "group flex h-8 shrink-0 items-center gap-2 rounded-full pr-3 pl-1 text-[12.5px] font-medium transition",
              active ? "bg-accent-soft text-fg" : reachable ? "text-muted hover:text-fg" : "text-faint",
            )}
          >
            <span
              className={cx(
                "grid size-6 place-items-center rounded-full font-mono text-[11px] transition",
                active ? "bg-accent text-accent-fg" : done ? "bg-good-soft text-good" : "border border-line text-faint",
              )}
            >
              {done && !active ? <Check className="size-3.5" /> : i + 1}
            </span>
            <span className={cx(!active && "hidden md:inline")}>{label}</span>
          </button>
        );
      })}
    </nav>
  );
}

export function Shell({ children }: { children: ReactNode }) {
  const theme = useStudio((s) => s.theme);
  const toggleTheme = useStudio((s) => s.toggleTheme);
  const python = useStudio((s) => s.python);
  const engineKind = useStudio((s) => s.engineKind);
  const [menu, setMenu] = useState(false);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  const dot = python.status === "ready" ? "bg-good" : python.status === "loading" ? "bg-warn pulse-ring" : python.status === "error" ? "bg-bad" : "bg-faint";

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-30 border-b border-line bg-[color-mix(in_srgb,var(--bg)_82%,transparent)] backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-[1320px] items-center gap-4 px-4 sm:px-6">
          <Logo />
          <div className="mx-auto min-w-0">
            <Stepper />
          </div>
          <div className="relative flex items-center gap-1.5">
            <button
              onClick={() => setMenu((m) => !m)}
              className="flex h-8 items-center gap-2 rounded-full border border-line bg-panel px-3 text-[12px] text-muted transition hover:text-fg"
              title={python.message || "Python engine"}
            >
              <span className={cx("size-2 rounded-full", dot)} />
              <Cpu className="size-3.5" />
              <span className="hidden lg:inline">{engineKind === "local" ? "Local Python" : "In-browser"}</span>
            </button>
            {menu && <EngineMenu onClose={() => setMenu(false)} />}
            <button onClick={toggleTheme} aria-label="Toggle theme" className="grid size-8 place-items-center rounded-full text-muted transition hover:bg-panel-2 hover:text-fg">
              {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
            </button>
          </div>
        </div>
      </header>
      <main className="mx-auto w-full max-w-[1320px] flex-1 px-4 py-8 sm:px-6 sm:py-10">{children}</main>
      <footer className="border-t border-line">
        <div className="mx-auto flex max-w-[1320px] flex-wrap items-center justify-between gap-2 px-4 py-4 text-[12px] text-faint sm:px-6">
          <span className="inline-flex items-center gap-1.5">
            <ShieldCheck className="size-3.5 text-good" /> Your data never leaves this device. Parsing, cleaning and training all run locally.
          </span>
          <span>Prόvlepsi AI · IDEAS, ISI Kolkata internship project</span>
        </div>
      </footer>
    </div>
  );
}
