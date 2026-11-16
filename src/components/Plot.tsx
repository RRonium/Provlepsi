"use client";
import { useEffect, useRef, useState } from "react";
import type { Config, Data, Layout } from "plotly.js";
import { useStudio } from "@/store/studio";
import { cx } from "./ui";

type PlotlyModule = {
  react: (el: HTMLElement, data: Data[], layout: Partial<Layout>, config?: Partial<Config>) => Promise<unknown>;
  purge: (el: HTMLElement) => void;
  Plots: { resize: (el: HTMLElement) => void };
};

let plotlyPromise: Promise<PlotlyModule> | null = null;
const loadPlotly = () => (plotlyPromise ??= import("plotly.js-dist-min").then((m) => (m.default ?? m) as unknown as PlotlyModule));

export const SERIES_COLORS = ["#ff7a45", "#5cc8ef", "#a78bfa", "#3ecf8e", "#f0b429", "#f472b6", "#94a3b8", "#22d3ee", "#fb7185", "#84cc16"];

function cssVar(name: string) {
  if (typeof window === "undefined") return "#888";
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
}

export function Plot({
  data,
  layout,
  height = 360,
  filename = "provlepsi-chart",
  className,
}: {
  data: Data[];
  layout?: Partial<Layout>;
  height?: number;
  filename?: string;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const theme = useStudio((s) => s.theme);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const el = ref.current;
    if (!el) return;
    loadPlotly().then((Plotly) => {
      if (cancelled) return;
      const fg = cssVar("--fg");
      const muted = cssVar("--muted");
      const grid = cssVar("--border");
      const base: Partial<Layout> = {
        height,
        autosize: true,
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { family: "var(--font-geist-sans), system-ui, sans-serif", color: muted, size: 12 },
        margin: { l: 56, r: 16, t: 16, b: 44 },
        hovermode: "x unified",
        hoverlabel: { bgcolor: cssVar("--panel"), bordercolor: cssVar("--border-strong"), font: { color: fg } },
        legend: { orientation: "h", y: 1.12, x: 0, font: { color: muted } },
        colorway: SERIES_COLORS,
        ...layout,
        xaxis: { gridcolor: grid, zerolinecolor: grid, linecolor: grid, ...(layout?.xaxis ?? {}) },
        yaxis: { gridcolor: grid, zerolinecolor: grid, linecolor: grid, ...(layout?.yaxis ?? {}) },
      };
      void Plotly.react(el, data, base, {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
        toImageButtonOptions: { filename, format: "png", scale: 2 },
      }).then(() => !cancelled && setReady(true));
    });
    return () => {
      cancelled = true;
    };
  }, [data, layout, height, theme, filename]);

  useEffect(() => {
    const el = ref.current;
    return () => {
      if (el) void loadPlotly().then((P) => P.purge(el));
    };
  }, []);

  return (
    <div className={cx("relative", className)} style={{ minHeight: height }}>
      {!ready && <div className="shimmer absolute inset-x-5 rounded-xl" style={{ height: height - 20 }} />}
      <div ref={ref} className="w-full" />
    </div>
  );
}
