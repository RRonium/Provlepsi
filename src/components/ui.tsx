"use client";
import clsx from "clsx";
import { Info, Loader2 } from "lucide-react";
import type { ButtonHTMLAttributes, ReactNode } from "react";

export function cx(...args: Parameters<typeof clsx>) {
  return clsx(...args);
}

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";

export function Button({
  variant = "secondary",
  size = "md",
  loading,
  icon,
  className,
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: ButtonVariant; size?: "sm" | "md" | "lg"; loading?: boolean; icon?: ReactNode }) {
  return (
    <button
      {...rest}
      disabled={rest.disabled || loading}
      className={cx(
        "inline-flex items-center justify-center gap-2 rounded-lg font-medium transition-[background,color,box-shadow,transform] select-none",
        "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:opacity-45 disabled:pointer-events-none active:translate-y-px",
        size === "sm" && "h-8 px-3 text-[13px]",
        size === "md" && "h-10 px-4 text-sm",
        size === "lg" && "h-12 px-6 text-[15px]",
        variant === "primary" && "bg-accent text-accent-fg hover:brightness-110 shadow-[0_6px_20px_-8px_var(--accent)]",
        variant === "secondary" && "bg-panel-2 text-fg border border-line hover:border-line-strong",
        variant === "ghost" && "text-muted hover:text-fg hover:bg-panel-2",
        variant === "danger" && "bg-bad-soft text-bad hover:brightness-110",
        className,
      )}
    >
      {loading ? <Loader2 className="size-4 animate-spin" /> : icon}
      {children}
    </button>
  );
}

export function Card({ className, children, ...rest }: { className?: string; children: ReactNode } & React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div {...rest} className={cx("rounded-2xl border border-line bg-panel shadow-panel", className)}>
      {children}
    </div>
  );
}

export function CardHeader({ title, subtitle, actions, icon }: { title: ReactNode; subtitle?: ReactNode; actions?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-3 px-5 pt-5 pb-3">
      <div className="flex min-w-0 items-start gap-3">
        {icon && <div className="mt-0.5 grid size-8 shrink-0 place-items-center rounded-lg bg-accent-soft text-accent">{icon}</div>}
        <div className="min-w-0">
          <h3 className="text-[15px] font-semibold tracking-tight">{title}</h3>
          {subtitle && <p className="mt-0.5 text-[13px] text-muted">{subtitle}</p>}
        </div>
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}

export function Badge({ tone = "neutral", children, className }: { tone?: "neutral" | "accent" | "good" | "bad" | "warn" | "cyan"; children: ReactNode; className?: string }) {
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium whitespace-nowrap",
        tone === "neutral" && "bg-panel-2 text-muted border border-line",
        tone === "accent" && "bg-accent-soft text-accent",
        tone === "good" && "bg-good-soft text-good",
        tone === "bad" && "bg-bad-soft text-bad",
        tone === "warn" && "bg-[color-mix(in_srgb,var(--warn)_14%,transparent)] text-warn",
        tone === "cyan" && "bg-[color-mix(in_srgb,var(--cyan)_14%,transparent)] text-cyan",
        className,
      )}
    >
      {children}
    </span>
  );
}

export function Label({ children, help }: { children: ReactNode; help?: string }) {
  return (
    <div className="mb-1.5 flex items-center gap-1.5 text-[12px] font-medium text-muted">
      {children}
      {help && (
        <span title={help} className="cursor-help text-faint">
          <Info className="size-3.5" />
        </span>
      )}
    </div>
  );
}

export function Select<T extends string>({
  value,
  onChange,
  options,
  className,
  ariaLabel,
}: {
  value: T | null;
  onChange: (v: T) => void;
  options: { value: T; label: string; disabled?: boolean }[];
  className?: string;
  ariaLabel?: string;
}) {
  return (
    <select
      aria-label={ariaLabel}
      value={value ?? ""}
      onChange={(e) => onChange(e.target.value as T)}
      className={cx(
        "h-10 w-full rounded-lg border border-line bg-panel-2 px-3 text-sm text-fg outline-none transition focus:border-accent",
        className,
      )}
    >
      {value === null && <option value="">Choose…</option>}
      {options.map((o) => (
        <option key={o.value} value={o.value} disabled={o.disabled}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

export function Segmented<T extends string>({
  value,
  onChange,
  options,
  size = "md",
}: {
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: ReactNode; icon?: ReactNode }[];
  size?: "sm" | "md";
}) {
  return (
    <div role="tablist" className="inline-flex max-w-full flex-wrap gap-1 rounded-xl border border-line bg-panel-2 p-1">
      {options.map((o) => (
        <button
          key={o.value}
          role="tab"
          aria-selected={o.value === value}
          onClick={() => onChange(o.value)}
          className={cx(
            "inline-flex items-center gap-1.5 rounded-lg font-medium transition",
            size === "sm" ? "h-7 px-2.5 text-[12px]" : "h-8 px-3 text-[13px]",
            o.value === value ? "bg-panel text-fg shadow-panel" : "text-muted hover:text-fg",
          )}
        >
          {o.icon}
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function NumberField({
  value,
  onChange,
  min,
  max,
  step,
  nullable,
  placeholder,
  ariaLabel,
}: {
  ariaLabel?: string;
  value: number | null;
  onChange: (v: number | null) => void;
  min?: number;
  max?: number;
  step?: number;
  nullable?: boolean;
  placeholder?: string;
}) {
  return (
    <input
      type="number"
      aria-label={ariaLabel}
      inputMode="decimal"
      value={value ?? ""}
      min={min}
      max={max}
      step={step}
      placeholder={placeholder ?? (nullable ? "None" : undefined)}
      onChange={(e) => {
        const raw = e.target.value;
        if (raw === "") return onChange(nullable ? null : (min ?? 0));
        const n = Number(raw);
        if (Number.isFinite(n)) onChange(n);
      }}
      className="tabular h-10 w-full rounded-lg border border-line bg-panel-2 px-3 text-sm outline-none transition focus:border-accent"
    />
  );
}

export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: ReactNode }) {
  return (
    <label className="inline-flex cursor-pointer items-center gap-2.5 text-sm">
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={cx("relative h-5 w-9 rounded-full transition", checked ? "bg-accent" : "bg-line-strong")}
      >
        <span className={cx("absolute top-0.5 size-4 rounded-full bg-white shadow transition-all", checked ? "left-[18px]" : "left-0.5")} />
      </button>
      <span className="text-muted">{label}</span>
    </label>
  );
}

export function Stat({ label, value, hint, tone }: { label: string; value: ReactNode; hint?: ReactNode; tone?: "accent" | "good" }) {
  return (
    <div className="rounded-xl border border-line bg-panel-2/60 px-4 py-3">
      <div className="text-[11px] font-medium tracking-wide text-faint uppercase">{label}</div>
      <div className={cx("tabular mt-1 text-xl font-semibold tracking-tight", tone === "accent" && "text-accent", tone === "good" && "text-good")}>{value}</div>
      {hint && <div className="mt-0.5 text-[12px] text-muted">{hint}</div>}
    </div>
  );
}

export function SectionTitle({ eyebrow, title, children }: { eyebrow: string; title: string; children?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <div className="mb-1.5 font-mono text-[11px] tracking-[0.18em] text-accent uppercase">{eyebrow}</div>
        <h1 className="text-2xl font-semibold tracking-tight sm:text-[28px]">{title}</h1>
      </div>
      {children}
    </div>
  );
}

export function Callout({ tone = "neutral", icon, children }: { tone?: "neutral" | "bad" | "warn" | "good"; icon?: ReactNode; children: ReactNode }) {
  return (
    <div
      className={cx(
        "flex gap-2.5 rounded-xl border px-3.5 py-3 text-[13px] leading-relaxed",
        tone === "neutral" && "border-line bg-panel-2/60 text-muted",
        tone === "bad" && "border-transparent bg-bad-soft text-bad",
        tone === "good" && "border-transparent bg-good-soft text-good",
        tone === "warn" && "border-transparent bg-[color-mix(in_srgb,var(--warn)_12%,transparent)] text-warn",
      )}
    >
      {icon && <div className="mt-0.5 shrink-0">{icon}</div>}
      <div className="min-w-0">{children}</div>
    </div>
  );
}
