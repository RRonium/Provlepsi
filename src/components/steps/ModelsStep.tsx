"use client";
import { ArrowLeft, Check, Cpu, FileCode2, Play, SlidersHorizontal } from "lucide-react";
import { FAMILY_LABEL, MODELS, defaultParams, type Family, type ModelSpec, type ParamSpec } from "@/lib/models";
import { useStudio } from "@/store/studio";
import { Badge, Button, Callout, Card, Label, NumberField, SectionTitle, cx } from "../ui";

function ParamField({ spec, value, onChange }: { spec: ParamSpec; value: number | string | null; onChange: (v: number | string | null) => void }) {
  const num = typeof value === "number" ? value : null;
  return (
    <div>
      <Label help={spec.help}>{spec.label}</Label>
      <div className="flex items-center gap-2">
        {spec.kind === "int" && spec.min !== undefined && spec.max !== undefined && spec.max - spec.min <= 60 && (
          <input
            type="range"
            min={spec.min}
            max={spec.max}
            step={spec.step ?? 1}
            value={num ?? spec.min}
            onChange={(e) => onChange(Number(e.target.value))}
            className="min-w-0 flex-1"
            aria-label={spec.label}
          />
        )}
        <div className={cx(spec.kind === "int" && spec.max !== undefined && spec.min !== undefined && spec.max - spec.min <= 60 ? "w-20" : "w-full")}>
          <NumberField
            ariaLabel={spec.label}
            value={num}
            min={spec.min}
            max={spec.max}
            step={spec.step}
            nullable={spec.kind === "nullable-int"}
            onChange={(v) => onChange(v === null ? null : spec.kind === "float" ? v : Math.round(v))}
          />
        </div>
      </div>
    </div>
  );
}

function ModelCard({ spec, disabledReason }: { spec: ModelSpec; disabledReason?: string }) {
  const params = useStudio((s) => s.selected[spec.id]);
  const toggle = useStudio((s) => s.toggleModel);
  const setParam = useStudio((s) => s.setParam);
  const on = !!params;
  return (
    <div className={cx("rounded-xl border transition", on ? "border-accent bg-accent-soft/40" : "border-line bg-panel", disabledReason && "opacity-55")}>
      <button disabled={!!disabledReason} onClick={() => toggle(spec.id)} className="flex w-full items-start gap-3 p-4 text-left" aria-pressed={on}>
        <span className={cx("mt-0.5 grid size-5 shrink-0 place-items-center rounded-md border transition", on ? "border-accent bg-accent text-accent-fg" : "border-line-strong")}>
          {on && <Check className="size-3.5" strokeWidth={3} />}
        </span>
        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-center gap-2">
            <span className="text-[14px] font-semibold">{spec.name}</span>
            {spec.original && (
              <Badge tone="good">
                <FileCode2 className="size-3" /> Original script
              </Badge>
            )}
            {spec.forecasts && <Badge tone="cyan">Forecasts ahead</Badge>}
          </span>
          <span className="mt-1 block text-[12.5px] leading-snug text-muted">{disabledReason ?? spec.blurb}</span>
        </span>
      </button>
      {on && spec.params.length > 0 && (
        <div className="grid gap-4 border-t border-line px-4 py-4 sm:grid-cols-2">
          {spec.params.map((p) => (
            <ParamField key={p.key} spec={p} value={params[p.key] ?? null} onChange={(v) => setParam(spec.id, p.key, v)} />
          ))}
          <button
            className="text-left text-[12px] text-faint underline-offset-2 hover:text-muted hover:underline sm:col-span-2"
            onClick={() => Object.entries(defaultParams(spec)).forEach(([k, v]) => setParam(spec.id, k, v))}
          >
            Reset to defaults
          </button>
        </div>
      )}
      {on && spec.params.length === 0 && <div className="border-t border-line px-4 py-3 text-[12px] text-faint">No hyperparameters to tune.</div>}
    </div>
  );
}

export function ModelsStep() {
  const prep = useStudio((s) => s.prep)!;
  const processed = useStudio((s) => s.processed);
  const selected = useStudio((s) => s.selected);
  const go = useStudio((s) => s.go);
  const runAll = useStudio((s) => s.runAll);
  const toggle = useStudio((s) => s.toggleModel);

  if (!processed) return <Callout tone="warn">Finish the preprocessing step first.</Callout>;
  const data = processed;
  const visible = MODELS.filter((m) => m.task === prep.task);
  const families = (Object.keys(FAMILY_LABEL) as Family[]).filter((f) => visible.some((m) => m.family === f));
  const count = Object.keys(selected).length;

  const reasonFor = (m: ModelSpec) => {
    if (m.needsFeatures && data.featureColumns.length === 0) return "Needs at least one feature column besides the target. Add one in preprocessing.";
    if (m.id === "qda" && data.classes && data.classes.length > 0) {
      const minClass = Math.min(...data.classes.map((c) => (data.data[data.target] as (string | number)[]).filter((v) => String(v) === c).length));
      if (minClass <= data.featureColumns.length) return "Every class needs more rows than there are features for QDA.";
    }
    return undefined;
  };

  return (
    <div>
      <SectionTitle eyebrow="Step 4 · Model selection" title="Pick models and tune them">
        <div className="flex gap-2">
          <Button variant="ghost" size="lg" onClick={() => go(2)} icon={<ArrowLeft className="size-4" />}>
            Back
          </Button>
          <Button
            variant="primary"
            size="lg"
            disabled={!count}
            onClick={() => {
              go(4);
              void runAll();
            }}
            icon={<Play className="size-4" />}
          >
            Train {count} model{count === 1 ? "" : "s"}
          </Button>
        </div>
      </SectionTitle>

      <div className="grid gap-6 xl:grid-cols-[1fr_300px]">
        <div className="space-y-8">
          {families.map((f) => {
            const models = visible.filter((m) => m.family === f);
            return (
              <section key={f}>
                <div className="mb-3 flex items-baseline justify-between gap-3">
                  <h2 className="text-[15px] font-semibold">{FAMILY_LABEL[f].title}</h2>
                  <span className="font-mono text-[11px] text-faint">{FAMILY_LABEL[f].runtime}</span>
                </div>
                <div className="grid gap-3 lg:grid-cols-2">
                  {models.map((m) => (
                    <ModelCard key={m.id} spec={m} disabledReason={reasonFor(m)} />
                  ))}
                </div>
              </section>
            );
          })}
        </div>

        <aside className="space-y-4 xl:sticky xl:top-24 xl:self-start">
          <Card className="p-5">
            <div className="flex items-center gap-2 text-[13px] font-semibold">
              <SlidersHorizontal className="size-4 text-accent" /> Run setup
            </div>
            <dl className="mt-4 space-y-2.5 text-[13px]">
              {[
                ["Target", data.target],
                ["Task", prep.task],
                ["Features", data.featureColumns.length],
                ["Rows", data.rows.toLocaleString()],
                ["Test split", `${Math.round(prep.testFraction * 100)}% (last rows)`],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between gap-3">
                  <dt className="text-muted">{k}</dt>
                  <dd className="truncate font-medium capitalize">{v}</dd>
                </div>
              ))}
            </dl>
            <div className="mt-4 flex gap-2">
              <Button
                size="sm"
                variant="secondary"
                className="flex-1"
                onClick={() => visible.forEach((m) => !selected[m.id] && !reasonFor(m) && toggle(m.id))}
              >
                Select all
              </Button>
              <Button size="sm" variant="ghost" className="flex-1" onClick={() => Object.keys(selected).forEach((id) => toggle(id))}>
                Clear
              </Button>
            </div>
          </Card>
          <Callout icon={<Cpu className="size-4" />}>
            Python models train one after another in a background worker. Neural networks train at the same time in a second worker on your GPU, so the page
            stays responsive.
          </Callout>
        </aside>
      </div>
    </div>
  );
}
