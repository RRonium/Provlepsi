"use client";
import { useStudio } from "@/store/studio";
import { Shell } from "./Shell";
import { UploadStep } from "./steps/UploadStep";
import { ExploreStep } from "./steps/ExploreStep";
import { PrepareStep } from "./steps/PrepareStep";
import { ModelsStep } from "./steps/ModelsStep";
import { TrainStep } from "./steps/TrainStep";
import { ResultsStep } from "./steps/ResultsStep";

export function Studio() {
  const step = useStudio((s) => s.step);
  const hasData = useStudio((s) => !!s.dataset);
  const current = !hasData ? 0 : step;
  return (
    <Shell>
      <div key={current} className="animate-[fadeIn_.25s_ease-out]">
        {current === 0 && <UploadStep />}
        {current === 1 && <ExploreStep />}
        {current === 2 && <PrepareStep />}
        {current === 3 && <ModelsStep />}
        {current === 4 && <TrainStep />}
        {current === 5 && <ResultsStep />}
      </div>
    </Shell>
  );
}
