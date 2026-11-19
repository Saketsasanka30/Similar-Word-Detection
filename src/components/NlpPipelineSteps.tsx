import { ChevronDown, ChevronUp } from 'lucide-react';
import { useState } from 'react';
import type { NlpPipelineStage } from '../types/nlp';

interface NlpPipelineStepsProps {
  stages: NlpPipelineStage[];
  defaultOpen?: boolean;
}

export function NlpPipelineSteps({ stages, defaultOpen = false }: NlpPipelineStepsProps) {
  const [open, setOpen] = useState(defaultOpen);

  if (!stages || stages.length === 0) return null;

  return (
    <div className="rounded-lg bg-[#071522]/75 border border-[rgba(207,221,216,0.10)] p-4 sm:p-5">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between text-left cursor-pointer"
      >
        <div>
          <span className="font-mono text-[10px] tracking-[0.14em] text-[#5BA4D6]">
            NLP PIPELINE STAGES
          </span>
          <span className="block text-xs text-[#91A4B0] mt-0.5">
            6-step processing trace from raw input to vectorization and interpretation
          </span>
        </div>
        <span className="inline-flex items-center gap-1 text-xs text-[#91A4B0]">
          {open ? (
            <>
              Hide steps <ChevronUp className="w-3.5 h-3.5" />
            </>
          ) : (
            <>
              Inspect 6 stages <ChevronDown className="w-3.5 h-3.5" />
            </>
          )}
        </span>
      </button>

      {open && (
        <div className="mt-4 pt-4 border-t border-[rgba(207,221,216,0.08)] grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {stages.map((stage) => (
            <div
              key={stage.step}
              className="p-3 rounded-md bg-[#010A12]/80 border border-[rgba(207,221,216,0.08)] space-y-1.5"
            >
              <div className="font-mono text-[10px] tracking-wider text-[#5BA4D6]">
                {stage.step}. {stage.name.toUpperCase()}
              </div>
              <div className="text-xs font-medium text-[#CFDDD8]">{stage.summary}</div>
              <ul className="space-y-0.5 text-[11px] font-mono text-[#91A4B0]">
                {stage.details.map((d, i) => (
                  <li key={i} className="truncate" title={d}>
                    · {d}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
