import { ArrowRight } from 'lucide-react';
import { useState } from 'react';
import { TermTooltip } from './TermTooltip';

const PIPELINE_STAGES = [
  {
    step: '01',
    title: 'INPUT',
    description: 'User enters two words or sentences for semantic or lexical comparison.',
  },
  {
    step: '02',
    title: 'PREPROCESSING',
    description: 'Text normalization, case folding, tokenization, and stopword cleaning.',
  },
  {
    step: '03',
    title: 'VECTORIZATION',
    description: 'Words or sentences are converted into sparse TF-IDF or dense neural vectors.',
  },
  {
    step: '04',
    title: 'SIMILARITY COMPUTATION',
    description: 'The selected algorithm calculates Wu-Palmer distance or vector cosine alignment.',
  },
  {
    step: '05',
    title: 'RESULT INTERPRETATION',
    description: 'The system contextualizes the relationship and presents grounded explanations.',
  },
];

export function HowItWorks() {
  const [angleDeg, setAngleDeg] = useState(28);
  const radians = (angleDeg * Math.PI) / 180;
  const cosineVal = Math.cos(radians);

  // Vector B endpoint on a 110px radius arc from origin (40, 140)
  const ox = 44;
  const oy = 148;
  const r = 104;
  const bx = ox + Math.cos(radians) * r;
  const by = oy - Math.sin(radians) * r;

  return (
    <section className="max-w-[1440px] mx-auto px-4 sm:px-8 py-14 sm:py-20 border-t border-[rgba(207,221,216,0.10)]">
      <div className="mb-10">
        <p className="font-mono text-[11px] tracking-[0.16em] text-[#5BA4D6] mb-2">
          HOW IT WORKS
        </p>
        <h2 className="font-display font-bold text-2xl sm:text-4xl tracking-tight text-[#CFDDD8]">
          The NLP Similarity Pipeline
        </h2>
        <p className="text-sm sm:text-base text-[#91A4B0] mt-2 max-w-2xl">
          From raw characters to geometric meaning: every query flows through five transparent computational stages.
        </p>
      </div>

      {/* Horizontal on desktop, vertical on mobile */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4 mb-12">
        {PIPELINE_STAGES.map((stage, idx) => (
          <div
            key={stage.step}
            className="relative rounded-lg bg-[#0B1925] border border-[rgba(207,221,216,0.12)] p-5 flex flex-col justify-between"
          >
            <div>
              <div className="flex items-center justify-between mb-3">
                <span className="font-mono text-xs font-semibold text-[#5BA4D6]">
                  {stage.step}
                </span>
                {idx < PIPELINE_STAGES.length - 1 && (
                  <ArrowRight className="hidden lg:block w-4 h-4 text-[#647987]" aria-hidden="true" />
                )}
              </div>
              <h3 className="font-display text-sm font-bold tracking-wider text-[#CFDDD8]">
                {stage.title}
              </h3>
              <p className="text-xs sm:text-sm text-[#91A4B0] leading-relaxed mt-2">
                {stage.description}
              </p>
            </div>
          </div>
        ))}
      </div>

      {/* Interactive Cosine Similarity Equation Explainer */}
      <div className="rounded-lg bg-[#0B1925] border border-[rgba(207,221,216,0.12)] p-6 sm:p-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
          <div className="lg:col-span-7 space-y-4">
            <p className="font-mono text-[11px] tracking-wider text-[#5BA4D6]">
              MATHEMATICAL FOUNDATION
            </p>
            <h3 className="font-display text-xl sm:text-2xl font-semibold text-[#CFDDD8]">
              Understanding <TermTooltip termKey="Cosine Similarity">Cosine Similarity</TermTooltip>
            </h3>

            <div className="p-4 rounded-md bg-[#071522] border border-[rgba(207,221,216,0.12)] font-mono text-sm sm:text-base text-[#CFDDD8] overflow-x-auto">
              Cosine Similarity = (A · B) / (||A|| × ||B||)
            </div>

            <p className="text-sm text-[#91A4B0] leading-relaxed">
              Think of each word or sentence as an arrow (vector) pointing in a particular direction in meaning space.
              Instead of measuring how long the arrows are, cosine similarity measures the{' '}
              <span className="text-[#CFDDD8] font-medium">angle between them</span>:
              when two vectors point in nearly the same direction (angle close to 0°), their cosine is near{' '}
              <span className="font-mono text-[#65C99A]">1.00</span>. When they point at 90°, they share no directional alignment (<span className="font-mono text-[#E9B96E]">0.00</span>).
            </p>

            <div className="pt-2">
              <label
                htmlFor="angle-slider"
                className="flex items-center justify-between text-xs font-mono text-[#91A4B0] mb-2"
              >
                <span>INTERACTIVE VECTOR ANGLE (θ)</span>
                <span className="text-[#CFDDD8] tabular-nums">
                  θ = {angleDeg}° · cos(θ) = {cosineVal.toFixed(3)} ({(cosineVal * 100).toFixed(1)}%)
                </span>
              </label>
              <input
                id="angle-slider"
                type="range"
                min={0}
                max={90}
                value={angleDeg}
                onChange={(e) => setAngleDeg(Number(e.target.value))}
                className="w-full accent-[#5BA4D6] cursor-pointer"
              />
            </div>
          </div>

          <div className="lg:col-span-5 flex justify-center">
            <div className="w-full max-w-[300px] p-4 rounded-md bg-[#071522] border border-[rgba(207,221,216,0.10)]">
              <svg viewBox="0 0 220 180" className="w-full h-auto" role="img" aria-label="Vector angle cosine similarity diagram">
                <line x1={ox} y1={20} x2={ox} y2={oy} stroke="rgba(207,221,216,0.15)" strokeWidth="1" />
                <line x1={ox} y1={oy} x2={195} y2={oy} stroke="rgba(207,221,216,0.15)" strokeWidth="1" />

                {/* Vector A along X-axis */}
                <line x1={ox} y1={oy} x2={ox + r} y2={oy} stroke="#65C99A" strokeWidth="2.2" />
                <circle cx={ox + r} cy={oy} r="3.5" fill="#65C99A" />
                <text x={ox + r + 8} y={oy + 4} fill="#65C99A" className="font-mono text-[11px]">
                  A
                </text>

                {/* Vector B at angle */}
                <line x1={ox} y1={oy} x2={bx} y2={by} stroke="#5BA4D6" strokeWidth="2.2" />
                <circle cx={bx} cy={by} r="3.5" fill="#5BA4D6" />
                <text x={bx + 6} y={by - 4} fill="#5BA4D6" className="font-mono text-[11px]">
                  B
                </text>

                {/* Angle readout */}
                <text x={ox + 26} y={oy - 12} fill="#CFDDD8" className="font-mono text-[10px]">
                  {angleDeg}°
                </text>
              </svg>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
