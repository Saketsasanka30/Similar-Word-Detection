import type { VectorGeometry2D } from '../types/nlp';

interface VectorAngleDiagramProps {
  geometry: VectorGeometry2D;
}

export function VectorAngleDiagram({ geometry }: VectorAngleDiagramProps) {
  const ox = 110;
  const oy = 270;
  const radius = 235;

  const pPrimary = geometry.points.find((p) => p.role === 'primary');
  const pSecondary = geometry.points.find((p) => p.role === 'secondary');
  const pContrast = geometry.points.find((p) => p.role === 'contrast');

  if (!pPrimary || !pSecondary) return null;

  const ax = ox + pPrimary.x * radius;
  const ay = oy - pPrimary.y * radius;

  const bx = ox + pSecondary.x * radius;
  const by = oy - pSecondary.y * radius;

  const cx = pContrast ? ox + pContrast.x * (radius * 0.92) : ox + 190;
  const cy = pContrast ? oy - pContrast.y * (radius * 0.92) : oy + 90;

  // Arc for angle θ between Vector A and Vector B
  const arcR = 90;
  const angA = Math.atan2(-pPrimary.y, pPrimary.x);
  const angB = Math.atan2(-pSecondary.y, pSecondary.x);
  const arcStartX = ox + Math.cos(angA) * arcR;
  const arcStartY = oy + Math.sin(angA) * arcR;
  const arcEndX = ox + Math.cos(angB) * arcR;
  const arcEndY = oy + Math.sin(angB) * arcR;
  const sweepFlag = angB >= angA ? 1 : 0;
  const midAng = (angA + angB) / 2;
  const thetaLabelX = ox + Math.cos(midAng) * (arcR + 24);
  const thetaLabelY = oy + Math.sin(midAng) * (arcR + 24);

  return (
    <div className="rounded-lg bg-[#071522]/85 border border-[rgba(207,221,216,0.12)] p-5 sm:p-6 space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="font-mono text-[11px] tracking-[0.14em] text-[#5BA4D6]">
            VECTOR GEOMETRY · COSINE ANGLE
          </p>
          <h4 className="font-display text-base sm:text-lg font-semibold text-[#CFDDD8] mt-0.5">
            Graphical representation relative to &ldquo;{geometry.primaryLabel}&rdquo;
          </h4>
        </div>
        <div className="font-mono text-xs sm:text-sm text-[#CFDDD8] tabular-nums bg-[#010A12] px-3.5 py-1.5 rounded border border-[rgba(207,221,216,0.12)]">
          cos(θ) = <span className="text-[#65C99A] font-semibold">{geometry.cosineSimilarity.toFixed(3)}</span> · θ ={' '}
          <span className="text-[#5BA4D6] font-semibold">{geometry.thetaDegrees.toFixed(1)}°</span>
        </div>
      </div>

      {/* Large Full-Width Coordinate Vector Graph */}
      <div className="w-full bg-[#010A12]/85 rounded-lg border border-[rgba(207,221,216,0.10)] p-3 sm:p-5">
        <svg
          viewBox="0 0 680 420"
          className="w-full h-auto min-h-[300px] sm:min-h-[380px] max-h-[480px] mx-auto"
          role="img"
          aria-label={`2D cosine similarity angle diagram between ${geometry.primaryLabel} and ${geometry.secondaryLabel}`}
        >
          <defs>
            <pattern id="vec-grid" width="24" height="24" patternUnits="userSpaceOnUse">
              <path
                d="M 24 0 L 0 0 0 24"
                fill="none"
                stroke="rgba(207,221,216,0.06)"
                strokeWidth="1"
              />
            </pattern>
            <marker id="arrow-a" markerWidth="10" markerHeight="10" refX="7" refY="4" orient="auto">
              <path d="M0,0 L0,8 L9,4 z" fill="#5BA4D6" />
            </marker>
            <marker id="arrow-b" markerWidth="10" markerHeight="10" refX="7" refY="4" orient="auto">
              <path d="M0,0 L0,8 L9,4 z" fill="#65C99A" />
            </marker>
            <marker id="arrow-c" markerWidth="10" markerHeight="10" refX="7" refY="4" orient="auto">
              <path d="M0,0 L0,8 L9,4 z" fill="#647987" />
            </marker>
          </defs>

          <rect x="0" y="0" width="680" height="420" fill="url(#vec-grid)" />

          {/* Axes */}
          <line x1={ox} y1="24" x2={ox} y2="396" stroke="rgba(207,221,216,0.32)" strokeWidth="1.8" />
          <line x1="36" y1={oy} x2="644" y2={oy} stroke="rgba(207,221,216,0.32)" strokeWidth="1.8" />
          <text x={ox - 24} y="36" fill="#91A4B0" fontSize="16" className="font-mono font-semibold">Y</text>
          <text x="628" y={oy + 24} fill="#91A4B0" fontSize="16" className="font-mono font-semibold">X</text>
          <text x={ox - 42} y={oy + 22} fill="#647987" fontSize="15" className="font-mono">0,0</text>

          {/* Contrasting Vector C */}
          {pContrast && (
            <>
              <line
                x1={ox}
                y1={oy}
                x2={cx}
                y2={cy}
                stroke="#647987"
                strokeWidth="2.6"
                strokeDasharray="6 4"
                markerEnd="url(#arrow-c)"
              />
              <text
                x={(ox + cx) / 2 - 8}
                y={(oy + cy) / 2 + 24}
                fill="#91A4B0"
                fontSize="16"
                className="font-mono font-bold"
              >
                C
              </text>
              <text
                x={Math.min(545, cx + 14)}
                y={Math.min(400, cy + 8)}
                fill="#647987"
                fontSize="15"
                className="font-mono"
              >
                [{geometry.contrastLabel}]
              </text>
            </>
          )}

          {/* Angle Arc θ */}
          {geometry.thetaDegrees > 2 && (
            <>
              <path
                d={`M ${arcStartX} ${arcStartY} A ${arcR} ${arcR} 0 0 ${sweepFlag} ${arcEndX} ${arcEndY}`}
                fill="none"
                stroke="#E9B96E"
                strokeWidth="2.2"
              />
              <text
                x={thetaLabelX}
                y={thetaLabelY}
                fill="#E9B96E"
                fontSize="17"
                className="font-mono font-bold"
              >
                θ
              </text>
            </>
          )}

          {/* Vector B (Secondary / Selected Match) */}
          <line
            x1={ox}
            y1={oy}
            x2={bx}
            y2={by}
            stroke="#65C99A"
            strokeWidth="3.2"
            markerEnd="url(#arrow-b)"
          />
          <text
            x={(ox + bx) / 2 + 10}
            y={(oy + by) / 2 + 22}
            fill="#65C99A"
            fontSize="16"
            className="font-mono font-bold"
          >
            B
          </text>
          <text
            x={Math.min(535, bx + 14)}
            y={Math.max(28, by + 6)}
            fill="#65C99A"
            fontSize="16"
            className="font-mono font-semibold"
          >
            [{geometry.secondaryLabel}]
          </text>

          {/* Vector A (Primary / Target Word) */}
          <line
            x1={ox}
            y1={oy}
            x2={ax}
            y2={ay}
            stroke="#5BA4D6"
            strokeWidth="3.4"
            markerEnd="url(#arrow-a)"
          />
          <text
            x={(ox + ax) / 2 - 22}
            y={(oy + ay) / 2 - 12}
            fill="#5BA4D6"
            fontSize="16"
            className="font-mono font-bold"
          >
            A
          </text>
          <text
            x={Math.min(535, ax + 14)}
            y={Math.max(24, ay - 2)}
            fill="#5BA4D6"
            fontSize="16"
            className="font-mono font-bold"
          >
            [{geometry.primaryLabel}]
          </text>
        </svg>
      </div>

      {/* Angle Interpretation & Cosine Readouts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs sm:text-sm">
        <div className="p-3.5 rounded bg-[#010A12]/75 border border-[rgba(207,221,216,0.08)]">
          <div className="font-mono text-[10px] tracking-wider text-[#5BA4D6]">
            ANGLE INTERPRETATION
          </div>
          <p className="text-[#CFDDD8] mt-1 leading-relaxed">
            Smaller angle <span className="font-mono text-[#E9B96E]">θ</span> means higher cosine similarity.{' '}
            <span className="text-[#5BA4D6] font-medium">{geometry.primaryLabel}</span> and{' '}
            <span className="text-[#65C99A] font-medium">{geometry.secondaryLabel}</span> are separated by{' '}
            <span className="font-mono">{geometry.thetaDegrees.toFixed(1)}°</span>.
          </p>
        </div>

        <div className="p-3.5 rounded bg-[#010A12]/75 border border-[rgba(207,221,216,0.08)] flex flex-col justify-center space-y-1.5">
          <div className="flex justify-between font-mono text-xs sm:text-sm">
            <span className="text-[#91A4B0]">
              cos({geometry.primaryLabel}, {geometry.secondaryLabel})
            </span>
            <span className="text-[#65C99A] font-semibold tabular-nums">
              {geometry.cosineSimilarity.toFixed(3)}
            </span>
          </div>
          <div className="flex justify-between font-mono text-xs sm:text-sm">
            <span className="text-[#91A4B0]">
              cos({geometry.primaryLabel}, {geometry.contrastLabel})
            </span>
            <span className="text-[#647987] tabular-nums">
              {geometry.contrastCosine.toFixed(3)}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
