interface SimilarityGaugeProps {
  score: number | null;
  label?: string;
  size?: number;
}

export function SimilarityGauge({
  score,
  label = 'Semantic Similarity',
  size = 176,
}: SimilarityGaugeProps) {
  const strokeWidth = 10;
  const center = size / 2;
  const radius = center - strokeWidth;
  const circumference = 2 * Math.PI * radius;
  const clamped = score === null ? 0 : Math.max(0, Math.min(100, score));
  const dashOffset = circumference - (clamped / 100) * circumference;

  const strokeColor =
    score === null
      ? '#647987'
      : clamped >= 70
        ? '#65C99A'
        : clamped >= 40
          ? '#5BA4D6'
          : '#E9B96E';

  return (
    <div className="flex flex-col items-center">
      <div className="relative" style={{ width: size, height: size }}>
        <svg
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          className="-rotate-90"
          role="img"
          aria-label={
            score !== null
              ? `${label}: ${score.toFixed(1)} percent`
              : `${label}: Not available`
          }
        >
          <circle
            cx={center}
            cy={center}
            r={radius}
            fill="none"
            stroke="rgba(207,221,216,0.10)"
            strokeWidth={strokeWidth}
          />
          {score !== null && (
            <circle
              cx={center}
              cy={center}
              r={radius}
              fill="none"
              stroke={strokeColor}
              strokeWidth={strokeWidth}
              strokeLinecap="round"
              strokeDasharray={circumference}
              strokeDashoffset={dashOffset}
              style={{ transition: 'stroke-dashoffset 600ms cubic-bezier(0.16, 1, 0.3, 1)' }}
            />
          )}
        </svg>

        <div className="absolute inset-0 flex flex-col items-center justify-center text-center px-3">
          <span className="font-mono text-3xl sm:text-4xl font-semibold text-[#CFDDD8] tabular-nums tracking-tight">
            {score !== null ? `${score.toFixed(1)}%` : 'N/A'}
          </span>
          <span className="text-[11px] text-[#91A4B0] mt-1">
            {label}
          </span>
        </div>
      </div>
    </div>
  );
}
