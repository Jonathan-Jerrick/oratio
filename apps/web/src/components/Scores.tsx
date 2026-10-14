import { DIMENSION_LABEL, DIMENSIONS, dimensionColor, scoreTone } from "../lib/report";

const TONE_COLOR = { good: "var(--color-good)", warn: "var(--color-warn)", bad: "var(--color-bad)" };

export function ScoreDial({ value, size = 112 }: { value: number; size?: number }) {
  const radius = size / 2 - 8;
  const circumference = 2 * Math.PI * radius;
  const tone = scoreTone(value, 100);
  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} role="img" aria-label={`Overall delivery score ${value.toFixed(0)} out of 100`}>
        <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="var(--color-line)" strokeWidth={8} />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={TONE_COLOR[tone]}
          strokeWidth={8}
          strokeLinecap="round"
          strokeDasharray={`${(circumference * value) / 100} ${circumference}`}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-[30px] font-semibold leading-none">{value.toFixed(0)}</span>
        <span className="mt-1 text-xs text-muted">of 100</span>
      </div>
    </div>
  );
}

export function ScorePill({ value }: { value: number | null }) {
  if (value === null || !Number.isFinite(value)) return null;
  const tone = scoreTone(value, 100);
  return (
    <span
      className="inline-flex h-7 items-center rounded-full px-2.5 text-sm font-semibold"
      style={{ color: TONE_COLOR[tone], background: `color-mix(in srgb, ${TONE_COLOR[tone]} 10%, white)` }}
    >
      {value.toFixed(0)}
    </span>
  );
}

export function RubricBars({
  dimensions,
  onPick,
}: {
  dimensions: Record<string, number>;
  onPick?: (dimension: string) => void;
}) {
  return (
    <ul className="space-y-3">
      {DIMENSIONS.map((dimension) => {
        const value = dimensions[dimension] ?? 10;
        return (
          <li key={dimension}>
            <button
              type="button"
              onClick={() => onPick?.(dimension)}
              className="group w-full text-left"
              disabled={!onPick}
            >
              <div className="mb-1 flex items-baseline justify-between text-sm">
                <span className="flex items-center gap-2 font-medium">
                  <span className="size-2.5 rounded-full" style={{ background: dimensionColor(dimension) }} />
                  {DIMENSION_LABEL[dimension]}
                </span>
                <span className="text-muted">
                  <span className="font-semibold text-ink">{value.toFixed(1)}</span> / 10
                </span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-surface">
                <div
                  className="h-full rounded-full"
                  style={{ width: `${(value / 10) * 100}%`, background: dimensionColor(dimension) }}
                />
              </div>
            </button>
          </li>
        );
      })}
    </ul>
  );
}
