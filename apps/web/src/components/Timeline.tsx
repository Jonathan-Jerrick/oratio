import { useEffect, useMemo, useRef, useState } from "react";

import {
  codeLabel,
  decodeTrack,
  dimensionColor,
  formatTime,
  sampleTrack,
  timeMap,
  type Region,
  type Report,
  type Track,
  type TruthLabel,
} from "../lib/report";

type Lane = {
  key: string;
  label: string;
  unit: string;
  height: number;
  participant: Track | null;
  baseline: Track | null;
};

type Props = {
  report: Report;
  truth: TruthLabel[] | null;
  time: number;
  onSeek: (t: number) => void;
  selected: string | null;
  onSelect: (regionId: string) => void;
};

const GUTTER = 104;
const WORD_LANE = 46;
const STRIP = 26;
const GAP = 10;

function robustRange(values: Float32Array[]): [number, number] {
  const all: number[] = [];
  for (const v of values) {
    for (let i = 0; i < v.length; i += 3) {
      const x = v[i] as number;
      if (Number.isFinite(x)) all.push(x);
    }
  }
  if (all.length === 0) return [0, 1];
  all.sort((a, b) => a - b);
  const lo = all[Math.floor(all.length * 0.02)] as number;
  const hi = all[Math.floor(all.length * 0.98)] as number;
  const pad = (hi - lo || 1) * 0.12;
  return [lo - pad, hi + pad];
}

function linePath(
  width: number,
  height: number,
  view: [number, number],
  range: [number, number],
  value: (t: number) => number,
): string {
  const [v0, v1] = view;
  const [lo, hi] = range;
  let d = "";
  let pen = false;
  const step = 1.5;
  for (let x = 0; x <= width; x += step) {
    const t = v0 + ((v1 - v0) * x) / width;
    const y = value(t);
    if (!Number.isFinite(y)) {
      pen = false;
      continue;
    }
    const py = height - ((Math.min(hi, Math.max(lo, y)) - lo) / (hi - lo)) * height;
    d += `${pen ? "L" : "M"}${x.toFixed(1)},${py.toFixed(1)}`;
    pen = true;
  }
  return d;
}

export function Timeline({ report, truth, time, onSeek, selected, onSelect }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(900);
  const [view, setView] = useState<[number, number]>([0, report.participant_seconds]);
  const [hover, setHover] = useState<number | null>(null);
  const duration = report.participant_seconds;

  useEffect(() => {
    const element = box.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry) setWidth(Math.max(320, entry.contentRect.width));
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => setView([0, report.participant_seconds]), [report]);

  const toBaseline = useMemo(() => timeMap(report), [report]);
  const lanes: Lane[] = useMemo(() => {
    const p = report.tracks.participant;
    const b = report.tracks.baseline;
    return [
      { key: "f0_st", label: "Pitch", unit: "st", height: 92, participant: decodeTrack(p, "f0_st"), baseline: decodeTrack(b, "f0_st") },
      { key: "energy_db", label: "Loudness", unit: "dB", height: 92, participant: decodeTrack(p, "energy_db"), baseline: decodeTrack(b, "energy_db") },
      { key: "hf_ratio", label: "Brightness", unit: "", height: 70, participant: decodeTrack(p, "hf_ratio"), baseline: decodeTrack(b, "hf_ratio") },
      { key: "cpp_db", label: "Voice clarity", unit: "dB", height: 70, participant: decodeTrack(p, "cpp_db"), baseline: decodeTrack(b, "cpp_db") },
    ];
  }, [report]);
  const ranges = useMemo(
    () =>
      lanes.map((lane) =>
        robustRange([lane.participant?.values, lane.baseline?.values].filter(Boolean) as Float32Array[]),
      ),
    [lanes],
  );

  const plot = width - GUTTER;
  const [v0, v1] = view;
  const x = (t: number) => ((t - v0) / (v1 - v0)) * plot;
  const tAt = (px: number) => v0 + ((v1 - v0) * px) / plot;
  const truthHeight = truth && truth.length > 0 ? STRIP + GAP : 0;
  const top = truthHeight + STRIP + GAP + WORD_LANE + GAP;
  const lanesTop = lanes.reduce<number[]>((acc, lane, i) => {
    acc.push(i === 0 ? top : (acc[i - 1] as number) + (lanes[i - 1] as Lane).height + GAP);
    return acc;
  }, []);
  const totalHeight = (lanesTop[lanesTop.length - 1] ?? top) + (lanes[lanes.length - 1]?.height ?? 0) + 22;

  const zoom = (factor: number) => {
    const center = hover ?? Math.min(Math.max(time, v0), v1);
    const span = Math.min(duration, Math.max(2, (v1 - v0) * factor));
    let start = center - (center - v0) * (span / (v1 - v0));
    start = Math.max(0, Math.min(duration - span, start));
    setView([start, start + span]);
  };
  const focus = (region: Region) => {
    const pad = Math.max(1.5, (region.end - region.start) * 0.8);
    const start = Math.max(0, region.start - pad);
    const end = Math.min(duration, region.end + pad);
    setView([start, end]);
  };

  const visibleWords = report.words.filter((w) => w.end >= v0 && w.start <= v1);
  const ticks = useMemo(() => {
    const span = v1 - v0;
    const step = [0.5, 1, 2, 5, 10, 15, 30, 60].find((s) => span / s <= 12) ?? 60;
    const out: number[] = [];
    for (let t = Math.ceil(v0 / step) * step; t <= v1; t += step) out.push(t);
    return out;
  }, [v0, v1]);

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
        <div className="flex items-center gap-4 text-muted">
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-0.5 w-5 bg-brand" /> Your delivery
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-0 w-5 border-t-2 border-dashed border-baseline" /> Reference, time-aligned
          </span>
        </div>
        <div className="ml-auto flex items-center gap-1">
          <button type="button" onClick={() => zoom(0.5)} className="h-8 rounded-full border border-line px-3 hover:bg-surface">
            Zoom in
          </button>
          <button type="button" onClick={() => zoom(2)} className="h-8 rounded-full border border-line px-3 hover:bg-surface">
            Zoom out
          </button>
          <button type="button" onClick={() => setView([0, duration])} className="h-8 rounded-full border border-line px-3 hover:bg-surface">
            Whole clip
          </button>
        </div>
      </div>
      <div ref={box} className="relative select-none">
        <svg
          width={width}
          height={totalHeight}
          role="img"
          aria-label="Time-series comparison of your delivery against the reference, with flagged regions"
          onMouseMove={(event) => {
            const px = event.clientX - event.currentTarget.getBoundingClientRect().left - GUTTER;
            setHover(px >= 0 ? tAt(px) : null);
          }}
          onMouseLeave={() => setHover(null)}
          onClick={(event) => {
            const px = event.clientX - event.currentTarget.getBoundingClientRect().left - GUTTER;
            if (px >= 0) onSeek(Math.max(0, Math.min(duration, tAt(px))));
          }}
        >
          <defs>
            <clipPath id="plot-clip">
              <rect x={0} y={0} width={plot} height={totalHeight} />
            </clipPath>
          </defs>
          <g transform={`translate(${GUTTER},0)`}>
            <g clipPath="url(#plot-clip)">
              {report.regions.map((region) => {
                const active = region.region_id === selected;
                return (
                  <rect
                    key={`band-${region.region_id}`}
                    x={x(region.start)}
                    y={truthHeight}
                    width={Math.max(2, x(region.end) - x(region.start))}
                    height={totalHeight - truthHeight - 20}
                    fill={dimensionColor(region.dimension)}
                    opacity={active ? 0.2 : 0.09}
                  />
                );
              })}
              {truth?.map((label, i) => (
                <g key={`truth-${i}`}>
                  <rect
                    x={x(label.start)}
                    y={2}
                    width={Math.max(2, x(label.end) - x(label.start))}
                    height={STRIP - 4}
                    rx={4}
                    fill="none"
                    stroke={dimensionColor(label.dimension)}
                    strokeWidth={1.5}
                    strokeDasharray="4 3"
                  />
                  <text x={x(label.start) + 6} y={STRIP / 2 + 4} fontSize={11} fill="var(--color-muted)">
                    {codeLabel(label.code)}
                  </text>
                </g>
              ))}
              {report.regions.map((region) => {
                const active = region.region_id === selected;
                const w = Math.max(2, x(region.end) - x(region.start));
                return (
                  <g
                    key={`chip-${region.region_id}`}
                    className="cursor-pointer"
                    onClick={(event) => {
                      event.stopPropagation();
                      onSelect(region.region_id);
                      focus(region);
                      onSeek(region.start);
                    }}
                  >
                    <rect
                      x={x(region.start)}
                      y={truthHeight + 2}
                      width={w}
                      height={STRIP - 4}
                      rx={4}
                      fill={dimensionColor(region.dimension)}
                      opacity={active ? 1 : 0.85}
                    />
                    {w > 60 && (
                      <text x={x(region.start) + 6} y={truthHeight + STRIP / 2 + 4} fontSize={11} fontWeight={600} fill="white">
                        {codeLabel(region.code)}
                      </text>
                    )}
                    <title>{`${codeLabel(region.code)}, ${formatTime(region.start)} to ${formatTime(region.end)}`}</title>
                  </g>
                );
              })}
              {visibleWords.map((word, i) => {
                const left = x(word.start);
                const w = x(word.end) - left;
                const flagged = report.regions.find((r) => word.end > r.start && word.start < r.end);
                const row = i % 2;
                return (
                  <g key={`w-${word.start}-${i}`}>
                    <rect
                      x={left}
                      y={top - WORD_LANE - GAP + 4 + row * 20}
                      width={Math.max(1, w)}
                      height={16}
                      rx={3}
                      fill={flagged ? dimensionColor(flagged.dimension) : "var(--color-surface)"}
                      opacity={flagged ? 0.25 : 1}
                    />
                    {w > 22 && (
                      <text x={left + 3} y={top - WORD_LANE - GAP + 16 + row * 20} fontSize={11} fill="var(--color-ink)">
                        {word.word.toLowerCase()}
                      </text>
                    )}
                  </g>
                );
              })}
              {lanes.map((lane, i) => {
                const y0 = lanesTop[i] as number;
                const range = ranges[i] as [number, number];
                return (
                  <g key={lane.key} transform={`translate(0,${y0})`}>
                    <rect width={plot} height={lane.height} fill="none" stroke="var(--color-line)" />
                    <path
                      d={linePath(plot, lane.height, view, range, (t) => sampleTrack(lane.baseline, toBaseline(t)))}
                      fill="none"
                      stroke="var(--color-baseline)"
                      strokeWidth={1.4}
                      strokeDasharray="4 3"
                    />
                    <path
                      d={linePath(plot, lane.height, view, range, (t) => sampleTrack(lane.participant, t))}
                      fill="none"
                      stroke="var(--color-brand)"
                      strokeWidth={1.6}
                    />
                  </g>
                );
              })}
              {ticks.map((t) => (
                <g key={`tick-${t}`} transform={`translate(${x(t)},${totalHeight - 18})`}>
                  <line y1={-4} y2={0} stroke="var(--color-muted)" />
                  <text y={13} fontSize={11} textAnchor="middle" fill="var(--color-muted)">
                    {formatTime(t)}
                  </text>
                </g>
              ))}
              <line x1={x(time)} x2={x(time)} y1={0} y2={totalHeight - 20} stroke="var(--color-ink)" strokeWidth={1.5} />
              {hover !== null && (
                <line x1={x(hover)} x2={x(hover)} y1={0} y2={totalHeight - 20} stroke="var(--color-muted)" strokeDasharray="2 3" />
              )}
            </g>
          </g>
          <g fontSize={12} fill="var(--color-ink)">
            {truthHeight > 0 && (
              <text x={0} y={STRIP / 2 + 4} fill="var(--color-muted)">
                Injected
              </text>
            )}
            <text x={0} y={truthHeight + STRIP / 2 + 4} fontWeight={600}>
              Detected
            </text>
            <text x={0} y={top - WORD_LANE / 2 - 2} fill="var(--color-muted)">
              Words
            </text>
            {lanes.map((lane, i) => (
              <g key={`label-${lane.key}`} transform={`translate(0,${(lanesTop[i] as number) + 16})`}>
                <text fontWeight={600}>{lane.label}</text>
                {hover !== null && (
                  <>
                    <text y={17} fontSize={11} fill="var(--color-brand)">
                      {Number.isFinite(sampleTrack(lane.participant, hover))
                        ? `${sampleTrack(lane.participant, hover).toFixed(1)} ${lane.unit}`
                        : "no voice"}
                    </text>
                    <text y={31} fontSize={11} fill="var(--color-muted)">
                      {Number.isFinite(sampleTrack(lane.baseline, toBaseline(hover)))
                        ? `${sampleTrack(lane.baseline, toBaseline(hover)).toFixed(1)} ${lane.unit} ref`
                        : "ref silent"}
                    </text>
                  </>
                )}
              </g>
            ))}
          </g>
        </svg>
        <div className="mt-1 flex items-center gap-3 pl-[104px] text-xs text-muted">
          <input
            type="range"
            aria-label="Pan the timeline"
            min={0}
            max={Math.max(0, duration - (v1 - v0))}
            step={0.05}
            value={v0}
            disabled={v1 - v0 >= duration - 0.01}
            onChange={(event) => {
              const start = Number(event.target.value);
              setView([start, start + (v1 - v0)]);
            }}
            className="w-full accent-[var(--color-brand)]"
          />
          <span className="whitespace-nowrap">
            {formatTime(v0)} to {formatTime(v1)}
          </span>
        </div>
      </div>
    </div>
  );
}
