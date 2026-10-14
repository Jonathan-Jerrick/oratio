import { codeLabel, DIMENSION_LABEL, dimensionColor, formatTime, type Region } from "../lib/report";
import { PlayIcon } from "./Icons";

function signed(value: number, digits = 2): string {
  return `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;
}

export function RegionCard({
  region,
  active,
  onSelect,
  onPlay,
  onPlayReference,
}: {
  region: Region;
  active: boolean;
  onSelect: () => void;
  onPlay: () => void;
  onPlayReference: (() => void) | null;
}) {
  const m = region.measurement;
  return (
    <article
      id={`region-${region.region_id}`}
      className={`rounded-2xl border bg-canvas p-5 ${active ? "border-ink shadow-[0_0_0_1px_var(--color-ink)]" : "border-line"}`}
    >
      <header className="flex flex-wrap items-start gap-3">
        <span
          className="mt-1.5 size-3 shrink-0 rounded-full"
          style={{ background: dimensionColor(region.dimension) }}
          aria-hidden="true"
        />
        <div className="min-w-0 flex-1">
          <button type="button" onClick={onSelect} className="text-left">
            <h3 className="text-[17px] font-semibold leading-snug">{codeLabel(region.code)}</h3>
          </button>
          <p className="text-sm text-muted">
            {DIMENSION_LABEL[region.dimension] ?? region.dimension}, {formatTime(region.start)} to{" "}
            {formatTime(region.end)}, “{region.words}”
          </p>
        </div>
        <span className="rounded-full bg-surface px-2.5 py-1 text-sm">
          −{region.deduction.toFixed(2)} pts
        </span>
      </header>

      {m && (
        <table className="mt-4 w-full text-sm">
          <caption className="sr-only">Measurement behind this flag</caption>
          <thead>
            <tr className="text-left text-muted">
              <th className="pb-1 font-normal">{m.label}</th>
              <th className="pb-1 text-right font-normal">You</th>
              <th className="pb-1 text-right font-normal">Reference</th>
              <th className="pb-1 text-right font-normal">Difference</th>
              <th className="pb-1 text-right font-normal">Deviation</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-t border-line">
              <td className="py-1.5 text-muted">{m.unit || "ratio"}</td>
              <td className="py-1.5 text-right font-semibold">{m.participant.toFixed(2)}</td>
              <td className="py-1.5 text-right">{m.baseline.toFixed(2)}</td>
              <td className="py-1.5 text-right">{signed(m.delta)}</td>
              <td className="py-1.5 text-right">
                <span className="font-semibold">{signed(m.z, 1)} σ</span>
                <span className="text-muted"> (limit {m.z_threshold.toFixed(1)})</span>
              </td>
            </tr>
          </tbody>
        </table>
      )}

      <p className="mt-4 text-[15px] leading-relaxed">{region.rationale}</p>
      <p className="mt-3 rounded-xl bg-surface px-4 py-3 text-[15px] leading-relaxed">
        <span className="font-semibold">Try this: </span>
        {region.fix}
      </p>
      <div className="mt-4 flex flex-wrap gap-2">
        <button
          type="button"
          onClick={onPlay}
          className="inline-flex h-9 items-center gap-1.5 rounded-full bg-ink px-3.5 text-sm font-medium text-white hover:bg-brand"
        >
          <PlayIcon className="size-4" /> Hear yours
        </button>
        {onPlayReference && (
          <button
            type="button"
            onClick={onPlayReference}
            className="inline-flex h-9 items-center gap-1.5 rounded-full border border-line px-3.5 text-sm font-medium hover:bg-surface"
          >
            <PlayIcon className="size-4" /> Hear the reference
          </button>
        )}
      </div>
    </article>
  );
}
