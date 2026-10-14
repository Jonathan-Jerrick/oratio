import { useEffect, useMemo, useState } from "react";

import { Notice, PageHeader } from "../components/Shell";
import { ScorePill } from "../components/Scores";
import { href } from "../lib/router";
import { loadSamples, type Sample } from "../lib/samples";

const LEVEL_NAME: Record<number, string> = {
  0: "Reference",
  1: "Near-perfect",
  2: "Minor issues",
  3: "Noticeable",
  4: "Poor",
  5: "Egregious",
};

const SOURCE_NAME: Record<string, string> = {
  "S-PD-COMMONS": "Public-domain speech",
  "S-LV": "LibriVox reading",
  "S-LV-ORATIONS": "LibriVox oration",
  "S-EARN": "Earnings call",
  demo: "Synthetic demo",
};

function Waveglyph({ seed }: { seed: string }) {
  const bars = useMemo(() => {
    let h = 0;
    for (const c of seed) h = (h * 31 + c.charCodeAt(0)) >>> 0;
    return Array.from({ length: 40 }, (_, i) => {
      h = (h * 1103515245 + 12345) >>> 0;
      return 0.25 + 0.75 * Math.abs(Math.sin(i / 3 + (h % 100) / 30));
    });
  }, [seed]);
  return (
    <svg viewBox="0 0 160 40" className="h-10 w-full" aria-hidden="true">
      {bars.map((b, i) => (
        <rect key={i} x={i * 4} y={20 - b * 18} width={2.4} height={b * 36} rx={1.2} fill="var(--color-brand)" opacity={0.35 + b * 0.5} />
      ))}
    </svg>
  );
}

export function Explore() {
  const [samples, setSamples] = useState<Sample[] | null>(null);
  const [query, setQuery] = useState("");

  useEffect(() => {
    void loadSamples().then(setSamples);
    const onSearch = (event: Event) => setQuery((event as CustomEvent<string>).detail);
    window.addEventListener("oratio:search", onSearch);
    return () => window.removeEventListener("oratio:search", onSearch);
  }, []);

  const shown = (samples ?? []).filter((s) =>
    `${s.title} ${s.speaker} ${SOURCE_NAME[s.source] ?? s.source}`.toLowerCase().includes(query.toLowerCase()),
  );

  return (
    <>
      <PageHeader
        title="Hear where a speech goes wrong, and why"
        subtitle="Oratio lines your delivery up against a strong reading of the same words, measures pitch, loudness, pace, pauses and clarity word by word, and marks each stretch that drifts beyond how good speakers vary."
      />
      <section className="px-4 pt-6 md:px-8">
        <div className="flex flex-wrap gap-3">
          <a href="#/analyze" className="inline-flex h-11 items-center rounded-full bg-brand px-5 text-sm font-semibold text-white hover:bg-ink">
            Analyze a recording
          </a>
          <a href="#/method" className="inline-flex h-11 items-center rounded-full border border-line px-5 text-sm font-semibold hover:bg-surface">
            How the scoring works
          </a>
        </div>
      </section>

      <section className="px-4 pt-10 md:px-8" aria-labelledby="samples-heading">
        <div className="mb-4 flex items-baseline justify-between">
          <h2 id="samples-heading" className="text-xl font-semibold">
            Analyzed samples
          </h2>
          <span className="text-sm text-muted">{samples ? `${shown.length} shown` : "Loading"}</span>
        </div>
        {samples && samples.length === 0 && (
          <Notice>No published samples yet. Upload a recording on the Analyze page to see a report.</Notice>
        )}
        <ul className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {shown.map((sample) => (
            <li key={sample.id}>
              <a
                href={href({ page: "report", source: "sample", id: sample.id })}
                className="block h-full rounded-2xl border border-line bg-canvas p-4 hover:border-ink"
              >
                <div className="rounded-xl bg-surface px-3 py-2">
                  <Waveglyph seed={sample.id} />
                </div>
                <div className="mt-3 flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <h3 className="truncate font-semibold">{sample.title}</h3>
                    <p className="truncate text-sm text-muted">{sample.speaker}</p>
                  </div>
                  <ScorePill value={sample.overall} />
                </div>
                <div className="mt-3 flex flex-wrap gap-1.5 text-xs">
                  <span className="rounded-full bg-surface px-2.5 py-1">{SOURCE_NAME[sample.source] ?? sample.source}</span>
                  {sample.level !== null && (
                    <span className="rounded-full bg-surface px-2.5 py-1">
                      Level {sample.level}: {LEVEL_NAME[sample.level] ?? ""}
                    </span>
                  )}
                  <span className="rounded-full bg-surface px-2.5 py-1">{sample.license.replace(/_/g, " ")}</span>
                </div>
              </a>
            </li>
          ))}
        </ul>
      </section>
    </>
  );
}
