import { useEffect, useState } from "react";

import { Notice, PageHeader } from "../components/Shell";
import { CODE_LABEL, DIMENSION_LABEL, DIMENSIONS, dimensionColor } from "../lib/report";
import { loadMetrics, type Metrics } from "../lib/samples";

const DATASET_URL = import.meta.env.VITE_DATASET_URL as string | undefined;

const LEVELS = [
  ["L0", "Reference", "The original reading, passed through the same processing chain with no edits."],
  ["L1", "Near-perfect", "One subtle flaw, just past how much good speakers naturally vary."],
  ["L2", "Minor issues", "Two mild flaws."],
  ["L3", "Noticeable", "Three or four moderate flaws."],
  ["L4", "Poor", "Five to seven strong flaws."],
  ["L5", "Egregious", "Eight or more severe flaws plus one that spans the whole clip."],
];

export function Dataset() {
  return (
    <>
      <PageHeader
        title="The contrastive delivery dataset"
        subtitle="Strong public readings, each mirrored into a graded set of flawed versions of the same words, with every flaw's start and end known to the millisecond."
        aside={
          DATASET_URL ? (
            <a href={DATASET_URL} className="inline-flex h-10 items-center rounded-full bg-ink px-4 text-sm font-medium text-white hover:bg-brand">
              Open on Hugging Face
            </a>
          ) : undefined
        }
      />
      <div className="grid gap-8 px-4 pt-8 md:px-8 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="space-y-8">
          <section>
            <h2 className="mb-3 text-lg font-semibold">Where the good readings come from</h2>
            <ul className="space-y-2 text-[15px] leading-relaxed">
              <li>Public-domain speeches by US presidents and other orators, from Wikimedia Commons and presidential libraries.</li>
              <li>LibriVox readings of famous orations, including 19 different readers of the same Gettysburg Address.</li>
              <li>Executives' prepared remarks from the Earnings-21 and Earnings-22 corpora, with speakers from many regions.</li>
              <li>European Parliament speech from VoxPopuli, used for how good speakers vary in general.</li>
            </ul>
          </section>
          <section>
            <h2 className="mb-3 text-lg font-semibold">The severity gradient</h2>
            <table className="w-full text-sm">
              <tbody>
                {LEVELS.map(([code, name, text]) => (
                  <tr key={code} className="border-t border-line align-top">
                    <td className="w-12 py-2.5 font-semibold">{code}</td>
                    <td className="w-36 py-2.5 font-medium">{name}</td>
                    <td className="py-2.5 text-muted">{text}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
          <section>
            <h2 className="mb-3 text-lg font-semibold">Flaws that are injected</h2>
            <div className="grid gap-3 sm:grid-cols-2">
              {DIMENSIONS.map((dimension) => (
                <div key={dimension} className="rounded-2xl border border-line p-4">
                  <h3 className="mb-2 flex items-center gap-2 font-semibold">
                    <span className="size-2.5 rounded-full" style={{ background: dimensionColor(dimension) }} />
                    {DIMENSION_LABEL[dimension]}
                  </h3>
                  <ul className="text-sm text-muted">
                    {Object.entries(CODE_LABEL)
                      .filter(([code]) => dimensionOf(code) === dimension)
                      .map(([code, label]) => (
                        <li key={code}>{label}</li>
                      ))}
                  </ul>
                </div>
              ))}
            </div>
          </section>
        </div>
        <aside className="space-y-4">
          <section className="rounded-2xl border border-line p-5 text-sm leading-relaxed">
            <h2 className="mb-2 font-semibold">How labels stay exact</h2>
            <p className="text-muted">
              Every edit that changes timing records a time map from the original to the edited audio. Labels are moved through
              that map, then the edited clip is aligned again from scratch; a clip whose words drift more than 30 ms from the
              label is rebuilt.
            </p>
          </section>
          <section className="rounded-2xl border border-line p-5 text-sm leading-relaxed">
            <h2 className="mb-2 font-semibold">Licences</h2>
            <p className="text-muted">
              Only public-domain, CC0 and CC BY-SA sources enter the dataset, and every file has a row in the rights register
              with evidence of its licence. No TED material and no voice cloning are used.
            </p>
          </section>
        </aside>
      </div>
    </>
  );
}

function dimensionOf(code: string): string {
  const prefix = code.split("-")[0];
  return (
    { PAC: "pacing", PAU: "pausing", INT: "intonation", ENE: "vocal_energy", EMP: "emphasis", FLU: "fluency", CLA: "clarity", VQ: "clarity" } as Record<
      string,
      string
    >
  )[prefix ?? ""] ?? "clarity";
}

export function Benchmarks() {
  const [metrics, setMetrics] = useState<Metrics | null | undefined>(undefined);
  useEffect(() => {
    void loadMetrics().then(setMetrics);
  }, []);
  const board = metrics?.leaderboard ?? [];
  const columns = board.length > 0 ? Object.keys(board[0] as object) : [];
  const scalars = metrics
    ? Object.entries(metrics).filter(([key, value]) => key !== "leaderboard" && (typeof value === "number" || typeof value === "string"))
    : [];
  return (
    <>
      <PageHeader
        title="Benchmarks"
        subtitle="Every model variant is scored on the same held-out passages, on real recordings with human-annotated disfluencies, and against expert ratings."
      />
      <div className="space-y-8 px-4 pt-8 md:px-8">
        {metrics === undefined && <p className="text-muted">Loading results…</p>}
        {metrics === null && (
          <Notice>
            Results appear here after the evaluation run finishes: event F1 for each flaw, timing error in milliseconds, false
            flags on clean speech, correlation with expert ratings, and stability under gain and pitch changes.
          </Notice>
        )}
        {board.length > 0 && (
          <section>
            <h2 className="mb-3 text-lg font-semibold">Leaderboard</h2>
            <div className="overflow-x-auto rounded-2xl border border-line">
              <table className="w-full min-w-[560px] text-sm">
                <thead className="bg-surface text-left">
                  <tr>
                    <th className="px-4 py-2.5 font-semibold">Rank</th>
                    {columns.map((column) => (
                      <th key={column} className="px-4 py-2.5 font-semibold">
                        {column.replace(/_/g, " ")}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {board.map((row, i) => (
                    <tr key={i} className="border-t border-line">
                      <td className="px-4 py-2.5">
                        <span
                          className="inline-flex size-6 items-center justify-center rounded-full text-xs font-semibold"
                          style={{ background: ["#f5c84c", "#c9ced6", "#d9a06b"][i] ?? "var(--color-surface)" }}
                        >
                          {i + 1}
                        </span>
                      </td>
                      {columns.map((column) => (
                        <td key={column} className="px-4 py-2.5">
                          {typeof row[column] === "number" ? (row[column] as number).toFixed(3) : String(row[column])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}
        {scalars.length > 0 && (
          <section>
            <h2 className="mb-3 text-lg font-semibold">Validation results</h2>
            <dl className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
              {scalars.map(([key, value]) => (
                <div key={key} className="rounded-2xl border border-line p-4">
                  <dt className="text-sm text-muted">{key.replace(/_/g, " ")}</dt>
                  <dd className="mt-1 text-xl font-semibold">{typeof value === "number" ? value.toFixed(3) : String(value)}</dd>
                </div>
              ))}
            </dl>
          </section>
        )}
      </div>
    </>
  );
}

export function Method() {
  return (
    <>
      <PageHeader title="How the scoring works" subtitle="Models find where delivery drifts; plain measurements explain why." />
      <article className="max-w-3xl space-y-6 px-4 pt-8 text-[15px] leading-relaxed md:px-8">
        <section>
          <h2 className="mb-2 text-lg font-semibold">1. Line the words up</h2>
          <p>
            A speech recognizer's character probabilities are aligned to the transcript, which gives every word a start and an
            end time in your recording and in the reference. Words are paired across the two readings, so each moment of your
            delivery has a counterpart in the reference.
          </p>
        </section>
        <section>
          <h2 className="mb-2 text-lg font-semibold">2. Measure each word</h2>
          <p>
            For every word Oratio measures pitch in semitones relative to your own median, loudness relative to your own level,
            duration, the pause after it, spectral brightness and voice clarity. Normalising to your own voice keeps the
            comparison fair across high and low, loud and soft speakers.
          </p>
        </section>
        <section>
          <h2 className="mb-2 text-lg font-semibold">3. Compare with natural variation</h2>
          <p>
            Each difference from the reference is divided by how much strong speakers vary on that measure. A word is flagged
            only when it falls outside that range, so two good but different styles are not punished.
          </p>
        </section>
        <section>
          <h2 className="mb-2 text-lg font-semibold">4. Ground, explain, score</h2>
          <p>
            Flagged words are merged into regions with exact start and end times. Each region carries the measurement behind
            it, the reference value, the size of the deviation and a concrete fix. The seven rubric scores start at ten and
            lose points only for these regions, so every lost point can be traced to a moment in the recording.
          </p>
        </section>
        <section>
          <h2 className="mb-2 text-lg font-semibold">5. Language models write, never score</h2>
          <p>
            The written feedback comes from a panel of language-model judges, one per dimension. They receive the
            measurements, not the audio, and every number they write is checked against the report before it is shown.
          </p>
        </section>
      </article>
    </>
  );
}
