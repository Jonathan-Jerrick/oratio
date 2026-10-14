import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { PauseIcon, PlayIcon } from "../components/Icons";
import { RegionCard } from "../components/RegionCard";
import { RubricBars, ScoreDial } from "../components/Scores";
import { Notice } from "../components/Shell";
import { Timeline } from "../components/Timeline";
import { judges, type JudgeFeedback } from "../lib/api";
import {
  codeLabel,
  DIMENSION_LABEL,
  dimensionColor,
  formatTime,
  iou,
  timeMap,
  type Report,
  type TruthLabel,
} from "../lib/report";
import { assetUrl, loadReport, loadSamples, loadTruth, recallUpload } from "../lib/samples";

type Loaded = {
  report: Report;
  truth: TruthLabel[] | null;
  title: string;
  subtitle: string;
  audio: string | null;
  reference: string | null;
};

const TABS = ["Overview", "Timeline", "Flaws", "Feedback", "Data", "Run info"] as const;
type Tab = (typeof TABS)[number];

function useLoaded(source: "sample" | "upload", id: string) {
  const [state, setState] = useState<Loaded | "missing" | null>(null);
  useEffect(() => {
    let cancelled = false;
    setState(null);
    (async () => {
      if (source === "upload") {
        const upload = recallUpload(id);
        if (!cancelled)
          setState(
            upload
              ? { report: upload.report, truth: null, title: upload.title, subtitle: "Your upload", audio: upload.audioUrl, reference: null }
              : "missing",
          );
        return;
      }
      const sample = (await loadSamples()).find((s) => s.id === id);
      const report = sample ? await loadReport(sample) : null;
      if (cancelled) return;
      if (!sample || !report) return setState("missing");
      setState({
        report,
        truth: await loadTruth(sample),
        title: sample.title,
        subtitle: sample.speaker,
        audio: assetUrl(sample, sample.audio),
        reference: assetUrl(sample, sample.baseline_audio),
      });
    })();
    return () => {
      cancelled = true;
    };
  }, [source, id]);
  return state;
}

export function ReportPage({ source, id }: { source: "sample" | "upload"; id: string }) {
  const loaded = useLoaded(source, id);
  if (loaded === null) return <div className="p-8 text-muted">Loading the report…</div>;
  if (loaded === "missing")
    return (
      <div className="p-8">
        <Notice tone="error">
          This report is not available. Uploaded reports live only in the tab that created them; open a sample from{" "}
          <a className="underline" href="#/">
            Explore
          </a>{" "}
          or analyze the recording again.
        </Notice>
      </div>
    );
  return <ReportView {...loaded} />;
}

function ReportView({ report, truth, title, subtitle, audio, reference }: Loaded) {
  const [tab, setTab] = useState<Tab>("Overview");
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [selected, setSelected] = useState<string | null>(report.regions[0]?.region_id ?? null);
  const player = useRef<HTMLAudioElement>(null);
  const refPlayer = useRef<HTMLAudioElement>(null);
  const stopAt = useRef<{ el: HTMLAudioElement; t: number } | null>(null);
  const toBaseline = useMemo(() => timeMap(report), [report]);

  const playRange = useCallback((el: HTMLAudioElement | null, start: number, end: number) => {
    if (!el) return;
    player.current?.pause();
    refPlayer.current?.pause();
    el.currentTime = Math.max(0, start - 0.3);
    stopAt.current = { el, t: end + 0.3 };
    void el.play();
  }, []);

  useEffect(() => {
    let frame = 0;
    const tick = () => {
      const el = player.current;
      if (el) setTime(el.currentTime);
      const stop = stopAt.current;
      if (stop && stop.el.currentTime >= stop.t) {
        stop.el.pause();
        stopAt.current = null;
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, []);

  const seek = (t: number) => {
    setTime(t);
    if (player.current) player.current.currentTime = t;
  };

  const matched = useMemo(() => {
    if (!truth) return null;
    const hits = truth.filter((label) => report.regions.some((r) => iou(r, label) >= 0.3)).length;
    const precise = report.regions.filter((r) => truth.some((label) => iou(r, label) >= 0.3)).length;
    return { hits, total: truth.length, precise, flagged: report.regions.length };
  }, [truth, report]);

  const worst = [...report.regions].sort((a, b) => b.deduction - a.deduction).slice(0, 3);

  return (
    <>
      <div className="border-b border-line px-4 pt-8 md:px-8">
        <div className="flex flex-wrap items-start gap-6">
          <div className="min-w-0 flex-1">
            <p className="text-sm text-muted">Delivery report</p>
            <h1 className="mt-1 text-[28px] font-semibold leading-tight tracking-tight">{title}</h1>
            <p className="mt-1 text-[15px] text-muted">
              {subtitle}. {formatTime(report.participant_seconds)} long, {report.words.length} words aligned,{" "}
              {report.regions.length} flagged {report.regions.length === 1 ? "region" : "regions"}.
            </p>
            <div className="mt-4 flex items-center gap-3">
              <button
                type="button"
                disabled={!audio}
                onClick={() => {
                  const el = player.current;
                  if (!el) return;
                  stopAt.current = null;
                  if (el.paused) void el.play();
                  else el.pause();
                }}
                className="inline-flex h-10 items-center gap-2 rounded-full bg-ink px-4 text-sm font-medium text-white hover:bg-brand disabled:opacity-40"
              >
                {playing ? <PauseIcon className="size-4" /> : <PlayIcon className="size-4" />}
                {playing ? "Pause" : "Play recording"}
              </button>
              <span className="text-sm text-muted">{formatTime(time)}</span>
            </div>
            {audio && (
              <audio
                ref={player}
                src={audio}
                preload="auto"
                onPlay={() => setPlaying(true)}
                onPause={() => setPlaying(false)}
              />
            )}
            {reference && <audio ref={refPlayer} src={reference} preload="auto" />}
          </div>
          <ScoreDial value={report.scores.overall} />
        </div>
        <div role="tablist" aria-label="Report sections" className="mt-6 flex gap-1 overflow-x-auto">
          {TABS.map((name) => (
            <button
              key={name}
              role="tab"
              type="button"
              aria-selected={tab === name}
              onClick={() => setTab(name)}
              className={`h-11 whitespace-nowrap border-b-2 px-3 text-sm ${
                tab === name ? "border-ink font-semibold text-ink" : "border-transparent text-muted hover:text-ink"
              }`}
            >
              {name}
            </button>
          ))}
        </div>
      </div>

      <div className="px-4 py-6 md:px-8">
        {tab === "Overview" && (
          <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_320px]">
            <div className="space-y-8">
              <section>
                <h2 className="mb-3 text-lg font-semibold">Where it went wrong</h2>
                <Timeline
                  report={report}
                  truth={truth}
                  time={time}
                  onSeek={seek}
                  selected={selected}
                  onSelect={(regionId) => {
                    setSelected(regionId);
                  }}
                />
              </section>
              <section>
                <h2 className="mb-3 text-lg font-semibold">Fix these first</h2>
                {worst.length === 0 ? (
                  <Notice>No region deviates beyond the natural variation of strong speakers.</Notice>
                ) : (
                  <ol className="space-y-3">
                    {worst.map((region) => (
                      <li key={region.region_id} className="flex gap-3 rounded-2xl border border-line p-4">
                        <span className="mt-1.5 size-3 shrink-0 rounded-full" style={{ background: dimensionColor(region.dimension) }} />
                        <div className="min-w-0">
                          <p className="font-semibold">
                            {codeLabel(region.code)} at {formatTime(region.start)}
                          </p>
                          <p className="text-[15px] text-muted">{region.fix}</p>
                          <button
                            type="button"
                            onClick={() => {
                              setSelected(region.region_id);
                              setTab("Flaws");
                            }}
                            className="mt-1 text-sm font-medium text-brand underline-offset-2 hover:underline"
                          >
                            See the measurement
                          </button>
                        </div>
                      </li>
                    ))}
                  </ol>
                )}
              </section>
            </div>
            <aside className="space-y-6">
              <section className="rounded-2xl border border-line p-5">
                <h2 className="mb-4 font-semibold">Rubric</h2>
                <RubricBars dimensions={report.scores.dimensions} />
              </section>
              {matched && (
                <section className="rounded-2xl border border-line p-5 text-sm">
                  <h2 className="mb-2 font-semibold">Detected against injected</h2>
                  <p className="text-muted">
                    This clip has known, injected flaws. Oratio found {matched.hits} of {matched.total}; {matched.precise} of its{" "}
                    {matched.flagged} flags overlap an injected flaw.
                  </p>
                </section>
              )}
              <section className="rounded-2xl border border-line p-5 text-sm">
                <h2 className="mb-2 font-semibold">About this report</h2>
                <dl className="space-y-1.5 text-muted">
                  <div className="flex justify-between gap-3">
                    <dt>Rubric</dt>
                    <dd className="text-ink">v{report.provenance.rubric_version}</dd>
                  </div>
                  <div className="flex justify-between gap-3">
                    <dt>Engine</dt>
                    <dd className="text-ink">speechgrade {report.provenance.speechgrade_version}</dd>
                  </div>
                  <div className="flex justify-between gap-3">
                    <dt>Report hash</dt>
                    <dd className="truncate text-ink">{report.provenance.report_sha256.slice(0, 12)}</dd>
                  </div>
                </dl>
              </section>
            </aside>
          </div>
        )}

        {tab === "Timeline" && (
          <Timeline report={report} truth={truth} time={time} onSeek={seek} selected={selected} onSelect={setSelected} />
        )}

        {tab === "Flaws" && (
          <div className="mx-auto max-w-3xl space-y-4">
            {report.regions.length === 0 && <Notice>Nothing was flagged in this recording.</Notice>}
            {report.regions.map((region) => (
              <RegionCard
                key={region.region_id}
                region={region}
                active={region.region_id === selected}
                onSelect={() => setSelected(region.region_id)}
                onPlay={() => playRange(player.current, region.start, region.end)}
                onPlayReference={
                  reference ? () => playRange(refPlayer.current, toBaseline(region.start), toBaseline(region.end)) : null
                }
              />
            ))}
          </div>
        )}

        {tab === "Feedback" && <Feedback report={report} />}

        {tab === "Data" && <WordTable report={report} />}

        {tab === "Run info" && (
          <div className="max-w-3xl space-y-4 text-sm">
            <p className="text-muted">
              Reports are deterministic. The same audio, transcript and engine version always produce the same hash, so a
              score can be reproduced and checked.
            </p>
            <pre className="overflow-x-auto rounded-xl bg-surface p-4 text-xs leading-relaxed">
              {JSON.stringify(
                {
                  schema_version: report.schema_version,
                  ...report.provenance,
                  participant_seconds: report.participant_seconds,
                  baseline_seconds: report.baseline_seconds,
                },
                null,
                2,
              )}
            </pre>
            <button
              type="button"
              onClick={() => {
                const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
                const url = URL.createObjectURL(blob);
                const link = document.createElement("a");
                link.href = url;
                link.download = `oratio-report-${report.provenance.report_sha256.slice(0, 12)}.json`;
                link.click();
                URL.revokeObjectURL(url);
              }}
              className="inline-flex h-10 items-center rounded-full border border-line px-4 font-medium hover:bg-surface"
            >
              Download the report as JSON
            </button>
          </div>
        )}
      </div>
    </>
  );
}

function Feedback({ report }: { report: Report }) {
  const [state, setState] = useState<{ kind: "idle" | "loading" } | { kind: "ok"; data: JudgeFeedback } | { kind: "error"; message: string }>({
    kind: "idle",
  });
  const run = () => {
    setState({ kind: "loading" });
    judges(report)
      .then((data) => setState({ kind: "ok", data }))
      .catch((error: unknown) => setState({ kind: "error", message: error instanceof Error ? error.message : String(error) }));
  };
  return (
    <div className="mx-auto max-w-3xl space-y-4">
      <p className="text-[15px] leading-relaxed text-muted">
        A panel of seven language-model judges, one per rubric dimension, turns the measurements into written feedback. Every
        number they quote is checked against the report before it is shown; anything that fails the check is replaced with
        the measured wording.
      </p>
      {state.kind === "idle" && (
        <button type="button" onClick={run} className="inline-flex h-10 items-center rounded-full bg-ink px-4 text-sm font-medium text-white hover:bg-brand">
          Write the feedback
        </button>
      )}
      {state.kind === "loading" && <Notice>The judges are writing.</Notice>}
      {state.kind === "error" && <Notice tone="error">{state.message}</Notice>}
      {state.kind === "ok" && (
        <>
          <section className="rounded-2xl border border-line p-5">
            <h2 className="mb-2 font-semibold">Overall</h2>
            <p className="text-[15px] leading-relaxed">{state.data.overall}</p>
            <p className="mt-3 text-xs text-muted">
              {state.data.verified ? "Every number was verified against the report." : "Some lines fell back to measured wording."}{" "}
              Written by {state.data.provider}.
            </p>
          </section>
          {Object.entries(state.data.dimensions).map(([dimension, text]) => (
            <section key={dimension} className="rounded-2xl border border-line p-5">
              <h3 className="mb-1 flex items-center gap-2 font-semibold">
                <span className="size-2.5 rounded-full" style={{ background: dimensionColor(dimension) }} />
                {DIMENSION_LABEL[dimension] ?? dimension}
              </h3>
              <p className="text-[15px] leading-relaxed">{text}</p>
            </section>
          ))}
        </>
      )}
    </div>
  );
}

function WordTable({ report }: { report: Report }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[560px] text-sm">
        <thead className="sticky top-16 bg-canvas text-left text-muted">
          <tr>
            <th className="py-2 font-normal">#</th>
            <th className="py-2 font-normal">Word</th>
            <th className="py-2 text-right font-normal">Start</th>
            <th className="py-2 text-right font-normal">End</th>
            <th className="py-2 text-right font-normal">Length</th>
            <th className="py-2 text-right font-normal">Alignment confidence</th>
            <th className="py-2 font-normal">Flag</th>
          </tr>
        </thead>
        <tbody>
          {report.words.map((word, i) => {
            const region = report.regions.find((r) => word.end > r.start && word.start < r.end);
            return (
              <tr key={`${word.start}-${i}`} className="border-t border-line">
                <td className="py-1.5 text-muted">{i + 1}</td>
                <td className="py-1.5 font-medium">{word.word.toLowerCase()}</td>
                <td className="py-1.5 text-right">{word.start.toFixed(2)} s</td>
                <td className="py-1.5 text-right">{word.end.toFixed(2)} s</td>
                <td className="py-1.5 text-right">{((word.end - word.start) * 1000).toFixed(0)} ms</td>
                <td className="py-1.5 text-right">{(word.score * 100).toFixed(0)}%</td>
                <td className="py-1.5">
                  {region && (
                    <span className="inline-flex items-center gap-1.5">
                      <span className="size-2 rounded-full" style={{ background: dimensionColor(region.dimension) }} />
                      {codeLabel(region.code)}
                    </span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
