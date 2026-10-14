import { useEffect, useRef, useState } from "react";

import { MicIcon } from "../components/Icons";
import { Notice, PageHeader } from "../components/Shell";
import { analyze, ApiError, warmUp, type StageEvent } from "../lib/api";
import { navigate } from "../lib/router";
import { loadReport, loadSamples, rememberUpload, type Sample } from "../lib/samples";
import { MAX_SECONDS, toWav16k } from "../lib/wav";

const STAGES = [
  { key: "align", label: "Aligning words to the audio" },
  { key: "features", label: "Measuring pitch, loudness, pace and clarity" },
  { key: "detect", label: "Comparing against the reference" },
  { key: "score", label: "Scoring and writing explanations" },
];

type Reference = { id: string; title: string; transcript: string };

async function loadReferences(): Promise<Reference[]> {
  const samples = await loadSamples();
  const byBaseline = new Map<string, Sample>();
  for (const sample of samples) if (sample.baseline_id && !byBaseline.has(sample.baseline_id)) byBaseline.set(sample.baseline_id, sample);
  const refs: Reference[] = [];
  for (const [id, sample] of byBaseline) {
    const report = await loadReport(sample);
    if (!report) continue;
    refs.push({ id, title: sample.title, transcript: report.baseline_words.map((w) => w.word.toLowerCase()).join(" ") });
  }
  return refs;
}

export function Analyze() {
  const [file, setFile] = useState<Blob | null>(null);
  const [fileName, setFileName] = useState("");
  const [transcript, setTranscript] = useState("");
  const [refs, setRefs] = useState<Reference[]>([]);
  const [refId, setRefId] = useState("");
  const [recording, setRecording] = useState(false);
  const [stages, setStages] = useState<Record<string, StageEvent>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);

  useEffect(() => {
    void warmUp();
    void loadReferences().then(setRefs);
  }, []);

  const pickReference = (id: string) => {
    setRefId(id);
    const ref = refs.find((r) => r.id === id);
    if (ref && !transcript.trim()) setTranscript(ref.transcript);
  };

  const toggleRecording = async () => {
    setError(null);
    if (recording) {
      recorder.current?.stop();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      chunks.current = [];
      rec.ondataavailable = (event) => chunks.current.push(event.data);
      rec.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        setFile(new Blob(chunks.current, { type: rec.mimeType }));
        setFileName("Microphone recording");
        setRecording(false);
      };
      rec.start();
      recorder.current = rec;
      setRecording(true);
    } catch {
      setError("Microphone access was refused. Allow it in the browser's site settings, or upload a file instead.");
    }
  };

  const submit = async () => {
    if (!file) return setError("Choose or record a recording first.");
    if (!transcript.trim()) return setError("Paste the words you read, so they can be aligned to the audio.");
    setBusy(true);
    setError(null);
    setStages({});
    try {
      const { wav } = await toWav16k(file);
      const report = await analyze(wav, transcript.trim(), refId || null, {
        onStage: (event) => setStages((prev) => ({ ...prev, [event.stage]: event })),
      });
      const id = rememberUpload(report, URL.createObjectURL(file), fileName || "Your recording");
      navigate({ page: "report", source: "upload", id });
    } catch (caught) {
      setError(caught instanceof ApiError || caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <PageHeader
        title="Analyze a recording"
        subtitle={`Upload or record up to ${MAX_SECONDS} seconds, add the words you read, and pick the reading to compare against.`}
      />
      <div className="grid gap-6 px-4 pt-6 md:px-8 lg:grid-cols-[minmax(0,1fr)_340px]">
        <div className="space-y-6">
          <section className="rounded-2xl border border-line p-5">
            <h2 className="font-semibold">1. Your recording</h2>
            <div className="mt-4 flex flex-wrap items-center gap-3">
              <label className="inline-flex h-10 cursor-pointer items-center rounded-full border border-line px-4 text-sm font-medium hover:bg-surface">
                Choose a file
                <input
                  type="file"
                  accept="audio/*,.wav,.mp3,.m4a,.ogg,.webm,.flac"
                  className="sr-only"
                  onChange={(event) => {
                    const chosen = event.target.files?.[0];
                    if (chosen) {
                      setFile(chosen);
                      setFileName(chosen.name);
                    }
                  }}
                />
              </label>
              <button
                type="button"
                onClick={() => void toggleRecording()}
                className={`inline-flex h-10 items-center gap-2 rounded-full px-4 text-sm font-medium ${
                  recording ? "bg-bad text-white" : "border border-line hover:bg-surface"
                }`}
              >
                <MicIcon className="size-4" />
                {recording ? "Stop recording" : "Record"}
              </button>
              <span className="text-sm text-muted">{fileName || "Nothing chosen yet"}</span>
            </div>
            {file && <audio className="mt-4 w-full" controls src={URL.createObjectURL(file)} />}
          </section>

          <section className="rounded-2xl border border-line p-5">
            <h2 className="font-semibold">2. The reference reading</h2>
            <p className="mt-1 text-sm text-muted">
              Your delivery is compared word for word with a strong reading of the same text. Without one, Oratio compares you
              with how good speakers vary in general.
            </p>
            <select
              value={refId}
              onChange={(event) => pickReference(event.target.value)}
              className="mt-3 h-10 w-full rounded-xl border border-line bg-canvas px-3 text-sm"
            >
              <option value="">No reference: use general norms</option>
              {refs.map((ref) => (
                <option key={ref.id} value={ref.id}>
                  {ref.title}
                </option>
              ))}
            </select>
          </section>

          <section className="rounded-2xl border border-line p-5">
            <h2 className="font-semibold">3. What you read</h2>
            <textarea
              value={transcript}
              onChange={(event) => setTranscript(event.target.value)}
              rows={6}
              maxLength={5000}
              placeholder="Paste the text exactly as you read it."
              className="mt-3 w-full rounded-xl border border-line p-3 text-[15px] leading-relaxed outline-none focus:border-brand"
            />
          </section>

          {error && <Notice tone="error">{error}</Notice>}

          <button
            type="button"
            disabled={busy}
            onClick={() => void submit()}
            className="inline-flex h-11 items-center rounded-full bg-brand px-6 text-sm font-semibold text-white hover:bg-ink disabled:opacity-50"
          >
            {busy ? "Analyzing…" : "Analyze delivery"}
          </button>
        </div>

        <aside className="space-y-4">
          <section className="rounded-2xl border border-line p-5" aria-live="polite">
            <h2 className="mb-3 font-semibold">Progress</h2>
            <ol className="space-y-2 text-sm">
              {STAGES.map((stage) => {
                const event = stages[stage.key];
                const done = event?.status === "done";
                const running = event?.status === "start";
                return (
                  <li key={stage.key} className="flex items-center gap-2.5">
                    <span
                      className={`size-2.5 rounded-full ${done ? "bg-good" : running ? "animate-pulse bg-brand" : "bg-line"}`}
                      aria-hidden="true"
                    />
                    <span className={done || running ? "text-ink" : "text-muted"}>{stage.label}</span>
                    {done && event?.elapsed_ms !== undefined && (
                      <span className="ml-auto text-muted">{(event.elapsed_ms / 1000).toFixed(1)} s</span>
                    )}
                  </li>
                );
              })}
            </ol>
          </section>
          <Notice>
            Audio is converted to 16 kHz mono in your browser and processed in memory on the server. It is not stored.
          </Notice>
        </aside>
      </div>
    </>
  );
}
