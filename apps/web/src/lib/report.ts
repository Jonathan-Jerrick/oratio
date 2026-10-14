export type Measurement = {
  feature: string;
  label: string;
  unit: string;
  participant: number;
  baseline: number;
  delta: number;
  sigma_good: number;
  z: number;
  z_threshold: number;
};

export type Region = {
  region_id: string;
  code: string;
  dimension: string;
  start: number;
  end: number;
  words: string;
  measurement: Measurement | null;
  z: number;
  deduction: number;
  rationale: string;
  fix: string;
};

export type WordRecord = {
  word: string;
  start: number;
  end: number;
  score: number;
  baseline_index: number | null;
};

export type Report = {
  schema_version: string;
  participant_seconds: number;
  baseline_seconds: number;
  words: WordRecord[];
  baseline_words: WordRecord[];
  regions: Region[];
  scores: { dimensions: Record<string, number>; overall: number };
  tracks: Record<string, Record<string, string>>;
  provenance: {
    speechgrade_version: string;
    rubric_version: string;
    rubric_sha256: string;
    report_sha256: string;
  };
};

export type TruthLabel = {
  code: string;
  dimension: string;
  start: number;
  end: number;
  magnitude: number;
};

export const DIMENSIONS = [
  "pacing",
  "pausing",
  "intonation",
  "vocal_energy",
  "emphasis",
  "fluency",
  "clarity",
] as const;

export const DIMENSION_LABEL: Record<string, string> = {
  pacing: "Pacing",
  pausing: "Pausing",
  intonation: "Intonation",
  vocal_energy: "Vocal energy",
  emphasis: "Emphasis",
  fluency: "Fluency",
  clarity: "Clarity",
};

export function dimensionColor(dimension: string): string {
  return `var(--color-dim-${dimension}, var(--color-muted))`;
}

export const CODE_LABEL: Record<string, string> = {
  "PAC-RUSH": "Rushing",
  "PAC-DRAG": "Dragging",
  "PAC-ERR": "Erratic tempo",
  "PAU-DROP": "Missing pause",
  "PAU-MID": "Pause inside a phrase",
  "PAU-LONG": "Hesitation",
  "INT-MONO": "Monotone",
  "INT-UP": "Uptalk",
  "INT-EXAG": "Sing-song pitch",
  "ENE-TRAIL": "Trailing off",
  "ENE-LOW": "Too quiet",
  "ENE-SPIKE": "Sudden shout",
  "ENE-FLAT": "Flat dynamics",
  "EMP-DROP": "Key word under-stressed",
  "EMP-MISP": "Stress on the wrong word",
  "FLU-REP": "Repetition",
  "FLU-FALSE": "False start",
  "FLU-PROL": "Prolonged sound",
  "FLU-OMIT": "Skipped word",
  "FLU-FILL": "Filler word",
  "CLA-MUFF": "Muffled",
  "CLA-SWAL": "Swallowed consonants",
  "VQ-FRY": "Vocal fry",
};

export function codeLabel(code: string): string {
  return CODE_LABEL[code] ?? code;
}

export type Track = { hop: number; values: Float32Array };

export function decodeTrack(encoded: Record<string, string> | undefined, name: string): Track | null {
  if (!encoded) return null;
  const data = encoded[name];
  const hop = Number(encoded.hop_seconds);
  if (!data || !Number.isFinite(hop)) return null;
  const binary = atob(data);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return { hop, values: new Float32Array(bytes.buffer, 0, Math.floor(bytes.length / 4)) };
}

/** Piecewise-linear map from participant time to baseline time, anchored on matched words. */
export function timeMap(report: Report): (t: number) => number {
  const anchors: Array<[number, number]> = [[0, 0]];
  for (const word of report.words) {
    if (word.baseline_index === null) continue;
    const partner = report.baseline_words[word.baseline_index];
    if (!partner) continue;
    anchors.push([word.start, partner.start], [word.end, partner.end]);
  }
  anchors.push([report.participant_seconds, report.baseline_seconds]);
  anchors.sort((a, b) => a[0] - b[0]);
  const clean: Array<[number, number]> = [];
  for (const anchor of anchors) {
    const last = clean[clean.length - 1];
    if (!last || (anchor[0] > last[0] && anchor[1] >= last[1])) clean.push(anchor);
  }
  return (t: number) => {
    if (clean.length < 2) return t;
    let lo = 0;
    let hi = clean.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if ((clean[mid] as [number, number])[0] <= t) lo = mid;
      else hi = mid;
    }
    const [x0, y0] = clean[lo] as [number, number];
    const [x1, y1] = clean[hi] as [number, number];
    if (x1 === x0) return y0;
    const f = Math.min(1, Math.max(0, (t - x0) / (x1 - x0)));
    return y0 + f * (y1 - y0);
  };
}

export function sampleTrack(track: Track | null, t: number): number {
  if (!track) return Number.NaN;
  const index = Math.round(t / track.hop);
  return index >= 0 && index < track.values.length ? (track.values[index] as number) : Number.NaN;
}

export function scoreTone(score: number, max: number): "good" | "warn" | "bad" {
  const ratio = score / max;
  if (ratio >= 0.85) return "good";
  if (ratio >= 0.65) return "warn";
  return "bad";
}

export function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

export function regionWordIndexes(report: Report, region: Region): number[] {
  return report.words
    .map((w, i) => ({ w, i }))
    .filter(({ w }) => w.end > region.start && w.start < region.end)
    .map(({ i }) => i);
}

export function iou(a: { start: number; end: number }, b: { start: number; end: number }): number {
  const inter = Math.max(0, Math.min(a.end, b.end) - Math.max(a.start, b.start));
  const union = Math.max(a.end, b.end) - Math.min(a.start, b.start);
  return union > 0 ? inter / union : 0;
}
