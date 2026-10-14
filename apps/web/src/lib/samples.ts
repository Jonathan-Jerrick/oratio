import type { Report, TruthLabel } from "./report";

export type Sample = {
  id: string;
  title: string;
  speaker: string;
  source: string;
  license: string;
  level: number | null;
  baseline_id: string | null;
  report: string;
  audio: string | null;
  baseline_audio: string | null;
  overall: number | null;
  truth?: string | null;
  base: string;
};

const SOURCES = ["/samples/", "/demo/"];

async function getJson<T>(url: string): Promise<T | null> {
  try {
    const response = await fetch(url, { cache: "no-cache" });
    if (!response.ok) return null;
    const type = response.headers.get("content-type") ?? "";
    if (!type.includes("json")) return null;
    return (await response.json()) as T;
  } catch {
    return null;
  }
}

let cache: Promise<Sample[]> | null = null;

/** Published samples first; the bundled synthetic demo is used only when none exist yet. */
export function loadSamples(): Promise<Sample[]> {
  cache ??= (async () => {
    for (const base of SOURCES) {
      const list = await getJson<Omit<Sample, "base">[]>(`${base}index.json`);
      if (list && list.length > 0) return list.map((s) => ({ ...s, base }));
    }
    return [];
  })();
  return cache;
}

export async function loadReport(sample: Sample): Promise<Report | null> {
  return getJson<Report>(`${sample.base}${sample.report}`);
}

export async function loadTruth(sample: Sample): Promise<TruthLabel[] | null> {
  if (!sample.truth) return null;
  return getJson<TruthLabel[]>(`${sample.base}${sample.truth}`);
}

export function assetUrl(sample: Sample, path: string | null): string | null {
  return path ? `${sample.base}${path}` : null;
}

export type Metrics = Record<string, unknown> & {
  leaderboard?: Array<Record<string, string | number>>;
};

export function loadMetrics(): Promise<Metrics | null> {
  return getJson<Metrics>("/samples/metrics.json");
}

const uploads = new Map<string, { report: Report; audioUrl: string | null; title: string }>();

export function rememberUpload(report: Report, audioUrl: string | null, title: string): string {
  const id = report.provenance.report_sha256.slice(0, 12);
  uploads.set(id, { report, audioUrl, title });
  return id;
}

export function recallUpload(id: string) {
  return uploads.get(id) ?? null;
}
