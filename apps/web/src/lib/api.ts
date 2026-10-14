import type { Report } from "./report";

export const API_URL = (import.meta.env.VITE_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export type StageEvent = { stage: string; status: string; elapsed_ms?: number };

export type AnalyzeHandlers = {
  onStage: (event: StageEvent) => void;
  signal?: AbortSignal;
};

export class ApiError extends Error {}

/** POST the clip and read the server-sent events until the report arrives. */
export async function analyze(
  wav: Blob,
  transcript: string,
  baselineId: string | null,
  handlers: AnalyzeHandlers,
): Promise<Report> {
  const form = new FormData();
  form.append("audio", wav, "clip.wav");
  form.append("transcript", transcript);
  if (baselineId) form.append("baseline_id", baselineId);
  const response = await fetch(`${API_URL}/api/analyze`, {
    method: "POST",
    body: form,
    signal: handlers.signal,
    headers: { accept: "text/event-stream" },
  });
  if (response.status === 404) throw new ApiError("The analysis service is not deployed yet.");
  if (!response.ok || !response.body) {
    const detail = await response.text().catch(() => "");
    throw new ApiError(detail || `The analysis service answered ${response.status}.`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let split = buffer.indexOf("\n\n");
    while (split !== -1) {
      const block = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      const event = /^event: (.*)$/m.exec(block)?.[1]?.trim() ?? "message";
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (event === "stage") handlers.onStage(JSON.parse(data) as StageEvent);
      if (event === "report") return JSON.parse(data) as Report;
      if (event === "error") throw new ApiError((JSON.parse(data) as { detail: string }).detail);
      split = buffer.indexOf("\n\n");
    }
  }
  throw new ApiError("The analysis ended without a report.");
}

export type JudgeFeedback = {
  overall: string;
  dimensions: Record<string, string>;
  verified: boolean;
  provider: string;
};

export async function judges(report: Report, signal?: AbortSignal): Promise<JudgeFeedback> {
  const response = await fetch(`${API_URL}/api/agents/judges`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ report }),
    signal,
  });
  if (response.status === 404) throw new ApiError("The feedback service is not deployed yet.");
  if (!response.ok) throw new ApiError(`The feedback service answered ${response.status}.`);
  return (await response.json()) as JudgeFeedback;
}

export async function warmUp(): Promise<void> {
  try {
    await fetch(`${API_URL}/api/health`, { cache: "no-store" });
  } catch {
    // A cold or unreachable API only matters once the user submits.
  }
}
