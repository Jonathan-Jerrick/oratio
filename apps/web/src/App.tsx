import { useEffect, useState } from "react";

const API_URL = (import.meta.env.VITE_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

type Health = { status: string; version: string; arch: string; image_digest: string };
type State =
  | { kind: "loading" }
  | { kind: "ok"; health: Health; elapsedMs: number }
  | { kind: "error"; message: string };

export function App() {
  const [state, setState] = useState<State>({ kind: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    const started = performance.now();
    fetch(`${API_URL}/api/health`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const health = (await response.json()) as Health;
        setState({ kind: "ok", health, elapsedMs: Math.round(performance.now() - started) });
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setState({ kind: "error", message: error instanceof Error ? error.message : String(error) });
      });
    return () => controller.abort();
  }, []);

  return (
    <main>
      <h1>Oratio</h1>
      <p>Deployment check. The real interface is not built yet.</p>
      <p>API: {API_URL}</p>
      {state.kind === "loading" && <p role="status">Contacting the API…</p>}
      {state.kind === "error" && <p role="alert">API unreachable: {state.message}</p>}
      {state.kind === "ok" && (
        <p role="status">
          API {state.health.status}, version {state.health.version}, {state.health.arch}, answered in{" "}
          {state.elapsedMs} ms.
        </p>
      )}
    </main>
  );
}
