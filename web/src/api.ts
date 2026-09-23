// Thin API client — contract is api/README.md (frozen). Base URL is env-driven
// so the same build works against the local API or a deployed one.
import { SseParser } from "./sse";
import type {
  HealthResponse,
  InputMode,
  QueryRequest,
  RunBundle,
  RunListItem,
  SeatsResponse,
  StageEvent,
  UploadResponse,
} from "./types";

export const API_BASE: string =
  (import.meta as { env?: Record<string, string> }).env?.VITE_API_BASE ??
  "http://127.0.0.1:8000";

export class ApiHttpError extends Error {
  status: number;
  slug: string;
  constructor(status: number, slug: string, detail: string) {
    super(detail);
    this.status = status;
    this.slug = slug;
  }
}

// The SSE connection itself failed — fetch rejected, missing body, reader
// error, or the stream ended without a done/error frame.
export class StreamTransportError extends Error {
  eventsSeen: number;
  constructor(eventsSeen: number, message?: string) {
    super(message ?? `stream transport failed after ${eventsSeen} events`);
    this.eventsSeen = eventsSeen;
  }
}

// The stream delivered an `error` SSE event — the run failed server-side.
export class StreamPipelineError extends Error {
  slug: string;
  detail: string;
  constructor(slug: string, detail: string) {
    super(detail);
    this.slug = slug;
    this.detail = detail;
  }
}

function isAbortError(e: unknown): boolean {
  return (
    typeof e === "object" &&
    e !== null &&
    (e as { name?: string }).name === "AbortError"
  );
}

async function errorFromResponse(res: Response): Promise<ApiHttpError> {
  let slug = `http_${res.status}`;
  let detail = res.statusText;
  try {
    const body = await res.json();
    if (body?.error) {
      slug = body.error;
      detail = body.detail ?? detail;
    }
  } catch {
    /* non-json error body */
  }
  return new ApiHttpError(res.status, slug, detail);
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) throw await errorFromResponse(res);
  return (await res.json()) as T;
}

// POST /query/stream — resolve with the `done` bundle, reject with
// StreamPipelineError on an `error` frame, StreamTransportError on transport
// failure. Stage frames are JSON.parsed and handed to onStage; malformed
// stage frames are skipped, unknown event names ignored.
export async function queryStream(
  q: QueryRequest,
  onStage: (e: StageEvent) => void,
  signal?: AbortSignal,
): Promise<RunBundle> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/query/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(q),
      signal,
    });
  } catch (e) {
    if (isAbortError(e) || signal?.aborted) throw e;
    throw new StreamTransportError(
      0,
      e instanceof Error ? e.message : String(e),
    );
  }
  if (!res.ok) throw await errorFromResponse(res);
  const reader = res.body?.getReader();
  if (!reader) throw new StreamTransportError(0, "response has no body");

  const decoder = new TextDecoder("utf-8");
  let eventsSeen = 0;
  return new Promise<RunBundle>((resolve, reject) => {
    const parser = new SseParser((frame) => {
      eventsSeen++;
      if (frame.event === "stage") {
        try {
          onStage(JSON.parse(frame.data) as StageEvent);
        } catch {
          /* malformed stage frame — skip, not fatal */
        }
      } else if (frame.event === "done") {
        try {
          const bundle = JSON.parse(frame.data) as RunBundle;
          void reader.cancel().catch(() => {});
          resolve(bundle);
        } catch {
          reject(new StreamTransportError(eventsSeen));
        }
      } else if (frame.event === "error") {
        try {
          const body = JSON.parse(frame.data) as {
            error?: string;
            detail?: string;
          };
          reject(
            new StreamPipelineError(
              body.error ?? "pipeline_error",
              body.detail ?? "",
            ),
          );
        } catch {
          reject(new StreamTransportError(eventsSeen));
        }
      }
      // unknown event names (e.g. a future `token`) are ignored
    });

    const pump = async () => {
      try {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          if (value) parser.push(decoder.decode(value, { stream: true }));
        }
        parser.push(decoder.decode());
        parser.end();
        reject(new StreamTransportError(eventsSeen));
      } catch (e) {
        if (isAbortError(e) || signal?.aborted) reject(e);
        else
          reject(
            new StreamTransportError(
              eventsSeen,
              e instanceof Error ? e.message : String(e),
            ),
          );
      }
    };
    void pump();
  });
}

export const api = {
  health: () => req<HealthResponse>("/health"),
  seats: () => req<SeatsResponse>("/seats"),
  setProfile: (profile: "local" | "cloud") =>
    req<SeatsResponse>("/seats", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile }),
    }),
  upload: (mode: InputMode, files: Record<string, File>) => {
    const fd = new FormData();
    fd.append("mode", mode);
    for (const [role, f] of Object.entries(files)) fd.append(role, f);
    return req<UploadResponse>("/upload", { method: "POST", body: fd });
  },
  query: (q: QueryRequest, signal?: AbortSignal) =>
    req<RunBundle>("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(q),
      signal,
    }),
  runs: (limit = 200) => req<RunListItem[]>(`/runs?limit=${limit}`),
  run: (id: string) => req<RunBundle>(`/runs/${encodeURIComponent(id)}`),
  artifactUrl: (rel: string) => `${API_BASE}${rel}`,
  previewUrl: (rel: string) => `${API_BASE}${rel}`,
};
