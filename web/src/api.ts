// Thin API client — contract is api/README.md (frozen). Base URL is env-driven
// so the same build works against the local API or a deployed one.
import type {
  HealthResponse,
  InputMode,
  QueryRequest,
  RunBundle,
  RunListItem,
  SeatsResponse,
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

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
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
    throw new ApiHttpError(res.status, slug, detail);
  }
  return (await res.json()) as T;
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
  query: (q: QueryRequest) =>
    req<RunBundle>("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(q),
    }),
  runs: (limit = 200) => req<RunListItem[]>(`/runs?limit=${limit}`),
  run: (id: string) => req<RunBundle>(`/runs/${encodeURIComponent(id)}`),
  artifactUrl: (rel: string) => `${API_BASE}${rel}`,
};
