// SatQuery API contract types — mirror api/schemas.py (frozen).
// Field names/nullability are load-bearing; do not invent fields.

export type InputMode = "single" | "bi-temporal" | "optical+sar";
export type SeatProfile = "local" | "cloud";

export interface SeatState {
  url: string;
  model: string | null;
  up: boolean | null;
  gguf_sha256?: string | null;
}

export interface HealthResponse {
  ok: boolean;
  seats: Record<string, SeatState>;
}

export interface SeatsResponse {
  profile: SeatProfile;
  seats: Record<string, SeatState>;
}

export interface DetectedBlock {
  gsd_m: number | null;
  gsd_source: string;
  crs: string | null;
  crs_note: string | null;
  bands: number | null;
  dtype: string | null;
  calibrated: boolean | null;
  files?: Record<string, Record<string, unknown>>;
}

export interface UploadPreview {
  role: string;
  url: string;
}

export interface UploadResponse {
  upload_id: string;
  workdir: string | null;
  detected: DetectedBlock;
  warnings: string[];
  ingest_note?: string;
  previews?: UploadPreview[]; // additive; older APIs omit it
}

export interface QueryRequest {
  query: string;
  input_mode: InputMode;
  scene?: number | null;
  upload_id?: string | null;
  sensor_profile?: string | null;
  live?: boolean;
}

export interface ArtifactRef {
  type: string; // image | overlay | geo_export
  path: string;
  name: string;
  url: string;
}

// Evidence-packet claim (shape from demo/evidence_packet.py).
export interface Claim {
  id: string;
  predicate: string;
  value: unknown;
  region?: string | null;
  confidence?: { level?: string; basis?: string };
  // measured_index | heuristic_estimate | learned_estimate (demo/tools.py)
  evidence_class?: string;
  provenance?: { tool?: string; model?: string; seat?: string; sha256?: string; gguf_sha256?: string };
  [k: string]: unknown;
}

export interface RunBundle {
  run_id: string;
  answer: string | null;
  visible_answer: string | null;
  plan: { supported?: boolean; tools?: string[]; refusal?: string; [k: string]: unknown };
  tool_outputs: Record<string, unknown>;
  evidence_packet: { claims?: Claim[]; limitations?: string[]; [k: string]: unknown } | null;
  trace: {
    query?: string;
    input_mode?: string;
    scene?: number | null;
    live?: boolean;
    first_token_s?: number | null;
    complete_s?: number | null;
    gsd?: Record<string, unknown> | null;
    misreg_check?: unknown;
    misreg_shift_px?: number | null;
    misregistration_note?: string | null;
    geo_exports?: unknown[];
    vlm?: Record<string, unknown>;
    overlay_path?: string | null;
    ground_overlay_path?: string | null;
    agreement_map_path?: string | null;
    images?: unknown[];
    metrics_badge?: unknown;
    ingest_note?: string | null;
    input_source?: string;
    narration_check?: Record<string, unknown> | null;
    [k: string]: unknown;
  };
  report: {
    findings?: string;
    measurement?: string;
    confidence?: string;
    limitations?: string[];
    narration_check?: Record<string, unknown> | null;
    evidence_packet_error?: string | null;
    [k: string]: unknown;
  };
  artifacts: ArtifactRef[];
  [k: string]: unknown;
}

export interface RunListItem {
  run_id: string;
  query: string | null;
  input_mode: string | null;
  ts: string | null;
  supported: boolean | null;
}

export interface ApiError {
  error: string;
  detail: string;
}

// POST /query/stream stage event (see api/README.md SSE wire format).
// stage: plan | bind | tool | narration | packet | done
// status: start | done | fail | withheld
export interface StageEvent {
  stage: string;
  status: string;
  ts?: number;
  tool?: string;
  data?: Record<string, unknown>;
}

// --- withold convention (demo/evidence_packet.py) ---
export function isWithheldClaim(c: Claim): boolean {
  const lvl = c.confidence?.level ?? "";
  const v = String(c.value ?? "");
  return (
    lvl === "withheld" ||
    v.startsWith("withheld") ||
    v === "not_determined" ||
    v.startsWith("inconclusive")
  );
}
