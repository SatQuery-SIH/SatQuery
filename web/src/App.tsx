import { useEffect, useRef, useState } from "react";
import {
  api,
  ApiHttpError,
  queryStream,
  StreamPipelineError,
  StreamTransportError,
} from "./api";
import type { InputMode, QueryRequest, RunBundle, UploadResponse } from "./types";
import { SeatBar } from "./components/SeatBar";
import { PresetPicker } from "./components/Presets";
import { DetectedCard, UploadPanel } from "./components/UploadPanel";
import { InputsStrip } from "./components/InputsStrip";
import { Results } from "./components/Results";
import { RunsPanel } from "./components/RunsPanel";
import { TraceView } from "./components/TraceView";
import {
  applyStageEvent,
  finalizeLiveTrace,
  initialLiveTrace,
  type LiveTrace,
} from "./liveTrace";
import { fetchPresetFile, mimeForName, type Preset } from "./presets";
import "./app.css";

const MODES: { id: InputMode; label: string }[] = [
  { id: "single", label: "single image" },
  { id: "bi-temporal", label: "bi-temporal pair" },
  { id: "optical+sar", label: "optical + SAR" },
];

// API role order per mode (mirrors api/README.md) — for the InputsStrip only.
const ROLE_ORDER: Record<InputMode, string[]> = {
  single: ["image"],
  "bi-temporal": ["before", "after"],
  "optical+sar": ["optical", "sar"],
};

// Rejected-before-stream request errors — a rejected request is not a stream
// failure and is never retried or fallen back from.
const REQUEST_REJECTION = new Set([
  "unknown_upload",
  "upload_gone",
  "mode_mismatch",
  "validation_error",
]);

interface RunError {
  slug: string;
  detail: string;
  stage?: string;
}

type RunView =
  | { kind: "idle" }
  | { kind: "live"; trace: LiveTrace; running: boolean }
  | { kind: "replay"; note?: string };

function isAbort(e: unknown): boolean {
  return (
    typeof e === "object" &&
    e !== null &&
    (e as { name?: string }).name === "AbortError"
  );
}

function RunErrorCard({ e }: { e: RunError }) {
  const rejected = REQUEST_REJECTION.has(e.slug);
  const narratorFailed = (e.stage ?? "").toLowerCase().includes("narrator");
  return (
    <div className="run-error" data-testid="run-error">
      <div className="run-error-head">
        <strong>{rejected ? "request rejected" : "run failed"}</strong>
        <span className="slug-chip">{e.slug}</span>
      </div>
      {e.stage && <div className="muted">failed stage: {e.stage}</div>}
      <div className="run-error-detail">{e.detail}</div>
      {narratorFailed && (
        <div className="run-error-hint">
          the narrator seat may be down or still loading — check the seat pills
          above
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [mode, setMode] = useState<InputMode>("single");
  const [query, setQuery] = useState("");
  const [upload, setUpload] = useState<UploadResponse | null>(null);
  const [scene, setScene] = useState<number | null>(null);
  const [boundPreset, setBoundPreset] = useState<Preset | null>(null);
  const [bundle, setBundle] = useState<RunBundle | null>(null);
  const [busy, setBusy] = useState(false);
  const [runError, setRunError] = useState<RunError | null>(null);
  const [presetPhase, setPresetPhase] = useState<{
    id: string;
    phase: "uploading" | "running";
  } | null>(null);
  const [runView, setRunView] = useState<RunView>({ kind: "idle" });
  const [runsCollapsed, setRunsCollapsed] = useState(false);
  // bumped whenever a run lands so the runs panel reloads itself
  const [runsRefresh, setRunsRefresh] = useState(0);

  const ctlRef = useRef<AbortController | null>(null);
  useEffect(() => () => ctlRef.current?.abort(), []);

  // Explicit-args run: mode/uploadId/scene are parameters, never read from
  // closure state after an await (preset clicks used to run with stale mode).
  const runQuery = async (args: {
    text: string;
    mode: InputMode;
    uploadId?: string;
    scene?: number;
  }) => {
    const text = args.text.trim();
    if (!text) return;
    ctlRef.current?.abort();
    const ctl = new AbortController();
    ctlRef.current = ctl;
    const signal = ctl.signal;
    setBusy(true);
    setRunError(null);
    setBundle(null);
    // per-run trace state — a late frame from an aborted run must never
    // patch the next run's view
    let trace = initialLiveTrace();
    setRunView({ kind: "live", trace, running: true });

    const req: QueryRequest = {
      query: text,
      input_mode: args.mode,
      upload_id: args.uploadId,
      scene: args.uploadId ? undefined : args.scene,
      live: true,
    };

    const patchTrace = (fn: (t: LiveTrace) => LiveTrace) => {
      if (signal.aborted) return;
      trace = fn(trace);
      const snap = trace;
      setRunView((v) => (v.kind === "live" ? { ...v, trace: snap } : v));
    };
    const stopRunning = () =>
      setRunView((v) => (v.kind === "live" ? { ...v, running: false } : v));

    const fallbackToQuery = async (reason: string) => {
      setRunView({
        kind: "replay",
        note: `live stream unavailable (${reason}) — fell back to POST /query`,
      });
      try {
        const b = await api.query(req, signal);
        if (signal.aborted) return;
        setBundle(b);
        setRunsRefresh((k) => k + 1);
      } catch (e) {
        if (signal.aborted || isAbort(e)) return;
        if (e instanceof ApiHttpError)
          setRunError({ slug: e.slug, detail: e.message });
        else
          setRunError({
            slug: "query_failed",
            detail: e instanceof Error ? e.message : String(e),
          });
      }
    };

    try {
      const b = await queryStream(
        req,
        (ev) => patchTrace((t) => applyStageEvent(t, ev)),
        signal,
      );
      if (signal.aborted) return;
      patchTrace((t) => finalizeLiveTrace(t, "done"));
      stopRunning();
      setBundle(b);
      setRunsRefresh((k) => k + 1);
    } catch (e) {
      if (signal.aborted || isAbort(e)) {
        /* aborted — silent */
      } else if (e instanceof StreamPipelineError) {
        patchTrace((t) => finalizeLiveTrace(t, "error"));
        stopRunning();
        const failed = trace.stages.find((s) => s.state === "failed");
        setRunError({ slug: e.slug, detail: e.detail, stage: failed?.label });
      } else if (e instanceof ApiHttpError && REQUEST_REJECTION.has(e.slug)) {
        setRunView({ kind: "idle" });
        setRunError({ slug: e.slug, detail: e.message });
      } else if (e instanceof StreamTransportError && e.eventsSeen > 0) {
        patchTrace((t) => finalizeLiveTrace(t, "error"));
        stopRunning();
        setRunError({
          slug: "stream_interrupted",
          detail: `connection lost after ${e.eventsSeen} events — the run may still finish server-side; check the runs panel`,
        });
      } else if (e instanceof StreamTransportError || e instanceof ApiHttpError) {
        const reason =
          e instanceof ApiHttpError
            ? `HTTP ${e.status} (${e.slug})`
            : e.message || "connection failed";
        await fallbackToQuery(reason);
      } else {
        setRunView({ kind: "idle" });
        setRunError({
          slug: "run_failed",
          detail: e instanceof Error ? e.message : String(e),
        });
      }
    } finally {
      if (ctlRef.current === ctl) setBusy(false);
    }
  };

  const runPreset = async (p: Preset) => {
    if (busy || presetPhase) return;
    setMode(p.mode);
    setQuery(p.query);
    setRunError(null);
    try {
      if (p.kind === "prepared") {
        setUpload(null);
        setScene(p.scene ?? null);
        setBoundPreset(p);
        setPresetPhase({ id: p.id, phase: "running" });
        await runQuery({ text: p.query, mode: p.mode, scene: p.scene });
      } else {
        setPresetPhase({ id: p.id, phase: "uploading" });
        const files: Record<string, File> = {};
        for (const f of p.files ?? []) {
          const blob = await fetchPresetFile(f);
          files[f.role] = new File([blob], f.name, { type: mimeForName(f.name) });
        }
        const up = await api.upload(p.mode, files);
        setUpload(up);
        setScene(null);
        setBoundPreset(null);
        setPresetPhase({ id: p.id, phase: "running" });
        await runQuery({ text: p.query, mode: p.mode, uploadId: up.upload_id });
      }
    } catch (e) {
      if (e instanceof ApiHttpError)
        setRunError({ slug: e.slug, detail: e.message });
      else
        setRunError({
          slug: "preset_failed",
          detail: e instanceof Error ? e.message : String(e),
        });
    } finally {
      setPresetPhase(null);
    }
  };

  const onLoadRun = (id: string) => {
    ctlRef.current?.abort();
    setRunError(null);
    api
      .run(id)
      .then((b) => {
        setBundle(b);
        setRunView({ kind: "replay" });
      })
      .catch((e) =>
        setRunError({
          slug: e instanceof ApiHttpError ? e.slug : "load_failed",
          detail: e instanceof Error ? e.message : String(e),
        }),
      );
  };

  const blocked = busy || presetPhase != null;

  const inputsNode = (() => {
    if (upload) {
      const items = ROLE_ORDER[mode].map((role) => {
        const pv = upload.previews?.find((p) => p.role === role);
        return {
          role,
          label: role === "sar" ? "SAR" : role,
          src: pv ? api.previewUrl(pv.url) : null,
        };
      });
      return (
        <InputsStrip
          title={`uploaded files · ${upload.upload_id.slice(0, 8)}`}
          source="upload"
          items={items}
          gsd={{
            gsd_m: upload.detected.gsd_m,
            gsd_source: upload.detected.gsd_source,
          }}
        />
      );
    }
    if (boundPreset && scene != null) {
      return (
        <InputsStrip
          title={`prepared scene ${scene} · built-in example`}
          source="prepared"
          items={(boundPreset.previews ?? []).map((p) => ({
            role: p.role,
            label: p.label,
            src: p.url,
          }))}
          gsd={null}
        />
      );
    }
    return (
      <div className="muted">
        no files bound — the API uses this mode's prepared scene
      </div>
    );
  })();

  return (
    <div className="app">
      <SeatBar />
      <div className="layout">
        <div className="col-inputs">
          <div className="section-label">1 · input mode</div>
          <div className="mode-tabs">
            {MODES.map((m) => (
              <button
                key={m.id}
                data-testid={`mode-tab-${m.id}`}
                className={mode === m.id ? "tab on" : "tab"}
                disabled={blocked}
                onClick={() => {
                  setMode(m.id);
                  setUpload(null);
                  setScene(null);
                  setBoundPreset(null);
                }}
              >
                {m.label}
              </button>
            ))}
          </div>

          <div className="section-label">2 · demo presets</div>
          <PresetPicker
            mode={mode}
            disabled={blocked}
            active={presetPhase}
            onRun={(p) => void runPreset(p)}
          />

          <div className="section-label">3 · or upload imagery</div>
          <UploadPanel
            key={mode}
            mode={mode}
            onUploaded={(u) => {
              setUpload(u);
              setScene(null);
              setBoundPreset(null);
            }}
          />

          <div className="section-label">bound inputs</div>
          {inputsNode}

          {upload && (
            <DetectedCard d={upload.detected} warnings={upload.warnings} />
          )}

          <div className="section-label">4 · ask</div>
          <div className="query-row">
            <input
              className="query-input"
              data-testid="query-input"
              value={query}
              placeholder="ask about the imagery…"
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) =>
                e.key === "Enter" &&
                void runQuery({
                  text: query,
                  mode,
                  uploadId: upload?.upload_id,
                  scene: upload ? undefined : (scene ?? undefined),
                })
              }
            />
            <button
              className="btn primary"
              data-testid="run-button"
              disabled={blocked || !query.trim()}
              onClick={() =>
                void runQuery({
                  text: query,
                  mode,
                  uploadId: upload?.upload_id,
                  scene: upload ? undefined : (scene ?? undefined),
                })
              }
            >
              {busy ? "running…" : "run"}
            </button>
          </div>
        </div>

        <div className="col-results">
          {runView.kind === "idle" && !bundle && !runError && (
            <div className="panel empty-state" data-testid="empty-state">
              <div className="empty-title">Results appear here.</div>
              <div className="muted">
                Pick a preset or upload imagery, then ask — the execution trace
                streams live from the pipeline.
              </div>
            </div>
          )}
          {runView.kind === "live" && (
            <TraceView mode="live" trace={runView.trace} running={runView.running} />
          )}
          {runView.kind === "replay" && bundle && (
            <TraceView mode="replay" bundle={bundle} note={runView.note} />
          )}
          {runView.kind === "replay" && !bundle && (
            <div className="panel pending-card">
              {runView.note && <div className="muted">{runView.note}</div>}
              <div className="muted">
                …waiting for POST /query; the trace will be a replay of the
                recorded run
              </div>
            </div>
          )}
          {runError && <RunErrorCard e={runError} />}
          {bundle && <Results bundle={bundle} />}
        </div>

        <aside>
          <RunsPanel
            onLoad={onLoadRun}
            collapsed={runsCollapsed}
            onToggle={() => setRunsCollapsed((c) => !c)}
            refreshKey={runsRefresh}
            currentRunId={bundle?.run_id}
          />
        </aside>
      </div>
    </div>
  );
}
