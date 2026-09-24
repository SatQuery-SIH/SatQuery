import { useEffect, useRef, useState } from "react";
import {
  api,
  ApiHttpError,
  queryStream,
  StreamPipelineError,
  StreamTransportError,
} from "./api";
import type {
  InputMode,
  QueryRequest,
  RunBundle,
  RunListItem,
  UploadResponse,
} from "./types";
import { SeatBar } from "./components/SeatBar";
import { PresetPicker } from "./components/Presets";
import { DetectedCard, UploadPanel } from "./components/UploadPanel";
import { Results } from "./components/Results";
import { RunsPanel } from "./components/RunsPanel";
import { TraceView } from "./components/TraceView";
import { Stepper } from "./components/Stepper";
import { ImageryStage } from "./components/ImageryStage";
import { Collapsible } from "./components/Collapsible";
import {
  applyStageEvent,
  baseName,
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

// API role order per mode (mirrors api/README.md).
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
  const writerFailed = /narrator|report writer/i.test(e.stage ?? "");
  return (
    <div className="run-error" data-testid="run-error">
      <div className="run-error-head">
        <strong>
          {rejected
            ? "request rejected"
            : `run failed${e.stage ? ` at ${e.stage}` : ""}`}
        </strong>
      </div>
      {writerFailed && (
        <div className="run-error-hint">
          the answer model may be offline or still loading — check the status
          dots at the top
        </div>
      )}
      <Collapsible className="run-error-details" summary="details">
        <div className="slug-chip">{e.slug}</div>
        <div className="run-error-detail">{e.detail}</div>
      </Collapsible>
    </div>
  );
}

export default function App() {
  const [mode, setMode] = useState<InputMode>("single");
  const [query, setQuery] = useState("");
  const [upload, setUpload] = useState<UploadResponse | null>(null);
  const [scene, setScene] = useState<number | null>(null);
  const [boundPreset, setBoundPreset] = useState<Preset | null>(null);
  const [inputNames, setInputNames] = useState<Record<string, string>>({});
  const [bundle, setBundle] = useState<RunBundle | null>(null);
  const [busy, setBusy] = useState(false);
  const [runError, setRunError] = useState<RunError | null>(null);
  const [presetPhase, setPresetPhase] = useState<{
    id: string;
    phase: "uploading" | "running";
  } | null>(null);
  const [runView, setRunView] = useState<RunView>({ kind: "idle" });
  const [traceOpen, setTraceOpen] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [runsErr, setRunsErr] = useState(false);
  // bumped whenever a run lands so the drawer reloads itself
  const [runsRefresh, setRunsRefresh] = useState(0);
  // session-only per-mode results — switching tabs restores that mode's
  // last finished run as an honest replay instead of losing it
  const [modeRuns, setModeRuns] = useState<
    Partial<Record<InputMode, RunBundle>>
  >({});
  const cacheBundle = (b: RunBundle, fallbackMode?: InputMode) => {
    const m = b.trace?.input_mode;
    const key: InputMode | undefined =
      m === "single" || m === "bi-temporal" || m === "optical+sar"
        ? m
        : fallbackMode;
    if (key) setModeRuns((r) => ({ ...r, [key]: b }));
  };

  const ctlRef = useRef<AbortController | null>(null);
  useEffect(() => () => ctlRef.current?.abort(), []);

  const loadRuns = () =>
    api.runs(50).then(setRuns).catch(() => setRunsErr(true));
  useEffect(() => {
    void loadRuns();
  }, [runsRefresh]);

  const running = runView.kind === "live" && runView.running;
  // a finished live run collapses the full trace back to the slim stepper
  const wasRunning = useRef(false);
  useEffect(() => {
    if (wasRunning.current && !running) setTraceOpen(false);
    wasRunning.current = running;
  }, [running]);

  // on run start bring the stepper into view — it sits just below the stage
  const stepperAnchor = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!running) return;
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    const el = stepperAnchor.current;
    if (typeof el?.scrollIntoView === "function")
      el.scrollIntoView({
        block: "nearest",
        behavior: reduce ? "auto" : "smooth",
      });
  }, [running]);

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
    setTraceOpen(false);
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
        note: `live updates unavailable (${reason}) — showing the recorded run`,
      });
      try {
        const b = await api.query(req, signal);
        if (signal.aborted) return;
        setBundle(b);
        cacheBundle(b, args.mode);
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
      cacheBundle(b, args.mode);
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
          detail: `connection lost after ${e.eventsSeen} events — the run may still finish server-side; check the runs drawer`,
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
        setInputNames(
          Object.fromEntries(
            (p.previews ?? []).map((x) => [x.role, baseName(x.url)]),
          ),
        );
        setPresetPhase({ id: p.id, phase: "running" });
        await runQuery({ text: p.query, mode: p.mode, scene: p.scene });
      } else {
        setPresetPhase({ id: p.id, phase: "uploading" });
        const files: Record<string, File> = {};
        for (const f of p.files ?? []) {
          const blob = await fetchPresetFile(f);
          files[f.role] = new File([blob], f.name, { type: mimeForName(f.name) });
        }
        setInputNames(
          Object.fromEntries((p.files ?? []).map((f) => [f.role, f.name])),
        );
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
    setTraceOpen(false);
    api
      .run(id)
      .then((b) => {
        setBundle(b);
        cacheBundle(b);
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
  const namesOrdered = ROLE_ORDER[mode]
    .map((r) => inputNames[r])
    .filter(Boolean) as string[];

  return (
    <div className="app">
      <SeatBar
        runsCount={runs.length}
        onOpenRuns={() => setDrawerOpen(true)}
      />
      <div className="page">
        <div className="controls">
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
                  setInputNames({});
                  setRunError(null);
                  setTraceOpen(false);
                  // restore that mode's last finished run as a replay; the
                  // bundle (not stale upload state) drives stage + answer
                  const cached = modeRuns[m.id];
                  if (cached) {
                    setBundle(cached);
                    setRunView({ kind: "replay" });
                  } else {
                    setBundle(null);
                    setRunView({ kind: "idle" });
                  }
                }}
              >
                {m.label}
              </button>
            ))}
          </div>

          <PresetPicker
            mode={mode}
            disabled={blocked}
            active={presetPhase}
            onRun={(p) => void runPreset(p)}
          />

          <UploadPanel
            key={mode}
            mode={mode}
            onUploaded={(u) => {
              setUpload(u);
              setScene(null);
              setBoundPreset(null);
              // new inputs invalidate the displayed run
              setBundle(null);
              setRunView({ kind: "idle" });
              setRunError(null);
              setTraceOpen(false);
            }}
            onFilesChange={setInputNames}
          />

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

        <ImageryStage
          mode={mode}
          bundle={bundle}
          upload={upload}
          boundPreset={boundPreset}
          inputNames={namesOrdered}
        />

        {upload && !bundle && (
          <Collapsible className="panel" summary="image details">
            <DetectedCard d={upload.detected} warnings={upload.warnings} />
          </Collapsible>
        )}

        {runView.kind === "live" && (
          <div ref={stepperAnchor}>
            <Stepper
              view={runView}
              expanded={traceOpen}
              onToggle={() => setTraceOpen((o) => !o)}
            />
            {traceOpen && (
              <TraceView
                mode="live"
                trace={runView.trace}
                running={runView.running}
                embedded
              />
            )}
          </div>
        )}
        {runView.kind === "replay" && bundle && (
          <div ref={stepperAnchor}>
            <Stepper
              view={{ kind: "replay", bundle }}
              expanded={traceOpen}
              onToggle={() => setTraceOpen((o) => !o)}
            />
            {runView.note && <div className="trace-note">{runView.note}</div>}
            {traceOpen && (
              <TraceView mode="replay" bundle={bundle} embedded />
            )}
          </div>
        )}
        {runView.kind === "replay" && !bundle && (
          <div className="panel pending-card">
            {runView.note && <div className="muted">{runView.note}</div>}
            <div className="muted">
              waiting for the result… steps will show as a replay of the
              recorded run
            </div>
          </div>
        )}
        {runError && <RunErrorCard e={runError} />}
        {bundle && <Results bundle={bundle} upload={upload} />}
      </div>

      <RunsPanel
        open={drawerOpen}
        runs={runs}
        error={runsErr}
        onClose={() => setDrawerOpen(false)}
        onLoad={onLoadRun}
        onRefresh={loadRuns}
        currentRunId={bundle?.run_id}
      />
    </div>
  );
}
