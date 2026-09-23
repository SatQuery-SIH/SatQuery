import { useEffect, useMemo, useRef, useState } from "react";
import type { RunBundle } from "../types";
import { isWithheldClaim } from "../types";
import { formatSecs } from "../format";
import type { LiveTrace } from "../liveTrace";
import { baseName, humanTool, stageDetail, stageDuration } from "../liveTrace";
import { humanPredicate, humanValue } from "../verified";

// The agentic-trace stage view, in two honest modes:
//  - live: driven by POST /query/stream stage events as they arrive
//  - replay: replays a completed run bundle, labeled as replay — never
//    fake "live" on a finished bundle.

type ReplayStageState = "pending" | "running" | "done" | "failed" | "withheld";

interface Stage {
  id: string;
  title: string;
  state: ReplayStageState;
  detail: React.ReactNode;
}

function claimText(c: { predicate?: string; value?: unknown; confidence?: { level?: string } }) {
  const s = humanValue(c.value);
  return `${humanPredicate(String(c.predicate))}: ${s}${s.length > 90 ? "…" : ""}`;
}

export function buildStages(bundle: RunBundle): Stage[] {
  const t = bundle.trace ?? {};
  const plan = bundle.plan ?? {};
  const tools = Object.keys(bundle.tool_outputs ?? {});
  const claims = bundle.evidence_packet?.claims ?? [];
  const refused = plan.supported === false;
  const stages: Stage[] = [];

  stages.push({
    id: "ingest",
    title: "your images",
    state: t.input_source ? "done" : "pending",
    detail: (
      <div className="stage-detail">
        source: {String(t.input_source ?? "—")} · gsd:{" "}
        {t.gsd ? `${(t.gsd as { gsd_m?: number }).gsd_m ?? "?"} m` : "—"}
        {t.misregistration_note ? ` · ${t.misregistration_note}` : ""}
      </div>
    ),
  });

  stages.push({
    id: "plan",
    title: "plan",
    state: refused ? "withheld" : "done",
    detail: (
      <div className="stage-detail">
        {refused
          ? `unsupported — ${plan.refusal ?? "refused"}`
          : `planned: ${(plan.tools ?? []).map((k) => humanTool(String(k))).join(" → ") || "—"}`}
      </div>
    ),
  });

  if (!refused) {
    stages.push({
      id: "tools",
      title: `tools (${tools.length})`,
      state: tools.length ? "done" : "pending",
      detail: (
        <div className="stage-detail">
          {tools.map((k) => (
            <span className="tool-chip" key={k} title={k}>
              {humanTool(k)}
            </span>
          ))}
        </div>
      ),
    });

    const nWithheld = claims.filter(isWithheldClaim).length;
    stages.push({
      id: "packet",
      title: `evidence (${claims.length} claims)`,
      state: nWithheld === claims.length && claims.length ? "withheld" : "done",
      detail: (
        <div className="stage-detail">
          {claims.slice(0, 6).map((c) => (
            <div
              key={c.id}
              className={`trace-claim ${isWithheldClaim(c) ? "withheld" : ""}`}
            >
              {claimText(c)}
            </div>
          ))}
          {claims.length > 6 && <div>…+{claims.length - 6} more</div>}
        </div>
      ),
    });

    stages.push({
      id: "narrate",
      title: "report writer",
      state: bundle.answer || bundle.visible_answer ? "done" : "pending",
      detail: (
        <div className="stage-detail">
          {(t.vlm as Record<string, unknown> | undefined)?.model
            ? `model: ${baseName(String((t.vlm as Record<string, unknown>).model))}`
            : ""}
          {t.first_token_s != null ? ` · first token ${t.first_token_s}s` : ""}
          {t.narration_check
            ? ` · report check ${(t.narration_check as { ok?: boolean }).ok ? "passed" : "flagged"}`
            : ""}
        </div>
      ),
    });
  }

  return stages;
}

function badgeText(state: string): string {
  return state === "skipped" ? "not run" : state;
}

function LiveTraceView({
  trace,
  running,
  embedded = false,
}: {
  trace: LiveTrace;
  running: boolean;
  embedded?: boolean;
}) {
  const [manual, setManual] = useState<boolean | null>(null);
  const anyFailed = trace.stages.some((s) => s.state === "failed");
  const autoExpanded = running || anyFailed;
  // embedded under the stepper: always the full stage list — expand/collapse
  // is owned by the stepper's toggle, so the badge/details button drop out
  const expanded = embedded ? true : (manual ?? autoExpanded);
  const failed = trace.stages.find((s) => s.state === "failed");
  const completeS = trace.summary?.complete_s;
  const headline = running
    ? "running…"
    : trace.outcome === "error"
      ? `live run · failed at ${failed?.label ?? "unknown stage"}`
      : `live run · completed in ${typeof completeS === "number" ? completeS : "?"} s${
          failed ? ` · ${failed.label} failed` : ""
        }`;

  return (
    <div className="trace-view" data-testid="trace-view">
      <div className="trace-head">
        <span className="trace-title">how it answered</span>
        {!embedded && (
          <span
            className={`live-badge${running ? " pulsing" : ""}${
              trace.outcome === "error" ? " failed" : ""
            }`}
          >
            ● LIVE
          </span>
        )}
        <span className="trace-note">{headline}</span>
        {!embedded && !autoExpanded && (
          <button className="btn-mini" onClick={() => setManual(!(manual ?? autoExpanded))}>
            {expanded ? "details ▴" : "details ▾"}
          </button>
        )}
      </div>
      {running && !embedded && (
        <div className="warming" data-testid="warming">
          the vision-language models can take up to a minute on a cold start.
        </div>
      )}
      {expanded ? (
        <ol className="trace-stages">
          {trace.stages.map((s) => {
            const dur = stageDuration(s);
            const det = stageDetail(s);
            return (
              <li
                key={s.key}
                className={`trace-stage state-${s.state}${s.state === "running" ? " running" : ""}`}
              >
                <div className="stage-row">
                  <span className={`stage-badge st-${s.state}`}>{badgeText(s.state)}</span>
                  <span className="stage-title">{s.label}</span>
                  {dur != null && <span className="stage-dur">{formatSecs(dur)}</span>}
                </div>
                {det && <div className="stage-detail">{det}</div>}
              </li>
            );
          })}
        </ol>
      ) : (
        <div className="step-chips">
          {trace.stages.map((s) => {
            const dur = stageDuration(s);
            return (
              <span key={s.key} className={`step-chip st-${s.state}`}>
                {s.label}
                {dur != null ? ` · ${formatSecs(dur)}` : ""}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}

function ReplayTraceView({
  bundle,
  note,
  embedded = false,
}: {
  bundle: RunBundle;
  note?: string;
  embedded?: boolean;
}) {
  const stages = useMemo(() => buildStages(bundle), [bundle]);
  const [reveal, setReveal] = useState(stages.length);
  const [replaying, setReplaying] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const timers = useRef<number[]>([]);

  useEffect(() => setReveal(stages.length), [bundle, stages.length]);
  useEffect(
    () => () => {
      for (const t of timers.current) clearTimeout(t);
    },
    [],
  );

  // staged reveal — labeled replay of a completed bundle (real data, real order)
  const replay = () => {
    setReplaying(true);
    setReveal(0);
    let i = 0;
    const step = () => {
      i++;
      setReveal(i);
      if (i < stages.length) timers.current.push(window.setTimeout(step, 550));
      else setReplaying(false);
    };
    timers.current.push(window.setTimeout(step, 550));
  };

  const showFull = embedded || expanded || replaying;
  // which stored run this is — the stage may still show unrelated inputs
  const t = bundle.trace ?? {};
  const q = String(t.query ?? "");
  const runCtx = `run ${bundle.run_id.slice(0, 8)} · ${String(
    t.input_mode ?? "?",
  )} · “${q.length > 60 ? `${q.slice(0, 60)}…` : q}”`;
  return (
    <div className="trace-view" data-testid="trace-view">
      <div className="trace-head">
        <span className="trace-title">how it answered</span>
        {!embedded && (
          <span className="replay-badge">▶ replay of recorded run</span>
        )}
        <span className="run-context">{runCtx}</span>
        <button className="btn-mini" onClick={replay} disabled={replaying}>
          {replaying ? "replaying…" : "▶ replay pipeline"}
        </button>
        {!embedded && (
          <button className="btn-mini" onClick={() => setExpanded((e) => !e)}>
            {expanded ? "details ▴" : "details ▾"}
          </button>
        )}
        {note && <span className="trace-note">{note}</span>}
      </div>
      {showFull ? (
        <ol className="trace-stages">
          {stages.slice(0, reveal).map((s) => (
            <li key={s.id} className={`trace-stage state-${s.state}`}>
              <div className="stage-row">
                <span className={`stage-badge st-${s.state}`}>{badgeText(s.state)}</span>
                <span className="stage-title">{s.title}</span>
              </div>
              {s.detail}
            </li>
          ))}
        </ol>
      ) : (
        <div className="step-chips">
          {stages.map((s) => (
            <span key={s.id} className={`step-chip st-${s.state}`}>{s.title}</span>
          ))}
        </div>
      )}
    </div>
  );
}

export function TraceView(
  props:
    | { mode: "live"; trace: LiveTrace; running: boolean; embedded?: boolean }
    | { mode: "replay"; bundle: RunBundle; note?: string; embedded?: boolean },
) {
  if (props.mode === "live")
    return (
      <LiveTraceView
        trace={props.trace}
        running={props.running}
        embedded={props.embedded}
      />
    );
  return (
    <ReplayTraceView
      bundle={props.bundle}
      note={props.note}
      embedded={props.embedded}
    />
  );
}
