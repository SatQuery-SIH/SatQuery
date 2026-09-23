// Slim live stepper — five plain-word steps mapped 1:1 to the real SSE stage
// ids (plan/bind/tool/packet/narration → plan/inputs/tools/evidence/answer).
// States come straight from the LiveTrace reducer (live) or buildStages
// (replay). One click expands the full metro-route TraceView; it collapses
// back when a live run finishes.
import type { LiveTrace, StageState } from "../liveTrace";
import type { RunBundle } from "../types";
import { buildStages } from "./TraceView";
import { formatSecs } from "../format";

export interface StepperStep {
  key: string;
  label: string;
  state: StageState;
}

function liveSteps(trace: LiveTrace): StepperStep[] {
  const byKey = (k: string) => trace.stages.find((s) => s.key === k);
  const toolRows = trace.stages.filter((s) => s.kind === "tool");
  let toolsState: StageState;
  if (!toolRows.length) toolsState = byKey("tools")?.state ?? "pending";
  else if (toolRows.some((s) => s.state === "running")) toolsState = "running";
  else if (toolRows.some((s) => s.state === "failed")) toolsState = "failed";
  else if (toolRows.every((s) => s.state === "done")) toolsState = "done";
  else toolsState = "pending";
  return [
    { key: "plan", label: "plan", state: byKey("plan")?.state ?? "pending" },
    { key: "inputs", label: "inputs", state: byKey("bind")?.state ?? "pending" },
    { key: "tools", label: "tools", state: toolsState },
    { key: "evidence", label: "evidence", state: byKey("packet")?.state ?? "pending" },
    { key: "answer", label: "answer", state: byKey("narration")?.state ?? "pending" },
  ];
}

// fixed display order — buildStages emits ingest first, but the stepper reads
// plan → inputs → tools → evidence → answer like the live run does
const REPLAY_ORDER: [string, string][] = [
  ["plan", "plan"],
  ["ingest", "inputs"],
  ["tools", "tools"],
  ["packet", "evidence"],
  ["narrate", "answer"],
];

function replaySteps(bundle: RunBundle): StepperStep[] {
  const byId = new Map(buildStages(bundle).map((s) => [s.id, s]));
  return REPLAY_ORDER.map(([id, label]) => ({
    key: label,
    label,
    state: (byId.get(id)?.state ?? "skipped") as StageState,
  }));
}

export function Stepper({
  view,
  expanded,
  onToggle,
}: {
  view:
    | { kind: "live"; trace: LiveTrace; running: boolean }
    | { kind: "replay"; bundle: RunBundle };
  expanded: boolean;
  onToggle: () => void;
}) {
  const live = view.kind === "live";
  const running = live && view.running;
  const steps = live ? liveSteps(view.trace) : replaySteps(view.bundle);
  const totalS = live
    ? (view.trace.summary?.complete_s as number | undefined)
    : (view.bundle.trace?.complete_s ?? undefined);
  const failedLabel = live
    ? view.trace.stages.find((s) => s.state === "failed")?.label
    : null;
  const status = running
    ? "running…"
    : live && view.trace.outcome === "error"
      ? `failed at ${failedLabel ?? "unknown step"}`
      : typeof totalS === "number"
        ? `completed in ${formatSecs(totalS)}`
        : "finished";

  return (
    <div className="stepper" data-testid="stepper">
      <div className="stepper-row">
        {live ? (
          <span
            className={`live-badge${running ? " pulsing" : ""}${
              view.trace.outcome === "error" ? " failed" : ""
            }`}
          >
            ● LIVE
          </span>
        ) : (
          <span className="replay-badge">▶ replay of recorded run</span>
        )}
        <ol className="stepper-steps">
          {steps.map((s) => (
            <li key={s.key} className={`stepper-step st-${s.state}`}>
              <span className="stepper-dot" />
              <span className="stepper-label">{s.label}</span>
            </li>
          ))}
        </ol>
        <span className="stepper-status">{status}</span>
        <button className="btn-mini" onClick={onToggle} data-testid="trace-toggle">
          how it answered {expanded ? "▴" : "▾"}
        </button>
      </div>
      {running && (
        <div className="stepper-note" data-testid="warming">
          the vision-language models can take up to a minute on a cold start.
        </div>
      )}
    </div>
  );
}
