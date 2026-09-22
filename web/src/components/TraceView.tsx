import { useEffect, useMemo, useState } from "react";
import type { RunBundle } from "../types";
import { isWithheldClaim } from "../types";

// The agentic-trace stage view. Today it replays the completed run bundle
// (labeled as replay — never fake "live" on a finished bundle). When the
// backend ships POST /query/stream SSE, the same component can be driven by
// live events: the stage list + per-stage detail rendering are identical.

type StageState = "pending" | "running" | "done" | "failed" | "withheld";

interface Stage {
  id: string;
  title: string;
  state: StageState;
  detail: React.ReactNode;
}

function claimText(c: { predicate?: string; value?: unknown; confidence?: { level?: string } }) {
  const v = c.value;
  const s = typeof v === "string" ? v : JSON.stringify(v);
  return `${c.predicate}: ${s}${s.length > 90 ? "…" : ""}`;
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
    title: "input binding + ingest contract",
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
    title: "planner",
    state: refused ? "withheld" : "done",
    detail: (
      <div className="stage-detail">
        {refused
          ? `unsupported — ${plan.refusal ?? "refused"}`
          : `tools: ${(plan.tools ?? []).join(", ") || "—"}`}
      </div>
    ),
  });

  if (!refused) {
    stages.push({
      id: "tools",
      title: `tool execution (${tools.length})`,
      state: tools.length ? "done" : "pending",
      detail: (
        <div className="stage-detail">
          {tools.map((k) => (
            <span className="tool-chip" key={k}>{k}</span>
          ))}
        </div>
      ),
    });

    const nWithheld = claims.filter(isWithheldClaim).length;
    stages.push({
      id: "packet",
      title: `evidence packet (${claims.length} claims)`,
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
      title: "frozen narrator",
      state: bundle.answer || bundle.visible_answer ? "done" : "pending",
      detail: (
        <div className="stage-detail">
          {t.vlm ? `seat: ${(t.vlm as Record<string, unknown>).url ?? "narrator"}` : ""}
          {t.first_token_s != null ? ` · first token ${t.first_token_s}s` : ""}
          {t.narration_check
            ? ` · audit ${(t.narration_check as { ok?: boolean }).ok ? "passed" : "flagged"}`
            : ""}
        </div>
      ),
    });
  }

  return stages;
}

export function TraceView({ bundle }: { bundle: RunBundle }) {
  const stages = useMemo(() => buildStages(bundle), [bundle]);
  const [reveal, setReveal] = useState(stages.length);
  const [replaying, setReplaying] = useState(false);

  // staged reveal — labeled replay of a completed bundle (real data, real order)
  const replay = () => {
    setReplaying(true);
    setReveal(0);
    let i = 0;
    const step = () => {
      i++;
      setReveal(i);
      if (i < stages.length) setTimeout(step, 550);
      else setReplaying(false);
    };
    setTimeout(step, 550);
  };

  useEffect(() => setReveal(stages.length), [bundle, stages.length]);

  return (
    <div className="trace-view" data-testid="trace-view">
      <div className="trace-head">
        <span className="trace-title">execution trace</span>
        <button className="btn-mini" onClick={replay} disabled={replaying}>
          {replaying ? "replaying…" : "▶ replay pipeline"}
        </button>
        <span className="trace-note">replay of recorded run — live SSE stream lands with the backend spec</span>
      </div>
      <ol className="trace-stages">
        {stages.slice(0, reveal).map((s) => (
          <li key={s.id} className={`trace-stage state-${s.state}`}>
            <div className="stage-row">
              <span className={`stage-badge st-${s.state}`}>{s.state}</span>
              <span className="stage-title">{s.title}</span>
            </div>
            {s.detail}
          </li>
        ))}
      </ol>
    </div>
  );
}
