// Live stage model for POST /query/stream — a pure, immutable reducer over
// StageEvent frames. Tool rows are created by real `tool` events only (never
// pre-created from plan.tools: "vqa" emits narration events, not tool events).
import type { StageEvent } from "./types";

export type StageState =
  | "pending"
  | "running"
  | "done"
  | "failed"
  | "withheld"
  | "skipped";

export interface LiveStage {
  key: string;
  kind: "plan" | "bind" | "tools" | "tool" | "narration" | "packet";
  label: string;
  tool?: string;
  state: StageState;
  data?: Record<string, unknown>;
  tStart?: number;
  tEnd?: number;
}

export interface LiveTrace {
  stages: LiveStage[];
  finished: boolean;
  summary?: Record<string, unknown>;
  events: number;
}

// Skeleton in real execution order; the `tools` row is a placeholder replaced
// by per-tool rows as `tool` events arrive.
export function initialLiveTrace(): LiveTrace {
  return {
    stages: [
      { key: "plan", kind: "plan", label: "planner", state: "pending" },
      { key: "bind", kind: "bind", label: "input binding + ingest", state: "pending" },
      { key: "tools", kind: "tools", label: "tool execution", state: "pending" },
      { key: "narration", kind: "narration", label: "frozen narrator", state: "pending" },
      { key: "packet", kind: "packet", label: "evidence packet", state: "pending" },
    ],
    finished: false,
    events: 0,
  };
}

export function applyStageEvent(t: LiveTrace, ev: StageEvent): LiveTrace {
  const next: LiveTrace = {
    ...t,
    events: t.events + 1,
    stages: t.stages.map((s) => ({ ...s })),
  };
  const stages = next.stages;
  const ts = ev.ts;
  const merge = (s: LiveStage) => {
    if (ev.data) s.data = { ...(s.data ?? {}), ...ev.data };
  };
  const insertToolRow = (tool: string): LiveStage => {
    const ph = stages.findIndex((s) => s.kind === "tools");
    if (ph !== -1) stages.splice(ph, 1);
    let key = `tool:${tool}`;
    if (stages.some((s) => s.key === key)) key = `tool:${tool}:2`;
    const row: LiveStage = {
      key,
      kind: "tool",
      label: tool,
      tool,
      state: "running",
    };
    const at = stages.findIndex((s) => s.kind === "narration");
    stages.splice(at === -1 ? stages.length : at, 0, row);
    return row;
  };

  switch (ev.stage) {
    case "plan": {
      const s = stages.find((x) => x.key === "plan");
      if (!s) break;
      if (ev.status === "start") {
        s.state = "running";
        s.tStart = ts;
      } else if (ev.status === "done" || ev.status === "withheld" || ev.status === "fail") {
        s.state =
          ev.status === "done" ? "done" : ev.status === "fail" ? "failed" : "withheld";
        s.tEnd = ts;
        merge(s);
      }
      break;
    }
    case "bind": {
      const s = stages.find((x) => x.key === "bind");
      if (!s) break;
      if (ev.status === "start") {
        s.state = "running";
        s.tStart = ts;
      } else if (ev.status === "done" || ev.status === "fail") {
        s.state = ev.status === "done" ? "done" : "failed";
        s.tEnd = ts;
        merge(s);
      }
      break;
    }
    case "tool": {
      const tool = ev.tool;
      if (!tool) break;
      if (ev.status === "start") {
        const running = stages.find(
          (x) => x.kind === "tool" && x.tool === tool && x.state === "running",
        );
        const s = running ?? insertToolRow(tool);
        s.state = "running";
        s.tStart = ts;
      } else if (ev.status === "done" || ev.status === "fail") {
        let s = stages.find(
          (x) => x.kind === "tool" && x.tool === tool && x.state === "running",
        );
        if (!s) s = insertToolRow(tool);
        s.state = ev.status === "done" ? "done" : "failed";
        s.tEnd = ts;
        merge(s);
      }
      break;
    }
    case "narration": {
      const s = stages.find((x) => x.key === "narration");
      if (!s) break;
      if (ev.status === "start") {
        s.state = "running";
        s.tStart = ts;
        merge(s);
      } else if (ev.status === "done" || ev.status === "withheld" || ev.status === "fail") {
        s.state =
          ev.status === "done" ? "done" : ev.status === "fail" ? "failed" : "withheld";
        s.tEnd = ts;
        merge(s);
      }
      break;
    }
    case "packet": {
      const s = stages.find((x) => x.key === "packet");
      if (!s) break;
      if (ev.status === "start") {
        s.state = "running";
        s.tStart = ts;
      } else if (ev.status === "done") {
        s.state = ev.data?.ok === false ? "failed" : "done";
        s.tEnd = ts;
        merge(s);
      } else if (ev.status === "fail") {
        s.state = "failed";
        s.tEnd = ts;
        merge(s);
      }
      break;
    }
    case "done": {
      next.finished = true;
      next.summary = ev.data;
      for (const s of stages) if (s.state === "pending") s.state = "skipped";
      break;
    }
    default:
      break; // unknown stages: only events++
  }
  return next;
}

export function finalizeLiveTrace(
  t: LiveTrace,
  outcome: "done" | "error",
): LiveTrace {
  const stages = t.stages.map((s) => {
    const n = { ...s };
    if (outcome === "done") {
      if (n.state === "pending") n.state = "skipped";
    } else {
      if (n.state === "running") n.state = "failed";
      else if (n.state === "pending") n.state = "skipped";
    }
    return n;
  });
  return { ...t, stages, finished: true };
}

function scalarEntries(
  d: Record<string, unknown>,
  skip: string,
  n: number,
): string[] {
  const out: string[] = [];
  for (const [k, v] of Object.entries(d)) {
    if (k === skip) continue;
    if (v === null || ["string", "number", "boolean"].includes(typeof v)) {
      const sv = typeof v === "string" && v.length > 60 ? `${v.slice(0, 60)}…` : String(v);
      out.push(`${k}=${sv}`);
      if (out.length >= n) break;
    }
  }
  return out;
}

export function stageDetail(s: LiveStage): string {
  if (s.state === "skipped") return "not run";
  if (s.state === "pending") return "";
  const d = s.data ?? {};
  switch (s.kind) {
    case "plan":
      if (s.state === "withheld") return `refused — ${d.refusal ?? ""}`;
      if (s.state === "done") {
        if (d.supported === false) return `unsupported · task ${d.task ?? "?"}`;
        const tools = Array.isArray(d.tools) ? d.tools.join(" → ") : "";
        return `task ${d.task ?? "?"} · plan: ${tools}`;
      }
      if (s.state === "failed") return `failed — ${d.error ?? ""}`;
      return "";
    case "bind":
      if (s.state === "failed") return `bind failed — ${d.error ?? ""}`;
      if (s.state === "done") {
        let out = `${d.source ?? "?"}${d.scene != null ? ` · scene ${d.scene}` : ""} · gsd ${
          d.gsd_m != null ? `${d.gsd_m} m (${d.gsd_source})` : `not detected (${d.gsd_source})`
        }`;
        if (d.misreg_shift_px != null) out += ` · misreg ≈ ${d.misreg_shift_px} px`;
        if (d.coreg_shift_px != null) out += ` · coreg shift ${d.coreg_shift_px} px`;
        return out;
      }
      return "";
    case "tool":
      if (s.state === "failed") return `failed — ${d.error ?? ""}`;
      if (s.state === "done") {
        const rest = scalarEntries(d, "latency_s", 3);
        return `${d.latency_s ?? "?"}s${rest.length ? ` · ${rest.join(" · ")}` : ""}`;
      }
      return "";
    case "narration":
      if (s.state === "running")
        return `waiting for narrator · ${d.url ?? "?"} · ${d.n_images ?? "?"} image(s)`;
      if (s.state === "done")
        return `first token ${d.first_token_s ?? "?"}s · complete ${d.complete_s ?? "?"}s${
          d.model ? ` · ${d.model}` : ""
        }`;
      if (s.state === "withheld") return `withheld — ${d.reason ?? ""}`;
      if (s.state === "failed") return `failed — ${d.error ?? ""}`;
      return "";
    case "packet":
      if (s.state === "failed") return `packet error — ${d.error ?? ""}`;
      if (s.state === "done") return "assembled";
      return "";
    case "tools":
      return "";
  }
}

export function stageDuration(s: LiveStage): number | null {
  if (s.kind === "tool" && typeof s.data?.latency_s === "number")
    return s.data.latency_s;
  if (s.tStart != null && s.tEnd != null) return s.tEnd - s.tStart;
  return null;
}
