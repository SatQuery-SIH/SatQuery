import { describe, expect, it } from "vitest";
import { parseSseText } from "../sse";
import {
  applyStageEvent,
  finalizeLiveTrace,
  initialLiveTrace,
  stageDetail,
  stageDuration,
  type LiveTrace,
} from "../liveTrace";
import type { StageEvent } from "../types";
import refusal from "./fixtures/sse_refusal.txt?raw";
import supported from "./fixtures/sse_supported.txt?raw";
import seatDown from "./fixtures/sse_seat_down.txt?raw";

function traceOf(txt: string): LiveTrace {
  let t = initialLiveTrace();
  for (const f of parseSseText(txt)) {
    if (f.event === "stage")
      t = applyStageEvent(t, JSON.parse(f.data) as StageEvent);
  }
  return t;
}

const byKey = (t: LiveTrace, key: string) =>
  t.stages.find((s) => s.key === key)!;

describe("liveTrace — refusal fixture", () => {
  const t = traceOf(refusal);
  it("plan withheld with refusal; bind done gsd 0.5; tools+narration skipped; packet done", () => {
    const plan = byKey(t, "plan");
    expect(plan.state).toBe("withheld");
    expect(plan.data?.refusal).toBeTruthy();

    const bind = byKey(t, "bind");
    expect(bind.state).toBe("done");
    expect(bind.data?.gsd_m).toBe(0.5);

    expect(byKey(t, "tools").state).toBe("skipped");
    expect(byKey(t, "narration").state).toBe("skipped");
    expect(byKey(t, "packet").state).toBe("done");

    expect(t.finished).toBe(true);
    expect(t.summary?.complete_s).toBe(0.095);
    expect(t.events).toBe(8); // stage events only — the done bundle isn't a stage
  });
});

describe("liveTrace — supported fixture", () => {
  const t = traceOf(supported);
  it("key order matches real execution order", () => {
    expect(t.stages.map((s) => s.key)).toEqual([
      "plan",
      "bind",
      "tool:water_highlight",
      "tool:area_calc",
      "narration",
      "packet",
    ]);
  });
  it("water_highlight duration comes from latency_s; narration withheld", () => {
    expect(stageDuration(byKey(t, "tool:water_highlight"))).toBeCloseTo(0.143);
    expect(byKey(t, "narration").state).toBe("withheld");
    expect(byKey(t, "packet").state).toBe("done");
    expect(t.finished).toBe(true);
  });
});

describe("liveTrace — seat_down fixture + finalize(error)", () => {
  it("narration failed with the real error text; packet skipped; nothing running", () => {
    let t = traceOf(seatDown);
    t = finalizeLiveTrace(t, "error");
    const narr = byKey(t, "narration");
    expect(narr.state).toBe("failed");
    expect(String(narr.data?.error)).toContain("WinError 10061");
    expect(byKey(t, "packet").state).toBe("skipped");
    expect(t.stages.every((s) => s.state !== "running")).toBe(true);
    expect(t.finished).toBe(true);
  });
});

describe("liveTrace — incremental + details", () => {
  it("after only plan/start: plan running, everything else pending", () => {
    const t = applyStageEvent(initialLiveTrace(), {
      stage: "plan",
      status: "start",
      ts: 1,
    });
    expect(byKey(t, "plan").state).toBe("running");
    expect(t.stages.slice(1).every((s) => s.state === "pending")).toBe(true);
    expect(t.events).toBe(1);
  });

  it("stageDetail spot checks on the recorded supported run", () => {
    const t = traceOf(supported);
    expect(stageDetail(byKey(t, "plan"))).toBe(
      "task water_highlight · plan: water_highlight → area_calc → vqa",
    );
    expect(stageDetail(byKey(t, "bind"))).toBe(
      "prepared · scene 1 · gsd not detected (none)",
    );
    const wh = stageDetail(byKey(t, "tool:water_highlight"));
    expect(wh.startsWith("0.143s")).toBe(true);
    expect(wh).toContain("water_pixels=15077");
    expect(stageDetail(byKey(t, "narration"))).toBe(
      "withheld — live=false (cached mode)",
    );
  });

  it("stageDetail on refusal + pending/skipped", () => {
    const t = traceOf(refusal);
    expect(stageDetail(byKey(t, "plan"))).toContain("refused — ");
    expect(stageDetail(byKey(t, "tools"))).toBe("not run");
    expect(stageDetail(byKey(t, "bind"))).toContain("gsd 0.5 m (benchmark_constant)");
    const fresh = initialLiveTrace();
    expect(stageDetail(byKey(fresh, "bind"))).toBe("");
  });
});
