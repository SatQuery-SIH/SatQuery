// v2 layout/copy pass — answer composition, stepper wiring, imagery modes,
// de-jargon surface guard, runs drawer grouping.
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  within,
} from "@testing-library/react";
import App from "../App";
import { Results } from "../components/Results";
import { ImageryStage } from "../components/ImageryStage";
import { Stepper } from "../components/Stepper";
import type { RunBundle } from "../types";
import biFlaggedJson from "./fixtures/bundle_bitemporal_flagged.json";
import optSarJson from "./fixtures/bundle_optsar.json";
import singleAreaJson from "./fixtures/bundle_single_area.json";
import refusedJson from "./fixtures/bundle_refused.json";
import supported from "./fixtures/sse_supported.txt?raw";

const biFlagged = biFlaggedJson as unknown as RunBundle;
const optSar = optSarJson as unknown as RunBundle;
const singleArea = singleAreaJson as unknown as RunBundle;
const refusedFx = refusedJson as unknown as RunBundle;

// ---------------------------------------------------------------- App fetch

const health = {
  ok: true,
  seats: {
    narrator: {
      url: "http://127.0.0.1:8080",
      model: "gates/qwen3vl/Qwen3VL-8B-Instruct-Q4_K_M.gguf",
      up: true,
    },
    canonical: {
      url: "http://127.0.0.1:8091",
      model: "gates/qwen3vl/canonical/Qwen3VL-8B-RSVQA-LRFOLD-Q4_K_M.gguf",
      up: true,
    },
  },
};

const json = (v: unknown, status = 200) =>
  new Response(JSON.stringify(v), { status });

function sseRes(text: string): Response {
  return new Response(
    new ReadableStream<Uint8Array>({
      start(c) {
        c.enqueue(new TextEncoder().encode(text));
        c.close();
      },
    }),
    { status: 200, headers: { "content-type": "text/event-stream" } },
  );
}

function installFetch(
  o: {
    stream?: () => Response | Promise<Response>;
    runs?: unknown[];
    run?: unknown;
  } = {},
) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: unknown, init?: RequestInit) => {
      const url = String(input);
      if (init?.method === "POST" && url.endsWith("/seats"))
        return json({ profile: "local", seats: health.seats });
      if (url.endsWith("/health")) return json(health);
      if (url.endsWith("/seats"))
        return json({ profile: "local", seats: health.seats });
      if (url.includes("/runs?")) return json(o.runs ?? []);
      if (url.includes("/runs/")) return json(o.run ?? {});
      if (url.endsWith("/upload"))
        return json({
          upload_id: "up1",
          workdir: "/w",
          detected: {
            gsd_m: 10,
            gsd_source: "geotransform",
            crs: "EPSG:32645",
            crs_note: null,
            bands: 4,
            dtype: "uint16",
            calibrated: null,
          },
          warnings: [],
          previews: [{ role: "image", url: "/uploads/up1/preview/image" }],
        });
      if (url.endsWith("/query/stream"))
        return o.stream ? o.stream() : sseRes(supported);
      if (url.endsWith("/query")) return json({});
      if (url.includes("/presets/"))
        return new Response(new Uint8Array([1, 2, 3, 4]), { status: 200 });
      return new Response("not stubbed", { status: 404 });
    }),
  );
}

beforeEach(() => installFetch());
afterEach(() => vi.unstubAllGlobals());

// ------------------------------------------------------- answer composition

describe("answer card composition (real fixtures)", () => {
  it("bi-temporal flagged: loud measured + withheld lines, flag on the collapsed summary", () => {
    render(<Results bundle={biFlagged} />);
    const card = screen.getByTestId("answer-card");

    const measured = screen.getAllByTestId("fact-measured");
    expect(measured[0]).toHaveTextContent(
      "change detected across 25.8% of the image",
    );
    expect(screen.getAllByTestId("fact-withheld")).toHaveLength(2);
    expect(card).toHaveTextContent("report check: flagged");

    const det = card.querySelector(
      "details.interpretation",
    ) as HTMLDetailsElement;
    expect(det).toBeTruthy();
    expect(det.open).toBe(false);
    const summary = det.querySelector("summary")!;
    expect(summary).toHaveTextContent(
      "interpretation — model-written, unverified",
    );
    // the pipeline's own flag stays visible while collapsed
    expect(summary).toHaveTextContent(
      "UNVERIFIED INTERPRETATION — JSON card wins.",
    );
    // model prose is not rendered until the section is opened
    expect(screen.queryByText(/residential/)).toBeNull();
    fireEvent.click(summary);
    expect(screen.getByText(/residential/)).toBeInTheDocument();
    expect(screen.getByText(/invented number 30\.2/)).toBeInTheDocument();
  });

  it("withheld-only card leads with the honest headline", () => {
    const b = {
      run_id: "w1",
      answer: "",
      visible_answer: "",
      plan: { supported: true, tools: ["sar_agreement"] },
      tool_outputs: {},
      evidence_packet: {
        claims: [
          {
            id: "E1",
            predicate: "built_up_agreement",
            value: "withheld_no_tool",
            confidence: { level: "withheld", basis: "no optical built-up tool" },
            provenance: { tool: "sar_agreement" },
          },
        ],
        limitations: [],
      },
      trace: { query: "built up?", input_mode: "optical+sar" },
      report: {},
      artifacts: [],
    } as unknown as RunBundle;
    render(<Results bundle={b} />);
    expect(
      screen.getByText("Nothing here could be measured with confidence."),
    ).toBeInTheDocument();
    expect(screen.queryAllByTestId("fact-measured")).toHaveLength(0);
    expect(screen.getAllByTestId("fact-withheld")).toHaveLength(1);
  });

  it("refused fixture → distinct refused card, no facts block", () => {
    const { container } = render(<Results bundle={refusedFx} />);
    expect(container.querySelector(".answer-card.refused")).toBeInTheDocument();
    expect(
      screen.getByText(/can't answer this with the available tools/),
    ).toBeInTheDocument();
    expect(screen.queryAllByTestId("fact-measured")).toHaveLength(0);
    expect(screen.queryAllByTestId("fact-withheld")).toHaveLength(0);
  });
});

// ----------------------------------------------------------- imagery stage

describe("ImageryStage — mode-driven panels from real bundle artifacts", () => {
  const props = {
    upload: null,
    boundPreset: null,
    inputNames: [] as string[],
  };

  it("single: one role panel + water overlay layer toggle", () => {
    render(<ImageryStage mode="single" bundle={singleArea} {...props} />);
    expect(screen.getByTestId("stage-panel-image")).toBeInTheDocument();
    const toggle = screen.getByTestId("overlay-toggle");
    expect(toggle).toHaveTextContent("water overlay");
    // overlay defaults ON as a layer on the image panel
    expect(document.querySelector(".stage-overlay.on")).toBeInTheDocument();
    fireEvent.click(toggle);
    expect(document.querySelector(".stage-overlay.on")).toBeNull();
    expect(document.querySelector(".stage-overlay")).toBeInTheDocument();
  });

  it("bi-temporal: before|after both visible; overlay is a layer on after", () => {
    render(<ImageryStage mode="bi-temporal" bundle={biFlagged} {...props} />);
    const before = screen.getByTestId("stage-panel-before");
    const after = screen.getByTestId("stage-panel-after");
    expect(before).toBeInTheDocument();
    expect(after).toBeInTheDocument();
    const toggle = screen.getByTestId("overlay-toggle");
    expect(toggle).toHaveTextContent("change overlay");
    // overlay lives on the AFTER panel, off/on never removes the pair
    expect(after.querySelector(".stage-overlay.on")).toBeInTheDocument();
    fireEvent.click(toggle);
    expect(after.querySelector(".stage-overlay")).toBeInTheDocument();
    expect(after.querySelector(".stage-overlay.on")).toBeNull();
    expect(before).toBeInTheDocument();
    fireEvent.click(toggle);
    expect(after.querySelector(".stage-overlay.on")).toBeInTheDocument();
  });

  it("optical+sar: optical | SAR | agreement map + legend", () => {
    render(<ImageryStage mode="optical+sar" bundle={optSar} {...props} />);
    expect(screen.getByTestId("stage-panel-optical")).toBeInTheDocument();
    expect(screen.getByTestId("stage-panel-sar")).toBeInTheDocument();
    const agree = screen.getByTestId("stage-panel-agreement");
    expect(agree).toBeInTheDocument();
    for (const l of [
      "water in both",
      "optical only",
      "SAR only",
      "disagreement",
      "neither",
    ])
      expect(within(agree).getByText(l)).toBeInTheDocument();
  });
});

// ------------------------------------------------------------- stepper wiring

describe("slim stepper ↔ full trace", () => {
  it("live steps mirror SSE events; expand toggles the full trace; finish auto-collapses", async () => {
    // a stream we hold open mid-run
    let ctl!: ReadableStreamDefaultController<Uint8Array>;
    const stream = new ReadableStream<Uint8Array>({
      start(c) {
        ctl = c;
      },
    });
    installFetch({
      stream: () =>
        new Response(stream, {
          status: 200,
          headers: { "content-type": "text/event-stream" },
        }),
    });
    const enc = new TextEncoder();
    const frames = supported.split("\n\n").filter((f) => f.trim());
    // frames 0..4: plan done, inputs done, first tool running
    const push = (n: number, from: number) =>
      ctl.enqueue(enc.encode(frames.slice(from, n).join("\n\n") + "\n\n"));

    render(<App />);
    fireEvent.change(screen.getByTestId("query-input"), {
      target: { value: "is there water" },
    });
    fireEvent.click(screen.getByTestId("run-button"));
    push(5, 0);

    await waitFor(() =>
      expect(screen.getByTestId("stepper")).toBeInTheDocument(),
    );
    const labels = [...document.querySelectorAll(".stepper-label")].map(
      (e) => e.textContent,
    );
    expect(labels).toEqual(["plan", "inputs", "tools", "evidence", "answer"]);
    const states = [...document.querySelectorAll(".stepper-step")].map((e) =>
      e.className.replace("stepper-step ", ""),
    );
    expect(states).toEqual([
      "st-done",
      "st-done",
      "st-running",
      "st-pending",
      "st-pending",
    ]);
    // cold-start note while running
    expect(screen.getByTestId("warming")).toHaveTextContent(/cold start/);

    // expand → full trace; collapse → gone
    fireEvent.click(screen.getByTestId("trace-toggle"));
    expect(document.querySelector(".trace-stages")).toBeInTheDocument();
    fireEvent.click(screen.getByTestId("trace-toggle"));
    expect(document.querySelector(".trace-stages")).toBeNull();

    // expand again, then finish — the trace collapses itself
    fireEvent.click(screen.getByTestId("trace-toggle"));
    expect(document.querySelector(".trace-stages")).toBeInTheDocument();
    ctl.enqueue(enc.encode(frames.slice(5).join("\n\n") + "\n\n"));
    ctl.close();
    await waitFor(() =>
      expect(screen.getByTestId("answer-card")).toBeInTheDocument(),
    );
    // finish → the expanded trace collapses itself back to the slim stepper
    await waitFor(() =>
      expect(document.querySelector(".trace-stages")).toBeNull(),
    );
    expect(screen.getByTestId("stepper")).toBeInTheDocument();
  });
});

describe("replay stepper — fixed order", () => {
  const stepLabels = () =>
    [...document.querySelectorAll(".stepper-label")].map((e) => e.textContent);
  const stepStates = () =>
    [...document.querySelectorAll(".stepper-step")].map((e) =>
      e.className.replace("stepper-step ", ""),
    );

  it("optical+sar replay reads plan → inputs → tools → evidence → answer", () => {
    render(
      <Stepper
        view={{ kind: "replay", bundle: optSar }}
        expanded={false}
        onToggle={() => {}}
      />,
    );
    expect(stepLabels()).toEqual([
      "plan",
      "inputs",
      "tools",
      "evidence",
      "answer",
    ]);
    expect(stepStates()).toEqual([
      "st-done",
      "st-done",
      "st-done",
      "st-done",
      "st-done",
    ]);
  });

  it("refused replay: plan withheld, missing stages render skipped", () => {
    render(
      <Stepper
        view={{ kind: "replay", bundle: refusedFx }}
        expanded={false}
        onToggle={() => {}}
      />,
    );
    expect(stepLabels()).toEqual([
      "plan",
      "inputs",
      "tools",
      "evidence",
      "answer",
    ]);
    expect(stepStates()).toEqual([
      "st-withheld",
      "st-done",
      "st-skipped",
      "st-skipped",
      "st-skipped",
    ]);
  });
});

// -------------------------------------------------------- stale-run clearing

describe("stale-run clearing", () => {
  it("switching mode tabs drops the previous run's answer + stage", async () => {
    render(<App />);
    fireEvent.click(screen.getByTestId("preset-sundarbans-single"));
    await waitFor(() =>
      expect(screen.getByTestId("answer-card")).toBeInTheDocument(),
    );
    fireEvent.click(screen.getByTestId("mode-tab-bi-temporal"));
    expect(screen.queryByTestId("answer-card")).toBeNull();
    expect(screen.queryByTestId("stepper")).toBeNull();
    // stage is bare again — the placeholder for the new mode, not old imagery
    expect(screen.getByTestId("imagery-stage")).toHaveTextContent(
      /pick an example or upload imagery/,
    );
    expect(screen.getByTestId("preset-levir-pair")).toBeInTheDocument();
  });
});

// -------------------------------------------------------------- de-jargon

describe("de-jargon guard", () => {
  it("no pipeline jargon anywhere on the page — idle and after a finished run", async () => {
    render(<App />);
    fireEvent.click(screen.getByTestId("preset-sundarbans-single"));
    await waitFor(() =>
      expect(screen.getByTestId("answer-card")).toBeInTheDocument(),
    );
    const text = (document.body.textContent ?? "").toLowerCase();
    for (const w of [
      "narrator",
      "canonical",
      "post /query",
      "execution trace",
      "bound inputs",
      "ingest contract",
      "evidence packet",
    ])
      expect(text).not.toContain(w);
  });
});

// -------------------------------------------------------------- runs drawer

describe("runs drawer", () => {
  const runs = [
    {
      run_id: "r1",
      query: "q water?",
      input_mode: "single",
      ts: "2026-09-22T13:34:02Z",
      supported: true,
    },
    {
      run_id: "r2",
      query: "q water?",
      input_mode: "single",
      ts: "2026-09-22T13:30:00Z",
      supported: true,
    },
    {
      run_id: "r3",
      query: "how many buildings changed?",
      input_mode: "bi-temporal",
      ts: "2026-09-22T13:00:00Z",
      supported: false,
    },
  ];

  it("opens from the header, groups duplicates behind ×N, closes on load", async () => {
    installFetch({ runs, run: singleAreaJson });
    render(<App />);
    expect(screen.queryByTestId("runs-drawer")).toBeNull();
    fireEvent.click(screen.getByTestId("runs-open"));
    const drawer = await screen.findByTestId("runs-drawer");

    // two groups: "q water?" (×2) and the refused bi-temporal one
    const expander = within(drawer).getByText(/×2/);
    expect(within(drawer).getByText("q water?")).toBeInTheDocument();
    expect(
      within(drawer).getByText(/how many buildings changed\?/),
    ).toBeInTheDocument();
    // refused styling
    expect(
      within(drawer).getByText(/refused/).className,
    ).toContain("refused");

    // expander lists the older duplicate (time-only row)
    fireEvent.click(expander);
    expect(
      within(drawer).getByText(/13:30/),
    ).toBeInTheDocument();

    // loading a run closes the drawer and shows the replay badge
    fireEvent.click(within(drawer).getByText("q water?"));
    await waitFor(() =>
      expect(screen.queryByTestId("runs-drawer")).toBeNull(),
    );
    await waitFor(() =>
      expect(screen.getByText(/replay of recorded run/)).toBeInTheDocument(),
    );
  });
});
