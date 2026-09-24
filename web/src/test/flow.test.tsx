import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import App from "../App";
import { makeBundle } from "./fixtures";
import supported from "./fixtures/sse_supported.txt?raw";
import seatDown from "./fixtures/sse_seat_down.txt?raw";

const health = {
  ok: true,
  seats: {
    narrator: { url: "http://127.0.0.1:8080", model: "m", up: true },
    canonical: { url: "http://127.0.0.1:8091", model: "m2", up: false },
  },
};

const uploadRes = {
  upload_id: "up_bt",
  workdir: "/w",
  detected: {
    gsd_m: 0.5,
    gsd_source: "geotransform",
    crs: null,
    crs_note: null,
    bands: 3,
    dtype: "uint8",
    calibrated: null,
  },
  warnings: [],
  previews: [
    { role: "before", url: "/uploads/up_bt/preview/before" },
    { role: "after", url: "/uploads/up_bt/preview/after" },
  ],
};

interface Call {
  url: string;
  method: string;
  body?: Record<string, unknown>;
}

function sseRes(text: string): Response {
  const bytes = new TextEncoder().encode(text);
  return new Response(
    new ReadableStream<Uint8Array>({
      start(c) {
        c.enqueue(bytes);
        c.close();
      },
    }),
    { status: 200, headers: { "content-type": "text/event-stream" } },
  );
}

const json = (v: unknown, status = 200) =>
  new Response(JSON.stringify(v), { status });

function installFetch(
  calls: Call[],
  o: {
    stream?: () => Response | Promise<Response>;
    query?: unknown;
    upload?: unknown;
    runs?: unknown[];
    run?: unknown;
  } = {},
) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: unknown, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method ?? "GET";
      calls.push({
        url,
        method,
        body:
          typeof init?.body === "string"
            ? (JSON.parse(init.body) as Record<string, unknown>)
            : undefined,
      });
      if (url.endsWith("/health")) return json(health);
      if (url.endsWith("/seats"))
        return json({ profile: "local", seats: health.seats });
      if (url.includes("/runs?")) return json(o.runs ?? []);
      if (url.includes("/runs/")) return json(o.run ?? makeBundle());
      if (url.endsWith("/upload")) return json(o.upload ?? uploadRes);
      if (url.endsWith("/query/stream"))
        return o.stream ? o.stream() : sseRes(supported);
      if (url.endsWith("/query")) return json(o.query ?? makeBundle());
      if (url.includes("/presets/"))
        return new Response(new Uint8Array([1, 2, 3, 4]), { status: 200 });
      return new Response("not stubbed", { status: 404 });
    }),
  );
}

let calls: Call[] = [];
beforeEach(() => {
  calls = [];
  installFetch(calls);
});
afterEach(() => vi.unstubAllGlobals());

const streamBodies = () =>
  calls
    .filter((c) => c.url.endsWith("/query/stream") && c.method === "POST")
    .map((c) => c.body!);

// Presets are load-only: a click binds inputs (real /upload for file
// presets); the run itself is a separate explicit click.
async function loadAndRun(presetId: string, query?: string) {
  fireEvent.click(screen.getByTestId(`preset-${presetId}`));
  await waitFor(() =>
    expect(screen.getByTestId("run-button")).not.toBeDisabled(),
  );
  if (query !== undefined)
    fireEvent.change(screen.getByTestId("query-input"), {
      target: { value: query },
    });
  fireEvent.click(screen.getByTestId("run-button"));
}

describe("App — live stream flow", () => {
  it("live success: ● LIVE trace, answer-card, no replay label", async () => {
    render(<App />);
    await loadAndRun("sundarbans-single", "highlight the water");
    await waitFor(() =>
      expect(screen.getByTestId("answer-card")).toBeInTheDocument(),
    );
    expect(screen.getByText("● LIVE")).toBeInTheDocument();
    expect(screen.queryByText(/replay of recorded run/)).toBeNull();
    expect(screen.queryByTestId("run-error")).toBeNull();
    // request went out with explicit args
    expect(streamBodies()[0]).toMatchObject({
      query: "highlight the water",
      input_mode: "single",
      live: true,
    });
  });

  it("stream transport failure → honest fallback to POST /query + replay label", async () => {
    installFetch(calls, {
      stream: () => {
        throw new TypeError("failed to fetch");
      },
    });
    render(<App />);
    await loadAndRun("sundarbans-single", "highlight the water");
    await waitFor(() =>
      expect(screen.getByTestId("answer-card")).toBeInTheDocument(),
    );
    expect(screen.getByText(/replay of recorded run/)).toBeInTheDocument();
    expect(
      screen.getByText(
        /live updates unavailable \(failed to fetch\) — showing the recorded run/,
      ),
    ).toBeInTheDocument();
    expect(calls.some((c) => c.url.endsWith("/query"))).toBe(true);
  });

  it("seat down: pipeline error card with the narrator hint, no answer-card", async () => {
    installFetch(calls, { stream: () => sseRes(seatDown) });
    render(<App />);
    await loadAndRun("sundarbans-single", "highlight the water");
    const card = await screen.findByTestId("run-error");
    // raw slug + detail live inside the collapsed "details" section
    fireEvent.click(within(card).getByText("details"));
    expect(card).toHaveTextContent("pipeline_error");
    expect(card).toHaveTextContent("WinError 10061");
    expect(card).toHaveTextContent(/answer model may be offline/);
    expect(screen.queryByTestId("answer-card")).toBeNull();
    // the trace tells the truth: failed at the report writer, never "completed"
    fireEvent.click(screen.getByTestId("trace-toggle"));
    expect(
      screen.getByText(/live run · failed at report writer/),
    ).toBeInTheDocument();
    expect(screen.queryByText(/completed in/)).toBeNull();
  });

  it("422 mode_mismatch → request rejected, POST /query never called", async () => {
    installFetch(calls, {
      stream: () =>
        json({ error: "mode_mismatch", detail: "upload is single" }, 422),
    });
    render(<App />);
    await loadAndRun("sundarbans-single", "q");
    const card = await screen.findByTestId("run-error");
    expect(card).toHaveTextContent("request rejected");
    fireEvent.click(within(card).getByText("details"));
    expect(card).toHaveTextContent("mode_mismatch");
    expect(calls.some((c) => c.url.endsWith("/query"))).toBe(false);
  });
});

describe("App — preset runs pass explicit args (stale-closure regression)", () => {
  it("levir-pair sends bi-temporal + upload_id; scene3 sends optical+sar + scene 3", async () => {
    render(<App />);
    // bi-temporal tab → levir-pair upload preset loads, then an explicit run
    fireEvent.click(screen.getByTestId("mode-tab-bi-temporal"));
    await loadAndRun("levir-pair");
    await waitFor(() => expect(streamBodies().length).toBe(1));
    expect(streamBodies()[0]).toMatchObject({
      query: "what changed between the two images",
      input_mode: "bi-temporal",
      upload_id: "up_bt",
    });
    expect(streamBodies()[0].scene).toBeUndefined();
    expect(streamBodies()[0].live).toBe(true);

    // wait for the first run to settle, then switch tabs
    await waitFor(() =>
      expect(screen.getByTestId("answer-card")).toBeInTheDocument(),
    );
    fireEvent.click(screen.getByTestId("mode-tab-optical+sar"));
    await loadAndRun("scene3");
    await waitFor(() => expect(streamBodies().length).toBe(2));
    expect(streamBodies()[1]).toMatchObject({
      query: "Is there water in this scene?",
      input_mode: "optical+sar",
      scene: 3,
    });
    expect(streamBodies()[1].upload_id).toBeUndefined();
  });

  it("a preset click loads inputs + query but never runs on its own", async () => {
    render(<App />);
    fireEvent.click(screen.getByTestId("preset-sundarbans-single"));
    // the real upload goes out (binding), but no query is fired
    await waitFor(() =>
      expect(calls.some((c) => c.url.endsWith("/upload"))).toBe(true),
    );
    await waitFor(() =>
      expect(screen.getByTestId("run-button")).not.toBeDisabled(),
    );
    expect(streamBodies()).toHaveLength(0);
    expect(calls.some((c) => c.url.endsWith("/query"))).toBe(false);
    // the preset's query is in the box, editable before running
    expect(screen.getByTestId("query-input")).toHaveValue(
      "is there a large water body in this image",
    );
  });

  it("run is disabled with no bound inputs — never falls back to a default scene", async () => {
    render(<App />);
    const run = screen.getByTestId("run-button");
    expect(run).toBeDisabled();
    expect(screen.getByTestId("input-hint")).toBeInTheDocument();
    // typing a question alone is not enough — nothing may run unbound
    fireEvent.change(screen.getByTestId("query-input"), {
      target: { value: "is there water?" },
    });
    expect(run).toBeDisabled();
    fireEvent.keyDown(screen.getByTestId("query-input"), { key: "Enter" });
    expect(streamBodies()).toHaveLength(0);
    expect(calls.some((c) => c.url.endsWith("/query"))).toBe(false);
    // binding a preset enables it
    fireEvent.click(screen.getByTestId("preset-sundarbans-single"));
    await waitFor(() => expect(run).not.toBeDisabled());
  });

  it("editing the preset query still runs against the preset's upload", async () => {
    render(<App />);
    await loadAndRun("sundarbans-single", "how much water covers the scene");
    await waitFor(() => expect(streamBodies().length).toBe(1));
    expect(streamBodies()[0]).toMatchObject({
      query: "how much water covers the scene",
      input_mode: "single",
      upload_id: "up_bt",
    });
    expect(streamBodies()[0].scene).toBeUndefined();
  });

  it("a file dropped in the slot auto-binds — run uses that upload", async () => {
    render(<App />);
    const input = screen
      .getByTestId("slot-image")
      .querySelector("input[type=file]")!;
    fireEvent.change(input, {
      target: {
        files: [new File(["x"], "mine.tif", { type: "image/tiff" })],
      },
    });
    fireEvent.change(screen.getByTestId("query-input"), {
      target: { value: "what do you see" },
    });
    // no "upload & inspect" click — the pick itself binds the input
    await waitFor(() =>
      expect(screen.getByTestId("run-button")).not.toBeDisabled(),
    );
    fireEvent.click(screen.getByTestId("run-button"));
    await waitFor(() => expect(streamBodies().length).toBe(1));
    expect(streamBodies()[0]).toMatchObject({
      query: "what do you see",
      input_mode: "single",
      upload_id: "up_bt",
    });
    expect(streamBodies()[0].scene).toBeUndefined();
  });

  it("tab switch changes which preset cards are visible", () => {
    render(<App />);
    expect(screen.getByTestId("preset-sundarbans-single")).toBeInTheDocument();
    expect(screen.queryByTestId("preset-levir-pair")).toBeNull();
    fireEvent.click(screen.getByTestId("mode-tab-bi-temporal"));
    expect(screen.getByTestId("preset-levir-pair")).toBeInTheDocument();
    expect(screen.getByTestId("preset-scene2")).toBeInTheDocument();
    expect(screen.queryByTestId("preset-sundarbans-single")).toBeNull();
    fireEvent.click(screen.getByTestId("mode-tab-optical+sar"));
    expect(screen.getByTestId("preset-sundarbans")).toBeInTheDocument();
    expect(screen.getByTestId("preset-scene3")).toBeInTheDocument();
  });
});

describe("App — runs history", () => {
  it("loading a stored run shows the replay label", async () => {
    installFetch(calls, {
      runs: [
        {
          run_id: "r1",
          query: "Is there water in this scene?",
          input_mode: "optical+sar",
          ts: "2026-09-22T13:34:02Z",
          supported: true,
        },
      ],
    });
    render(<App />);
    fireEvent.click(screen.getByTestId("runs-open"));
    const drawer = await screen.findByTestId("runs-drawer");
    fireEvent.click(
      within(drawer).getByText(/Is there water in this scene\?/),
    );
    await waitFor(() =>
      expect(screen.queryByTestId("runs-drawer")).toBeNull(),
    );
    await waitFor(() =>
      expect(screen.getByText(/replay of recorded run/)).toBeInTheDocument(),
    );
    expect(screen.getByTestId("answer-card")).toBeInTheDocument();
  });
});
