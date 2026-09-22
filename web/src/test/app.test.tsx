
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import App from "../App";
import { DetectedCard } from "../components/UploadPanel";
import { Results } from "../components/Results";
import { buildStages } from "../components/TraceView";
import { isWithheldClaim } from "../types";
import { fakeSarUpload, fakeUpload, makeBundle, refusedBundle } from "./fixtures";

// --- fetch stub: API is never contacted in tests ---
const health = {
  ok: true,
  seats: {
    narrator: { url: "http://127.0.0.1:8080", model: "m", up: true },
    canonical: { url: "http://127.0.0.1:8091", model: "m2", up: false },
  },
};

beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      if (String(url).endsWith("/health"))
        return new Response(JSON.stringify(health), { status: 200 });
      if (String(url).endsWith("/seats"))
        return new Response(
          JSON.stringify({ profile: "local", seats: health.seats }),
          { status: 200 },
        );
      if (String(url).endsWith("/runs?limit=50"))
        return new Response(JSON.stringify([]), { status: 200 });
      return new Response(JSON.stringify({ error: "x", detail: "y" }), { status: 404 });
    }),
  );
});

describe("App shell", () => {
  it("renders three input modes", async () => {
    render(<App />);
    expect(screen.getByText("single image")).toBeInTheDocument();
    expect(screen.getByText("bi-temporal pair")).toBeInTheDocument();
    expect(screen.getByText("optical + SAR")).toBeInTheDocument();
  });

  it("seat pills show canonical down + local profile", async () => {
    render(<App />);
    await waitFor(() =>
      expect(screen.getByText(/canonical: down/)).toBeInTheDocument(),
    );
    expect(screen.getByText(/narrator: up/)).toBeInTheDocument();
  });

  it("upload slots follow the selected mode", async () => {
    render(<App />);
    // default single: one slot
    expect(screen.getByText("image")).toBeInTheDocument();
    fireEvent.click(screen.getByText("optical + SAR"));
    expect(screen.getByText("optical")).toBeInTheDocument();
    expect(screen.getByText(/sar \(tif\/npz\)/)).toBeInTheDocument();
    fireEvent.click(screen.getByText("bi-temporal pair"));
    expect(screen.getByText("before")).toBeInTheDocument();
    expect(screen.getByText("after")).toBeInTheDocument();
  });
});

describe("DetectedCard (ingest instrumentation)", () => {
  it("renders gsd/crs/bands/dtype as a card", () => {
    render(<DetectedCard d={fakeUpload.detected} warnings={[]} />);
    expect(screen.getByTestId("detected-card")).toBeInTheDocument();
    expect(screen.getByText("10 m/px")).toBeInTheDocument();
    expect(screen.getByText("EPSG:32645")).toBeInTheDocument();
    expect(screen.getByText("uint16")).toBeInTheDocument();
  });

  it("shows SAR calibration + crs note", () => {
    render(
      <DetectedCard
        d={{ ...fakeSarUpload.detected, crs_note: "LOCAL_CS parsed" }}
        warnings={["x"]}
      />,
    );
    expect(screen.getByText("yes")).toBeInTheDocument();
    expect(screen.getByText("LOCAL_CS parsed")).toBeInTheDocument();
  });
});

describe("Results — evidence + withholding", () => {
  it("renders claims with provenance + withheld styling", () => {
    render(<Results bundle={makeBundle()} />);
    // withheld claim gets the badge + class
    const badge = document.querySelector(".withheld-badge");
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveTextContent("withheld");
    // provenance visible: seat + model sha
    expect(screen.getByText(/seat=127\.0\.0\.1:8091/)).toBeInTheDocument();
    expect(screen.getByText(/a8797686/)).toBeInTheDocument();
    // claim counts surfaced
    expect(screen.getByText(/3 claims · 1 withheld/)).toBeInTheDocument();
  });

  it("refused runs render distinctly", () => {
    render(<Results bundle={refusedBundle} />);
    expect(screen.getByText(/bi-temporal pair/)).toBeInTheDocument();
    expect(screen.getByText(/unsupported/)).toBeInTheDocument();
  });

  it("trace stages built from the bundle with withheld marker", () => {
    const stages = buildStages(makeBundle());
    const ids = stages.map((s) => s.id);
    expect(ids).toEqual(["ingest", "plan", "tools", "packet", "narrate"]);
    // packet has a withheld claim inside but not all → still "done"
    expect(stages.find((s) => s.id === "packet")?.state).toBe("done");
    const rs = buildStages(refusedBundle);
    expect(rs.find((s) => s.id === "plan")?.state).toBe("withheld");
  });
});

describe("warming + withheld helpers", () => {
  it("isWithheldClaim catches the conventions", () => {
    expect(
      isWithheldClaim({ id: "x", predicate: "p", value: "withheld_no_tool" }),
    ).toBe(true);
    expect(
      isWithheldClaim({ id: "x", predicate: "p", value: "yes" }),
    ).toBe(false);
    expect(
      isWithheldClaim({
        id: "x",
        predicate: "p",
        value: "v",
        confidence: { level: "withheld" },
      }),
    ).toBe(true);
  });
});
