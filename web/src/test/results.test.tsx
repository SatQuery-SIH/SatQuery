import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { Results } from "../components/Results";
import { makeBundle, refusedBundle } from "./fixtures";

beforeEach(() =>
  // useContentLength does a real fetch per geo_export tile — never hit network
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response("nope", { status: 404 })),
  ),
);
afterEach(() => vi.unstubAllGlobals());

describe("Results — answer card + refusal", () => {
  it("refused run renders the refused card (amber), not the red run-error", () => {
    const { container } = render(<Results bundle={refusedBundle} />);
    const card = container.querySelector(".answer-card.refused");
    expect(card).toBeInTheDocument();
    expect(screen.getByTestId("answer-card")).toHaveTextContent(
      "unsupported query — refused by the planner",
    );
    expect(container.querySelector(".run-error")).toBeNull();
    expect(screen.queryByTestId("run-error")).toBeNull();
  });

  it("answer card carries audit + duration chips when present", () => {
    render(<Results bundle={makeBundle()} />);
    const card = screen.getByTestId("answer-card");
    expect(card).toHaveTextContent("narration audit passed");
  });
});

describe("ArtifactsGrid — groups, lightbox, GeoTIFF hints", () => {
  it("groups by API type with fixed-order headers", () => {
    render(<Results bundle={makeBundle()} />);
    expect(screen.getByTestId("artifact-group-overlay")).toHaveTextContent(
      "overlays (1)",
    );
    expect(screen.getByTestId("artifact-group-geo_export")).toHaveTextContent(
      "GeoTIFF exports (1)",
    );
  });

  it("clicking an image tile opens the lightbox; Escape closes it", () => {
    render(<Results bundle={makeBundle()} />);
    fireEvent.click(screen.getByText("overlay_live.png"));
    const lb = screen.getByTestId("lightbox");
    expect(lb).toBeInTheDocument();
    expect(lb).toHaveTextContent("overlay_live.png · overlay");
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByTestId("lightbox")).toBeNull();
  });

  it("GeoTIFF tile shows a size hint from content-length", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response("", {
          status: 200,
          headers: { "content-length": "1258291" },
        }),
      ),
    );
    render(<Results bundle={makeBundle()} />);
    await waitFor(() => expect(screen.getByText("1.2 MB")).toBeInTheDocument());
    expect(screen.getByText(/GeoTIFF · download/)).toBeInTheDocument();
  });
});
