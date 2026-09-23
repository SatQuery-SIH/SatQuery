import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { UploadPanel } from "../components/UploadPanel";
import { InputsStrip } from "../components/InputsStrip";
import { API_BASE } from "../api";

let blobN = 0;
const createURL = vi.fn((_f: Blob) => `blob:mock-${++blobN}`);
const revokeURL = vi.fn();

beforeEach(() => {
  (URL as unknown as Record<string, unknown>).createObjectURL = createURL;
  (URL as unknown as Record<string, unknown>).revokeObjectURL = revokeURL;
  createURL.mockClear();
  revokeURL.mockClear();
  blobN = 0;
});

afterEach(() => vi.unstubAllGlobals());

function pickFile(role: string, file: File) {
  const slot = screen.getByTestId(`slot-${role}`);
  const input = slot.querySelector("input[type=file]")!;
  fireEvent.change(input, { target: { files: [file] } });
}

const previewOf = (role: string) =>
  screen.getByTestId(`slot-${role}`).querySelector(".slot-preview")!;

const uploadRes = {
  upload_id: "up1",
  workdir: "/w",
  detected: {
    gsd_m: 10,
    gsd_source: "geotransform",
    crs: null,
    crs_note: null,
    bands: 1,
    dtype: "uint8",
    calibrated: null,
  },
  warnings: [],
  previews: [{ role: "image", url: "/uploads/up1/preview/image" }],
};

describe("UploadPanel slot previews", () => {
  it("PNG pick → local blob thumbnail", async () => {
    render(<UploadPanel mode="single" onUploaded={() => {}} />);
    pickFile("image", new File(["x"], "a.png", { type: "image/png" }));
    await waitFor(() => {
      const img = previewOf("image").querySelector("img");
      expect(img?.src.startsWith("blob:")).toBe(true);
    });
    expect(previewOf("image").getAttribute("data-state")).toBe("local");
  });

  it("TIFF pick → pending-server (no local render)", () => {
    render(<UploadPanel mode="single" onUploaded={() => {}} />);
    pickFile("image", new File(["x"], "a.tif", { type: "image/tiff" }));
    expect(previewOf("image").getAttribute("data-state")).toBe("pending-server");
    expect(previewOf("image").textContent).toContain("preview rendered by the API");
    expect(previewOf("image").querySelector("img")).toBeNull();
  });

  it("upload with previews → server img at API_BASE + url", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) =>
        String(url).endsWith("/upload")
          ? new Response(JSON.stringify(uploadRes), { status: 200 })
          : new Response("x", { status: 404 }),
      ),
    );
    const onUploaded = vi.fn();
    render(<UploadPanel mode="single" onUploaded={onUploaded} />);
    pickFile("image", new File(["x"], "a.tif", { type: "image/tiff" }));
    fireEvent.click(screen.getByText("upload & inspect"));
    await waitFor(() =>
      expect(previewOf("image").getAttribute("data-state")).toBe("server"),
    );
    const img = previewOf("image").querySelector("img")!;
    expect(img.src).toBe(`${API_BASE}/uploads/up1/preview/image`);
    expect(onUploaded).toHaveBeenCalledWith(
      expect.objectContaining({ upload_id: "up1" }),
    );
  });

  it("upload without previews → no-preview", async () => {
    const res = { ...uploadRes };
    delete (res as Record<string, unknown>).previews;
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify(res), { status: 200 })),
    );
    render(<UploadPanel mode="single" onUploaded={() => {}} />);
    pickFile("image", new File(["x"], "a.tif", { type: "image/tiff" }));
    fireEvent.click(screen.getByText("upload & inspect"));
    await waitFor(() =>
      expect(previewOf("image").getAttribute("data-state")).toBe("no-preview"),
    );
  });

  it("replacing a PNG revokes the old object URL; unmount revokes too", async () => {
    const { unmount } = render(
      <UploadPanel mode="single" onUploaded={() => {}} />,
    );
    pickFile("image", new File(["x"], "a.png", { type: "image/png" }));
    await waitFor(() => expect(createURL).toHaveBeenCalledTimes(1));
    pickFile("image", new File(["y"], "b.png", { type: "image/png" }));
    await waitFor(() =>
      expect(revokeURL).toHaveBeenCalledWith("blob:mock-1"),
    );
    unmount();
    expect(revokeURL).toHaveBeenCalledWith("blob:mock-2");
  });

  it("img error → error state; resets when the file changes", async () => {
    render(<UploadPanel mode="single" onUploaded={() => {}} />);
    pickFile("image", new File(["x"], "a.png", { type: "image/png" }));
    await waitFor(() => previewOf("image").querySelector("img"));
    fireEvent.error(previewOf("image").querySelector("img")!);
    await waitFor(() =>
      expect(previewOf("image").getAttribute("data-state")).toBe("error"),
    );
    pickFile("image", new File(["y"], "b.png", { type: "image/png" }));
    await waitFor(() =>
      expect(previewOf("image").getAttribute("data-state")).toBe("local"),
    );
  });
});

describe("InputsStrip", () => {
  it("role labels + GSD chip for uploads", () => {
    render(
      <InputsStrip
        title="uploaded files · abc12345"
        source="upload"
        items={[{ role: "image", label: "image", src: "/x.png" }]}
        gsd={{ gsd_m: 10, gsd_source: "geotransform" }}
      />,
    );
    expect(screen.getByTestId("gsd-chip")).toHaveTextContent(
      "GSD 10 m/px · geotransform",
    );
    expect(screen.getByText("image")).toBeInTheDocument();
    expect(screen.getByText("uploaded files · abc12345")).toBeInTheDocument();
  });

  it("prepared scene: title, previews, no gsd chip", () => {
    render(
      <InputsStrip
        title="prepared scene 2 · built-in example"
        source="prepared"
        items={[
          { role: "before", label: "before", src: "/presets/scene2_before.png" },
          { role: "after", label: "after", src: "/presets/scene2_after.png" },
        ]}
        gsd={null}
      />,
    );
    expect(screen.getByText(/prepared scene 2/)).toBeInTheDocument();
    expect(screen.queryByTestId("gsd-chip")).toBeNull();
  });

  it("muted 'GSD not detected' for uploads without gsd", () => {
    render(
      <InputsStrip
        title="uploaded files · x"
        source="upload"
        items={[{ role: "image", label: "image", src: null }]}
        gsd={{ gsd_m: null, gsd_source: "none" }}
      />,
    );
    expect(screen.getByTestId("gsd-chip")).toHaveTextContent("GSD not detected");
    expect(screen.getByText("no preview")).toBeInTheDocument();
  });
});
