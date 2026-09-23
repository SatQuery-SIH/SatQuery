import { describe, expect, it } from "vitest";
import { mimeForName, PRESETS, presetsForMode } from "../presets";
import type { InputMode } from "../types";

const MODES: InputMode[] = ["single", "bi-temporal", "optical+sar"];

describe("presets data", () => {
  it("presetsForMode returns only that mode, and every mode is non-empty", () => {
    for (const m of MODES) {
      const ps = presetsForMode(m);
      expect(ps.length).toBeGreaterThan(0);
      expect(ps.every((p) => p.mode === m)).toBe(true);
    }
    expect(new Set(PRESETS.map((p) => p.id)).size).toBe(PRESETS.length);
  });

  it("bi-temporal has levir-pair (upload, PNG files) + scene2 (prepared)", () => {
    const ps = presetsForMode("bi-temporal");
    const levir = ps.find((p) => p.id === "levir-pair")!;
    expect(levir.kind).toBe("upload");
    expect(levir.files?.map((f) => f.role)).toEqual(["before", "after"]);
    expect(levir.files?.every((f) => f.url.endsWith(".png"))).toBe(true);
    const s2 = ps.find((p) => p.id === "scene2")!;
    expect(s2.kind).toBe("prepared");
    expect(s2.scene).toBe(2);
    expect(s2.previews?.length).toBe(2);
  });

  it("mimeForName maps by extension", () => {
    expect(mimeForName("a.tif")).toBe("image/tiff");
    expect(mimeForName("a.TIFF")).toBe("image/tiff");
    expect(mimeForName("a.png")).toBe("image/png");
    expect(mimeForName("a.jpg")).toBe("image/jpeg");
    expect(mimeForName("a.jpeg")).toBe("image/jpeg");
    expect(mimeForName("a.npz")).toBe("application/octet-stream");
    expect(mimeForName("a.bin")).toBe("");
  });
});
