import { describe, expect, it, vi, afterEach } from "vitest";
import {
  fetchPresetFile,
  mimeForName,
  PRESETS,
  presetsForMode,
} from "../presets";
import type { InputMode } from "../types";

const MODES: InputMode[] = ["single", "bi-temporal", "optical+sar"];

afterEach(() => vi.unstubAllGlobals());

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

describe("fetchPresetFile — embedded fallback", () => {
  const sha256 = async (blob: Blob) => {
    const hash = await crypto.subtle.digest(
      "SHA-256",
      await blob.arrayBuffer(),
    );
    return [...new Uint8Array(hash)]
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
  };

  it("falls back to the embedded copy when the fetch is empty (204)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(null, { status: 204 })),
    );
    const f = PRESETS.find((p) => p.id === "sundarbans-single")!.files![0];
    const blob = await fetchPresetFile(f);
    expect(blob.size).toBe(525066);
    expect(blob.type).toBe("image/tiff");
  });

  it("embedded bytes are byte-exact with the committed public GeoTIFFs", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(null, { status: 204 })),
    );
    const cases: [string, string][] = [
      [
        "sundarbans_optical.tiff",
        "47cc51d8e532d105efe244f379641c456520658d206498f6c361f32719cb50b0",
      ],
      [
        "sundarbans_sar.tiff",
        "5fb7616cf07e372d5986ee4a7e24bc616a217f998dad5b8742bc54d0228dfa69",
      ],
    ];
    for (const [name, sha] of cases) {
      const f = PRESETS.flatMap((p) => p.files ?? []).find(
        (x) => x.name === name,
      )!;
      expect(await sha256(await fetchPresetFile(f))).toBe(sha);
    }
  });
});
