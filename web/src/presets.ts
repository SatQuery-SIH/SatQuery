// Demo presets — prepared scenes ride the contract's `scene` param; upload
// presets bundle real files under public/presets and go through the real
// /upload + ingest path. No fake client-side results.
// (Lives outside components/ on purpose: Windows resolves "./Presets"
// case-insensitively, which would collide with components/Presets.tsx.)
import type { InputMode } from "./types";
import opticalTiffB64 from "./assets/sundarbans_optical.tiff.b64?raw";
import sarTiffB64 from "./assets/sundarbans_sar.tiff.b64?raw";

// Embedded copies of the bundled GeoTIFFs (base64). Some managed-network
// filters strip image/tiff responses entirely — the public fetch comes
// back 204 with no body — so fetchPresetFile falls back to these bytes,
// which never traverse the network.
const PRESET_INLINE: Record<string, string> = {
  "/presets/sundarbans_optical.tiff": opticalTiffB64,
  "/presets/sundarbans_sar.tiff": sarTiffB64,
};

function base64Blob(b64: string, type: string): Blob {
  const bin = atob(b64.trim());
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type });
}

export interface PresetFile {
  role: string;
  url: string;
  name: string;
}

export interface PresetPreview {
  role: string;
  url: string;
  label: string;
}

export interface Preset {
  id: string;
  label: string;
  mode: InputMode;
  query: string;
  kind: "upload" | "prepared";
  scene?: number;
  files?: PresetFile[];
  previews?: PresetPreview[];
  blurb: string;
}

export const PRESETS: Preset[] = [
  {
    id: "sundarbans-single",
    label: "Sundarbans estuary — Sentinel-2 optical",
    mode: "single",
    query: "is there a large water body in this image",
    kind: "upload",
    files: [
      {
        role: "image",
        url: "/presets/sundarbans_optical.tiff",
        name: "sundarbans_optical.tiff",
      },
    ],
    blurb: "bundled real GeoTIFF → live /upload + ingest",
  },
  {
    id: "levir-pair",
    label: "LEVIR-CD building-change pair",
    mode: "bi-temporal",
    query: "what changed between the two images",
    kind: "upload",
    files: [
      {
        role: "before",
        url: "/presets/scene2_before.png",
        name: "scene2_before.png",
      },
      {
        role: "after",
        url: "/presets/scene2_after.png",
        name: "scene2_after.png",
      },
    ],
    blurb: "bundled real PNG pair → live /upload + ingest",
  },
  {
    id: "scene2",
    label: "Prepared scene 2 — bi-temporal change",
    mode: "bi-temporal",
    query: "what changed between the two images",
    kind: "prepared",
    scene: 2,
    previews: [
      { role: "before", url: "/presets/scene2_before.png", label: "before" },
      { role: "after", url: "/presets/scene2_after.png", label: "after" },
    ],
    blurb: "built-in example (API scene=2)",
  },
  {
    id: "sundarbans",
    label: "Sundarbans estuary — Sentinel-1/2 pair",
    mode: "optical+sar",
    query: "Is there water in this scene?",
    kind: "upload",
    files: [
      {
        role: "optical",
        url: "/presets/sundarbans_optical.tiff",
        name: "sundarbans_optical.tiff",
      },
      {
        role: "sar",
        url: "/presets/sundarbans_sar.tiff",
        name: "sundarbans_sar.tiff",
      },
    ],
    blurb: "bundled real GeoTIFF pair → live /upload + ingest",
  },
  {
    id: "scene3",
    label: "Prepared scene 3 — optical + SAR",
    mode: "optical+sar",
    query: "Is there water in this scene?",
    kind: "prepared",
    scene: 3,
    previews: [
      {
        role: "optical",
        url: "/presets/scene3_optical.png",
        label: "optical",
      },
      {
        role: "sar",
        url: "/presets/scene3_sar_vv.png",
        label: "SAR VV (render)",
      },
    ],
    blurb: "built-in example (API scene=3)",
  },
];

export function presetsForMode(mode: InputMode): Preset[] {
  return PRESETS.filter((p) => p.mode === mode);
}

export async function fetchPresetFile(file: PresetFile): Promise<Blob> {
  try {
    const res = await fetch(file.url);
    if (res.ok) {
      const blob = await res.blob();
      if (blob.size > 0) return blob;
    }
  } catch {
    /* fall through to the embedded copy */
  }
  const inline = PRESET_INLINE[file.url];
  if (inline) return base64Blob(inline, mimeForName(file.name));
  throw new Error(`preset fetch failed: ${file.url}`);
}

export function mimeForName(name: string): string {
  const n = name.toLowerCase();
  if (n.endsWith(".tif") || n.endsWith(".tiff")) return "image/tiff";
  if (n.endsWith(".png")) return "image/png";
  if (n.endsWith(".jpg") || n.endsWith(".jpeg")) return "image/jpeg";
  if (n.endsWith(".npz")) return "application/octet-stream";
  return "";
}
