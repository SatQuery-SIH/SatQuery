import { useState } from "react";
import { api } from "../api";
import type { InputMode, UploadResponse } from "../types";

// Presets: prepared scenes ride the contract's `scene` param; real GeoTIFF
// presets bundle real files and go through the real /upload path — the same
// ingest an evaluator's own files get. No fake client-side results.
export interface Preset {
  id: string;
  label: string;
  mode: InputMode;
  query: string;
  scene?: number; // prepared-scene path
  files?: { role: string; url: string; name: string }[]; // bundled upload path
}

export const PRESETS: Preset[] = [
  {
    id: "sundarbans",
    label: "Sundarbans estuary — real Sentinel-1/2 pair",
    mode: "optical+sar",
    query: "Is there water in this scene?",
    files: [
      { role: "optical", url: "/presets/sundarbans_optical.tiff", name: "sundarbans_optical.tiff" },
      { role: "sar", url: "/presets/sundarbans_sar.tiff", name: "sundarbans_sar.tiff" },
    ],
  },
  {
    id: "sundarbans-single",
    label: "Sundarbans optical (single image)",
    mode: "single",
    query: "is there a large water body in this image",
    files: [
      { role: "image", url: "/presets/sundarbans_optical.tiff", name: "sundarbans_optical.tiff" },
    ],
  },
  {
    id: "scene2",
    label: "Prepared scene — bi-temporal change",
    mode: "bi-temporal",
    query: "what changed between the two images",
    scene: 2,
  },
  {
    id: "scene3",
    label: "Prepared scene — optical+SAR",
    mode: "optical+sar",
    query: "Is there water in this scene?",
    scene: 3,
  },
];

export function PresetPicker({
  onPick,
  onUploaded,
  onQuery,
}: {
  onPick: (p: Preset) => void;
  onUploaded: (u: UploadResponse) => void;
  onQuery: (q: string) => void;
}) {
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const run = async (p: Preset) => {
    setBusy(p.id);
    setErr(null);
    onPick(p);
    try {
      if (p.files) {
        // fetch bundled assets → real /upload → real ingest
        const rec: Record<string, File> = {};
        for (const f of p.files) {
          const blob = await (await fetch(f.url)).blob();
          rec[f.role] = new File([blob], f.name, { type: "image/tiff" });
        }
        const up = await api.upload(p.mode, rec);
        onUploaded(up);
      }
      onQuery(p.query);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="presets">
      <span className="presets-label">demo presets:</span>
      {PRESETS.map((p) => (
        <button
          key={p.id}
          className="preset-btn"
          disabled={busy !== null}
          onClick={() => run(p)}
          title={p.files ? "bundled real GeoTIFF → live upload path" : "prepared scene"}
        >
          {busy === p.id ? "running…" : p.label}
        </button>
      ))}
      {err && <span className="err">{err}</span>}
    </div>
  );
}
