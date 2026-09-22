import { useRef, useState } from "react";
import { api } from "../api";
import type { DetectedBlock, InputMode, UploadResponse } from "../types";

const MODE_ROLES: Record<InputMode, { role: string; label: string; accept: string }[]> = {
  single: [{ role: "image", label: "image", accept: ".png,.jpg,.jpeg,.tif,.tiff" }],
  "bi-temporal": [
    { role: "before", label: "before", accept: ".png,.jpg,.jpeg,.tif,.tiff" },
    { role: "after", label: "after", accept: ".png,.jpg,.jpeg,.tif,.tiff" },
  ],
  "optical+sar": [
    { role: "optical", label: "optical", accept: ".png,.jpg,.jpeg,.tif,.tiff" },
    { role: "sar", label: "sar (tif/npz)", accept: ".tif,.tiff,.npz,.png" },
  ],
};

export function DetectedCard({ d, warnings }: { d: DetectedBlock; warnings?: string[] }) {
  const rows: [string, React.ReactNode][] = [
    ["gsd", d.gsd_m != null ? `${d.gsd_m} m/px` : "—"],
    ["gsd source", d.gsd_source],
    ["crs", d.crs ?? "—"],
    ["bands", d.bands ?? "—"],
    ["dtype", d.dtype ?? "—"],
  ];
  if (d.calibrated != null) rows.push(["sar calibrated", d.calibrated ? "yes" : "no"]);
  return (
    <div className="detected-card" data-testid="detected-card">
      <div className="detected-title">detected — ingest contract</div>
      <div className="detected-grid">
        {rows.map(([k, v]) => (
          <div className="detected-row" key={k}>
            <span className="detected-key">{k}</span>
            <span className="detected-val">{v}</span>
          </div>
        ))}
      </div>
      {d.crs_note && <div className="detected-note">{d.crs_note}</div>}
      {(warnings ?? []).map((w, i) => (
        <div className="detected-warn" key={i}>⚠ {w}</div>
      ))}
    </div>
  );
}

export function UploadPanel({
  mode,
  onUploaded,
}: {
  mode: InputMode;
  onUploaded: (u: UploadResponse) => void;
}) {
  const [files, setFiles] = useState<Record<string, File>>({});
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState<string | null>(null);
  const inputs = useRef<Record<string, HTMLInputElement | null>>({});

  const setFile = (role: string, f: File | undefined) => {
    setFiles((s) => {
      const n = { ...s };
      if (f) n[role] = f;
      else delete n[role];
      return n;
    });
  };

  const complete = MODE_ROLES[mode].every((r) => files[r.role]);

  const doUpload = async () => {
    setBusy(true);
    setErr(null);
    try {
      const res = await api.upload(mode, files);
      onUploaded(res);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="upload-panel">
      {MODE_ROLES[mode].map((r) => (
        <div
          key={r.role}
          className={`drop-slot ${files[r.role] ? "has-file" : ""} ${dragOver === r.role ? "drag" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(r.role);
          }}
          onDragLeave={() => setDragOver(null)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(null);
            setFile(r.role, e.dataTransfer.files?.[0]);
          }}
          onClick={() => inputs.current[r.role]?.click()}
        >
          <input
            type="file"
            accept={r.accept}
            hidden
            ref={(el) => {
              inputs.current[r.role] = el;
            }}
            onChange={(e) => setFile(r.role, e.target.files?.[0] ?? undefined)}
          />
          <span className="drop-role">{r.label}</span>
          <span className="drop-name">
            {files[r.role] ? files[r.role].name : "drop file or click"}
          </span>
        </div>
      ))}
      <button className="btn" disabled={!complete || busy} onClick={doUpload}>
        {busy ? "uploading + ingesting…" : "upload & inspect"}
      </button>
      {err && <div className="err">{err}</div>}
    </div>
  );
}
