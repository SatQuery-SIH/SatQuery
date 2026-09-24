import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { sanitizeCopy } from "../verified";
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
      <div className="detected-title">image details</div>
      <div className="detected-grid">
        {rows.map(([k, v]) => (
          <div className="detected-row" key={k}>
            <span className="detected-key">{k}</span>
            <span className="detected-val">{v}</span>
          </div>
        ))}
      </div>
      {d.crs_note && (
        <div className="detected-note">{sanitizeCopy(d.crs_note)}</div>
      )}
      {(warnings ?? []).map((w, i) => (
        <div className="detected-warn" key={i}>⚠ {sanitizeCopy(w)}</div>
      ))}
    </div>
  );
}

function isInstantImage(f: File): boolean {
  if (f.type === "image/png" || f.type === "image/jpeg") return true;
  return /\.(png|jpe?g)$/i.test(f.name);
}

// Object URL lifecycle is owned by an effect keyed on the File — revoke on
// replace / clear / unmount / mode change (StrictMode-safe).
function LocalThumb({ file, onError }: { file: File; onError: () => void }) {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    const u = URL.createObjectURL(file);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [file]);
  return url ? <img src={url} alt={file.name} onError={onError} /> : null;
}

type SlotState =
  | "empty"
  | "local"
  | "pending-server"
  | "server"
  | "no-preview"
  | "error";

function SlotPreview({
  state,
  file,
  serverUrl,
  onImgError,
}: {
  state: SlotState;
  file?: File;
  serverUrl?: string;
  onImgError: () => void;
}) {
  return (
    <div className="slot-preview" data-state={state}>
      {state === "empty" && <span className="slot-hint">drop file or click</span>}
      {state === "local" && file && (
        <LocalThumb file={file} onError={onImgError} />
      )}
      {state === "pending-server" && (
        <span className="slot-hint">preview rendered by the API after upload</span>
      )}
      {state === "server" && serverUrl && (
        <img src={serverUrl} alt={file?.name ?? "preview"} onError={onImgError} />
      )}
      {state === "no-preview" && <span className="slot-hint">no preview</span>}
      {state === "error" && <span className="slot-hint">preview failed to load</span>}
    </div>
  );
}

export function UploadPanel({
  mode,
  onUploaded,
  onFilesChange,
}: {
  mode: InputMode;
  onUploaded: (u: UploadResponse) => void;
  // lifted so the imagery stage can name the files in its "your images" line
  onFilesChange?: (names: Record<string, string>) => void;
}) {
  const [files, setFiles] = useState<Record<string, File>>({});
  // null = no upload yet this panel; per-role url | null once uploaded
  const [serverPrev, setServerPrev] = useState<Record<string, string | null> | null>(null);
  // role -> File | server url that failed to render (resets when it changes)
  const [errFor, setErrFor] = useState<Record<string, File | string | null>>({});
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState<string | null>(null);
  const inputs = useRef<Record<string, HTMLInputElement | null>>({});

  // Monotonic upload id — if the user swaps a file mid-upload, only the
  // newest upload may bind; a stale response must not overwrite it.
  const uploadSeq = useRef(0);

  const setFile = (role: string, f: File | undefined) => {
    const n = { ...files };
    if (f) n[role] = f;
    else delete n[role];
    setFiles(n);
    onFilesChange?.(
      Object.fromEntries(Object.entries(n).map(([r, x]) => [r, x.name])),
    );
    // replacing any file invalidates this panel's server previews
    setServerPrev(null);
    // auto-bind: once every slot has a file, upload immediately — a staged
    // thumbnail must never look bound while nothing is actually uploaded
    if (MODE_ROLES[mode].every((r) => n[r.role])) void doUpload(n);
  };

  const complete = MODE_ROLES[mode].every((r) => files[r.role]);

  const doUpload = async (fset: Record<string, File> = files) => {
    const seq = ++uploadSeq.current;
    setBusy(true);
    setErr(null);
    try {
      const res = await api.upload(mode, fset);
      if (seq !== uploadSeq.current) return; // superseded by a newer pick
      const map: Record<string, string | null> = {};
      for (const r of MODE_ROLES[mode])
        map[r.role] =
          res.previews?.find((p) => p.role === r.role)?.url ?? null;
      setServerPrev(map);
      onUploaded(res);
    } catch (e) {
      if (seq === uploadSeq.current)
        setErr(e instanceof Error ? e.message : String(e));
    } finally {
      if (seq === uploadSeq.current) setBusy(false);
    }
  };

  const slotState = (role: string): SlotState => {
    const f = files[role];
    const sp = serverPrev?.[role];
    const src = sp ? api.previewUrl(sp) : null;
    const curKey: File | string | null = src ?? f ?? null;
    if (curKey != null && errFor[role] === curKey) return "error";
    if (src) return "server";
    if (!f) return "empty";
    if (serverPrev && sp === null) return "no-preview";
    return isInstantImage(f) ? "local" : "pending-server";
  };

  return (
    <div className="upload-panel">
      {MODE_ROLES[mode].map((r) => {
        const f = files[r.role];
        const sp = serverPrev?.[r.role];
        const state = slotState(r.role);
        return (
          <div
            key={r.role}
            data-testid={`slot-${r.role}`}
            className={`drop-slot ${f ? "has-file" : ""} ${dragOver === r.role ? "drag" : ""}`}
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
            <SlotPreview
              state={state}
              file={f}
              serverUrl={sp ? api.previewUrl(sp) : undefined}
              onImgError={() =>
                setErrFor((s) => ({
                  ...s,
                  [r.role]: (sp ? api.previewUrl(sp) : f) ?? null,
                }))
              }
            />
            {f && <span className="drop-name">{f.name}</span>}
          </div>
        );
      })}
      <button className="btn" disabled={!complete || busy} onClick={doUpload}>
        {busy ? "uploading + ingesting…" : "upload & inspect"}
      </button>
      {err && <div className="err">{err}</div>}
    </div>
  );
}
