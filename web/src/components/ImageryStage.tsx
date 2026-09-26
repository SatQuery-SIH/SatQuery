// The hero: mode-driven imagery panels. Before a run they show the bound
// upload previews / prepared-scene previews; after a run they come from the
// bundle's real artifacts (matched by trace basenames, never guessed).
// Bi-temporal keeps before|after side by side with the change overlay as a
// toggleable layer on the after panel — never a switcher.
import { useState } from "react";
import { api } from "../api";
import { baseName } from "../liveTrace";
import { imageSummary } from "../verified";
import type { InputMode, RunBundle, UploadResponse } from "../types";
import type { Preset } from "../presets";
import { Lightbox } from "./Lightbox";

const ROLE_LABELS: Record<InputMode, string[]> = {
  single: ["Image"],
  "bi-temporal": ["Before", "After"],
  "optical+sar": ["Optical", "SAR"],
};

// which panel an overlay stacks on, per mode
const OVERLAY_ROLE: Record<InputMode, string> = {
  single: "image",
  "bi-temporal": "after",
  "optical+sar": "optical",
};

const OVERLAY_LABEL: Record<InputMode, string> = {
  single: "water overlay",
  "bi-temporal": "change overlay",
  "optical+sar": "water overlay",
};

// agreement-map legend (colours from demo/pipeline.py _agreement_map_image)
const LEGEND: [string, string][] = [
  ["#1eb43c", "water in both"],
  ["#3c78e6", "optical only"],
  ["#f0961e", "SAR only"],
  ["#dc2828", "disagreement"],
  ["#2d2d2d", "neither"],
];

interface Panel {
  role: string;
  label: string;
  src: string | null;
  name: string;
}

function artifactByName(bundle: RunBundle, name: string | null) {
  if (!name) return null;
  return (bundle.artifacts ?? []).find((a) => a.name === name) ?? null;
}

export function ImageryStage({
  mode,
  bundle,
  upload,
  boundPreset,
  inputNames,
}: {
  mode: InputMode;
  bundle: RunBundle | null;
  upload: UploadResponse | null;
  boundPreset: Preset | null;
  inputNames: string[];
}) {
  const [overlayOn, setOverlayOn] = useState(true);
  const [lb, setLb] = useState<{ src: string; name: string; type?: string; withOverlay?: boolean } | null>(null);
  const runId = bundle?.run_id ?? null;
  // overlay defaults ON for each new run that ships one — reset during render
  // when the displayed run changes (no post-render effect needed)
  const [overlayRun, setOverlayRun] = useState(runId);
  if (overlayRun !== runId) {
    setOverlayRun(runId);
    setOverlayOn(true);
  }

  // the stage shows the bundle's own mode for replays of another tab's run
  const bm = bundle?.trace?.input_mode;
  const stageMode: InputMode =
    bm === "single" || bm === "bi-temporal" || bm === "optical+sar" ? bm : mode;
  const roles = ROLE_LABELS[stageMode];

  // pre-run sources: upload previews, then prepared preset previews
  const preRun: Record<string, string> = {};
  if (upload?.previews)
    for (const p of upload.previews) preRun[p.role] = api.previewUrl(p.url);
  else if (boundPreset?.previews)
    for (const p of boundPreset.previews) preRun[p.role] = p.url;

  const trace = bundle?.trace ?? {};
  const agreeName = trace.agreement_map_path ? baseName(String(trace.agreement_map_path)) : null;
  const overlayArt = bundle
    ? artifactByName(bundle, trace.overlay_path ? baseName(String(trace.overlay_path)) : null)
    : null;
  const agreeArt = bundle ? artifactByName(bundle, agreeName) : null;

  // role panels from bundle artifacts, in trace.images order, falling back to
  // the pre-run previews when a given artifact is absent
  const imageNames = Array.isArray(trace.images)
    ? trace.images.map(String).map(baseName).filter((n) => n !== agreeName)
    : [];
  const panels: Panel[] = roles.map((label, i) => {
    const art =
      bundle && imageNames[i]
        ? artifactByName(bundle, imageNames[i])
        : null;
    const src = art ? api.artifactUrl(art.url) : (preRun[label.toLowerCase()] ?? preRun[label] ?? null);
    return {
      role: label.toLowerCase(),
      label,
      src,
      name: art?.name ?? (imageNames[i] ?? inputNames[i] ?? label),
    };
  });
  if (agreeArt)
    panels.push({
      role: "agreement",
      label: "Agreement map",
      src: api.artifactUrl(agreeArt.url),
      name: agreeArt.name,
    });

  const overlaySrc = overlayArt ? api.artifactUrl(overlayArt.url) : null;
  const overlayRole = OVERLAY_ROLE[stageMode];
  // A presence-gated grounding box isn't a water overlay — label it so the
  // chip can't pass the estimate off as a measured mask.
  const groundOut = (bundle?.tool_outputs as Record<string, unknown> | undefined)
    ?.ground as Record<string, unknown> | undefined;
  const overlayLabel =
    groundOut && !groundOut.withheld && groundOut.box01
      ? "grounding box (estimate)"
      : OVERLAY_LABEL[stageMode];

  // "your images" line — names + one-line honest summary
  const g = (trace.gsd ?? null) as Record<string, unknown> | null;
  let metaLine: string;
  if (bundle) {
    const names = g?.path
      ? [baseName(String(g.path))]
      : panels.map((p) => p.name);
    metaLine = imageSummary({
      names,
      gsd_m: typeof g?.gsd_m === "number" ? (g.gsd_m as number) : null,
      crs: typeof g?.crs === "string" ? (g.crs as string) : null,
    });
  } else if (upload) {
    metaLine = imageSummary({
      names: inputNames.length ? inputNames : roles,
      gsd_m: upload.detected.gsd_m,
      crs: upload.detected.crs,
      calibrated: upload.detected.calibrated,
    });
  } else if (boundPreset) {
    metaLine = `built-in example · ${inputNames.join(", ") || boundPreset.label}`;
  } else {
    metaLine =
      "pick an example or upload imagery — without files the built-in example for this mode is used";
  }

  const bare = !bundle && !upload && !boundPreset;
  return (
    <div className="imagery-stage" data-testid="imagery-stage">
      {bare ? (
        <div className="stage-placeholder">
          {metaLine}
        </div>
      ) : (
      <div
        className={`stage-panels mode-${stageMode === "bi-temporal" ? "bi" : stageMode === "optical+sar" ? "optsar" : "single"}`}
      >
        {panels.map((p) => (
          <figure
            key={p.role}
            className="stage-panel"
            data-testid={`stage-panel-${p.role}`}
          >
            <div className="stage-frame">
              {p.src ? (
                <button
                  className="stage-img-btn"
                  onClick={() => setLb({ src: p.src!, name: p.name, withOverlay: p.role === overlayRole && !!overlaySrc })}
                  title={`expand ${p.label}`}
                >
                  <img src={p.src} alt={p.label} />
                </button>
              ) : (
                <div className="stage-empty">no imagery yet</div>
              )}
              {overlaySrc && p.role === overlayRole && (
                <img
                  className={`stage-overlay${overlayOn ? " on" : ""}`}
                  src={overlaySrc}
                  alt={`${p.label} overlay`}
                />
              )}
              <span className="stage-caption">{p.label}</span>
              <span className="stage-actions">
                {p.src && (
                  <button
                    className="stage-icon"
                    title={`expand ${p.label}`}
                    aria-label={`expand ${p.label}`}
                    onClick={() => setLb({ src: p.src!, name: p.name, withOverlay: p.role === overlayRole && !!overlaySrc })}
                  >
                    ⤢
                  </button>
                )}
                {p.src && (
                  <a
                    className="stage-icon"
                    href={p.src}
                    download={p.name}
                    title={`download ${p.name}`}
                    aria-label={`download ${p.name}`}
                  >
                    ⬇
                  </a>
                )}
              </span>
              {overlaySrc && p.role === overlayRole && (
                <button
                  className={`overlay-chip${overlayOn ? " on" : ""}`}
                  data-testid="overlay-toggle"
                  aria-pressed={overlayOn}
                  onClick={() => setOverlayOn((o) => !o)}
                >
                  {overlayLabel}: {overlayOn ? "on" : "off"}
                </button>
              )}
            </div>
            {p.role === "agreement" && (
              <div className="stage-legend">
                {LEGEND.map(([c, l]) => (
                  <span key={l} className="legend-item">
                    <i style={{ background: c }} />
                    {l}
                  </span>
                ))}
              </div>
            )}
          </figure>
        ))}
      </div>
      )}
      {!bare && (
      <div className="stage-meta">
        <span className="stage-meta-label">your images</span>
        <span className="stage-meta-line">{metaLine}</span>
      </div>
      )}
      {lb && (
        <Lightbox
          src={lb.src}
          name={lb.name}
          type={lb.type}
          overlaySrc={lb.withOverlay ? overlaySrc : null}
          overlayOn={overlayOn}
          overlayLabel={overlayLabel}
          onToggleOverlay={() => setOverlayOn((o) => !o)}
          onClose={() => setLb(null)}
        />
      )}
    </div>
  );
}
