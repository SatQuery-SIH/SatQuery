import { useEffect } from "react";

// Shared image lightbox — used by the imagery stage and the artifacts grid.
// When the opened panel carries an overlay (grounding box / water mask),
// overlaySrc stacks it on the same box so the alignment is pixel-exact, and
// the same on/off toggle is offered inside the lightbox too.
export function Lightbox({
  src,
  name,
  type,
  overlaySrc,
  overlayOn,
  overlayLabel,
  onToggleOverlay,
  onClose,
}: {
  src: string;
  name: string;
  type?: string;
  overlaySrc?: string | null;
  overlayOn?: boolean;
  overlayLabel?: string;
  onToggleOverlay?: () => void;
  onClose: () => void;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div
      className="lightbox"
      role="dialog"
      aria-modal="true"
      data-testid="lightbox"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="lightbox-body">
        <div className="lightbox-frame">
          <img src={src} alt={name} />
          {overlaySrc && overlayOn && (
            <img className="lightbox-overlay" src={overlaySrc} alt={`${name} overlay`} />
          )}
        </div>
        <div className="lightbox-caption">
          {name}
          {type ? ` · ${type}` : ""}
        </div>
        <div className="lightbox-actions">
          <a href={src} target="_blank" rel="noreferrer">
            open original ↗
          </a>
          {overlaySrc && onToggleOverlay && (
            <button
              className={`overlay-chip${overlayOn ? " on" : ""}`}
              aria-pressed={overlayOn}
              onClick={onToggleOverlay}
            >
              {overlayLabel ?? "overlay"}: {overlayOn ? "on" : "off"}
            </button>
          )}
          <button className="btn-mini" autoFocus onClick={onClose}>
            close
          </button>
        </div>
      </div>
    </div>
  );
}
