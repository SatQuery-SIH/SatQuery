import { useEffect } from "react";

// Shared image lightbox — used by the imagery stage and the artifacts grid.
export function Lightbox({
  src,
  name,
  type,
  onClose,
}: {
  src: string;
  name: string;
  type?: string;
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
        <img src={src} alt={name} />
        <div className="lightbox-caption">
          {name}
          {type ? ` · ${type}` : ""}
        </div>
        <div className="lightbox-actions">
          <a href={src} target="_blank" rel="noreferrer">
            open original ↗
          </a>
          <button className="btn-mini" autoFocus onClick={onClose}>
            close
          </button>
        </div>
      </div>
    </div>
  );
}
