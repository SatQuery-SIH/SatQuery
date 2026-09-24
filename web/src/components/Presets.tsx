import type { InputMode } from "../types";
import { presetsForMode, type Preset } from "../presets";

export type { Preset } from "../presets";

// Cards only — the fetch/upload binding lives in App.loadPreset: a click
// fills the query and binds the inputs, it never runs the query itself.
export function PresetPicker({
  mode,
  disabled,
  active,
  onLoad,
}: {
  mode: InputMode;
  disabled?: boolean;
  active: { id: string; phase: "uploading" } | null;
  onLoad: (p: Preset) => void;
}) {
  const presets = presetsForMode(mode);
  if (!presets.length) return null;
  return (
    <div className="preset-list">
      {presets.map((p) => {
        const isActive = active?.id === p.id;
        return (
          <button
            key={p.id}
            data-testid={`preset-${p.id}`}
            className={`preset-card${isActive ? " active" : ""}`}
            disabled={disabled || active != null}
            onClick={() => onLoad(p)}
          >
            <span className="preset-label">{p.label}</span>
            <span className="preset-tag">
              {p.kind === "upload" ? "real upload" : `prepared scene ${p.scene}`}
            </span>
            <span className="preset-blurb">{p.blurb}</span>
            {isActive && <span className="preset-phase">loading…</span>}
          </button>
        );
      })}
    </div>
  );
}
