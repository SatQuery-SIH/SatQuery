import type { InputMode } from "../types";
import { presetsForMode, type Preset } from "../presets";

export type { Preset } from "../presets";

// Cards only — the fetch/upload/query flow lives in App.runPreset so mode,
// upload_id and scene are passed explicitly (never read from stale closure).
export function PresetPicker({
  mode,
  disabled,
  active,
  onRun,
}: {
  mode: InputMode;
  disabled?: boolean;
  active: { id: string; phase: "uploading" | "running" } | null;
  onRun: (p: Preset) => void;
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
            onClick={() => onRun(p)}
          >
            <span className="preset-label">{p.label}</span>
            <span className="preset-tag">
              {p.kind === "upload" ? "real upload" : `prepared scene ${p.scene}`}
            </span>
            <span className="preset-blurb">{p.blurb}</span>
            {isActive && (
              <span className="preset-phase">
                {active.phase === "uploading" ? "uploading…" : "running…"}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
