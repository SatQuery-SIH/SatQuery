import { useEffect, useState } from "react";
import { api } from "../api";
import type { RunListItem } from "../types";

// "2026-09-22T13:34:02Z" → "09-22 13:34"
function shortTs(ts: string | null): string {
  if (!ts) return "";
  return ts.slice(5, 16).replace("T", " ");
}

export function RunsPanel({
  onLoad,
  collapsed,
  onToggle,
  refreshKey,
  currentRunId,
}: {
  onLoad: (runId: string) => void;
  collapsed: boolean;
  onToggle: () => void;
  refreshKey: number;
  currentRunId?: string | null;
}) {
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [err, setErr] = useState(false);
  const refresh = () => api.runs(50).then(setRuns).catch(() => setErr(true));
  useEffect(() => {
    void refresh();
    // refreshKey bumps whenever a run lands so new runs appear without ↻
  }, [refreshKey]);

  if (collapsed) {
    return (
      <div className="runs-panel collapsed">
        <button className="runs-rail" title="runs" onClick={onToggle}>
          <span className="runs-rail-text">runs</span>
          <span className="runs-rail-count">{runs.length || ""}</span>
        </button>
      </div>
    );
  }

  return (
    <div className="runs-panel">
      <div className="runs-head">
        <span>runs</span>
        <span className="runs-head-actions">
          <button className="btn-mini" onClick={refresh} title="refresh">
            ↻
          </button>
          <button className="btn-mini" onClick={onToggle} title="collapse">
            »
          </button>
        </span>
      </div>
      {err && <div className="muted">API unreachable</div>}
      <ul>
        {runs.map((r) => {
          const refused = r.supported === false;
          return (
            <li key={r.run_id}>
              <button
                className={`run-item${r.run_id === currentRunId ? " current" : ""}`}
                onClick={() => onLoad(r.run_id)}
              >
                <span className="run-q">{(r.query ?? "").slice(0, 60)}</span>
                <span className={`run-meta${refused ? " refused" : ""}`}>
                  {shortTs(r.ts)} · {r.input_mode} ·{" "}
                  {refused ? "refused" : "ok"}
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
