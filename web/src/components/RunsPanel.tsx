import { useEffect, useState } from "react";
import { api } from "../api";
import type { RunListItem } from "../types";

export function RunsPanel({ onLoad }: { onLoad: (runId: string) => void }) {
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [err, setErr] = useState(false);
  const refresh = () => api.runs(50).then(setRuns).catch(() => setErr(true));
  useEffect(() => { void refresh(); }, []);

  return (
    <div className="runs-panel">
      <div className="runs-head">
        <span>runs</span>
        <button className="btn-mini" onClick={refresh}>↻</button>
      </div>
      {err && <div className="muted">API unreachable</div>}
      <ul>
        {runs.map((r) => (
          <li key={r.run_id}>
            <button className="run-item" onClick={() => onLoad(r.run_id)}>
              <span className="run-q">{(r.query ?? "").slice(0, 60)}</span>
              <span className="run-meta">
                {r.input_mode} · {r.supported === false ? "refused" : "ok"}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
