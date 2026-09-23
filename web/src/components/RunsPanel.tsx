import { useEffect, useState } from "react";
import type { RunListItem } from "../types";

// "2026-09-22T13:34:02Z" → "09-22 13:34"
function shortTs(ts: string | null): string {
  if (!ts) return "";
  return ts.slice(5, 16).replace("T", " ");
}

const groupKey = (r: RunListItem) =>
  `${r.query}|${r.input_mode}|${r.supported}`;

// Slide-over runs drawer: one row per (query, mode, outcome) group — the
// latest run is the row, older duplicates collapse behind a ×N expander
// (time-only entries, each still loadable).
export function RunsPanel({
  open,
  runs,
  error,
  onClose,
  onLoad,
  onRefresh,
  currentRunId,
}: {
  open: boolean;
  runs: RunListItem[];
  error: boolean;
  onClose: () => void;
  onLoad: (runId: string) => void;
  onRefresh: () => void;
  currentRunId?: string | null;
}) {
  const [openGroup, setOpenGroup] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const groups: { key: string; items: RunListItem[] }[] = [];
  for (const r of runs) {
    const k = groupKey(r);
    const g = groups.find((x) => x.key === k);
    if (g) g.items.push(r);
    else groups.push({ key: k, items: [r] });
  }

  const load = (id: string) => {
    onLoad(id);
    onClose();
  };

  const row = (r: RunListItem, timeOnly = false) => {
    const refused = r.supported === false;
    return (
      <button
        className={`run-item${r.run_id === currentRunId ? " current" : ""}`}
        onClick={() => load(r.run_id)}
        title={r.query ?? undefined}
      >
        {!timeOnly && <span className="run-q">{(r.query ?? "").slice(0, 60)}</span>}
        <span className={`run-meta${refused ? " refused" : ""}`}>
          {shortTs(r.ts)}
          {!timeOnly && <> · {r.input_mode} · {refused ? "refused" : "ok"}</>}
        </span>
      </button>
    );
  };

  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} />
      <div className="runs-drawer" data-testid="runs-drawer" role="dialog" aria-label="runs">
        <div className="runs-head">
          <span>runs</span>
          <span className="runs-head-actions">
            <button className="btn-mini" onClick={onRefresh} title="refresh">
              ↻
            </button>
            <button className="btn-mini" onClick={onClose} title="close" data-testid="runs-close">
              ×
            </button>
          </span>
        </div>
        {error && <div className="muted">API unreachable</div>}
        {!error && runs.length === 0 && (
          <div className="muted">no runs yet</div>
        )}
        <ul>
          {groups.map((g) => {
            const latest = g.items[0];
            const extras = g.items.slice(1);
            const expanded = openGroup === g.key;
            return (
              <li key={g.key}>
                <div className="run-group">
                  {row(latest)}
                  {extras.length > 0 && (
                    <button
                      className="run-count"
                      title={`${extras.length} earlier identical run(s)`}
                      onClick={() => setOpenGroup(expanded ? null : g.key)}
                    >
                      ×{g.items.length} {expanded ? "▴" : "▾"}
                    </button>
                  )}
                </div>
                {expanded && (
                  <div className="run-extras">
                    {extras.map((r) => (
                      <div className="run-extra" key={r.run_id}>
                        {row(r, true)}
                      </div>
                    ))}
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      </div>
    </>
  );
}
