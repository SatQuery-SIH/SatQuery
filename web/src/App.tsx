import { useState } from "react";
import { api } from "./api";
import type { InputMode, RunBundle, UploadResponse } from "./types";
import { SeatBar } from "./components/SeatBar";
import { PresetPicker, type Preset } from "./components/Presets";
import { DetectedCard, UploadPanel } from "./components/UploadPanel";
import { Results } from "./components/Results";
import { RunsPanel } from "./components/RunsPanel";
import "./app.css";

const MODES: { id: InputMode; label: string }[] = [
  { id: "single", label: "single image" },
  { id: "bi-temporal", label: "bi-temporal pair" },
  { id: "optical+sar", label: "optical + SAR" },
];

function WarmingNote({ q }: { q: boolean }) {
  return (
    <div className="warming" data-testid="warming">
      <div className="warming-title">
        {q ? "pipeline running…" : "waking up serverless GPU nodes & loading model weights"}
      </div>
      <div className="warming-sub">
        two 8B vision-language models (narrator + canonical) — first boot can take a minute.
      </div>
      <div className="warming-stages">
        <span>ingest</span>→<span>plan</span>→<span>tools</span>→<span>packet</span>→<span>narrate</span>
      </div>
    </div>
  );
}

export default function App() {
  const [mode, setMode] = useState<InputMode>("single");
  const [query, setQuery] = useState("");
  const [scene, setScene] = useState<number | null>(null);
  const [upload, setUpload] = useState<UploadResponse | null>(null);
  const [bundle, setBundle] = useState<RunBundle | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const doQuery = async (q?: string) => {
    const text = (q ?? query).trim();
    if (!text) return;
    setBusy(true);
    setErr(null);
    setBundle(null);
    try {
      const b = await api.query({
        query: text,
        input_mode: mode,
        upload_id: upload?.upload_id ?? undefined,
        scene: upload ? undefined : (scene ?? undefined),
        live: true,
      });
      setBundle(b);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const onPreset = (p: Preset) => {
    setMode(p.mode);
    setQuery(p.query);
    setScene(p.scene ?? null);
    if (p.scene != null) setUpload(null); // prepared scene, not an upload
  };

  return (
    <div className="app">
      <SeatBar />
      <div className="layout">
        <main>
          <div className="mode-tabs">
            {MODES.map((m) => (
              <button
                key={m.id}
                className={mode === m.id ? "tab on" : "tab"}
                onClick={() => {
                  setMode(m.id);
                  setUpload(null);
                  setScene(null);
                }}
              >
                {m.label}
              </button>
            ))}
          </div>

          <PresetPicker
            onPick={onPreset}
            onUploaded={setUpload}
            onQuery={(q) => {
              setQuery(q);
              void doQuery(q);
            }}
          />

          <UploadPanel
            mode={mode}
            onUploaded={(u) => {
              setUpload(u);
              setScene(null);
            }}
          />
          {upload && (
            <DetectedCard d={upload.detected} warnings={upload.warnings} />
          )}
          {upload && (
            <div className="muted">bound upload: {upload.upload_id}</div>
          )}
          {scene != null && !upload && (
            <div className="muted">prepared scene {scene} selected</div>
          )}

          <div className="query-row">
            <input
              className="query-input"
              value={query}
              placeholder="ask about the imagery…"
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && void doQuery()}
            />
            <button className="btn primary" disabled={busy || !query.trim()} onClick={() => void doQuery()}>
              {busy ? "running…" : "run"}
            </button>
          </div>

          {busy && <WarmingNote q />}
          {err && <div className="err">query failed: {err}</div>}
          {bundle && <Results bundle={bundle} />}
        </main>
        <aside>
          <RunsPanel
            onLoad={(id) => {
              setErr(null);
              api.run(id).then(setBundle).catch((e) => setErr(String(e)));
            }}
          />
        </aside>
      </div>
    </div>
  );
}
