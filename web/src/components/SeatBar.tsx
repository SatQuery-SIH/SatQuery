import { useEffect, useState } from "react";
import { api, ApiHttpError } from "../api";
import { baseName } from "../liveTrace";
import type { HealthResponse, SeatProfile, SeatState, SeatsResponse } from "../types";

const PROFILE_TIP = "local = llama.cpp · cloud = Modal GPU";

// two status dots, no seat names on the surface — the model + state live in
// the tooltip/aria-label: "answer model — <basename> · online|offline|unknown"
function StatusDot({
  role,
  seat,
}: {
  role: string;
  seat?: SeatState;
}) {
  const state = seat?.up === true ? "online" : seat?.up === false ? "offline" : "unknown";
  const model = seat?.model ? baseName(seat.model) : "no model reported";
  const label = `${role} — ${model} · ${state}`;
  return (
    <span
      className={`status-dot st-${state}`}
      title={label}
      aria-label={label}
      role="img"
    />
  );
}

export function SeatBar({
  runsCount,
  onOpenRuns,
}: {
  runsCount: number;
  onOpenRuns: () => void;
}) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [seats, setSeats] = useState<SeatsResponse | null>(null);
  const [pendingNote, setPendingNote] = useState<{ text: string; title: string } | null>(null);
  const [apiDown, setApiDown] = useState(false);

  const refresh = () => {
    api.health().then(setHealth).catch(() => setApiDown(true));
    api.seats().then(setSeats).catch(() => {});
  };
  useEffect(() => { void refresh(); }, []);

  const toggle = async (profile: SeatProfile) => {
    setPendingNote(null);
    try {
      const r = await api.setProfile(profile);
      setSeats(r);
    } catch (e) {
      // cloud → 501 until CLOUD-SEAT lands: render as pending, never an error toast
      if (e instanceof ApiHttpError && e.status === 501) {
        setPendingNote({ text: "cloud not available yet", title: e.message });
      } else
        setPendingNote({
          text: "seat switch failed",
          title: e instanceof Error ? e.message : String(e),
        });
    }
  };

  const profile = seats?.profile ?? "local";
  return (
    <div className="seat-bar">
      <span className="brand">SatQuery AI</span>
      <span className="tagline">agentic remote-sensing assistant</span>
      <span className="seat-bar-right">
        {apiDown && <span className="seat-pill seat-down">API unreachable</span>}
        <StatusDot role="answer model" seat={health?.seats?.narrator} />
        <StatusDot role="remote-sensing model" seat={health?.seats?.canonical} />
        <span className="profile-toggle" title={PROFILE_TIP}>
          <button
            className={profile === "local" ? "on" : ""}
            onClick={() => profile !== "local" && toggle("local")}
            title={PROFILE_TIP}
          >
            local
          </button>
          <button
            className={profile === "cloud" ? "on" : ""}
            onClick={() => profile !== "cloud" && toggle("cloud")}
            title={PROFILE_TIP}
          >
            cloud
          </button>
        </span>
        <button
          className="btn-mini runs-open"
          data-testid="runs-open"
          onClick={onOpenRuns}
        >
          runs{runsCount ? ` (${runsCount})` : ""}
        </button>
      </span>
      {pendingNote && (
        <div className="pending-note" title={pendingNote.title}>
          {pendingNote.text}
        </div>
      )}
    </div>
  );
}
