import { useEffect, useState } from "react";
import { api, ApiHttpError } from "../api";
import type { HealthResponse, SeatProfile, SeatsResponse } from "../types";

const PROFILE_TIP = "local = llama.cpp · cloud = Modal GPU";

function SeatPill({
  name,
  seat,
}: {
  name: string;
  seat?: { url: string; model: string | null; up: boolean | null };
}) {
  const up = seat?.up === true;
  const down = seat?.up === false;
  return (
    <span
      className={`seat-pill ${up ? "seat-up" : down ? "seat-down" : "seat-unknown"}`}
      title={seat ? `${seat.url}${seat.model ? ` — ${seat.model}` : ""}` : "no seat"}
    >
      <span className="seat-dot" />
      {name}: {up ? "up" : down ? "down" : "?"}
    </span>
  );
}

export function SeatBar() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [seats, setSeats] = useState<SeatsResponse | null>(null);
  const [pendingNote, setPendingNote] = useState<string | null>(null);
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
        setPendingNote(`cloud profile pending — ${e.message}`);
      } else setPendingNote(`seat switch failed: ${e instanceof Error ? e.message : e}`);
    }
  };

  const profile = seats?.profile ?? "local";
  return (
    <div className="seat-bar">
      <span className="brand">SatQuery AI</span>
      <span className="tagline">agentic remote-sensing assistant</span>
      <span className="seat-bar-right">
        {apiDown && <span className="seat-pill seat-down">API unreachable</span>}
        <SeatPill name="narrator" seat={health?.seats?.narrator} />
        <SeatPill name="canonical" seat={health?.seats?.canonical} />
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
      </span>
      {pendingNote && <div className="pending-note">{pendingNote}</div>}
    </div>
  );
}
