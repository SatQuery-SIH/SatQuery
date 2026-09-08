"""Should-do free-ask probes for FINALE-HARDEN-10 (planner only, no GPU)."""
from __future__ import annotations

import json
from pathlib import Path

from planner import plan

DEMO = Path(__file__).resolve().parent
PROBES = [
    ("single", "How many buildings are in this image?"),
    ("single", "Count the cars in this photo."),
    ("single", "Describe the land cover and major objects."),
    ("single", "Is this scene in India?"),
    ("bi-temporal", "What changed between these two dates, and where?"),
    ("bi-temporal", "How much built-up area increased, in km2?"),
    ("optical+sar", "Identify water-covered regions."),
    ("single", "Generate a 3D city model from this pair."),
    ("single", "What will the weather be tomorrow over this scene?"),
    ("optical+sar", "Area of water in km2 on this optical+SAR pair."),
]


def main() -> None:
    rows = []
    for mode, q in PROBES:
        p = plan(q, mode)
        rows.append(
            {
                "query": q,
                "input_mode": mode,
                "supported": p["supported"],
                "task": p["task"],
                "tools": p["tools"],
                "refusal": p["refusal"],
            }
        )
    out = DEMO / "traces" / "harden_probes.json"
    payload = {
        "n": len(rows),
        "buildings_refused": any(
            r["query"].lower().startswith("how many buildings") and not r["supported"] for r in rows
        ),
        "india_note": (
            "No Indian Sentinel-2 screenshot on disk; BEN Scene 3 is Lithuania. "
            "Skipped live base-model India pass (would need a new download; spec forbids extra imagery)."
        ),
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "n": payload["n"], "buildings_refused": payload["buildings_refused"]}))


if __name__ == "__main__":
    main()
