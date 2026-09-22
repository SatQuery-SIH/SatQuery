"""Post-NARR-COREG rehearsal rerun: 3 optical+sar pairs, artifacts here."""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent))  # reuse run_rehearsal helpers
from run_rehearsal import BASE, live_run  # noqa: E402


def main() -> None:
    picks = {
        "cell_25725_p0015": ("train", "Is there water in this scene?"),
        "cell_21156_p0011": ("train", "What is the area of water in km2?"),
        "cell_48387_p0015": ("val", "Highlight the flooded region using SAR backscatter."),
    }
    runs = []
    for pid, (split, q) in picks.items():
        p = {
            "patch_id": pid,
            "split": split,
            "optical": str(BASE / split / "optical" / f"{pid}.tiff"),
            "sar": str(BASE / split / "sar" / f"{pid}.tiff"),
        }
        # live_run writes packet/trace beside run_rehearsal; re-point OUT
        import run_rehearsal

        run_rehearsal.OUT = OUT
        rec = live_run(p, q, "optical+sar", "post")
        tf = json.loads((OUT / f"post_{pid}.trace.json").read_text(encoding="utf-8"))
        rec["coreg"] = (tf.get("tool_outputs") or {}).get("coreg_check")
        rec["sar_verdicts"] = (
            (tf.get("tool_outputs") or {}).get("sar_agreement") or {}
        ).get("verdicts")
        runs.append(rec)
        print("done", pid, flush=True)
    (OUT / "postfix_summary.json").write_text(
        json.dumps(runs, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps([
        {
            "patch": r["patch_id"],
            "narr_ok": (r.get("narration_check") or {}).get("ok"),
            "coreg": r.get("coreg"),
            "verdicts": r.get("sar_verdicts"),
            "validate": r.get("validate_issues"),
        }
        for r in runs
    ], indent=1))


if __name__ == "__main__":
    main()
