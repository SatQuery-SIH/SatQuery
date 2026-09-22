"""REHEARSAL (s1s2_india): upload -> ingest -> plan -> tools -> packet -> narrate
on real S1/S2-like GeoTIFF pairs. All local; $0. Writes artifacts beside this
file. Run from repo root with demo/ on sys.path.

NOTE: s1_date ~= s2_date -> SAME-DATE pairs. They are never fed to bi-temporal.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

SAT = Path(__file__).resolve().parents[2]
DEMO = SAT / "demo"
sys.path.insert(0, str(DEMO))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

OUT = Path(__file__).resolve().parent
BASE = SAT / "gates" / "_cache" / "data" / "rehearsal" / "s1s2_india" / "cross_modal_dataset"


def pick_patches() -> list[dict]:
    rows = list(csv.DictReader((BASE / "metadata.csv").open(encoding="utf-8")))
    by_split: dict[str, dict] = {}
    for r in rows:
        by_split.setdefault(r["split"], {})[r["parent_scene_id"]] = r
    quota = {"train": 6, "val": 4, "test": 4}
    picked = []
    for sp, n in quota.items():
        cells = sorted(
            by_split[sp].values(),
            key=lambda r: (float(r["longitude"]) + float(r["latitude"])),
        )
        step = max(1, len(cells) // n)
        for c in cells[::step][:n]:
            picked.append(
                {
                    "patch_id": c["patch_id"],
                    "split": sp,
                    "cell": c["parent_scene_id"],
                    "lon": float(c["longitude"]),
                    "lat": float(c["latitude"]),
                    "s1": c["s1_date"],
                    "s2": c["s2_date"],
                    "tdiff_days": c["time_diff_days"],
                    "optical": str(BASE / sp / "optical" / f"{c['patch_id']}.tiff"),
                    "sar": str(BASE / sp / "sar" / f"{c['patch_id']}.tiff"),
                }
            )
    return picked


def png_stats(path) -> dict:
    p = Path(path)
    if not p.is_file():
        return {"exists": False}
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
    return {
        "exists": True,
        "shape": list(a.shape),
        "mean": round(float(a.mean()), 2),
        "std": round(float(a.std()), 2),
        "nonzero_frac": round(float((a > 0).mean()), 4),
        "black": bool(a.std() < 1.0),
    }


def ingest_checks(p: dict) -> dict:
    from pipeline import bind_inputs

    res = {"patch_id": p["patch_id"], "split": p["split"], "lon": p["lon"], "lat": p["lat"]}
    b1 = bind_inputs("single", uploads={"image": p["optical"]})
    g = b1.get("gsd") or {}
    res["single"] = {
        "ok": b1.get("ok"),
        "error": b1.get("error"),
        "gsd_source": g.get("source"),
        "gsd_m": g.get("gsd_m"),
        "crs": g.get("crs"),
        "crs_note": g.get("crs_note"),
        "preview": png_stats((b1.get("paths") or {}).get("image")),
        "note": b1.get("ingest_note"),
        "source_original": (b1.get("paths") or {}).get("source_original"),
    }
    b2 = bind_inputs("optical+sar", uploads={"optical": p["optical"], "sar": p["sar"]})
    g2 = b2.get("gsd") or {}
    sa = b2.get("sar_arrays") or {}
    vv = sa.get("vv")
    vh = sa.get("vh")
    res["sar"] = {
        "ok": b2.get("ok"),
        "error": b2.get("error"),
        "gsd_source": g2.get("source"),
        "gsd_m": g2.get("gsd_m"),
        "crs": g2.get("crs"),
        "crs_note": g2.get("crs_note"),
        "calibrated": sa.get("calibrated"),
        "vv_shape": None if vv is None else list(np.asarray(vv).shape),
        "vv_dtype": None if vv is None else str(np.asarray(vv).dtype),
        "vv_minmax": None if vv is None else [round(float(np.min(vv)), 2), round(float(np.max(vv)), 2)],
        "vh_minmax": None if vh is None else [round(float(np.min(vh)), 2), round(float(np.max(vh)), 2)],
        "vv_eq_vh": None if vv is None or vh is None else bool(np.array_equal(vv, vh)),
        "sar_preview": png_stats((b2.get("paths") or {}).get("sar_vv")),
        "optical_preview": png_stats((b2.get("paths") or {}).get("optical")),
        "misreg_fields": {k: b2.get(k) for k in ("misreg_check", "misreg_shift_px", "misregistration_note")},
        "note": b2.get("ingest_note"),
    }
    return res


def live_run(p: dict, query: str, mode: str, tag: str) -> dict:
    from pipeline import run_query

    uploads = {"image": p["optical"]} if mode == "single" else {
        "optical": p["optical"], "sar": p["sar"]
    }
    t0 = time.perf_counter()
    trace = run_query(query, mode, uploads=uploads, live=True)
    dt = round(time.perf_counter() - t0, 2)
    pkt = trace.get("evidence_packet") or {}
    tout = trace.get("tool_outputs") or {}
    rec = {
        "tag": tag,
        "patch_id": p["patch_id"],
        "split": p["split"],
        "mode": mode,
        "query": query,
        "wall_s": dt,
        "plan_tools": (trace.get("plan") or {}).get("tools"),
        "supported": (trace.get("plan") or {}).get("supported"),
        "refusal": (trace.get("plan") or {}).get("refusal"),
        "canonical_vqa": tout.get("canonical_vqa"),
        "sar_agreement_verdicts": (tout.get("sar_agreement") or {}).get("verdicts"),
        "canonical_answer": pkt.get("canonical_answer"),
        "claim_predicates": [c.get("predicate") for c in pkt.get("claims") or []],
        "limitations": pkt.get("limitations"),
        "validate_issues": None,
        "narration_check": trace.get("narration_check"),
        "narrator_url": (trace.get("vlm") or {}).get("url"),
        "answer_head": (trace.get("answer") or "")[:200],
    }
    try:
        from evidence_packet import validate_packet

        rec["validate_issues"] = validate_packet(pkt) if pkt else None
    except Exception as e:
        rec["validate_issues"] = [f"validate crashed: {e}"]
    stem = f"{tag}_{p['patch_id']}"
    (OUT / f"{stem}.packet.json").write_text(
        json.dumps(pkt, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (OUT / f"{stem}.trace.json").write_text(
        json.dumps(
            {k: v for k, v in trace.items() if k != "evidence_packet"},
            indent=2, default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return rec


def main() -> None:
    picked = pick_patches()
    (OUT / "patches.json").write_text(
        json.dumps(picked, indent=2) + "\n", encoding="utf-8"
    )
    print(f"picked {len(picked)} patches", flush=True)

    ingest = []
    for p in picked:
        ingest.append(ingest_checks(p))
        print("ingest", p["patch_id"], flush=True)
    (OUT / "ingest_checks.json").write_text(
        json.dumps(ingest, indent=2, default=str) + "\n", encoding="utf-8"
    )

    by_id = {p["patch_id"]: p for p in picked}
    runs = []
    single_q = [
        ("s1", "cell_25725_p0015", "Is there a road in this image?"),
        ("s2", "cell_11582_p0015", "Are there any buildings in the image?"),
        ("s3", "cell_30332_p0015", "How large is the water body in this image?"),
        ("s4", "cell_18886_p0015", "Is the area rural or urban?"),
        ("s5", "cell_30118_p0015", "How many buildings do you see?"),
    ]
    for tag, pid, q in single_q:
        runs.append(live_run(by_id[pid], q, "single", tag))
        print("live", tag, pid, flush=True)

    sar_q = [
        ("x1", "cell_25725_p0015", "Is there water in this scene?"),
        ("x2", "cell_07835_p0015", "Identify water-covered regions."),
        ("x3", "cell_21156_p0011", "What is the area of water in km2?"),
        ("x4", "cell_17093_p0015", "Compare optical vs SAR water evidence."),
        ("x5", "cell_48387_p0015", "Highlight the flooded region using SAR backscatter."),
        ("x6", "cell_16356_p0011", "Show me water bodies via radar."),
    ]
    for tag, pid, q in sar_q:
        runs.append(live_run(by_id[pid], q, "optical+sar", tag))
        print("live", tag, pid, flush=True)

    # --- withholding probes ---
    from planner import plan

    probes = []
    p1 = plan("How many buildings changed between the two dates?", "bi-temporal")
    probes.append({"probe": "bi-temporal count", "supported": p1["supported"],
                   "tools": p1["tools"], "refusal": p1["refusal"]})
    # mode/data mismatch: change question on single optical upload
    runs.append(live_run(by_id["cell_30439_p0011"],
                         "What changed between before and after?", "single", "p_mismatch_single"))
    # mode mismatch: change question on the optical+sar pair
    runs.append(live_run(by_id["cell_30439_p0011"],
                         "What changed between the two dates?", "optical+sar", "p_mismatch_sar"))
    # unsupported-evidence probe: object count the tools cannot produce
    runs.append(live_run(by_id["cell_32465_p0015"],
                         "How many helicopters are in this image?", "single", "p_unsupported"))

    (OUT / "probes.json").write_text(
        json.dumps(probes, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "runs_summary.json").write_text(
        json.dumps(runs, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({"runs": len(runs), "probes": len(probes)}))


if __name__ == "__main__":
    main()
