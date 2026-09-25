"""PRODUCT-LEVEL GROUNDING EVAL — measures the shipped tools.ground
contract end-to-end, not the bare box model.

Motivation (external review, 2026-09-26): the n=300 matched-geometry run
measured the box seat alone. The product is presence oracle (:8091
canonical_vqa) -> narrator box (:8080) -> frame/full-frame gate, so the
honest number is the gated pipeline on the same frozen items.

Per item the harness asks the presence oracle TWICE — once with the full
referring expression, once with its class-level head noun — and makes at
most ONE narrator box call (temp-0 deterministic; the same box applies to
both variants). Each variant then runs through tools.ground_decide with
its own presence answer, reproducing exactly what the shipped tool would
emit had that variant been wired in.

Pre-registered pick rule (fixed before the run):
  ship the variant with the best PRESENT-set acc@0.5 PROVIDED its
  ABSENT-set false-box rate <= 0.30; if neither variant satisfies the
  false-box bound, keep the current full-phrase gate pending redesign.
  Withheld items count as misses on the present set; an emitted box of
  any quality counts as a false box on the absent set (upper-bound
  caveat inherited from the mismatched-pair design).

Present set: the same seeded n=300 frozen referring ids as
ground_measure_local. Absent set: absent_target_probe.mismatched_pairs(150)
— same construction as the seat probes.

Usage:
  python ground_product_local.py [--n-present 300] [--n-absent 150]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SAT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(SAT / "demo"))

import ground_measure_local as g  # noqa: E402
import absent_target_probe as ap  # noqa: E402
import tools  # noqa: E402  demo/tools.py — the shipped implementation

BOX_URL = "http://127.0.0.1:8080"
PICK_FALSE_BOX_MAX = 0.30


def _presence(pq: str, img_path: Path) -> dict:
    """canonical_vqa presence call reduced to the fields ground_decide
    consumes, with the question/seat kept for the audit record."""
    for attempt in range(3):
        try:
            r = tools.canonical_vqa(pq, [img_path])
            return {
                "question": pq,
                "available": r.get("available"),
                "answer": r.get("answer"),
                "seat": r.get("seat"),
                "error": r.get("error"),
            }
        except Exception as ex:
            if attempt == 2:
                return {"question": pq, "available": False,
                        "answer": None, "seat": "127.0.0.1:8091",
                        "error": f"{type(ex).__name__}: {ex}"}
            time.sleep(10)


def _box_call(target: str, img_path: Path) -> tuple[str | None, str | None]:
    """Narrator-seat box call, matching the shipped tool's decode."""
    for attempt in range(3):
        try:
            r = tools.vqa_nonstream(
                tools.GROUND_REF_PROMPT.format(q=target),
                [img_path],
                url=BOX_URL,
                max_tokens=128,
                timeout=120,
            )
            return r.get("text") or "", None
        except Exception as ex:
            if attempt == 2:
                return None, f"{type(ex).__name__}: {ex}"
            time.sleep(10)


def run(items, absent: bool, label: str) -> tuple[list[dict], dict]:
    from PIL import Image

    dims_cache: dict[str, tuple] = {}
    recs = []
    t0 = time.time()
    for i, (eid, e, q_img) in enumerate(items, 1):
        img = e["image_id"]
        p = g.IMG_DIR / img
        if img not in dims_cache:
            dims_cache[img] = Image.open(p).size
        wh = dims_cache[img]
        target = e["question"]
        tq = time.time()
        pres = {
            m: _presence(tools._ground_presence_question(target, mode=m), p)
            for m in ("full", "head")
        }
        wants_box = any(
            (pr.get("answer") or "").startswith("yes") for pr in pres.values()
        )
        box_text, box_err = (None, None)
        if wants_box:
            box_text, box_err = _box_call(target, p)
        dt = time.time() - tq
        rec = {
            "id": eid, "image_id": img, "target": target,
            "presence_full": pres["full"], "presence_head": pres["head"],
            "box_text": (box_text or "")[:400], "box_error": box_err,
            "sec": round(dt, 1),
            "q_source_image": q_img if absent else img,
        }
        for m in ("full", "head"):
            # Reproduce the shipped gate for this variant: no box call is
            # considered to have happened unless ITS presence said yes.
            pv = pres[m]
            bv = box_text if (
                pv.get("available")
                and (pv.get("answer") or "").startswith("yes")
            ) else None
            dec = tools.ground_decide(target, pv, bv, wh)
            if absent:
                iou = None
            else:
                gt01 = g.parse_gt_box(e["ground_truth"])
                iou = g.iou_official(dec.get("box01"), gt01) if gt01 else 0.0
            rec[f"{m}_decision"] = {
                "withheld": dec.get("withheld"),
                "withheld_reason": dec.get("withheld_reason"),
                "frame_tag": dec.get("frame_tag"),
                "box01": dec.get("box01"),
                "iou": iou,
            }
        print(
            f"[{label} {i}/{len(items)}] {eid} "
            f"full={'W:' + str(rec['full_decision']['withheld_reason']) if rec['full_decision']['withheld'] else 'box iou=' + str(rec['full_decision']['iou'])} "
            f"head={'W:' + str(rec['head_decision']['withheld_reason']) if rec['head_decision']['withheld'] else 'box iou=' + str(rec['head_decision']['iou'])} "
            f"{dt:.1f}s", flush=True)
        recs.append(rec)
    return recs, {"wall_seconds": round(time.time() - t0, 1)}


def _summarize(recs: list[dict], absent: bool) -> dict:
    out = {}
    for m in ("full", "head"):
        ds = [r[f"{m}_decision"] for r in recs]
        n = len(ds)
        emitted = [d for d in ds if not d["withheld"] and d["box01"]]
        row = {
            "withheld_rate": round(sum(1 for d in ds if d["withheld"]) / n, 4),
            "box_emitted": len(emitted),
            "withhold_reasons": {
                w: sum(1 for d in ds if d["withheld_reason"] == w)
                for w in {d["withheld_reason"] for d in ds if d["withheld"]}
            },
        }
        if absent:
            row["false_box_rate"] = round(len(emitted) / n, 4)
        else:
            row["acc_iou_0.5"] = round(
                sum(1 for d in ds if (d["iou"] or 0) >= 0.5) / n, 4)
            row["mean_iou"] = round(
                sum(d["iou"] or 0 for d in ds) / n, 4)
        out[m] = row
    return out


def main():
    prs = argparse.ArgumentParser()
    prs.add_argument("--n-present", type=int, default=300)
    prs.add_argument("--n-absent", type=int, default=150)
    args = prs.parse_args()

    present = [(eid, e, e["image_id"]) for eid, e in
               g.pick_ids(args.n_present)]
    absent = [
        (eid, {**e, "question": q}, q_img)
        for eid, e, q, q_img in ap.mismatched_pairs(args.n_absent)
    ]

    present_recs, pw = run(present, absent=False, label="present")
    absent_recs, aw = run(absent, absent=True, label="absent")

    res = {
        "label": "product-ground-v1", "seat_box": BOX_URL,
        "seat_presence": "127.0.0.1:8091",
        "n_present": len(present_recs), "n_absent": len(absent_recs),
        "present": _summarize(present_recs, absent=False),
        "absent": _summarize(absent_recs, absent=True),
        "pick_rule": (
            "ship best present acc@0.5 with absent false_box_rate <= "
            f"{PICK_FALSE_BOX_MAX}; else keep full-phrase gate"),
        "frozen_ids": g.IDS_PATH.name, "gt_sha256": g.GT_SHA256,
        "sample_seed": g.SEED,
        "note": ("local Q4 llama.cpp product-level eval — shipped gate "
                 "replayed offline per presence variant; NOT the bf16 "
                 "reportable column"),
        "wall_seconds": pw["wall_seconds"] + aw["wall_seconds"],
        "preds_present": present_recs, "preds_absent": absent_recs,
    }
    g.OUT_DIR.mkdir(exist_ok=True)
    out = g.OUT_DIR / f"ground_product_{len(present_recs)}p_{len(absent_recs)}a.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in (
        "n_present", "n_absent", "present", "absent", "wall_seconds")},
        indent=2))
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
