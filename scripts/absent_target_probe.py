"""ABSENT-TARGET PROBE — does a seat invent a box when the described
target is not in the image?

Motivation (external review, 2026-09-26): the VRSBench referring column
rewards never-refusing — every question's target exists. In the product,
"find the airport" on an image with no airport must withhold, not emit a
confident box. Canonical produced 0 parse fails / narrator 9 refusals on
the matched pass — this probe measures each seat's refusal-vs-invention
behavior when the target is actually absent.

Method: same seeded n=300 sample as ground_measure_local, but each image
is paired with a referring expression drawn from a DIFFERENT image in the
sample (deterministic offset; pairs whose image coincidentally matches
are re-offset). ~Caveat: a mismatched expression may coincidentally
describe something present (e.g. "tennis court" where another tennis
court exists) — so the box-invention rate is an UPPER BOUND, not a clean
hallucination rate. The seat-level contrast is still decision-grade.

Output: eval_ground_local/absent_probe_<label>_<n>.json with per-item
raw output, parse tag, and invented_box flag.

Usage:
  python absent_target_probe.py --seat 8080 --label narrator-base-q4 --n 150
  python absent_target_probe.py --seat 8091 --label canonical-lrfold-q4 --n 150
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ground_measure_local as g  # noqa: E402


def mismatched_pairs(n: int):
    """(image_item, question) pairs where the question's target is
    almost surely absent — question comes from a different image."""
    items = g.pick_ids(300)
    pairs = []
    for i, (eid, e) in enumerate(items):
        if len(pairs) >= n:
            break
        for off in range(150, 300):
            _jid, ej = items[(i + off) % 300]
            if ej["image_id"] != e["image_id"]:
                pairs.append((eid, e, ej["question"], ej["image_id"]))
                break
    return pairs


def run_probe(port: int, label: str, n: int) -> dict:
    from PIL import Image

    pairs = mismatched_pairs(n)
    b64_cache: dict[str, str] = {}
    recs = []
    t0 = time.time()
    for i, (eid, e, q, q_img) in enumerate(pairs, 1):
        img = e["image_id"]
        if img not in b64_cache:
            b64_cache[img] = base64.b64encode(
                (g.IMG_DIR / img).read_bytes()
            ).decode("ascii")
        tq = time.time()
        raw = ""
        for attempt in range(3):
            try:
                raw = g.ask(port, b64_cache[img], q)
                break
            except Exception as ex:
                if attempt == 2:
                    raw = f"__transport_error__ {type(ex).__name__}: {ex}"
                else:
                    time.sleep(10)
        dt = time.time() - tq
        w, h = Image.open(g.IMG_DIR / img).size
        pred01, tag = g.parse_pred_box(raw, w, h, w, h, model_key="qwen3vl8b")
        invented = pred01 is not None
        recs.append({
            "image_id": img, "id": eid, "q_source_image": q_img,
            "question": q, "raw": raw, "tag": tag,
            "invented_box": invented, "sec": round(dt, 1),
        })
        print(f"[{label} {i}/{len(pairs)}] {img}<-q:{q_img} "
              f"invented={invented} tag={tag} {dt:.1f}s raw={raw[:70]!r}",
              flush=True)
    nrec = len(recs)
    res = {
        "seat": f"127.0.0.1:{port}", "label": label, "n": nrec,
        "invented_boxes": sum(1 for r in recs if r["invented_box"]),
        "invention_rate": round(
            sum(1 for r in recs if r["invented_box"]) / nrec, 4),
        "refusals": sum(1 for r in recs if not r["invented_box"]),
        "parse_tags": {t: sum(1 for r in recs if r["tag"] == t)
                       for t in {r["tag"] for r in recs}},
        "method": ("mismatched pairing — image gets a referring "
                   "expression from a different image; invention rate is "
                   "an UPPER BOUND (coincidental matches possible)"),
        "sample_seed": g.SEED, "wall_seconds": round(time.time() - t0, 1),
        "preds": recs,
    }
    g.OUT_DIR.mkdir(exist_ok=True)
    out = g.OUT_DIR / f"absent_probe_{label}_{nrec}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in (
        "seat", "n", "invented_boxes", "invention_rate", "refusals",
        "parse_tags", "wall_seconds")}))
    print(f"[out] {out}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seat", type=int, required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()
    run_probe(args.seat, args.label or f"seat{args.seat}", args.n)


if __name__ == "__main__":
    main()
