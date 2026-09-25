"""GROUND-MEASURE (local, decision-grade) — both Q4 llama.cpp seats on a
seeded subset of the frozen VRSBench referring ids.

Purpose (D-014): answer "did the LoRA break box output, and which served
seat should own the `ground` tool?" Both seats are served through
IDENTICAL llama.cpp preprocessing, so the comparison is apples-to-apples.
The absolute acc is NOT comparable to the bf16 vLLM 0.6114 column
(different quantization + backend) — the reportable bf16 number needs the
Modal run; this decides the seat.

Scoring is verbatim from scripts/modal_vrsbench_eval.py
(parse_pred_box / parse_gt_box / iou_official / iou_float / _clip01 /
_smart_resize / _resized_dims / REF_PROMPT / DECODING["referring"]) —
identical protocol, only the transport differs (llama.cpp HTTP instead of
vLLM batch).

Seat geometry (--factor/--min-px/--max-px): the <|box_start|> frame is the
*resized* image. Canonical seat runs --image-min-tokens 384 -> min pixels
~= 384 * factor^2 (llama.cpp tokenizes at patch*merge). Pass the per-seat
values; a short --probe run prints raw outputs + parse tags so a wrong
frame assumption is visible before the full pass.

Usage:
  python ground_measure_local.py --fetch --n 300        # stage images once
  python ground_measure_local.py --seat 8091 --min-px 393216 --n 300
  python ground_measure_local.py --seat 8080 --n 300
  python ground_measure_local.py --seat 8091 --probe 8 --min-px 393216
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

SAT = Path(__file__).resolve().parent.parent
IDS_PATH = SAT / "gates" / "_cache" / "eval_ids" / "vrsbench_referring_eval_ids.json"
GT_PATH = SAT / "gates" / "_cache" / "data" / "vrsbench" / "VRSBench_EVAL_referring.json"
GT_SHA256 = "fd63f7c6b77a158f4cc933a1ead88fa63aa23ebcf30f2ec9be111f3567ff1b44"
IMG_DIR = SAT / "gates" / "_cache" / "data" / "vrsbench" / "images_val"
OUT_DIR = SAT / "eval_ground_local"
SEED = 20260925

REF_PROMPT = (
    "Please provide the bounding box coordinate of the region this "
    "sentence describes: {q}"
)
DECODING = {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": 128}

VOL = "satquery-data"          # proxynanmaga holds Images_val
VOL_PREFIX = "vrsbench/Images_val"
VOL_PROFILE = "proxynanmaga"   # read source only; the job itself is local

# ------------------------------------------------- verbatim scoring copy
# from scripts/modal_vrsbench_eval.py (protocol must not drift)

_NUM_RE = r"-?\d+(?:\.\d+)?"

_BBOX_KEY_RE = re.compile(
    r'"(?:bbox_2d|bbox|bounding_box|box)"\s*:\s*\[([^\]]+)\]', re.I)
_BOX_TOK_RE = re.compile(r"<\|box_start\|>(.*?)<\|box_end\|>", re.S)
_ARR4_RE = re.compile(
    rf"[\[\(<]?\s*({_NUM_RE})\s*,\s*({_NUM_RE})\s*,\s*({_NUM_RE})\s*,\s*"
    rf"({_NUM_RE})\s*[\]\)>]?")


def _clip01(box):
    x1, y1, x2, y2 = box
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return [min(max(x1, 0.0), 1.0), min(max(y1, 0.0), 1.0),
            min(max(x2, 0.0), 1.0), min(max(y2, 0.0), 1.0)]


def parse_pred_box(text: str, img_w: int, img_h: int, res_w: int, res_h: int,
                   model_key: str = "qwen3vl8b"):
    v = None
    tag = None
    m = _BOX_TOK_RE.search(text)
    if m:
        nums = re.findall(_NUM_RE, m.group(1))
        if len(nums) >= 4:
            v = [float(x) for x in nums[:4]]
            tag = "qwen_native_px"
            box = [v[0] / res_w, v[1] / res_h, v[2] / res_w, v[3] / res_h]
            return _clip01(box), tag
    if v is None:
        m = _BBOX_KEY_RE.search(text)
        if m:
            nums = re.findall(_NUM_RE, m.group(1))
            if len(nums) >= 4:
                v = [float(x) for x in nums[:4]]
                if model_key == "qwen25vl7b":
                    tag = "bbox2d_resized_px"
                    return _clip01(
                        [v[0] / res_w, v[1] / res_h,
                         v[2] / res_w, v[3] / res_h]), tag
                tag = "bbox2d_0_1000"
                return _clip01([x / 1000 for x in v]), tag
    if v is None:
        m = _ARR4_RE.search(text)
        if m:
            v = [float(m.group(i)) for i in range(1, 5)]
            tag = "tuple4"
    if v is None:
        nums = re.findall(_NUM_RE, text)
        if len(nums) >= 4:
            v = [float(x) for x in nums[:4]]
            tag = "first4"
    if v is None:
        return None, "parse_fail"
    mx = max(abs(x) for x in v)
    if mx <= 1.5:
        gtag, box = "01", v
    elif mx <= 100.5:
        gtag, box = "100", [x / 100 for x in v]
    elif mx <= 1000.5:
        gtag, box = "1000", [x / 1000 for x in v]
    else:
        gtag = "px"
        box = [v[0] / img_w, v[1] / img_h, v[2] / img_w, v[3] / img_h]
    return _clip01(box), f"{tag}_{gtag}"


def parse_gt_box(gt: str):
    nums = re.findall(_NUM_RE, gt)
    if len(nums) < 4:
        return None
    return [float(x) / 100.0 for x in nums[:4]]  # official grid 0-100


def iou_official(pred01, gt01) -> float:
    p = [round(v * 100) for v in pred01]
    g = [round(v * 100) for v in gt01]
    ix1, iy1 = max(p[0], g[0]), max(p[1], g[1])
    ix2, iy2 = min(p[2], g[2]), min(p[3], g[3])
    inter = max(0, ix2 - ix1 + 1) * max(0, iy2 - iy1 + 1)
    a = (p[2] - p[0] + 1) * (p[3] - p[1] + 1)
    b = (g[2] - g[0] + 1) * (g[3] - g[1] + 1)
    return inter / max(a + b - inter, 1e-9)


def iou_float(pred01, gt01) -> float:
    ix1, iy1 = max(pred01[0], gt01[0]), max(pred01[1], gt01[1])
    ix2, iy2 = min(pred01[2], gt01[2]), min(pred01[3], gt01[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a = (pred01[2] - pred01[0]) * (pred01[3] - pred01[1])
    b = (gt01[2] - gt01[0]) * (gt01[3] - gt01[1])
    return inter / max(a + b - inter, 1e-9)


def _smart_resize(h: int, w: int, factor: int, min_px: int, max_px: int):
    """Qwen-VL image resize rule (matches qwen_vl_utils.smart_resize)."""
    h_bar = max(factor, round(h / factor) * factor)
    w_bar = max(factor, round(w / factor) * factor)
    if h_bar * w_bar > max_px:
        beta = math.sqrt(h * w / max_px)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_px:
        beta = math.sqrt(min_px / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return int(h_bar), int(w_bar)


def _resized_dims(img_w: int, img_h: int, factor: int, min_px: int, max_px: int):
    return _smart_resize(img_h, img_w, factor, min_px, max_px)[::-1]

# ------------------------------------------------- end verbatim copy


def load_inputs():
    ids_blob = json.loads(IDS_PATH.read_text())
    sha = hashlib.sha256(GT_PATH.read_bytes()).hexdigest()
    assert sha == GT_SHA256, f"GT sha mismatch {sha}"
    lut = {}
    for i, e in enumerate(json.loads(GT_PATH.read_text())):
        qid = e.get("question_id", i)
        lut[f"{e['image_id']}#{qid}"] = e
    return ids_blob["ids"], lut


def pick_ids(n: int):
    ids, lut = load_inputs()
    sample = random.Random(SEED).sample(ids, min(n, len(ids)))
    return [(i, lut[i]) for i in sample]


def fetch_images(needed: set[str]) -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    missing = sorted(i for i in needed if not (IMG_DIR / i).is_file())
    if not missing:
        print(f"[fetch] all {len(needed)} images already local")
        return
    os.environ.setdefault("MODAL_PROFILE", VOL_PROFILE)
    import modal

    vol = modal.Volume.from_name(VOL)
    print(f"[fetch] {len(missing)} images from {VOL_PROFILE}:{VOL}/{VOL_PREFIX}")
    t0 = time.time()
    for k, name in enumerate(missing, 1):
        data = None
        for attempt in range(4):
            try:
                data = b"".join(vol.read_file(f"{VOL_PREFIX}/{name}"))
                break
            except Exception as e:
                if attempt == 3:
                    raise
                wait = 2 ** attempt * 5
                print(f"[fetch] {name} retry {attempt+1} after "
                      f"{type(e).__name__}; wait {wait}s", flush=True)
                time.sleep(wait)
        (IMG_DIR / name).write_bytes(data)
        if k % 25 == 0 or k == len(missing):
            print(f"[fetch] {k}/{len(missing)} {time.time()-t0:.0f}s", flush=True)


def ask(port: int, img_b64: str, question: str) -> str:
    body = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": REF_PROMPT.format(q=question)},
            ],
        }],
        "temperature": DECODING["temperature"],
        "top_p": DECODING["top_p"],
        "max_tokens": DECODING["max_tokens"],
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"]


def run(port: int, items, factor: int, min_px: int, max_px: int,
        label: str) -> dict:
    from PIL import Image

    b64_cache: dict[str, str] = {}
    dims_cache: dict[str, tuple] = {}
    preds = []
    t0 = time.time()
    for i, (eid, e) in enumerate(items, 1):
        img = e["image_id"]
        if img not in b64_cache:
            p = IMG_DIR / img
            b64_cache[img] = base64.b64encode(p.read_bytes()).decode("ascii")
            w, h = Image.open(p).size
            dims_cache[img] = (w, h, *_resized_dims(w, h, factor, min_px, max_px))
        img_w, img_h, res_w, res_h = dims_cache[img]
        tq = time.time()
        raw = ""
        for attempt in range(3):
            try:
                raw = ask(port, b64_cache[img], e["question"])
                break
            except Exception as ex:
                if attempt == 2:
                    raw = f"__transport_error__ {type(ex).__name__}: {ex}"
                else:
                    time.sleep(10)
        dt = time.time() - tq
        pred01, tag = parse_pred_box(raw, img_w, img_h, res_w, res_h,
                                     model_key="qwen3vl8b")
        gt01 = parse_gt_box(e["ground_truth"])
        iou = iou_official(pred01, gt01) if pred01 and gt01 else 0.0
        fiou = iou_float(pred01, gt01) if pred01 and gt01 else 0.0
        preds.append({
            "id": eid, "raw": raw, "tag": tag, "pred01": pred01,
            "gt01": gt01, "iou": iou, "fiou": fiou, "sec": round(dt, 1),
        })
        print(f"[{label} {i}/{len(items)}] {eid} iou={iou:.2f} "
              f"tag={tag} {dt:.1f}s raw={raw[:90]!r}", flush=True)
    n = len(preds)
    tags: dict[str, int] = {}
    for p in preds:
        tags[p["tag"]] = tags.get(p["tag"], 0) + 1
    res = {
        "seat": f"127.0.0.1:{port}", "label": label, "n": n,
        "acc_iou_0.5": round(sum(p["iou"] >= 0.5 for p in preds) / n, 4),
        "acc_iou_0.7": round(sum(p["iou"] >= 0.7 for p in preds) / n, 4),
        "mean_iou": round(sum(p["iou"] for p in preds) / n, 4),
        "acc_iou_0.5_float": round(sum(p["fiou"] >= 0.5 for p in preds) / n, 4),
        "mean_iou_float": round(sum(p["fiou"] for p in preds) / n, 4),
        "parse_fail": sum(1 for p in preds if p["tag"] == "parse_fail"),
        "parse_tags": tags,
        "prompt": REF_PROMPT, "decoding": DECODING,
        "geometry": {"factor": factor, "min_px": min_px, "max_px": max_px},
        "frozen_ids": str(IDS_PATH.name), "gt_sha256": GT_SHA256,
        "sample_seed": SEED, "sample_n": n,
        "note": ("local Q4 llama.cpp decision-grade subset — NOT comparable "
                 "to the bf16 vLLM 0.6114 column"),
        "wall_seconds": round(time.time() - t0, 1),
        "preds": preds,
    }
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"ground_local_{label}_{n}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in (
        "seat", "n", "acc_iou_0.5", "acc_iou_0.7", "mean_iou",
        "acc_iou_0.5_float", "parse_fail", "parse_tags", "wall_seconds")}))
    print(f"[out] {out}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seat", type=int, default=8091, help="llama.cpp port")
    ap.add_argument("--label", default=None,
                    help="output label (default: port number)")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--probe", type=int, default=0,
                    help="run only first K ids of the sample")
    ap.add_argument("--factor", type=int, default=32)
    ap.add_argument("--min-px", type=int, default=4096)
    ap.add_argument("--max-px", type=int, default=16_777_216)
    ap.add_argument("--fetch", action="store_true",
                    help="only stage the needed images, then exit")
    args = ap.parse_args()

    items = pick_ids(args.probe or args.n)
    if args.fetch:
        fetch_images({e["image_id"] for _i, e in items})
        return
    run(args.seat, items, args.factor, args.min_px, args.max_px,
        args.label or f"seat{args.seat}")


if __name__ == "__main__":
    main()
