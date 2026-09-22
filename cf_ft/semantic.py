"""SEMANTIC-FROMTO: per-timestamp 6-class SECOND segmenter + transition compiler.

Column only. No demo attach. No router change. ``built_up_direction`` stays
``not_determined``. CDVQA gold is scorer-only; the compiler never opens
label1/label2 and never reads answers.

Class RGB triples come from ``dataset.SECOND_RGB_TO_CLASS`` (captain-whu SCD /
rs15164095 Table 2). Token strings are the CDVQA gold vocabulary inspected
read-only from questions/answers *formats* (not used as compiler inputs).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from cf_ft.dataset import (
    CF_CACHE,
    CHOSEN_RULE,
    IMG_SIZE,
    SATQUERY,
    SECOND_CLASS_MAX,
    SECOND_RGB_TO_CLASS,
    binary_change_mask,
    decode_second_label,
    load_rgb,
    overlap_cdvqa_eval,
    pair_is_readable,
    sha256_file,
)

UNIQUE_IMAGES = SATQUERY / "gates" / "_cache" / "prod10" / "unique_images.json"
VAL_JSON = CF_CACHE / "second_val_ids.json"
CDVQA_EVAL = SATQUERY / "gates" / "cdvqa_eval_ids.json"
CDVQA_VAL_IMAGES = SATQUERY / "gates" / "_cache" / "prod10" / "cdvqa_raw" / "Val_images.json"
QA_DIR = SATQUERY / "gates" / "_cache" / "scoreboard" / "cdvqa_qa"
V1_PREDS = SATQUERY / "gates" / "_cache" / "scoreboard" / "cdvqa_preds.jsonl"
PROTOCOL_PATH = CF_CACHE / "semantic_protocol.json"
ENTRY_GATES_PATH = CF_CACHE / "semantic_entry_gates.json"
COST_PATH = CF_CACHE / "semantic_cost_projection.json"
SKIP_CORRUPT_TRAIN = ("08301.png",)
TEAM_SECOND_SHA = "cbfefe6564613571e824cb2cee0f971b4793ca0dda6c1a5c5741cbcdf4c76fe6"
IGNORE_INDEX = 255
NUM_CLASSES = 6  # logits 0..5 ↔ land-cover ids 1..6
BUILT_UP_CLASS_ID = 5  # buildings
SEM_LR = 1e-4
SEM_BATCH = 16
SEM_EPOCHS = 12
SEM_SEED = 42
SEM_WEIGHT_DECAY = 0.01
PRIOR_USD = 1.224286
RUN_CAP_USD = 4.0
LANE_CAP_USD = 8.0
N_BOOT = 2000
BOOT_SEED = 42

# CDVQA gold tokens (lowercase). Source: read-only Test_* format inspection.
CLASS_ID_TO_TOKEN: dict[int, str] = {
    1: "water",
    2: "nvg_surface",
    3: "low_vegetation",
    4: "trees",
    5: "buildings",
    6: "playgrounds",
}
TOKEN_TO_CLASS_ID: dict[str, int] = {v: k for k, v in CLASS_ID_TO_TOKEN.items()}

# Longest phrase first.
CLASS_PHRASES: tuple[tuple[str, int], ...] = (
    ("non-vegetated ground surface", 2),
    ("low vegetation", 3),
    ("playgrounds", 6),
    ("buildings", 5),
    ("trees", 4),
    ("water", 1),
)
T2_MARKERS: tuple[str, ...] = (
    "post-change image",
    "post-event image",
    "second image",
    "post-change",
    "post-event",
)
T1_MARKERS: tuple[str, ...] = (
    "pre-change image",
    "pre-event image",
    "first image",
    "pre-change",
    "pre-event",
)
SEMANTIC_FAMILIES: tuple[str, ...] = (
    "change_to_what",
    "change_ratio_types",
    "largest_change",
    "smallest_change",
    "increase_or_not",
    "decrease_or_not",
)

# Frozen BEFORE train. ``0`` is exact-zero; other bins are (lo, hi] of share in [0,1].
RATIO_BINS: tuple[dict[str, Any], ...] = (
    {"token": "0", "lo": 0.0, "hi": 0.0, "exact_zero": True},
    {"token": "0_to_10", "lo": 0.0, "hi": 0.10, "exact_zero": False},
    {"token": "10_to_20", "lo": 0.10, "hi": 0.20, "exact_zero": False},
    {"token": "20_to_30", "lo": 0.20, "hi": 0.30, "exact_zero": False},
    {"token": "30_to_40", "lo": 0.30, "hi": 0.40, "exact_zero": False},
    {"token": "40_to_50", "lo": 0.40, "hi": 0.50, "exact_zero": False},
    {"token": "50_to_60", "lo": 0.50, "hi": 0.60, "exact_zero": False},
    {"token": "60_to_70", "lo": 0.60, "hi": 0.70, "exact_zero": False},
    {"token": "70_to_80", "lo": 0.70, "hi": 0.80, "exact_zero": False},
    {"token": "80_to_90", "lo": 0.80, "hi": 0.90, "exact_zero": False},
    {"token": "90_to_100", "lo": 0.90, "hi": 1.00, "exact_zero": False},
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODEL_NAME = "torchvision.deeplabv3_mobilenet_v3_large"
MODEL_LICENSE = (
    "torchvision DeepLabV3-MobileNetV3-Large (BSD-3-Clause). "
    "Pretrained encoder+heads: DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1 "
    "(ImageNet backbone + COCO-VOC). Classifier head replaced for 6 SECOND classes. "
    "CDVQA QA Apache-2.0. SECOND academic SCD benchmark."
)


def refuse_label_path(path: Path | str) -> Path:
    """Compiler I/O guard: label1/label2 are scorer-only."""
    p = Path(path)
    parts = {x.lower() for x in p.parts}
    if "label1" in parts or "label2" in parts or p.name.lower() in {"label1", "label2"}:
        raise RuntimeError(f"STOP: compiler refused label path {p}")
    return p


def ids_to_train_target(class_ids: np.ndarray) -> np.ndarray:
    """Land-cover 1..6 → 0..5; white/unmatched → IGNORE_INDEX."""
    ids = np.asarray(class_ids)
    out = np.full(ids.shape, IGNORE_INDEX, dtype=np.int64)
    valid = (ids >= 1) & (ids <= SECOND_CLASS_MAX)
    out[valid] = ids[valid] - 1
    return out


def logits_to_class_ids(pred_idx: np.ndarray) -> np.ndarray:
    return np.asarray(pred_idx, dtype=np.int32) + 1


def tensorize_imagenet(arr: np.ndarray):
    import torch

    t = torch.from_numpy(np.asarray(arr).astype(np.float32) / 255.0)
    if t.ndim != 3:
        raise ValueError(f"RGB array must be HxWx3, got {t.shape}")
    t = t.permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


def _resize_rgb(arr: np.ndarray, size: int) -> np.ndarray:
    im = Image.fromarray(arr).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(im)


def _resize_ids(arr: np.ndarray, size: int) -> np.ndarray:
    im = Image.fromarray(np.asarray(arr).astype(np.uint8), mode="L")
    if im.size != (size, size):
        im = im.resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(im).astype(np.int32)


def bin_ratio(share: float, bins: Sequence[dict[str, Any]] | None = None) -> str:
    x = float(share)
    if x <= 0.0:
        return "0"
    seq = list(bins) if bins is not None else list(RATIO_BINS)
    last = "90_to_100"
    for b in seq:
        if b.get("exact_zero"):
            continue
        last = str(b["token"])
        if x <= float(b["hi"]) + 1e-12:
            return str(b["token"])
    return last


def extract_class_id(question: str | None) -> int | None:
    q = (question or "").lower()
    for phrase, cid in CLASS_PHRASES:
        if phrase in q:
            return cid
    return None


def extract_time_side(question: str | None) -> str | None:
    """'t1' / 't2' / None (unspecified)."""
    q = (question or "").lower()
    for m in T2_MARKERS:
        if m in q:
            return "t2"
    for m in T1_MARKERS:
        if m in q:
            return "t1"
    return None


def from_to_token(from_id: int, to_id: int) -> str:
    """Frozen (from,to) → answer token: the TO class name (CDVQA gold format)."""
    return CLASS_ID_TO_TOKEN.get(int(to_id), "unknown")


def features_from_maps(
    cls_a: np.ndarray,
    cls_b: np.ndarray,
    change: np.ndarray,
) -> dict[str, Any]:
    """Spatial maps → compiler features. No gold. No label paths."""
    a = np.asarray(cls_a)
    b = np.asarray(cls_b)
    m = np.asarray(change).astype(bool)
    if m.ndim == 3:
        m = m[..., 0]
    if a.shape != b.shape or a.shape[:2] != m.shape[:2]:
        raise ValueError(f"shape mismatch {a.shape} {b.shape} {m.shape}")
    n = int(a.size)
    changed = int(m.sum())
    from_to = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    from_hist = np.zeros(NUM_CLASSES, dtype=np.int64)
    to_hist = np.zeros(NUM_CLASSES, dtype=np.int64)
    count_a = np.zeros(NUM_CLASSES, dtype=np.int64)
    count_b = np.zeros(NUM_CLASSES, dtype=np.int64)
    for cid in range(1, NUM_CLASSES + 1):
        count_a[cid - 1] = int((a == cid).sum())
        count_b[cid - 1] = int((b == cid).sum())
    if changed:
        aa = a[m]
        bb = b[m]
        va = (aa >= 1) & (aa <= NUM_CLASSES)
        vb = (bb >= 1) & (bb <= NUM_CLASSES)
        both = va & vb
        if int(both.sum()):
            fi = aa[both].astype(np.int64) - 1
            ti = bb[both].astype(np.int64) - 1
            np.add.at(from_to, (fi, ti), 1)
            np.add.at(from_hist, fi, 1)
            np.add.at(to_hist, ti, 1)
    return {
        "changed_pixels": changed,
        "total_pixels": n,
        "from_to": from_to.tolist(),
        "from_hist": from_hist.tolist(),
        "to_hist": to_hist.tolist(),
        "count_a": count_a.tolist(),
        "count_b": count_b.tolist(),
    }


def _argmax_hist(hist: Sequence[int], *, reverse: bool) -> int | None:
    vals = [int(x) for x in hist]
    if reverse:
        best = min((c for c in vals if c > 0), default=None)
        if best is None:
            return None
        for i, c in enumerate(vals):
            if c == best:
                return i + 1
        return None
    best = max(vals, default=0)
    if best <= 0:
        return None
    for i, c in enumerate(vals):
        if c == best:
            return i + 1
    return None


def compile_from_features(
    feat: dict[str, Any],
    official_type: str | None,
    question: str | None,
    protocol: dict[str, Any] | None = None,
) -> tuple[str, str | None]:
    """Deterministic compiler. Features + question text only. No gold."""
    proto = protocol or {}
    t = str(official_type or "").strip()
    if feat.get("error"):
        return "", str(feat.get("error"))
    changed = int(feat.get("changed_pixels") or 0)
    from_to = np.asarray(feat.get("from_to") or np.zeros((NUM_CLASSES, NUM_CLASSES)), dtype=np.int64)
    from_hist = [int(x) for x in (feat.get("from_hist") or [0] * NUM_CLASSES)]
    to_hist = [int(x) for x in (feat.get("to_hist") or [0] * NUM_CLASSES)]
    count_a = [int(x) for x in (feat.get("count_a") or [0] * NUM_CLASSES)]
    count_b = [int(x) for x in (feat.get("count_b") or [0] * NUM_CLASSES)]
    cid = extract_class_id(question)
    side = extract_time_side(question)

    if t == "change_to_what":
        if cid is None:
            return "unknown", None
        row = from_to[cid - 1]
        if int(row.sum()) <= 0:
            return "unknown", None
        to_i = int(np.argmax(row))
        return from_to_token(cid, to_i + 1), None

    if t == "change_ratio_types":
        if cid is None:
            return "unknown", None
        if changed <= 0:
            return "0", None
        hist = to_hist if side == "t2" else from_hist
        share = float(hist[cid - 1]) / float(changed)
        return bin_ratio(share, bins=proto.get("ratio_bins")), None

    if t in ("largest_change", "smallest_change"):
        reverse = t == "smallest_change"
        if side == "t2":
            hid = _argmax_hist(to_hist, reverse=reverse)
        elif side == "t1":
            hid = _argmax_hist(from_hist, reverse=reverse)
        else:
            combo = [from_hist[i] + to_hist[i] for i in range(NUM_CLASSES)]
            hid = _argmax_hist(combo, reverse=reverse)
        if hid is None:
            return "unknown", None
        return CLASS_ID_TO_TOKEN[hid], None

    if t in ("increase_or_not", "decrease_or_not"):
        if cid is None:
            cid = int(proto.get("built_up_class_id") or BUILT_UP_CLASS_ID)
        a_n = count_a[cid - 1]
        b_n = count_b[cid - 1]
        if t == "increase_or_not":
            return ("yes" if b_n > a_n else "no"), None
        return ("yes" if b_n < a_n else "no"), None

    return "unknown", None


def compile_answer(
    official_type: str | None,
    question: str | None,
    *,
    cls_a: np.ndarray | None = None,
    cls_b: np.ndarray | None = None,
    change: np.ndarray | None = None,
    feat: dict[str, Any] | None = None,
    protocol: dict[str, Any] | None = None,
    label_path: Path | str | None = None,
) -> tuple[str, str | None]:
    if label_path is not None:
        refuse_label_path(label_path)
    if feat is None:
        if cls_a is None or cls_b is None or change is None:
            return "unknown", "missing_maps"
        feat = features_from_maps(cls_a, cls_b, change)
    return compile_from_features(feat, official_type, question, protocol=protocol)


def exact_token(pred: str | None, gold: str | None) -> int:
    return int((pred or "").strip().lower() == (gold or "").strip().lower() and bool((pred or "").strip()))


def bootstrap_ci(bits: list[int], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict[str, Any]:
    n = len(bits)
    if n == 0:
        return {"lo": None, "hi": None, "n_boot": n_boot, "seed": seed}
    rng = random.Random(seed)
    accs = []
    for _ in range(n_boot):
        s = 0
        for __ in range(n):
            s += bits[rng.randrange(n)]
        accs.append(s / n)
    accs.sort()
    lo = accs[int(0.025 * (n_boot - 1))]
    hi = accs[int(0.975 * (n_boot - 1))]
    return {"lo": lo, "hi": hi, "n_boot": n_boot, "seed": seed}


def second_a_names(unique_path: Path | None = None) -> list[str]:
    blob = json.loads((unique_path or UNIQUE_IMAGES).read_text(encoding="utf-8"))
    return sorted(
        {
            Path(str(p).replace("\\", "/")).name
            for p in (blob.get("second") or [])
            if "/A/" in str(p).replace("\\", "/") and str(p).lower().endswith(".png")
        }
    )


def frozen_val_ids() -> list[str]:
    rec = json.loads(VAL_JSON.read_text(encoding="utf-8"))
    return list(rec.get("ids") or [])


def train_ids_complement() -> tuple[list[str], dict[str, Any]]:
    a_names = second_a_names()
    val = frozen_val_ids()
    val_set = set(val)
    train = [n for n in a_names if n not in val_set]
    return train, {
        "n_second_A": len(a_names),
        "n_val": len(val),
        "n_train": len(train),
        "val_seed_file": str(VAL_JSON),
    }


def cdvqa_test_ids() -> list[str]:
    data = json.loads(CDVQA_EVAL.read_text(encoding="utf-8"))
    return sorted({Path(str(x)).name for x in (data.get("pair_ids_test_union") or [])})


def cdvqa_val_ids() -> list[str]:
    obj = json.loads(CDVQA_VAL_IMAGES.read_text(encoding="utf-8"))
    rows = obj.get("images") or obj
    if isinstance(rows, dict):
        rows = list(rows.values())
    names: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        fn = str(row.get("file_name") or row.get("filename") or "")
        if fn:
            names.append(Path(fn).name)
    return sorted(set(names))


def protocol_dict() -> dict[str, Any]:
    from_to_map = {
        f"{i}->{j}": from_to_token(i, j)
        for i in range(1, NUM_CLASSES + 1)
        for j in range(1, NUM_CLASSES + 1)
    }
    rgb = {f"{r},{g},{b}": cid for (r, g, b), cid in SECOND_RGB_TO_CLASS.items()}
    return {
        "task": "SEMANTIC-FROMTO",
        "frozen_before_train": True,
        "chosen_rule": CHOSEN_RULE,
        "class_name_source": (
            "RGB triples: cf_ft/dataset.py SECOND_RGB_TO_CLASS "
            "(captain-whu SCD / rs15164095 Table 2). "
            "Token strings: CDVQA gold vocabulary from read-only Test_questions/"
            "Test_answers *format* inspection (not compiler inputs). "
            "id4 token is 'trees' (CDVQA), not 'tree'."
        ),
        "rgb_to_class": rgb,
        "class_id_to_token": {str(k): v for k, v in CLASS_ID_TO_TOKEN.items()},
        "built_up_class_id": BUILT_UP_CLASS_ID,
        "built_up_token": CLASS_ID_TO_TOKEN[BUILT_UP_CLASS_ID],
        "ignore_index": IGNORE_INDEX,
        "white_is_ignore": True,
        "ratio_bins": list(RATIO_BINS),
        "from_to_token_map": from_to_map,
        "from_to_token_rule": "answer token = TO class name (CDVQA change_to_what gold is a single class)",
        "compiler_rules": {
            "change_to_what": (
                "parse FROM class from the question; among predicted-change pixels "
                "with cls_a==FROM, emit majority cls_b token"
            ),
            "change_ratio_types": (
                "parse class + time side; share = hist[class] / changed_pixels "
                "(from_hist if t1/unspecified, to_hist if t2); bin with RATIO_BINS; "
                "no change → '0'"
            ),
            "largest_change": "argmax class area among changed pixels (t1=from, t2=to, else from+to); ties → lower id",
            "smallest_change": "argmin among classes with count>0; same side rule",
            "increase_or_not": (
                "whole-image predicted class counts B>A → yes. "
                "Class from question; fallback frozen built_up_class_id"
            ),
            "decrease_or_not": "whole-image predicted class counts B<A → yes",
        },
        "where_mask": "team_second binary prediction; transitions only under predicted change",
        "no_new_binary_head": True,
        "built_up_direction": "not_determined",
        "approved_for_demo": False,
        "column_not_attach": True,
        "labels_scorer_only": True,
    }


def project_semantic_cost(
    prior_usd: float = PRIOR_USD,
    run_cap_usd: float = RUN_CAP_USD,
    lane_cap_usd: float = LANE_CAP_USD,
    usd_per_hour: float = 1.95,
    n_train: int = 1152,
    n_val: int = 127,
    n_cdvqa: int = 968,
    epochs: int = SEM_EPOCHS,
    seconds_per_train_image: float = 0.04,
    seconds_per_val_image: float = 0.02,
    seconds_per_cf_pair: float = 0.15,
    safety: float = 1.6,
) -> dict[str, Any]:
    """Conservative upper bound written BEFORE GPU. Not a bid."""
    train_images = n_train * 2 * epochs
    val_images = n_val * 2 * epochs
    train_s = train_images * seconds_per_train_image
    val_s = val_images * seconds_per_val_image
    cdvqa_s = n_cdvqa * (seconds_per_cf_pair + 2 * seconds_per_val_image)
    raw = train_s + val_s + cdvqa_s
    projected_s = raw * safety
    new_usd = projected_s / 3600.0 * usd_per_hour
    rec = {
        "method": "conservative_mobilenet_deeplab_plus_team_second_cdvqa",
        "epochs": epochs,
        "n_train": n_train,
        "n_val": n_val,
        "n_cdvqa": n_cdvqa,
        "usd_per_hour": usd_per_hour,
        "gpu_preference": "L40S then A100-40GB",
        "seconds_per_train_image": seconds_per_train_image,
        "safety": safety,
        "projected_seconds": round(projected_s, 3),
        "projected_new_usd": round(new_usd, 6),
        "prior_usd": prior_usd,
        "projected_cumulative_usd": round(prior_usd + new_usd, 6),
        "run_cap_usd": run_cap_usd,
        "lane_cap_usd": lane_cap_usd,
        "fits_run_cap": new_usd <= run_cap_usd + 1e-9,
        "fits_lane_cap": (prior_usd + new_usd) <= lane_cap_usd + 1e-9,
    }
    rec["fits_cap"] = bool(rec["fits_run_cap"] and rec["fits_lane_cap"])
    return rec


def write_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    rec = protocol_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    rec["path"] = str(path)
    rec["sha256"] = sha256_file(path)
    return rec


def write_entry_gates(path: Path = ENTRY_GATES_PATH) -> dict[str, Any]:
    train, meta = train_ids_complement()
    val = frozen_val_ids()
    errors: list[str] = []
    if meta["n_second_A"] != 1280:
        errors.append(f"n_second_A={meta['n_second_A']} != 1280")
    if meta["n_val"] != 128 or len(val) != 128:
        errors.append(f"n_val={meta['n_val']} len={len(val)} != 128")
    if meta["n_train"] != 1152 or len(train) != 1152:
        errors.append(f"n_train={meta['n_train']} != 1152")
    if CHOSEN_RULE != "rgb6_decode_unequal":
        errors.append(f"CHOSEN_RULE={CHOSEN_RULE!r}")
    test_ids = cdvqa_test_ids()
    val_cdvqa = cdvqa_val_ids()
    ov_test = sorted(set(train) & set(test_ids))
    ov_cdvqa_val = sorted(set(train) & set(val_cdvqa))
    ov_our_val = sorted(set(train) & set(val))
    eval_obj = json.loads(CDVQA_EVAL.read_text(encoding="utf-8"))
    ov_eval = overlap_cdvqa_eval(train, eval_obj)
    if ov_test or ov_cdvqa_val or ov_our_val or ov_eval:
        errors.append(
            f"overlap train∩cdvqa_test={ov_test[:8]} "
            f"train∩cdvqa_val={ov_cdvqa_val[:8]} train∩our_val={ov_our_val[:8]} "
            f"overlap_cdvqa_eval={ov_eval[:8]}"
        )
    skipped = [n for n in SKIP_CORRUPT_TRAIN if n in train]
    ids_blob = "\n".join(train).encode("utf-8")
    proj = project_semantic_cost()
    if not proj["fits_cap"]:
        errors.append(
            f"projection does not fit cap new={proj['projected_new_usd']} "
            f"cum={proj['projected_cumulative_usd']}"
        )
    rec = {
        "task": "SEMANTIC-FROMTO",
        "ok": len(errors) == 0,
        "errors": errors,
        "n_train": len(train),
        "n_val": len(val),
        "n_complete": meta["n_second_A"],
        "train_ids_sha256": hashlib.sha256(ids_blob).hexdigest(),
        "val_ids_unchanged": True,
        "skip_corrupt_train": skipped,
        "skip_corrupt_documented": list(SKIP_CORRUPT_TRAIN),
        "disjoint_cdvqa_test": ov_test == [],
        "disjoint_cdvqa_val": ov_cdvqa_val == [],
        "disjoint_our_val": ov_our_val == [],
        "overlap_cdvqa_eval": len(ov_eval),
        "overlap_cdvqa_test": ov_test,
        "overlap_cdvqa_val": ov_cdvqa_val,
        "overlap_our_val": ov_our_val,
        "n_cdvqa_test": len(test_ids),
        "n_cdvqa_val": len(val_cdvqa),
        "chosen_rule": CHOSEN_RULE,
        "projection": proj,
        "meta": meta,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    COST_PATH.write_text(json.dumps(proj, indent=2) + "\n", encoding="utf-8")
    rec["path"] = str(path)
    rec["train_ids"] = train
    rec["val_ids"] = val
    return rec


def semantic_config(imported_note: str = "none") -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "license": MODEL_LICENSE,
        "num_classes": NUM_CLASSES,
        "ignore_index": IGNORE_INDEX,
        "optimizer": "AdamW",
        "lr": SEM_LR,
        "weight_decay": SEM_WEIGHT_DECAY,
        "betas": [0.9, 0.999],
        "batch": SEM_BATCH,
        "epochs": SEM_EPOCHS,
        "seed": SEM_SEED,
        "img_size": IMG_SIZE,
        "norm": "imagenet_mean_std",
        "loss": "CE-ignore",
        "white": "ignore",
        "init": "COCO_WITH_VOC_LABELS_V1 then 6-class head",
        "imported_note": imported_note,
        "approved_for_demo": False,
        "built_up_direction": "not_determined",
        "where_ckpt_sha256": TEAM_SECOND_SHA,
    }


def config_hash(cfg: dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def build_model(pretrained: bool = True):
    """Compact DeepLabV3-MobileNetV3-Large, 6-class head. ≤~50M params."""
    import torch.nn as nn
    from torchvision.models.segmentation import (
        DeepLabV3_MobileNet_V3_Large_Weights,
        deeplabv3_mobilenet_v3_large,
    )

    if pretrained:
        weights = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
        model = deeplabv3_mobilenet_v3_large(weights=weights)
        last = model.classifier[-1]
        model.classifier[-1] = nn.Conv2d(last.in_channels, NUM_CLASSES, kernel_size=1)
        if model.aux_classifier is not None:
            aux_last = model.aux_classifier[-1]
            model.aux_classifier[-1] = nn.Conv2d(aux_last.in_channels, NUM_CLASSES, kernel_size=1)
    else:
        model = deeplabv3_mobilenet_v3_large(weights=None, num_classes=NUM_CLASSES)
    n = count_params(model)
    if n > 50_000_000:
        raise RuntimeError(f"model params {n} exceed ~50M laptop-forwardable bound")
    return model


class SecondSemanticDataset:
    """One timestamp per item (A+label1 and B+label2). Compiler never uses this."""

    def __init__(
        self,
        root: str | Path,
        ids: Sequence[str],
        img_size: int = IMG_SIZE,
        skip: Iterable[str] = (),
    ) -> None:
        self.root = Path(root)
        self.img_size = int(img_size)
        skip_set = set(skip)
        items: list[tuple[str, str]] = []
        for name in ids:
            if name in skip_set:
                continue
            items.append((name, "A"))
            items.append((name, "B"))
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        name, side = self.items[index]
        img_split = "A" if side == "A" else "B"
        lab_split = "label1" if side == "A" else "label2"
        rgb = _resize_rgb(load_rgb(self.root / img_split / name), self.img_size)
        ids = _resize_ids(decode_second_label(self.root / lab_split / name), self.img_size)
        target = ids_to_train_target(ids)
        return {
            "image": tensorize_imagenet(rgb),
            "target": torch.from_numpy(target),
            "name": name,
            "side": side,
        }


def collate_semantic(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    return {
        "image": torch.stack([x["image"] for x in batch], dim=0),
        "target": torch.stack([x["target"] for x in batch], dim=0),
        "name": [x["name"] for x in batch],
        "side": [x["side"] for x in batch],
    }


def confusion_add(conf: np.ndarray, pred: np.ndarray, target: np.ndarray) -> None:
    p = np.asarray(pred).reshape(-1)
    t = np.asarray(target).reshape(-1)
    valid = (t >= 0) & (t < NUM_CLASSES)
    p = p[valid]
    t = t[valid]
    ok = (p >= 0) & (p < NUM_CLASSES)
    p = p[ok]
    t = t[ok]
    np.add.at(conf, (t, p), 1)


def iou_from_conf(conf: np.ndarray) -> dict[str, Any]:
    per = []
    for c in range(NUM_CLASSES):
        tp = float(conf[c, c])
        fp = float(conf[:, c].sum() - tp)
        fn = float(conf[c, :].sum() - tp)
        den = tp + fp + fn
        per.append(None if den <= 0 else tp / den)
    present = [x for x in per if x is not None]
    return {
        "per_class_iou": per,
        "per_class_token": [CLASS_ID_TO_TOKEN[i + 1] for i in range(NUM_CLASSES)],
        "mIoU": (sum(present) / len(present)) if present else 0.0,
        "n_classes_present": len(present),
    }


def majority_baseline_conf(targets: list[np.ndarray], majority_idx: int) -> dict[str, Any]:
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t in targets:
        pred = np.full(t.shape, int(majority_idx), dtype=np.int64)
        confusion_add(conf, pred, t)
    return iou_from_conf(conf)


def ce_ignore_loss(out, target, ignore_index: int = IGNORE_INDEX):
    import torch.nn.functional as F

    logits = out["out"] if isinstance(out, dict) else out
    loss = F.cross_entropy(logits, target, ignore_index=ignore_index)
    if isinstance(out, dict) and out.get("aux") is not None:
        loss = loss + 0.4 * F.cross_entropy(out["aux"], target, ignore_index=ignore_index)
    return loss


def _resize_ids_hw(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    im = Image.fromarray(np.asarray(arr).astype(np.uint8), mode="L")
    if im.size != (w, h):
        im = im.resize((w, h), Image.Resampling.NEAREST)
    return np.asarray(im).astype(np.int32)


def predict_class_map(model, rgb: np.ndarray, device: str, img_size: int = IMG_SIZE) -> np.ndarray:
    import torch

    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    x = tensorize_imagenet(_resize_rgb(rgb, img_size)).unsqueeze(0)
    if device == "cuda":
        x = x.cuda()
    model.eval()
    with torch.no_grad():
        out = model(x)
        logits = out["out"] if isinstance(out, dict) else out
        idx = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    ids = logits_to_class_ids(idx)
    if ids.shape != (h, w):
        ids = _resize_ids_hw(ids, h, w)
    return ids.astype(np.int32)


def train_majority_from_root(root: Path, train_ids: Sequence[str], skip: Iterable[str] = ()) -> dict[str, Any]:
    """TRAIN labels only. Frozen before CDVQA scoring. Not a compiler input at eval."""
    skip_set = set(skip)
    class_hist = np.zeros(NUM_CLASSES, dtype=np.int64)
    from_to = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    n_pairs = 0
    for name in train_ids:
        if name in skip_set:
            continue
        ok, _err = pair_is_readable(root, name)
        if not ok:
            continue
        l1 = decode_second_label(root / "label1" / name)
        l2 = decode_second_label(root / "label2" / name)
        change = binary_change_mask(l1, l2, rule="rgb6_decode_unequal").astype(bool)
        for lab in (l1, l2):
            for cid in range(1, NUM_CLASSES + 1):
                class_hist[cid - 1] += int((lab == cid).sum())
        if change.any():
            aa = l1[change]
            bb = l2[change]
            both = (aa >= 1) & (aa <= NUM_CLASSES) & (bb >= 1) & (bb <= NUM_CLASSES)
            if both.any():
                np.add.at(from_to, (aa[both] - 1, bb[both] - 1), 1)
        n_pairs += 1
    maj_i = int(np.argmax(class_hist))
    ft = from_to.copy()
    np.fill_diagonal(ft, 0)
    if int(ft.sum()) > 0:
        flat = int(np.argmax(ft))
        fr, to = divmod(flat, NUM_CLASSES)
    else:
        fr, to = 0, 0
    return {
        "n_pairs": n_pairs,
        "class_hist": class_hist.tolist(),
        "majority_class_id": maj_i + 1,
        "majority_token": CLASS_ID_TO_TOKEN[maj_i + 1],
        "majority_transition_from_id": fr + 1,
        "majority_transition_to_id": to + 1,
        "majority_transition_token": from_to_token(fr + 1, to + 1),
        "from_to": from_to.tolist(),
        "note": "TRAIN labels only; frozen before CDVQA scoring; not used by the compiler at inference",
    }


def evaluate_semantic(
    model,
    root: Path,
    ids: Sequence[str],
    device: str,
    skip: Iterable[str] = (),
) -> dict[str, Any]:
    import torch

    model.eval()
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    n = 0
    skip_set = set(skip)
    for name in ids:
        if name in skip_set:
            continue
        ok, err = pair_is_readable(root, name)
        if not ok:
            continue
        for split, lab in (("A", "label1"), ("B", "label2")):
            rgb = load_rgb(root / split / name)
            target_ids = decode_second_label(root / lab / name)
            if target_ids.shape[:2] != rgb.shape[:2]:
                target_ids = _resize_ids_hw(target_ids, int(rgb.shape[0]), int(rgb.shape[1]))
            pred = predict_class_map(model, rgb, device=device)
            t = ids_to_train_target(target_ids)
            p = ids_to_train_target(pred)
            p = np.where(p == IGNORE_INDEX, 0, p)
            confusion_add(conf, p, t)
            n += 1
    metrics = iou_from_conf(conf)
    metrics["n_maps"] = n
    metrics["confusion"] = conf.tolist()
    return metrics


def run_semantic_train(
    root: Path,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    out_dir: Path,
    device: str = "cuda",
    epochs: int = SEM_EPOCHS,
    batch: int = SEM_BATCH,
    lr: float = SEM_LR,
    seed: int = SEM_SEED,
    pretrained: bool = True,
    skip: Iterable[str] = SKIP_CORRUPT_TRAIN,
    prior_usd: float = PRIOR_USD,
    usd_per_hour: float = 1.95,
    commit_fn=None,
) -> dict[str, Any]:
    import time

    import torch
    from torch.utils.data import DataLoader

    t0 = time.time()
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    majority = train_majority_from_root(root, train_ids, skip=skip)
    ds = SecondSemanticDataset(root, train_ids, skip=skip)
    loader = DataLoader(
        ds,
        batch_size=int(batch),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_semantic,
        drop_last=False,
    )
    model = build_model(pretrained=pretrained)
    n_params = count_params(model)
    if device == "cuda":
        model = model.cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=SEM_WEIGHT_DECAY)
    cfg = semantic_config()
    cfg_hash = config_hash(cfg)
    epoch_rows: list[dict[str, Any]] = []
    best_miou = -1.0
    best_path = out_dir / "best_semantic.pt"
    gpu_name = None
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    for epoch in range(1, int(epochs) + 1):
        model.train()
        e0 = time.time()
        losses = []
        for batch_d in loader:
            img = batch_d["image"]
            tgt = batch_d["target"]
            if device == "cuda":
                img = img.cuda()
                tgt = tgt.cuda()
            opt.zero_grad(set_to_none=True)
            out = model(img)
            loss = ce_ignore_loss(out, tgt)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_m = evaluate_semantic(model, root, val_ids, device=device, skip=skip)
        row = {
            "epoch": epoch,
            "loss_mean": round(float(sum(losses) / max(len(losses), 1)), 6),
            "val_mIoU": round(float(val_m["mIoU"]), 6),
            "val_per_class_iou": val_m["per_class_iou"],
            "seconds": round(time.time() - e0, 3),
            "cumulative_seconds": round(time.time() - t0, 3),
        }
        epoch_rows.append(row)
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "val_mIoU": val_m["mIoU"],
            "config": cfg,
            "config_hash": cfg_hash,
            "n_params": n_params,
        }
        ep_path = out_dir / f"semantic_epoch_{epoch:03d}.pt"
        torch.save(ckpt, ep_path)
        if float(val_m["mIoU"]) > best_miou:
            best_miou = float(val_m["mIoU"])
            torch.save(ckpt, best_path)
        print(json.dumps({"epoch": epoch, "val_mIoU": row["val_mIoU"], "loss": row["loss_mean"]}), flush=True)
        if commit_fn is not None:
            commit_fn()

    seconds = time.time() - t0
    usd = round(seconds / 3600.0 * float(usd_per_hour), 4)
    final = evaluate_semantic(model, root, val_ids, device=device, skip=skip)
    maj_idx = int(majority["majority_class_id"]) - 1
    # Majority baseline on the same val maps (targets only).
    base_targets = []
    skip_set = set(skip)
    for name in val_ids:
        if name in skip_set:
            continue
        ok, _err = pair_is_readable(root, name)
        if not ok:
            continue
        for lab in ("label1", "label2"):
            ids = decode_second_label(root / lab / name)
            base_targets.append(ids_to_train_target(ids))
    baseline = majority_baseline_conf(base_targets, maj_idx)
    rec = {
        "ok": True,
        "error": None,
        "approved_for_demo": False,
        "built_up_direction": "not_determined",
        "n_params": n_params,
        "model": MODEL_NAME,
        "license": MODEL_LICENSE,
        "config": cfg,
        "config_hash": cfg_hash,
        "epochs": int(epochs),
        "epoch_rows": epoch_rows,
        "best_val_mIoU": best_miou,
        "final_val_mIoU": float(final["mIoU"]),
        "final_per_class_iou": final["per_class_iou"],
        "final_per_class_token": final["per_class_token"],
        "majority_baseline": majority,
        "majority_baseline_val_mIoU": float(baseline["mIoU"]),
        "majority_baseline_per_class_iou": baseline["per_class_iou"],
        "best_ckpt": str(best_path),
        "seconds": round(seconds, 3),
        "usd": usd,
        "usd_per_hour": usd_per_hour,
        "prior_usd": prior_usd,
        "cumulative_lane_usd": round(prior_usd + usd, 6),
        "gpu": gpu_name,
        "n_train_items": len(ds),
        "n_val_maps": final.get("n_maps"),
    }
    return rec


def load_semantic_model(ckpt_path: Path, device: str = "cpu", pretrained: bool = True):
    """Load a trained 6-class DeepLab. ``pretrained=True`` matches the train graph (aux head)."""
    import torch

    model = build_model(pretrained=pretrained)
    blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(state, strict=True)
    model.eval()
    if device == "cuda":
        model = model.cuda()
    return model, blob if isinstance(blob, dict) else {}


QA_QUESTION_FILES = (
    ("test1", "Test_questions.json", "Test_images.json"),
    ("test2", "Test2_questions.json", "Test2_images.json"),
)
QA_ANSWER_FILES = {
    "test1": "Test_answers.json",
    "test2": "Test2_answers.json",
}


def iter_cdvqa_questions(qa_dir: Path | None = None) -> list[dict[str, Any]]:
    """Questions + pair ids only. Does not load answers."""
    d = qa_dir or QA_DIR
    rows: list[dict[str, Any]] = []
    for split, qn, imn in QA_QUESTION_FILES:
        qs = json.loads((d / qn).read_text(encoding="utf-8"))
        ims = json.loads((d / imn).read_text(encoding="utf-8"))
        imgs = {int(im["id"]): Path(str(im["file_name"])).name for im in ims["images"]}
        for q in qs["questions"]:
            qid = int(q["id"])
            rows.append(
                {
                    "qa_id": f"{split}:{qid}",
                    "split": split,
                    "pair_id": imgs.get(int(q["img_id"])),
                    "official_type": str(q.get("type") or ""),
                    "question": str(q.get("question") or ""),
                }
            )
    return rows


def load_cdvqa_gold(qa_dir: Path | None = None) -> dict[str, str]:
    """Scorer-only."""
    d = qa_dir or QA_DIR
    gold: dict[str, str] = {}
    for split, fname in QA_ANSWER_FILES.items():
        blob = json.loads((d / fname).read_text(encoding="utf-8"))
        for a in blob["answers"]:
            gold[f"{split}:{int(a['question_id'])}"] = str(a.get("answer") or "")
    return gold


def compile_preds_from_features(
    questions: list[dict[str, Any]],
    feat_by_pair: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for q in questions:
        pid = str(q.get("pair_id") or "")
        feat = feat_by_pair.get(pid) or {"error": f"missing_pair_features:{pid}"}
        pred, err = compile_from_features(
            feat, q.get("official_type"), q.get("question"), protocol=protocol
        )
        if feat.get("error") and not err:
            err = str(feat.get("error"))
            pred = ""
        out.append(
            {
                "qa_id": q.get("qa_id"),
                "pair_id": pid,
                "official_type": q.get("official_type"),
                "pred": pred,
                "error": err,
                "supported": str(q.get("official_type") or "") in SEMANTIC_FAMILIES,
            }
        )
    return out


def score_preds_against_gold(
    preds: list[dict[str, Any]],
    gold_by_qa: dict[str, str],
    families: Sequence[str] = SEMANTIC_FAMILIES,
) -> dict[str, Any]:
    """Scorer-only. Gold never entered the compiler."""
    by: dict[str, list[int]] = {f: [] for f in families}
    all_bits: list[int] = []
    for row in preds:
        t = str(row.get("official_type") or "")
        if t not in by:
            continue
        qa = str(row.get("qa_id") or "")
        gold = gold_by_qa.get(qa)
        bit = exact_token(row.get("pred"), gold) if not row.get("error") else 0
        by[t].append(bit)
        all_bits.append(bit)
    families_out = {}
    for t, bits in by.items():
        n = len(bits)
        acc = (sum(bits) / n) if n else 0.0
        families_out[t] = {
            "n": n,
            "correct": int(sum(bits)),
            "accuracy": acc,
            "ci95": bootstrap_ci(bits),
        }
    n = len(all_bits)
    combined = {
        "n": n,
        "correct": int(sum(all_bits)),
        "accuracy": (sum(all_bits) / n) if n else 0.0,
        "ci95": bootstrap_ci(all_bits),
        "note": "disclosed micro-average over the semantic families only; not a full CDVQA headline",
    }
    return {"families": families_out, "combined": combined}


def baseline_unknown_bits(n: int) -> list[int]:
    return [0] * int(n)


def baseline_majority_pred(official_type: str, majority: dict[str, Any]) -> str:
    t = str(official_type or "")
    tok = str(majority.get("majority_transition_token") or majority.get("majority_token") or "nvg_surface")
    if t in ("change_to_what", "largest_change", "smallest_change"):
        return tok
    if t == "change_ratio_types":
        return "0"
    if t in ("increase_or_not", "decrease_or_not"):
        return "no"
    return "unknown"


# --- SEMANTIC-ATTACH (additive): train-stat ratio re-bin + live helpers ---

FITLIST_PATH = SATQUERY / "gates" / "_cache" / "scoreboard" / "cdvqa_threshold_fitlist.json"
RATIO_BINS_V2_PATH = CF_CACHE / "semantic_ratio_bins_v2.json"
TRAIN_SHARES_PATH = CF_CACHE / "semantic_train_shares.json"
SCORES_SEM_V1_PATH = CF_CACHE / "cdvqa_scores_sem.json"
SCORES_SEM_V2_PATH = CF_CACHE / "cdvqa_scores_sem_v2.json"
PREDS_SEM_V1_PATH = CF_CACHE / "cdvqa_preds_sem.jsonl"
PREDS_SEM_V2_PATH = CF_CACHE / "cdvqa_preds_sem_v2.jsonl"
PAIR_FEATURES_PATH = CF_CACHE / "semantic" / "cdvqa_pair_features.jsonl"
SECOND_SEMANTIC_SHA = "438cac09be7c630254a12278550b64f86254ecc131ee0cdc723fd526210764d8"
BUILT_UP_FRAC_THRESH = 0.005
RATIO_BIN_TOKENS = (
    "0",
    "0_to_10",
    "10_to_20",
    "20_to_30",
    "30_to_40",
    "40_to_50",
    "50_to_60",
    "60_to_70",
    "70_to_80",
    "80_to_90",
    "90_to_100",
)


def percentile_linear(values: Sequence[float], q: float) -> float:
    """Numpy-default linear percentile (C=1): idx = q/100 * (n-1)."""
    xs = sorted(float(v) for v in values)
    if not xs:
        raise ValueError("empty percentile input")
    if q < 0.0 or q > 100.0:
        raise ValueError(f"percentile q out of range: {q}")
    if len(xs) == 1:
        return xs[0]
    idx = (q / 100.0) * (len(xs) - 1)
    lo = int(math.floor(idx))
    hi = min(lo + 1, len(xs) - 1)
    w = idx - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def ids_canonical_sha256(ids: Sequence[str]) -> str:
    blob = json.dumps(list(ids), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_fitlist(path: Path | None = None) -> dict[str, Any]:
    p = Path(path or FITLIST_PATH)
    rec = json.loads(p.read_text(encoding="utf-8"))
    ids = list(rec.get("ids") or [])
    if len(ids) != 1152:
        raise RuntimeError(f"STOP: fitlist n={len(ids)} != 1152")
    return rec


def assert_fitlist_disjoint(ids: Sequence[str] | None = None) -> dict[str, Any]:
    """Re-assert train fit ids disjoint from CDVQA test/val and our frozen val."""
    fit = list(ids) if ids is not None else list(load_fitlist()["ids"])
    test_ids = cdvqa_test_ids()
    val_cdvqa = cdvqa_val_ids()
    our_val = frozen_val_ids()
    ov_test = sorted(set(fit) & set(test_ids))
    ov_cdvqa_val = sorted(set(fit) & set(val_cdvqa))
    ov_our_val = sorted(set(fit) & set(our_val))
    errors: list[str] = []
    if ov_test:
        errors.append(f"fit∩cdvqa_test={ov_test[:8]}")
    if ov_cdvqa_val:
        errors.append(f"fit∩cdvqa_val={ov_cdvqa_val[:8]}")
    if ov_our_val:
        errors.append(f"fit∩our_val={ov_our_val[:8]}")
    if "02524.png" in set(fit):
        errors.append("scene4 id 02524.png leaked into fit")
    rec = {
        "ok": len(errors) == 0,
        "errors": errors,
        "n_fit": len(fit),
        "n_cdvqa_test": len(test_ids),
        "n_cdvqa_val": len(val_cdvqa),
        "n_our_val": len(our_val),
        "disjoint_cdvqa_test": ov_test == [],
        "disjoint_cdvqa_val": ov_cdvqa_val == [],
        "disjoint_our_val": ov_our_val == [],
        "overlap_cdvqa_test": ov_test,
        "overlap_cdvqa_val": ov_cdvqa_val,
        "overlap_our_val": ov_our_val,
        "ids_sha256": ids_canonical_sha256(fit),
    }
    if errors:
        raise RuntimeError("STOP: fitlist leakage " + "; ".join(errors))
    return rec


def dominant_transition_from_from_to(from_to: Any) -> dict[str, Any]:
    ft = np.asarray(from_to, dtype=np.int64)
    if ft.size != NUM_CLASSES * NUM_CLASSES:
        ft = ft.reshape(NUM_CLASSES, NUM_CLASSES)
    if int(ft.sum()) <= 0:
        return {
            "from_id": None,
            "to_id": None,
            "from_token": None,
            "to_token": None,
            "count": 0,
            "token": None,
        }
    flat = int(np.argmax(ft))
    fr, to = divmod(flat, NUM_CLASSES)
    return {
        "from_id": fr + 1,
        "to_id": to + 1,
        "from_token": CLASS_ID_TO_TOKEN[fr + 1],
        "to_token": CLASS_ID_TO_TOKEN[to + 1],
        "count": int(ft[fr, to]),
        "token": from_to_token(fr + 1, to + 1),
    }


def dominant_transition_share_from_labels(l1: np.ndarray, l2: np.ndarray) -> dict[str, Any]:
    """TRAIN gold only. Share = max from_to cell / changed_pixels (rgb6)."""
    from cf_ft.dataset import binary_change_mask

    change = binary_change_mask(l1, l2, rule="rgb6_decode_unequal")
    feat = features_from_maps(l1, l2, change)
    changed = int(feat["changed_pixels"])
    dom = dominant_transition_from_from_to(feat["from_to"])
    share = (float(dom["count"]) / float(changed)) if changed else 0.0
    return {
        **feat,
        **{f"dominant_{k}": v for k, v in dom.items()},
        "dominant_share": share,
    }


def histogram_deciles(pos: Sequence[float]) -> dict[str, float]:
    out: dict[str, float] = {
        "p0_min": round(min(pos), 6),
        "p5": round(percentile_linear(pos, 5.0), 6),
    }
    for q in range(10, 100, 10):
        out[f"p{q}"] = round(percentile_linear(pos, float(q)), 6)
    out["p100_max"] = round(max(pos), 6)
    return out


def ratio_bins_from_deciles(deciles: dict[str, float]) -> list[dict[str, Any]]:
    """Token 0 stays exact-zero; other 10 names keep, edges at p10..p90."""
    edges = [float(deciles[f"p{q}"]) for q in range(10, 100, 10)]
    bins: list[dict[str, Any]] = [
        {"token": "0", "lo": 0.0, "hi": 0.0, "exact_zero": True},
    ]
    los = [0.0] + edges
    his = edges + [1.0]
    tokens = [t for t in RATIO_BIN_TOKENS if t != "0"]
    for tok, lo, hi in zip(tokens, los, his):
        bins.append(
            {
                "token": tok,
                "lo": float(lo),
                "hi": float(hi),
                "exact_zero": False,
            }
        )
    return bins


def freeze_ratio_bins_v2(
    stats: dict[str, Any],
    dest: Path | None = None,
    fitlist: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write semantic_ratio_bins_v2.json BEFORE any CDVQA rescore. TRAIN stats only."""
    fit = fitlist or load_fitlist()
    ids = list(fit["ids"])
    disjoint = assert_fitlist_disjoint(ids)
    allow = set(ids)
    rows = list(stats.get("rows") or [])
    extra = [r.get("id") for r in rows if r.get("id") not in allow]
    if extra:
        raise RuntimeError(f"STOP: train-share row outside fitlist {extra[:8]}")
    missing = sorted(allow - {r.get("id") for r in rows})
    if missing:
        raise RuntimeError(f"STOP: train-share missing fit ids {missing[:8]}")
    shares: list[float] = []
    n_zero = 0
    n_err = 0
    err_ids: list[str] = []
    for r in rows:
        if r.get("error"):
            n_err += 1
            err_ids.append(str(r.get("id")))
            continue
        changed = int(r.get("changed_pixels") or 0)
        if changed <= 0:
            n_zero += 1
            continue
        shares.append(float(r["dominant_share"]))
    if not shares:
        raise RuntimeError("STOP: no positive-change train pairs for ratio deciles")
    deciles = histogram_deciles(shares)
    bins = ratio_bins_from_deciles(deciles)
    rec = {
        "task": "SEMANTIC-ATTACH",
        "frozen_before_rescore": True,
        "fitlist_path": str(FITLIST_PATH),
        "fitlist_ids_sha256": fit.get("ids_sha256") or disjoint["ids_sha256"],
        "ids_sha256": disjoint["ids_sha256"],
        "n_fit": 1152,
        "n_rows": len(rows),
        "n_positive_change": len(shares),
        "n_zero_change": n_zero,
        "n_error": n_err,
        "error_ids": err_ids,
        "deciles": deciles,
        "ratio_bins": bins,
        "token_names_unchanged": list(RATIO_BIN_TOKENS),
        "edge_rule": (
            "token 0 stays exact-zero; remaining 10 tokens keep names; "
            "edges at the 10th/20th/.../90th percentiles of train dominant-transition "
            "share among pairs with change > 0"
        ),
        "share_definition": (
            "max(from_to[i,j]) / changed_pixels on rgb6 gold labels; TRAIN fitlist only"
        ),
        "CHOSEN_RULE": CHOSEN_RULE,
        "never_opens_cdvqa_eval_labels": True,
        "never_opens_frozen_val_labels_for_fit": True,
        "never_uses_cdvqa_gold_to_pick_bins": True,
        "disjoint_cdvqa_test": disjoint["disjoint_cdvqa_test"],
        "disjoint_cdvqa_val": disjoint["disjoint_cdvqa_val"],
        "disjoint_our_val": disjoint["disjoint_our_val"],
        "configured_deferred": True,
        "configured_note": "configured = max combined after rescore; not chosen here",
    }
    path = Path(dest or RATIO_BINS_V2_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    rec["path"] = str(path)
    rec["sha256"] = sha256_file(path)
    rec["mtime"] = path.stat().st_mtime
    return rec


def load_pair_features(path: Path | None = None) -> dict[str, dict[str, Any]]:
    p = Path(path or PAIR_FEATURES_PATH)
    out: dict[str, dict[str, Any]] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pid = str(row.get("pair_id") or "")
            if pid:
                out[pid] = row
    return out


def assert_mtime_order(frozen: Path, later: Path) -> None:
    if not frozen.is_file() or not later.is_file():
        raise RuntimeError(f"STOP: missing freeze or later file {frozen} {later}")
    if later.stat().st_mtime + 1e-6 < frozen.stat().st_mtime:
        raise RuntimeError("STOP: rescore mtime before freeze")


def protocol_with_bins(bins: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    proto = protocol_dict()
    if bins is not None:
        proto["ratio_bins"] = list(bins)
    return proto


def built_up_direction_from_counts(
    bld_a: int,
    bld_b: int,
    total_pixels: int,
    thresh_frac: float = BUILT_UP_FRAC_THRESH,
) -> str:
    """Frozen 3-way rule. increase iff (B-A) > 0.005*total; decrease iff < -0.005*total."""
    total = int(total_pixels)
    delta = int(bld_b) - int(bld_a)
    thr = float(thresh_frac) * float(total)
    if delta > thr:
        return "increase"
    if delta < -thr:
        return "decrease"
    return "no_change"


def classify_semantic_family(question: str | None) -> str | None:
    """Type-family path detector. None → binary WHERE only. No ML classifier."""
    q = (question or "").strip().lower()
    if not q:
        return None
    if "change ratio" in q or "how much area" in q:
        return "change_ratio_types"
    if "mainly changed to" in q or "changed to what" in q or "changed to?" in q:
        return "change_to_what"
    if re.search(r"\bchanged to\b", q):
        return "change_to_what"
    if "largest change" in q:
        return "largest_change"
    if "smallest change" in q:
        return "smallest_change"
    if re.search(r"\bincreas", q) and re.search(r"\bdecreas", q):
        return "built_up_direction"
    if re.search(r"\bincreas", q):
        return "increase_or_not"
    if re.search(r"\bdecreas", q):
        return "decrease_or_not"
    return None


def configured_choice(v1_combined: float, v2_combined: float) -> str:
    """Max combined only. Ties keep v1. Per-family keep-max is forbidden."""
    if float(v2_combined) > float(v1_combined):
        return "v2"
    return "v1"


def load_ratio_bins_v2(path: Path | None = None) -> list[dict[str, Any]]:
    rec = json.loads(Path(path or RATIO_BINS_V2_PATH).read_text(encoding="utf-8"))
    bins = rec.get("ratio_bins")
    if not bins:
        raise RuntimeError("STOP: semantic_ratio_bins_v2.json missing ratio_bins")
    return list(bins)


def rescore_sem_v2(
    *,
    bins_path: Path | None = None,
    features_path: Path | None = None,
    dest: Path | None = None,
    preds_dest: Path | None = None,
) -> dict[str, Any]:
    """Re-score EXISTING pair features with frozen v2 bins. Does not rewrite v1 files."""
    frozen = Path(bins_path or RATIO_BINS_V2_PATH)
    if not frozen.is_file():
        raise RuntimeError("STOP: refuse to score before freeze")
    frozen_mtime = frozen.stat().st_mtime
    v1_scores_path = SCORES_SEM_V1_PATH
    v1_preds_path = PREDS_SEM_V1_PATH
    v1_scores_mtime = v1_scores_path.stat().st_mtime
    v1_preds_mtime = v1_preds_path.stat().st_mtime
    v1_scores_sha = sha256_file(v1_scores_path)
    v1_preds_sha = sha256_file(v1_preds_path)
    v1 = json.loads(v1_scores_path.read_text(encoding="utf-8"))
    bins = load_ratio_bins_v2(frozen)
    proto = protocol_with_bins(bins)
    feat_by_pair = load_pair_features(features_path)
    questions = iter_cdvqa_questions()
    preds = compile_preds_from_features(questions, feat_by_pair, proto)
    gold = load_cdvqa_gold()
    model = score_preds_against_gold(preds, gold)
    majority = {
        "majority_transition_token": "buildings",
        "majority_token": "nvg_surface",
    }
    base_a = {
        "families": {
            t: {
                "n": model["families"][t]["n"],
                "accuracy": 0.0,
                "ci95": bootstrap_ci([0] * int(model["families"][t]["n"])),
                "rule": "always unknown",
            }
            for t in SEMANTIC_FAMILIES
        },
        "combined": {
            "n": model["combined"]["n"],
            "accuracy": 0.0,
            "ci95": bootstrap_ci([0] * int(model["combined"]["n"])),
        },
    }
    maj_preds = []
    for q in questions:
        t = str(q.get("official_type") or "")
        if t not in SEMANTIC_FAMILIES:
            continue
        maj_preds.append(
            {
                "qa_id": q.get("qa_id"),
                "official_type": t,
                "pred": baseline_majority_pred(t, majority),
                "error": None,
            }
        )
    base_b = score_preds_against_gold(maj_preds, gold)
    for t, rec in base_b["families"].items():
        rec["rule"] = "train-majority-transition / majority class / 'no' / bin '0'"
        rec["token"] = baseline_majority_pred(t, majority)

    v1_combined = float(v1["model"]["combined"]["accuracy"])
    v2_combined = float(model["combined"]["accuracy"])
    choice = configured_choice(v1_combined, v2_combined)
    per_family = {}
    for t in SEMANTIC_FAMILIES:
        per_family[t] = {
            "v1": v1["model"]["families"][t],
            "v2": model["families"][t],
            "note": "configured is max combined; this family is not independently maximised",
        }
    summary = {
        "task": "SEMANTIC-ATTACH",
        "frozen_bins_path": str(frozen),
        "frozen_bins_sha256": sha256_file(frozen),
        "frozen_before_rescore": True,
        "n_preds": len(preds),
        "n_pair_features": len(feat_by_pair),
        "model": model,
        "baseline_a_unknown": base_a,
        "baseline_b_train_majority": {
            "families": base_b["families"],
            "combined": base_b["combined"],
        },
        "v1": {
            "combined": v1["model"]["combined"],
            "families": v1["model"]["families"],
            "scores_sha256": v1_scores_sha,
            "preds_sha256": v1_preds_sha,
        },
        "v1_vs_v2": {
            "combined_v1": v1_combined,
            "combined_v2": v2_combined,
            "configured": choice,
            "configured_rule": "max combined; ties keep v1; per-family keep-max forbidden",
            "per_family": per_family,
        },
        "configured": choice,
        "v1_untouched": True,
        "approved_for_demo": False,
    }
    dest_path = Path(dest or SCORES_SEM_V2_PATH)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    preds_path = Path(preds_dest or PREDS_SEM_V2_PATH)
    with preds_path.open("w", encoding="utf-8") as f:
        for row in preds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if frozen.stat().st_mtime != frozen_mtime:
        raise RuntimeError("STOP: bins file mutated during rescore")
    if v1_scores_path.stat().st_mtime != v1_scores_mtime or sha256_file(v1_scores_path) != v1_scores_sha:
        raise RuntimeError("STOP: v1 cdvqa_scores_sem.json was rewritten")
    if v1_preds_path.stat().st_mtime != v1_preds_mtime or sha256_file(v1_preds_path) != v1_preds_sha:
        raise RuntimeError("STOP: v1 cdvqa_preds_sem.jsonl was rewritten")
    assert_mtime_order(frozen, dest_path)
    summary["path"] = str(dest_path)
    summary["preds_path"] = str(preds_path)
    summary["bins_mtime"] = frozen_mtime
    summary["scores_mtime"] = dest_path.stat().st_mtime
    return summary


def class_hist_tokens(counts: Sequence[int]) -> dict[str, int]:
    return {CLASS_ID_TO_TOKEN[i + 1]: int(counts[i]) for i in range(NUM_CLASSES)}


def semantic_trace_from_maps(
    cls_a: np.ndarray,
    cls_b: np.ndarray,
    change: np.ndarray,
    *,
    ckpt_sha: str,
    ckpt_path: str | None = None,
) -> dict[str, Any]:
    """Live compiler features + buildings counts. No gold."""
    a = np.asarray(cls_a)
    b = np.asarray(cls_b)
    uniq_ok = set(range(1, NUM_CLASSES + 1))
    valid = (
        a.size > 0
        and a.shape == b.shape
        and set(int(x) for x in np.unique(a).tolist()).issubset(uniq_ok)
        and set(int(x) for x in np.unique(b).tolist()).issubset(uniq_ok)
    )
    feat = features_from_maps(a, b, change)
    dom = dominant_transition_from_from_to(feat["from_to"])
    bld_a = int(feat["count_a"][BUILT_UP_CLASS_ID - 1])
    bld_b = int(feat["count_b"][BUILT_UP_CLASS_ID - 1])
    total = int(feat["total_pixels"])
    direction = built_up_direction_from_counts(bld_a, bld_b, total)
    return {
        "rung": "semantic_live",
        "checkpoint": ckpt_path,
        "checkpoint_sha256": ckpt_sha,
        "valid_6class_maps": bool(valid),
        "hist_a": class_hist_tokens(feat["count_a"]),
        "hist_b": class_hist_tokens(feat["count_b"]),
        "dominant_transition": dom,
        "buildings_a": bld_a,
        "buildings_b": bld_b,
        "buildings_delta": bld_b - bld_a,
        "total_pixels": total,
        "built_up_frac_thresh": BUILT_UP_FRAC_THRESH,
        "built_up_direction": direction,
        "features": feat,
    }


def gold_built_up_direction_from_labels(l1: np.ndarray, l2: np.ndarray) -> dict[str, Any]:
    """Scorer-only. Never a compiler input."""
    a = np.asarray(l1)
    b = np.asarray(l2)
    bld_a = int((a == BUILT_UP_CLASS_ID).sum())
    bld_b = int((b == BUILT_UP_CLASS_ID).sum())
    total = int(a.size)
    return {
        "buildings_a": bld_a,
        "buildings_b": bld_b,
        "total_pixels": total,
        "delta": bld_b - bld_a,
        "thresh_pixels": float(BUILT_UP_FRAC_THRESH) * float(total),
        "rule": "increase iff (bld_B-bld_A)>0.005*total; decrease iff < -0.005*total; else no_change",
        "built_up_direction": built_up_direction_from_counts(bld_a, bld_b, total),
        "scorer_only": True,
    }
