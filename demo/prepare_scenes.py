"""Export the 3 demo scene assets from already-downloaded caches (DEMO-SPEC-05 §4).

Scene 1: VRSBench val PNGs already at gates/_cache/vrsbench/images (no new download).
Scene 2: LEVIR-CD test pair from gates/_cache/levir_cd (HF satellite-image-deep-learning/LEVIR-CD).
Scene 3: co-registered S1+S2 from Lithuania LMDB already on disk.

Also scores ChangeFormer (or classical fallback) on a fixed n=20 LEVIR-CD subset.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

DEMO = Path(__file__).resolve().parent
SATQUERY = DEMO.parent
sys.path.insert(0, str(SATQUERY))
sys.path.insert(0, str(DEMO))

from gates.data_prep import (  # noqa: E402
    IMAGE_META_NAME,
    _find_image_metadata,
    _open_lmdb,
    ensure_assets,
    load_s2_rgb,
    s2_rgb_to_pil,
)
from tools import (  # noqa: E402
    LEVIR_DIR,
    LEVIR_GSD_M,
    S2_GSD_M,
    area_calc,
    change_detect,
    load_mask,
    load_rgb,
    mask_metrics,
    overlay_mask,
    sar_read,
    vv_to_preview,
)

DATA = DEMO / "data"
SCENE1 = DATA / "scene1"
SCENE2 = DATA / "scene2"
SCENE3 = DATA / "scene3"
VRS_IMAGES = SATQUERY / "gates" / "_cache" / "vrsbench" / "images"
N_LEVIR_SCORE = 20
LEVIR_SCORE_SEED = 42

SCENE1_IDS = [
    "05867_0000.png",  # overpass / transport
    "05933_0000.png",  # tennis courts / recreation
    "05945_0000.png",  # parking lot / urban
]


def _copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def prepare_scene1() -> dict:
    missing = [i for i in SCENE1_IDS if not (VRS_IMAGES / i).is_file()]
    if missing:
        raise FileNotFoundError(f"VRSBench images missing: {missing} under {VRS_IMAGES}")
    SCENE1.mkdir(parents=True, exist_ok=True)
    files = []
    for name in SCENE1_IDS:
        dest = SCENE1 / name
        _copy(VRS_IMAGES / name, dest)
        im = Image.open(dest)
        files.append({"name": name, "size": list(im.size), "mode": im.mode, "bytes": dest.stat().st_size})
    manifest = {
        "scene": 1,
        "source": "VRSBench EVAL val PNGs (xiang709/VRSBench, already cached)",
        "gsd_m": None,
        "gsd_note": "benchmark PNG; GSD not used (no area_calc on Scene 1)",
        "files": files,
        "primary": SCENE1_IDS[0],
    }
    (SCENE1 / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _levir_pairs() -> list[dict]:
    a_dir = LEVIR_DIR / "A"
    names = sorted(p.name for p in a_dir.glob("test_*.png"))
    pairs = []
    for name in names:
        ap, bp, lp = LEVIR_DIR / "A" / name, LEVIR_DIR / "B" / name, LEVIR_DIR / "label" / name
        if ap.is_file() and bp.is_file() and lp.is_file():
            gt = load_mask(lp)
            pairs.append({"name": name, "A": ap, "B": bp, "label": lp, "gt_changed": int(gt.sum())})
    if len(pairs) < 20:
        raise RuntimeError(f"LEVIR-CD test pairs found={len(pairs)} (<20) under {LEVIR_DIR}")
    return pairs


def prepare_scene2() -> dict:
    pairs = _levir_pairs()
    # Money-scene pair: most GT change pixels (visually obvious built-up growth).
    demo_pair = max(pairs, key=lambda p: p["gt_changed"])
    SCENE2.mkdir(parents=True, exist_ok=True)
    before = SCENE2 / "before.png"
    after = SCENE2 / "after.png"
    gt = SCENE2 / "gt_mask.png"
    _copy(demo_pair["A"], before)
    _copy(demo_pair["B"], after)
    _copy(demo_pair["label"], gt)
    im = Image.open(before)
    manifest = {
        "scene": 2,
        "source": "LEVIR-CD test split (HF satellite-image-deep-learning/LEVIR-CD test.zip, md5 07d5dd89e46f5c1359e2eca746989ed9)",
        "pair": demo_pair["name"],
        "n_test_pairs_available": len(pairs),
        "gt_changed_pixels": demo_pair["gt_changed"],
        "size": list(im.size),
        "gsd_m": LEVIR_GSD_M,
        "gsd_note": "LEVIR-CD official 0.5 m/px",
        "files": {"before": "before.png", "after": "after.png", "gt_mask": "gt_mask.png"},
    }
    (SCENE2 / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _labels_of(row) -> list[str]:
    for col in ("labels", "label", "class_names", "original_labels"):
        if col in row.index:
            val = row[col]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            if isinstance(val, (list, tuple, np.ndarray)):
                return [str(x) for x in val]
            s = str(val)
            if s.startswith("["):
                try:
                    parsed = json.loads(s.replace("'", '"'))
                    if isinstance(parsed, list):
                        return [str(x) for x in parsed]
                except Exception:
                    pass
            return [s]
    return []


def prepare_scene3() -> dict:
    assets = ensure_assets()
    lmdb_path = Path(assets["lmdb"])
    meta_path = Path(assets["image_metadata"])
    import pandas as pd

    df = pd.read_parquet(meta_path)
    water_needles = ("water", "sea", "ocean", "lagoon", "estuar", "water course", "inland wetland")
    hits = []
    for idx, row in df.iterrows():
        labels = _labels_of(row)
        blob = " ".join(labels).lower()
        if any(n in blob for n in water_needles):
            hits.append((idx, row, labels))
    if not hits:
        # fallback: still pick a train patch so Scene 3 has a real S1+S2 pair
        row = df.iloc[0]
        hits = [(0, row, _labels_of(row))]
        water_note = "NO water-class label found; using first metadata row (still a real S1+S2 pair)"
    else:
        water_note = f"{len(hits)} patches with a water-related CORINE label"

    env = _open_lmdb(lmdb_path)
    chosen = None
    try:
        from safetensors.numpy import load as safetensor_load

        for idx, row, labels in hits:
            pid = str(row["patch_id"])
            s1 = str(row["s1_name"]) if "s1_name" in row.index else None
            if not s1 or s1 in {"nan", "None"}:
                continue
            with env.begin(write=False, buffers=True) as txn:
                s2_raw = txn.get(pid.encode("utf-8"))
                s1_raw = txn.get(s1.encode("utf-8"))
            if s2_raw is None or s1_raw is None:
                continue
            rgb = load_s2_rgb(env, pid)
            tensor = safetensor_load(bytes(s1_raw))
            vv, vh = tensor["VV"], tensor["VH"]
            chosen = {
                "patch_id": pid,
                "s1_name": s1,
                "labels": labels,
                "split": str(row["split"]) if "split" in row.index else None,
                "rgb": rgb,
                "vv": np.asarray(vv),
                "vh": np.asarray(vh),
            }
            break
    finally:
        env.close()
    if chosen is None:
        raise RuntimeError("STOP: Lithuania LMDB has no patch with both S2 and S1 keys readable.")

    SCENE3.mkdir(parents=True, exist_ok=True)
    optical = SCENE3 / "optical.png"
    chosen["rgb"].save(optical)
    vv_img = vv_to_preview(chosen["vv"])
    vh_img = vv_to_preview(chosen["vh"])
    vv_img.save(SCENE3 / "sar_vv.png")
    vh_img.save(SCENE3 / "sar_vh.png")
    np.savez_compressed(SCENE3 / "sar_arrays.npz", vv=chosen["vv"], vh=chosen["vh"])
    sar_out = sar_read(chosen["vv"], chosen["vh"])
    water_mask = sar_out["water_mask"]
    Image.fromarray((water_mask * 255).astype(np.uint8), mode="L").save(SCENE3 / "water_mask.png")
    overlay_mask(np.asarray(chosen["rgb"].resize(vv_img.size) if chosen["rgb"].size != vv_img.size else chosen["rgb"]),
                 water_mask, color=(30, 90, 220), alpha=0.5).save(SCENE3 / "water_overlay.png")
    area = area_calc(water_mask, gsd_m=S2_GSD_M, label="water")
    # drop the raw mask from json
    sar_json = {k: v for k, v in sar_out.items() if k != "water_mask"}
    manifest = {
        "scene": 3,
        "source": "hackelle/BigEarthNetV2-Lithuania-Summer-LMDB (already on disk; no new download)",
        "lmdb": str(lmdb_path),
        "patch_id": chosen["patch_id"],
        "s1_name": chosen["s1_name"],
        "labels": chosen["labels"],
        "split": chosen["split"],
        "water_note": water_note,
        "gsd_m": S2_GSD_M,
        "gsd_note": "Sentinel-2 10 m/px (B04/B03/B02); S1 resampled to the same grid in reBEN",
        "optical_size": list(chosen["rgb"].size),
        "sar_shape": list(chosen["vv"].shape),
        "fusion": "late-fusion-by-design (structured SAR stats + VLM narration); NOT learned pixel fusion",
        "sar_preview": sar_json,
        "water_area_from_tool": area,
        "files": {
            "optical": "optical.png",
            "sar_vv": "sar_vv.png",
            "sar_vh": "sar_vh.png",
            "sar_arrays": "sar_arrays.npz",
            "water_mask": "water_mask.png",
        },
    }
    (SCENE3 / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def score_levir_subset(device: str = "cuda", n: int = N_LEVIR_SCORE, seed: int = LEVIR_SCORE_SEED) -> dict:
    pairs = _levir_pairs()
    # Prefer pairs that actually contain change so IoU is not 0/0 dominated.
    changed = [p for p in pairs if p["gt_changed"] >= (1024 * 1024 * 0.005)]
    pool = changed if len(changed) >= n else pairs
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    idx.sort()
    chosen = [pool[int(i)] for i in idx]
    rows = []
    ious, f1s = [], []
    rung = None
    for p in chosen:
        before = load_rgb(p["A"])
        after = load_rgb(p["B"])
        gt = load_mask(p["label"])
        out = change_detect(before, after, prefer="changeformer", device=device, gt_mask=gt)
        rung = out["rung"]
        met = out["vs_gt"]
        ious.append(met["iou"])
        f1s.append(met["f1"])
        rows.append(
            {
                "name": p["name"],
                "gt_changed": p["gt_changed"],
                "pred_changed": out["changed_pixels"],
                "iou": met["iou"],
                "f1": met["f1"],
            }
        )
        print(f"  {p['name']} iou={met['iou']:.4f} f1={met['f1']:.4f} rung={out['rung']}", flush=True)
    summary = {
        "n": len(rows),
        "seed": seed,
        "pool": "gt_change>=0.5% of 1024^2" if len(changed) >= n else "all test pairs",
        "n_test_pairs": len(pairs),
        "n_changed_pool": len(changed),
        "rung": rung,
        "mean_iou": float(np.mean(ious)) if ious else None,
        "mean_f1": float(np.mean(f1s)) if f1s else None,
        "names": [r["name"] for r in rows],
        "rows": rows,
        "device": device,
        "gsd_m": LEVIR_GSD_M,
    }
    (SCENE2 / "levir_score.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def score_demo_pair(device: str = "cpu") -> dict:
    before = load_rgb(SCENE2 / "before.png")
    after = load_rgb(SCENE2 / "after.png")
    gt = load_mask(SCENE2 / "gt_mask.png")
    out = change_detect(before, after, prefer="changeformer", device=device, gt_mask=gt)
    mask = out["mask"]
    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(SCENE2 / "pred_mask.png")
    overlay_mask(after, mask).save(SCENE2 / "overlay.png")
    area = area_calc(mask, gsd_m=LEVIR_GSD_M, label="built-up_change")
    (SCENE2 / "pred_area.json").write_text(json.dumps(area, indent=2), encoding="utf-8")
    slim = {k: v for k, v in out.items() if k != "mask"}
    slim["area"] = area
    (SCENE2 / "pred_trace.json").write_text(json.dumps(slim, indent=2, default=str), encoding="utf-8")
    return slim


def main() -> None:
    print("=== Scene 1 ===", flush=True)
    s1 = prepare_scene1()
    print(json.dumps(s1, indent=2), flush=True)
    print("=== Scene 2 assets ===", flush=True)
    s2 = prepare_scene2()
    print(json.dumps(s2, indent=2), flush=True)
    print("=== Scene 3 ===", flush=True)
    s3 = prepare_scene3()
    print(json.dumps({k: v for k, v in s3.items() if k not in {"sar_preview"}}, indent=2, default=str), flush=True)
    print("=== Scene 2 demo-pair ChangeFormer (cpu first; cuda if available) ===", flush=True)
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, flush=True)
    demo = score_demo_pair(device=device)
    print(json.dumps({k: v for k, v in demo.items() if k != "mask"}, indent=2, default=str), flush=True)
    print("=== LEVIR n=20 score ===", flush=True)
    summary = score_levir_subset(device=device)
    print(
        json.dumps(
            {k: summary[k] for k in ("n", "rung", "mean_iou", "mean_f1", "device", "names")},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
