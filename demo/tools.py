"""Specialist tools for the SatQuery demo (DEMO-SPEC-05).

Each function returns a structured dict (values + provenance).
The VLM never computes these numbers.

Tools:
  vqa              — llama-server Qwen3-VL-8B (OpenAI-compatible)
  change_detect    — ChangeFormerV6 LEVIR-CD pretrained; classical OpenCV fallback
  water_highlight  — RGB Otsu / NDWI water mask (no m²; area_calc does GSD)
  area_calc        — pixel count * GSD^2
  sar_read         — VV/VH boxcar + SDWI + water threshold (no learned fusion)
  sar_agreement    — agreement between independent optical/SAR water masks
  cdvqa_map        — second_semantic packet -> CDVQA canonical answer (no VLM)
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import mimetypes
import re
import sys
import time
import unittest
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

DEMO_DIR = Path(__file__).resolve().parent
SATQUERY = DEMO_DIR.parent
VENDOR_CF = DEMO_DIR / "_vendor" / "ChangeFormer"
CF_CKPT_DIR = SATQUERY / "gates" / "_cache" / "changeformer"
CF_REGISTRY = SATQUERY / "gates" / "_cache" / "cf_ft" / "ckpt_registry.json"
TEAM_SECOND_SHA = "cbfefe6564613571e824cb2cee0f971b4793ca0dda6c1a5c5741cbcdf4c76fe6"
IMPORTED_LEVIR_SHA = "db0dd783c3c3f27d02f55f24fa8199d2b2b47c422c7f7bd2e4d96e2fa65d69f2"
SECOND_SEMANTIC_SHA = "438cac09be7c630254a12278550b64f86254ecc131ee0cdc723fd526210764d8"
# Display layer only. Registry roles / SHA constants / rung keys stay as written above.
MODEL_DISPLAY: dict[str, str] = {
    "team_second": "Housing Change Specialist (SECOND-tuned)",
    "second_semantic": "Land-Cover Change Describer (6-class)",
    "imported_levir": "General Change Model (imported)",
    "second_like": "SECOND-style detailed pairs",
    "second": "SECOND-style detailed pairs",
}
RUNG_DISPLAY: dict[str, str] = {
    "team_second_live": "housing-change (live)",
    "semantic_live": "land-cover (live)",
    "changeformer_v6": "general change (live)",
    "cached_mask": "cached mask",
    "classical_otsu": "classical baseline",
}
SECOND_FOOTNOTE = "SECOND = Semantic Change Detection dataset"
UPLOAD_DOMAIN_LIMITATION = (
    "Housing Change Specialist (SECOND-tuned) is for SECOND-style detailed pairs "
    "(SECOND = Semantic Change Detection dataset); "
    "LEVIR-style pairs use the General Change Model (imported); "
    "the Housing Change Specialist (SECOND-tuned) regresses on LEVIR — disclosed"
)
SEMANTIC_UPLOAD_LIMITATION = (
    "Land-Cover Change Describer (6-class) is SECOND-style detailed pairs "
    "type-family only; uploads skip it — disclosed"
)
LEVIR_DIR = SATQUERY / "gates" / "_cache" / "levir_cd"
DEFAULT_VLM_URL = "http://127.0.0.1:8080"
DEFAULT_VLM_MODEL = "qwen3vl"
LEVIR_GSD_M = 0.5
S2_GSD_M = 10.0
CHANGEFORMER_TILE = 256
WATER_OVERLAY_RGB = (30, 90, 220)
DIRECTION_EPS = 8.0
SAR_WATER_DB = -16.0

# Rung recorded at import-of-prepare time; change_detect fills this per call.
LAST_CD_RUNG = "uninitialized"


def display_name_for_role(role: str | None) -> str:
    """Human label for a registry role or domain key. Unknown keys echo through."""
    key = "" if role is None else str(role)
    if not key:
        return ""
    return MODEL_DISPLAY.get(key, key)


def display_name_for_rung(rung: str | None) -> str:
    """Human label for a rung key. Unknown keys echo through."""
    key = "" if rung is None else str(rung)
    if not key:
        return ""
    return RUNG_DISPLAY.get(key, key)


def attach_rung_display(record: dict[str, Any]) -> dict[str, Any]:
    """Additive display fields alongside rung keys. Never replaces the keys."""
    if "rung" in record:
        record["rung_display"] = display_name_for_rung(record.get("rung"))
    if "semantic_rung" in record:
        record["semantic_rung_display"] = display_name_for_rung(record.get("semantic_rung"))
    return record


def _mime(path: Path) -> str:
    guess, _ = mimetypes.guess_type(str(path))
    if guess:
        return guess
    return {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(
        path.suffix.lower(), "image/png"
    )


def load_rgb(path: str | Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.asarray(im)


def load_mask(path: str | Path) -> np.ndarray:
    im = Image.open(path)
    arr = np.asarray(im)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 127).astype(np.uint8)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(220, 30, 30), alpha: float = 0.45) -> Image.Image:
    """Red overlay = change / water. Returns PIL RGB."""
    base = np.asarray(rgb, dtype=np.float32)
    if base.ndim != 3:
        raise ValueError("rgb must be HxWx3")
    m = (np.asarray(mask) > 0).astype(np.float32)
    if m.shape[:2] != base.shape[:2]:
        m_img = Image.fromarray((m * 255).astype(np.uint8)).resize(
            (base.shape[1], base.shape[0]), Image.Resampling.NEAREST
        )
        m = np.asarray(m_img).astype(np.float32) / 255.0
    paint = np.zeros_like(base)
    paint[..., 0] = color[0]
    paint[..., 1] = color[1]
    paint[..., 2] = color[2]
    out = base * (1.0 - alpha * m[..., None]) + paint * (alpha * m[..., None])
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def mask_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    """Binary IoU + F1. pred/gt are 0/1 arrays of the same shape."""
    p = np.asarray(pred).astype(bool).reshape(-1)
    g = np.asarray(gt).astype(bool).reshape(-1)
    if p.size != g.size:
        raise ValueError(f"shape mismatch pred={pred.shape} gt={gt.shape}")
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, np.logical_not(g)).sum())
    fn = int(np.logical_and(np.logical_not(p), g).sum())
    tn = int(np.logical_and(np.logical_not(p), np.logical_not(g)).sum())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": round(float(iou), 6),
        "precision": round(float(prec), 6),
        "recall": round(float(rec), 6),
        "f1": round(float(f1), 6),
        "n_pixels": int(p.size),
        "provenance": "tools.mask_metrics (pixel confusion matrix, not VLM)",
    }


def area_calc(
    mask: np.ndarray,
    gsd_m: float | None,
    label: str = "changed",
    gsd_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pixel count × GSD². Also reports which image quadrant holds most of the class.

    If gsd_m is None / non-finite / <= 0, pixels and percent still compute;
    area_m2 / area_km2 are withheld (no silent LEVIR/S2 constant).
    """
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    binary = m > 0
    h, w = binary.shape
    n = int(binary.sum())
    total = int(binary.size)
    pct = (100.0 * n / total) if total else 0.0
    mid_r, mid_c = h // 2, w // 2
    quads = {
        "NW": int(binary[:mid_r, :mid_c].sum()),
        "NE": int(binary[:mid_r, mid_c:].sum()),
        "SW": int(binary[mid_r:, :mid_c].sum()),
        "SE": int(binary[mid_r:, mid_c:].sum()),
    }
    dominant = max(quads, key=quads.get) if n else "none"
    north_frac = ((quads["NW"] + quads["NE"]) / n) if n else 0.0
    gsd_ok = gsd_m is not None and math.isfinite(float(gsd_m)) and float(gsd_m) > 0.0
    if gsd_ok:
        gsd_f = float(gsd_m)
        px_m2 = gsd_f * gsd_f
        area_m2 = n * px_m2
        area_km2 = area_m2 / 1_000_000.0
        formula = "area_m2 = count(mask>0) * gsd_m^2"
    else:
        gsd_f = None
        px_m2 = None
        area_m2 = None
        area_km2 = None
        formula = "percent = count/total; area_m2 withheld (no GSD)"
    meta = gsd_meta or {}
    provenance = "tools.area_calc (raster math; deterministic tool measurement)"
    if meta.get("provenance"):
        provenance = provenance + " | " + str(meta["provenance"])
    return {
        "label": label,
        "changed_pixels": n,
        "total_pixels": total,
        "gsd_m": gsd_f,
        "gsd_source": meta.get("source"),
        "pixel_area_m2": px_m2,
        "area_m2": area_m2,
        "area_km2": area_km2,
        "percent_of_image": pct,
        "quadrants": quads,
        "dominant_quadrant": dominant,
        "north_fraction": north_frac,
        "formula": formula,
        "provenance": provenance,
    }


def _uniform3(x: np.ndarray) -> np.ndarray:
    """3×3 boxcar / uniform filter. cv2.blur if present, else numpy pad-mean."""
    arr = np.asarray(x, dtype=np.float32)
    try:
        import cv2

        return cv2.blur(arr, (3, 3))
    except Exception:
        pad = np.pad(arr, 1, mode="edge")
        acc = np.zeros_like(arr)
        for i in range(3):
            for j in range(3):
                acc += pad[i : i + arr.shape[0], j : j + arr.shape[1]]
        return acc / 9.0


def _rgb_water_index_mask(rgb: np.ndarray) -> tuple[np.ndarray, str]:
    """Otsu (or fallback threshold) on (B−R). Blue water vs dark/red land.

    On land-dominated RGB scenes Otsu's auto-threshold sits low and shadows /
    asphalt / dark vegetation pass as "water". Two guards: a floor on the
    (B−R) cut, and a 3x3 morphological open to drop single-pixel speckle.
    """
    r = rgb[..., 0].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    index = b - r
    idx_u8 = np.clip(index, 0, 255).astype(np.uint8)
    try:
        import cv2

        thr, _binm = cv2.threshold(idx_u8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr_eff = max(float(thr), 18.0)
        mask = (index > thr_eff).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        method = (
            f"RGB Otsu on (B-R), threshold={thr_eff:.1f} (floored); 3x3 open"
        )
    except Exception:
        mask = (index > 20.0).astype(np.uint8)
        method = "RGB heuristic (B-R) > 20 (Otsu unavailable)"
    return mask, method


def water_highlight(
    rgb: np.ndarray | str | Path,
    source_path: str | Path | None = None,
    sensor_profile: str | None = None,
) -> dict[str, Any]:
    """Optical water mask. No m² — pipeline calls area_calc with ingest GSD.

    `source_path` is the original file (GeoTIFF) when `rgb` is a materialized
    PNG. Band identity comes from `sensor_profile` or the file's own metadata
    (ingest.resolve_band_map) — never from band count. When the bands cannot
    be identified the result withholds (`no_spectral_basis`) instead of
    running the RGB heuristic on an arbitrary band order.
    """
    inspect = Path(source_path) if source_path is not None else None
    if not isinstance(rgb, np.ndarray):
        rgb_path = Path(rgb)
        rgb = load_rgb(rgb)
        if inspect is None:
            inspect = rgb_path
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("water_highlight needs HxWx3 RGB")
    method = ""
    mask = None
    evidence_class = "heuristic_estimate"
    withheld_reason = None
    if inspect is not None:
        p = Path(inspect)
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff", ".geotiff"}:
            from ingest import resolve_band_map

            band_map, note = resolve_band_map(p, sensor_profile)
            if band_map and {"green", "nir"} <= set(band_map):
                try:
                    import rasterio

                    with rasterio.open(p) as ds:
                        green = ds.read(band_map["green"]).astype(np.float32)
                        nir = ds.read(band_map["nir"]).astype(np.float32)
                except Exception as e:
                    green = nir = None
                    note = note + f"; band read failed {type(e).__name__}: {e}"
                if green is not None and nir is not None:
                    denom = green + nir
                    ndwi = np.divide(
                        green - nir, denom, out=np.zeros_like(green), where=denom != 0
                    )
                    mask = (ndwi > 0.0).astype(np.uint8)
                    if mask.shape[:2] != rgb.shape[:2]:
                        mask = (
                            np.asarray(
                                Image.fromarray((mask * 255).astype(np.uint8)).resize(
                                    (rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST
                                )
                            )
                            > 0
                        ).astype(np.uint8)
                    method = (
                        "McFeeters NDWI=(G-NIR)/(G+NIR); water = NDWI>0. " + note
                    )
                    evidence_class = "measured_index"
            elif band_map is None and sensor_profile:
                withheld_reason = (
                    "unknown_sensor_profile"
                    if "unknown sensor profile" in note
                    else "no_spectral_basis"
                )
            elif band_map is None:
                withheld_reason = "no_spectral_basis"
            elif not {"red", "green", "blue"} <= set(band_map):
                withheld_reason = "no_spectral_basis"
            if mask is None and withheld_reason is None:
                method = note + ". "
    if withheld_reason is not None:
        z = np.zeros(rgb.shape[:2], dtype=np.uint8)
        method = (method + f" withheld: {withheld_reason}.").strip()
        return {
            "mask": z,
            "water_pixels": None,
            "mask_shape": list(z.shape),
            "method": method,
            "withheld": True,
            "withheld_reason": withheld_reason,
            "evidence_class": "none",
            "provenance": (
                "tools.water_highlight (withheld — no spectral claim). " + method
            ),
        }
    if mask is None:
        mask, rgb_method = _rgb_water_index_mask(rgb)
        method = (method + rgb_method).strip()
        evidence_class = "heuristic_estimate"
    n = int(mask.sum())
    return {
        "mask": mask,
        "water_pixels": n,
        "mask_shape": list(mask.shape),
        "method": method,
        "withheld": False,
        "evidence_class": evidence_class,
        "provenance": (
            "tools.water_highlight (optical mask; deterministic tool measurement). "
            + method
        ),
    }


def change_direction_proxy(
    before: np.ndarray,
    after: np.ndarray,
    mask: np.ndarray,
    epsilon: float = DIRECTION_EPS,
) -> dict[str, Any]:
    """Radiometric proxy inside the change mask. Not a class map.

    after-mean − before-mean on RGB in changed pixels. |delta| < epsilon →
    similar. Brighter after → radiometric_label=brighter_after.
    built_up_direction is always not_determined.
    """
    b = np.asarray(before, dtype=np.float32)
    a = np.asarray(after, dtype=np.float32)
    m = np.asarray(mask) > 0
    if m.ndim == 3:
        m = m[..., 0]
    n = int(m.sum())
    if n == 0:
        return {
            "before_mean": None,
            "after_mean": None,
            "delta_mean": 0.0,
            "radiometric_delta": 0.0,
            "radiometric_label": "unknown",
            "built_up_direction": "not_determined",
            "direction_n_pixels": 0,
            "direction_epsilon": float(epsilon),
            "direction_provenance": (
                "radiometric_label is mean-RGB brightness inside the change mask, "
                "NOT built-up increase/decrease. built_up_direction=not_determined "
                "until a class-aware tool exists."
            ),
        }
    before_mean = float(b[m].mean())
    after_mean = float(a[m].mean())
    delta = after_mean - before_mean
    if abs(delta) < float(epsilon):
        radio = "similar"
    elif delta > 0:
        radio = "brighter_after"
    else:
        radio = "darker_after"
    return {
        "before_mean": before_mean,
        "after_mean": after_mean,
        "delta_mean": delta,
        "radiometric_delta": delta,
        "radiometric_label": radio,
        "built_up_direction": "not_determined",
        "direction_n_pixels": n,
        "direction_epsilon": float(epsilon),
        "direction_provenance": (
            "radiometric_label is mean-RGB brightness inside the change mask "
            f"(delta={delta:.2f}, eps={float(epsilon):.1f}, n={n}); "
            "NOT construction / unchanged built-up. "
            "built_up_direction=not_determined until a class-aware tool exists."
        ),
    }


def change_detect_classical(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    """Rung 3: per-channel abs-diff + Otsu + morphology (OpenCV)."""
    import cv2

    a = np.asarray(before)
    b = np.asarray(after)
    if a.shape != b.shape:
        b_img = Image.fromarray(b).resize((a.shape[1], a.shape[0]), Image.Resampling.BILINEAR)
        b = np.asarray(b_img)
    diff = cv2.absdiff(a, b)
    mag = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY) if diff.ndim == 3 else diff
    # Otsu
    _thr, raw = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    mask = (closed > 0).astype(np.uint8)
    rec = {
        "mask": mask,
        "rung": "classical_otsu",
        "rung_label": "classical baseline (abs-diff + Otsu + morphology)",
        "otsu_threshold": float(_thr),
        "method": "cv2.absdiff -> grayscale -> THRESH_OTSU -> morph open/close 5x5",
        "provenance": "tools.change_detect_classical",
    }
    attach_rung_display(rec)
    return rec


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _registry_entry(role: str) -> dict | None:
    if not CF_REGISTRY.is_file():
        return None
    rec = json.loads(CF_REGISTRY.read_text(encoding="utf-8"))
    for ent in rec.get("checkpoints") or []:
        if ent.get("role") == role:
            return ent
    return None


def _registry_ckpt_path(role: str) -> Path | None:
    ent = _registry_entry(role)
    if not ent or not ent.get("path"):
        return None
    p = Path(ent["path"])
    if not p.is_absolute():
        p = SATQUERY / p
    return p if p.is_file() else None


def _find_changeformer_ckpt(domain: str | None = None) -> Path | None:
    """Registry + domain. LEVIR/default = imported. second_like = team_second if approved."""
    dom = (domain or "levir").strip().lower()
    if dom in {"second", "second_like"}:
        ent = _registry_entry("team_second")
        if (
            ent
            and ent.get("approved_for_demo") is True
            and (
                ent.get("route_scope") == "second_like"
                or "second_like" in list(ent.get("domains") or [])
            )
            and str(ent.get("sha256") or "") == TEAM_SECOND_SHA
        ):
            p = _registry_ckpt_path("team_second")
            if p is not None and sha256_file(p) == TEAM_SECOND_SHA:
                return p
    imported = _registry_ckpt_path("imported_levir")
    if imported is not None:
        return imported
    for p in sorted(CF_CKPT_DIR.rglob("best_ckpt.pt")):
        if p.is_file() and p.stat().st_size > 1_000_000:
            return p
    for p in sorted(CF_CKPT_DIR.rglob("*.pt")):
        if p.is_file() and p.stat().st_size > 1_000_000:
            return p
    return None


def _find_second_semantic_ckpt(*, require_approved: bool = True) -> Path | None:
    """Registry second_semantic. Live demo requires approved + route_scope second_like_type."""
    ent = _registry_entry("second_semantic")
    if not ent or not ent.get("path"):
        return None
    if str(ent.get("sha256") or "") != SECOND_SEMANTIC_SHA:
        return None
    if require_approved:
        if ent.get("approved_for_demo") is not True:
            return None
        scope_ok = ent.get("route_scope") == "second_like_type" or (
            "second_like_type" in list(ent.get("domains") or [])
        )
        if not scope_ok:
            return None
    p = _registry_ckpt_path("second_semantic")
    if p is None:
        return None
    if sha256_file(p) != SECOND_SEMANTIC_SHA:
        return None
    return p


_SEM_NET = None
_SEM_DEVICE = None
_SEM_CKPT = None


def semantic_live_forward(
    before: np.ndarray | str | Path,
    after: np.ndarray | str | Path,
    change_mask: np.ndarray,
    device: str = "cpu",
    ckpt_path: Path | None = None,
    require_approved: bool = True,
) -> dict[str, Any]:
    """Live per-timestamp 6-class maps. Transitions under the provided WHERE mask.

    Measurement may pass require_approved=False (SHA still checked). Demo route
    must use require_approved=True.
    """
    global _SEM_NET, _SEM_DEVICE, _SEM_CKPT
    if str(SATQUERY) not in sys.path:
        sys.path.insert(0, str(SATQUERY))
    from cf_ft.semantic import (  # noqa: E402
        SECOND_SEMANTIC_SHA as SEM_SHA,
        load_semantic_model,
        predict_class_map,
        semantic_trace_from_maps,
    )

    if not isinstance(before, np.ndarray):
        before = load_rgb(before)
    if not isinstance(after, np.ndarray):
        after = load_rgb(after)
    ckpt = Path(ckpt_path) if ckpt_path is not None else _find_second_semantic_ckpt(
        require_approved=require_approved
    )
    if ckpt is None:
        raise FileNotFoundError("second_semantic checkpoint not eligible")
    sha = sha256_file(ckpt)
    if sha != SEM_SHA:
        raise RuntimeError(f"STOP: second_semantic SHA {sha} != {SEM_SHA}")
    if _SEM_NET is None or _SEM_DEVICE != device or _SEM_CKPT != ckpt:
        model, _blob = load_semantic_model(ckpt, device=device, pretrained=True)
        _SEM_NET, _SEM_DEVICE, _SEM_CKPT = model, device, ckpt
    else:
        model = _SEM_NET
    cls_a = predict_class_map(model, before, device=device)
    cls_b = predict_class_map(model, after, device=device)
    rec = semantic_trace_from_maps(
        cls_a, cls_b, change_mask, ckpt_sha=sha, ckpt_path=str(ckpt)
    )
    rec["cls_a"] = cls_a
    rec["cls_b"] = cls_b
    return rec


_CF_NET = None
_CF_DEVICE = None
_CF_CKPT = None
_CF_ERROR = None


def _load_changeformer(device: str = "cpu", ckpt_path: Path | None = None, domain: str | None = None):
    """Lazy-load ChangeFormerV6. Returns (net, device, ckpt_path) or raises."""
    global _CF_NET, _CF_DEVICE, _CF_CKPT, _CF_ERROR
    ckpt = Path(ckpt_path) if ckpt_path is not None else _find_changeformer_ckpt(domain=domain)
    if ckpt is None:
        raise FileNotFoundError(f"No ChangeFormer .pt under {CF_CKPT_DIR}")
    if _CF_NET is not None and _CF_DEVICE == device and _CF_CKPT == ckpt:
        return _CF_NET, _CF_DEVICE, _CF_CKPT
    if not VENDOR_CF.is_dir():
        raise FileNotFoundError(f"ChangeFormer source missing at {VENDOR_CF}")
    vendor = str(VENDOR_CF)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    import torch
    from models.ChangeFormer import ChangeFormerV6

    net = ChangeFormerV6(input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=256)
    blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    state = blob["model_G_state_dict"] if isinstance(blob, dict) and "model_G_state_dict" in blob else blob
    net.load_state_dict(state, strict=True)
    net.eval()
    if device == "cuda":
        net = net.to("cuda")
    _CF_NET, _CF_DEVICE, _CF_CKPT, _CF_ERROR = net, device, ckpt, None
    return net, device, ckpt


def _tensorize_pair(tile_a: np.ndarray, tile_b: np.ndarray, device: str):
    import torch
    import torchvision.transforms.functional as TF

    def one(arr: np.ndarray):
        im = Image.fromarray(arr).convert("RGB")
        t = TF.to_tensor(im)
        t = TF.normalize(t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return t

    batch_a = one(tile_a).unsqueeze(0)
    batch_b = one(tile_b).unsqueeze(0)
    if device == "cuda":
        batch_a = batch_a.cuda()
        batch_b = batch_b.cuda()
    return batch_a, batch_b


def change_detect_changeformer(
    before: np.ndarray,
    after: np.ndarray,
    device: str = "cpu",
    tile: int = CHANGEFORMER_TILE,
    ckpt_path: Path | None = None,
    domain: str | None = None,
) -> dict[str, Any]:
    """ChangeFormerV6, 256 tiles, ImageNet-style [-1,1] norm. Domain selects ckpt after attach."""
    import torch

    net, device, ckpt = _load_changeformer(device=device, ckpt_path=ckpt_path, domain=domain)
    a = np.asarray(before)
    b = np.asarray(after)
    if a.shape != b.shape:
        b = np.asarray(Image.fromarray(b).resize((a.shape[1], a.shape[0]), Image.Resampling.BILINEAR))
    h, w = a.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    t0 = time.perf_counter()
    n_tiles = 0
    with torch.no_grad():
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                y2 = min(y + tile, h)
                x2 = min(x + tile, w)
                ta = a[y:y2, x:x2]
                tb = b[y:y2, x:x2]
                ph, pw = ta.shape[0], ta.shape[1]
                if ph != tile or pw != tile:
                    pad_a = np.zeros((tile, tile, 3), dtype=ta.dtype)
                    pad_b = np.zeros((tile, tile, 3), dtype=tb.dtype)
                    pad_a[:ph, :pw] = ta
                    pad_b[:ph, :pw] = tb
                    ta, tb = pad_a, pad_b
                ba, bb = _tensorize_pair(ta, tb, device)
                out = net(ba, bb)
                if isinstance(out, (list, tuple)):
                    logits = out[-1]
                else:
                    logits = out
                pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
                mask[y:y2, x:x2] = pred[:ph, :pw]
                n_tiles += 1
    elapsed = time.perf_counter() - t0
    sha = sha256_file(ckpt)
    team = sha == TEAM_SECOND_SHA
    return {
        "mask": mask,
        "rung": "team_second_live" if team else "changeformer_v6",
        "rung_label": (
            "housing-change (live) — Housing Change Specialist (SECOND-tuned)"
            if team
            else "general change (live) — General Change Model (imported); "
            "ChangeFormerV6 pretrained on LEVIR-CD (zero extra training)"
        ),
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha,
        "domain": (domain or "levir"),
        "device": device,
        "tile": tile,
        "n_tiles": n_tiles,
        "elapsed_s": round(elapsed, 3),
        "norm": "to_tensor then mean=0.5 std=0.5 (ChangeFormer eval protocol)",
        "provenance": "tools.change_detect_changeformer",
    }


def change_detect(
    before: np.ndarray | str | Path,
    after: np.ndarray | str | Path,
    prefer: str = "changeformer",
    device: str = "cpu",
    gt_mask: np.ndarray | None = None,
    ckpt_path: Path | None = None,
    domain: str | None = None,
) -> dict[str, Any]:
    """Public tool. Tries ChangeFormer; on failure walks the spec §5 ladder."""
    global LAST_CD_RUNG
    if not isinstance(before, np.ndarray):
        before = load_rgb(before)
    if not isinstance(after, np.ndarray):
        after = load_rgb(after)
    errors: list[str] = []
    result = None
    if prefer == "changeformer":
        try:
            result = change_detect_changeformer(
                before, after, device=device, ckpt_path=ckpt_path, domain=domain
            )
        except Exception as e:
            errors.append(f"ChangeFormer failed: {type(e).__name__}: {e}")
            result = None
    if result is None:
        # Rung 2 (Siamese-UNet) skipped: no verified pretrained weights downloaded.
        result = change_detect_classical(before, after)
        result["fallback_errors"] = errors
        result["siamese_unet"] = "skipped — no verified pretrained weights located"
    LAST_CD_RUNG = result["rung"]
    mask = result["mask"]
    out = {k: v for k, v in result.items() if k != "mask"}
    out["mask"] = mask
    out["mask_shape"] = list(mask.shape)
    out["changed_pixels"] = int(mask.sum())
    out.update(change_direction_proxy(before, after, mask))
    if gt_mask is not None:
        out["vs_gt"] = mask_metrics(mask, gt_mask)
    attach_rung_display(out)
    return out


def sar_read(
    vv: np.ndarray,
    vh: np.ndarray,
    water_db_threshold: float = SAR_WATER_DB,
    calibrated: bool = True,
) -> dict[str, Any]:
    """Classical VV/VH stats + water threshold. No learned optical↔SAR fusion.

    3×3 boxcar on dB before thresholding. SDWI = ln(10·VV_lin·VH_lin)−8
    (linear power). Water mask = speckle-filtered VV_dB < threshold + 3×3 morph
    open. SDWI is reported, not used for the mask.

    If calibrated is False (8-bit PNG/JPEG preview DN), do not apply −16 dB
    science and do not use median>5 as a unit guess.
    """
    vv_in = np.asarray(vv)
    vh_in = np.asarray(vh)
    uncal = (not calibrated) or vv_in.dtype == np.uint8 or vh_in.dtype == np.uint8
    vv = np.asarray(vv_in, dtype=np.float32)
    vh = np.asarray(vh_in, dtype=np.float32)
    if vv.ndim == 3:
        vv = vv[..., 0]
    if vh.ndim == 3:
        vh = vh[..., 0]

    def _stats(x: np.ndarray) -> dict[str, float]:
        f = x[np.isfinite(x)]
        if f.size == 0:
            return {"mean": float("nan"), "std": float("nan"), "p5": float("nan"), "p95": float("nan")}
        return {
            "mean": float(np.mean(f)),
            "std": float(np.std(f)),
            "p5": float(np.percentile(f, 5)),
            "p95": float(np.percentile(f, 95)),
        }

    if uncal:
        method = (
            "8-bit/preview DN; -16 dB threshold not applied; "
            "not a calibrated water product; stats are preview DN"
        )
        z = np.zeros(vv.shape, dtype=np.uint8)
        return {
            "vv_db_stats": None,
            "vh_db_stats": None,
            "dn_stats": _stats(vv),
            "sdwi_stats": None,
            "sdwi_finite": False,
            "sdwi_finite_n": 0,
            "sdwi_label": "SDWI",
            "sdwi_units": None,
            "water_db_threshold": None,
            "water_mask": z,
            "water_pixels": None,
            "water_fraction": None,
            "water_calibrated": False,
            "speckle": "not applied (uncalibrated preview)",
            "method": method,
            "provenance": "tools.sar_read (preview DN; deterministic tool measurement). " + method,
        }

    def _to_db(x: np.ndarray) -> np.ndarray:
        finite = x[np.isfinite(x)]
        if finite.size == 0:
            return x
        # reBEN S1 is typically already dB-ish (negative to low positive). If
        # values look linear (mostly > 5), convert. Never used as sole authority
        # on 8-bit data (gated above).
        if float(np.nanmedian(finite)) > 5.0:
            return 10.0 * np.log10(np.clip(x, 1e-10, None))
        return x

    vv_db = _uniform3(_to_db(vv))
    vh_db = _uniform3(_to_db(vh))

    vv_lin = np.power(10.0, np.clip(vv_db, -80.0, 20.0) / 10.0)
    vh_lin = np.power(10.0, np.clip(vh_db, -80.0, 20.0) / 10.0)
    prod = np.clip(10.0 * vv_lin * vh_lin, 1e-30, None)
    sdwi = np.log(prod) - 8.0
    bad = ~np.isfinite(vv_db) | ~np.isfinite(vh_db)
    sdwi[bad] = np.nan
    sdwi_finite_n = int(np.isfinite(sdwi).sum())

    water = (vv_db < water_db_threshold).astype(np.uint8)
    try:
        import cv2

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        water = (cv2.morphologyEx(water * 255, cv2.MORPH_OPEN, k) > 0).astype(np.uint8)
    except Exception:
        pass
    n = int(water.sum())
    frac = n / float(water.size) if water.size else 0.0
    method = (
        f"3x3 boxcar on VV/VH dB; water = (filtered VV_dB < {water_db_threshold}) "
        "+ 3x3 morph open; SDWI reported as letters SDWI, not used for mask"
    )
    return {
        "vv_db_stats": _stats(vv_db),
        "vh_db_stats": _stats(vh_db),
        "sdwi_stats": _stats(sdwi),
        "sdwi_finite": sdwi_finite_n > 0,
        "sdwi_finite_n": sdwi_finite_n,
        "sdwi_label": "SDWI",
        "sdwi_units": "ln(10*VV_linear*VH_linear)-8; VV_lin=10**(dB/10)",
        "water_db_threshold": water_db_threshold,
        "water_mask": water,
        "water_pixels": n,
        "water_fraction": frac,
        "water_calibrated": True,
        "speckle": "3x3 boxcar (uniform) on dB before threshold",
        "method": method,
        "provenance": "tools.sar_read (classical backscatter; deterministic tool measurement). " + method,
    }


# sar_agreement thresholds (module level, documented):
#   AGREE_IOU_T     — IoU at/above this means both modalities support water
#   AGREE_MIN_FRAC  — water fraction below this counts as "absent" for a modality
#   AGREE_MISREG_PX — ingest misregistration above this many px withholds agreement
AGREE_IOU_T = 0.25
AGREE_MIN_FRAC = 0.005
AGREE_MISREG_PX = 5.0


def sar_agreement(
    optical_water_mask: np.ndarray,
    sar_water_mask: np.ndarray,
    *,
    sar_calibrated: bool = True,
    misreg_shift_px: float | None = None,
) -> dict[str, Any]:
    """Mask agreement between two independent deterministic water tools.

    Compares tools.water_highlight (optical) and tools.sar_read (SAR) masks on a
    common grid. This is NOT pixel fusion and NOT learned fusion: it is an
    agreement check between two masks that already exist.

    If the mask shapes differ, the finer mask is resampled to the coarser shape
    with PIL NEAREST for comparison only; the input arrays are untouched and the
    resample is disclosed in `grid`.
    """
    om = np.asarray(optical_water_mask)
    sm = np.asarray(sar_water_mask)
    if om.ndim == 3:
        om = om[..., 0]
    if sm.ndim == 3:
        sm = sm[..., 0]
    om = om > 0
    sm = sm > 0
    grid = "same_grid"
    if om.shape != sm.shape:
        # Comparison-only resample of the derived mask; inputs untouched.
        if om.size > sm.size:
            om = np.asarray(
                Image.fromarray(om.astype(np.uint8) * 255).resize(
                    (sm.shape[1], sm.shape[0]), Image.Resampling.NEAREST
                )
            ) > 0
        else:
            sm = np.asarray(
                Image.fromarray(sm.astype(np.uint8) * 255).resize(
                    (om.shape[1], om.shape[0]), Image.Resampling.NEAREST
                )
            ) > 0
        grid = f"resampled_to_{om.shape[0]}x{om.shape[1]}_for_comparison"
    total = int(om.size)
    opt_px = int(om.sum())
    sar_px = int(sm.sum())
    opt_frac = (opt_px / total) if total else 0.0
    sar_frac = (sar_px / total) if total else 0.0
    inter = int(np.logical_and(om, sm).sum())
    union = int(np.logical_or(om, sm).sum())
    iou = (inter / union) if union else None

    if not sar_calibrated:
        water_verdict = "withheld_uncalibrated"
    elif misreg_shift_px is not None and float(misreg_shift_px) > AGREE_MISREG_PX:
        water_verdict = "withheld_misregistered"
    elif opt_frac < AGREE_MIN_FRAC and sar_frac < AGREE_MIN_FRAC:
        water_verdict = "both_absent"
    elif opt_frac >= AGREE_MIN_FRAC and sar_frac < AGREE_MIN_FRAC:
        water_verdict = "optical_only"
    elif sar_frac >= AGREE_MIN_FRAC and opt_frac < AGREE_MIN_FRAC:
        water_verdict = "sar_only"
    elif iou is not None and iou >= AGREE_IOU_T:
        water_verdict = "both_support"
    else:
        water_verdict = "disagree"

    return {
        "verdicts": {
            "water": water_verdict,
            # No optical built-up tool and no calibrated SAR built-up mask exist;
            # withheld is honesty, not a gap.
            "built_up": "withheld_no_tool",
        },
        "optical_water_px": opt_px,
        "sar_water_px": sar_px,
        "optical_water_fraction": round(float(opt_frac), 6),
        "sar_water_fraction": round(float(sar_frac), 6),
        "intersection_px": inter,
        "union_px": union,
        "iou": round(float(iou), 6) if iou is not None else None,
        "grid": grid,
        "optical_mask": om.astype(np.uint8),
        "sar_mask": sm.astype(np.uint8),
        "sar_calibrated": bool(sar_calibrated),
        "misreg_shift_px": misreg_shift_px,
        "thresholds": {
            "iou_t": AGREE_IOU_T,
            "min_frac": AGREE_MIN_FRAC,
            "misreg_px": AGREE_MISREG_PX,
        },
        "method": (
            "independent optical and SAR water masks compared on a common grid "
            "(per-pixel intersection/union IoU and per-modality water fractions)"
        ),
        "provenance": (
            "tools.sar_agreement (mask agreement; deterministic tool "
            "measurement). optical=tools.water_highlight mask; "
            "sar=tools.sar_read mask. Not pixel fusion."
        ),
    }


# --- coreg_check: optical+SAR co-registration basis -------------------------
#
# Two independent, measured levels — neither realigns anything:
#   transform level — compare affine transforms (origin offset in metres,
#     resolution equality) when both inputs are GeoTIFFs; unavailable is an
#     honest status, not silence.
#   pixel level — FFT phase cross-correlation between optical grayscale and
#     SAR VV normalized to 0-1 on the same grid, with parabolic sub-pixel
#     refinement. Cross-modal correlation is noisy: the estimate is reported
#     as measured-with-caveat, never asserted as zero.
# The pixel-level shift feeds sar_agreement's misreg_shift_px gate so a large
# measured misregistration withholds agreement instead of scoring silently.
COREG_MIN_STD = 1e-3
COREG_MIN_PEAK = 0.05


def _affine_status(src: Any) -> tuple[Any, str]:
    """(affine, status) for a raster source; status: ok|identity|unreadable."""
    try:
        import rasterio
    except ImportError:
        return None, "no_rasterio"
    try:
        with rasterio.open(str(src)) as ds:
            t = ds.transform
    except Exception:
        return None, "unreadable"
    if t is None or t == rasterio.Affine.identity():
        return None, "identity"
    return t, "ok"


def _norm01(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    lo = float(np.min(a))
    hi = float(np.max(a))
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _phase_corr_subpx(a01: np.ndarray, b01: np.ndarray) -> dict[str, Any]:
    """Phase correlation on same-shape 0-1 grids; sub-pixel via parabolic fit.

    Mirrors ingest._estimate_shift (normalized cross-power spectrum) plus a
    3-point parabolic interpolation around the integer peak.
    """
    a = np.asarray(a01, dtype=np.float32)
    b = np.asarray(b01, dtype=np.float32)
    base: dict[str, Any] = {
        "status": "inconclusive",
        "shift_px": None,
        "dx_px": None,
        "dy_px": None,
        "peak": None,
    }
    if a.shape != b.shape or a.ndim != 2:
        return {**base, "reason": "shape"}
    std_a = float(np.std(a))
    std_b = float(np.std(b))
    if std_a < COREG_MIN_STD or std_b < COREG_MIN_STD:
        return {
            **base,
            "reason": "uniform",
            "std_optical": std_a,
            "std_sar": std_b,
        }
    a = a - float(a.mean())
    b = b - float(b.mean())
    cross = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    corr = np.fft.ifft2(cross / (np.abs(cross) + 1e-12)).real
    iy, ix = np.unravel_index(int(np.argmax(corr)), corr.shape)
    peak = float(corr[iy, ix])
    h, w = corr.shape
    dy = int(iy) - (h if iy > h // 2 else 0)
    dx = int(ix) - (w if ix > w // 2 else 0)
    if peak < COREG_MIN_PEAK:
        return {
            **base,
            "reason": "weak_peak",
            "dx_px": float(dx),
            "dy_px": float(dy),
            "peak": round(peak, 4),
        }

    def _parab(c_m1: float, c_0: float, c_p1: float) -> float:
        denom = c_m1 - 2.0 * c_0 + c_p1
        return 0.0 if abs(denom) < 1e-12 else float(0.5 * (c_m1 - c_p1) / denom)

    dx_f = dx + _parab(float(corr[iy, (ix - 1) % w]), peak, float(corr[iy, (ix + 1) % w]))
    dy_f = dy + _parab(float(corr[(iy - 1) % h, ix]), peak, float(corr[(iy + 1) % h, ix]))
    return {
        "status": "measured",
        "shift_px": round(float(math.hypot(dx_f, dy_f)), 3),
        "dx_px": round(dx_f, 3),
        "dy_px": round(dy_f, 3),
        "peak": round(peak, 4),
    }


def coreg_check(
    optical_src: Any,
    sar_src: Any,
    *,
    optical_rgb: np.ndarray | None = None,
    sar_vv: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measured co-registration basis for an optical+SAR pair. No realignment."""
    t_opt, s_opt = _affine_status(optical_src)
    t_sar, s_sar = _affine_status(sar_src)
    if s_opt == "ok" and s_sar == "ok":
        dx_m = float(t_opt.c) - float(t_sar.c)
        dy_m = float(t_opt.f) - float(t_sar.f)
        transform = {
            "status": "measured",
            "offset_m": round(float(math.hypot(dx_m, dy_m)), 3),
            "dx_m": round(dx_m, 3),
            "dy_m": round(dy_m, 3),
            "same_res": bool(
                abs(t_opt.a - t_sar.a) < 1e-6 and abs(t_opt.e - t_sar.e) < 1e-6
            ),
            "optical_res_m": [round(abs(float(t_opt.a)), 6), round(abs(float(t_opt.e)), 6)],
            "sar_res_m": [round(abs(float(t_sar.a)), 6), round(abs(float(t_sar.e)), 6)],
            "basis": "affine transform comparison (rasterio .transform)",
        }
    else:
        transform = {
            "status": "unavailable",
            "reason": f"optical_transform={s_opt} sar_transform={s_sar}",
            "offset_m": None,
            "same_res": None,
            "basis": "affine transform comparison (rasterio .transform)",
        }

    if optical_rgb is None or sar_vv is None:
        pixel = {
            "status": "inconclusive",
            "reason": "inputs_missing",
            "shift_px": None,
            "dx_px": None,
            "dy_px": None,
            "peak": None,
        }
    else:
        rgb = np.asarray(optical_rgb, dtype=np.float32)
        g = rgb.mean(axis=-1) if rgb.ndim == 3 else rgb
        pixel = _phase_corr_subpx(_norm01(g), _norm01(sar_vv))
    pixel["method"] = (
        "FFT phase cross-correlation (normalized cross-power spectrum) on 0-1 "
        "optical-grayscale vs SAR-VV grids; sub-pixel via parabolic peak fit"
    )
    pixel["caveat"] = (
        "cross-modal optical-vs-SAR correlation is noisy; measured-with-caveat, "
        "not ground truth"
    )
    return {
        "transform": transform,
        "pixel": pixel,
        "shift_px": pixel.get("shift_px") if pixel.get("status") == "measured" else None,
        "provenance": (
            "tools.coreg_check (deterministic tool measurement). "
            "transform level: " + str(transform.get("basis"))
        ),
    }


# --- export_mask_geotiff: GeoTIFF sidecar for mask/grid artifacts -----------
#
# Additive export only: PNGs stay for the UI; the .tif carries the source
# raster's own CRS + affine (scaled native->mask grid when the preview was
# downscaled). A source with no real transform records geo_export=
# skipped_no_transform — a recorded event, not a silent omission.
def export_mask_geotiff(
    mask: np.ndarray,
    src_raster: Any,
    out_path: Any,
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> dict[str, Any]:
    """Write a single-band uint8 GeoTIFF of `mask` on `src_raster`'s grid.

    The affine is `src.transform * Affine.scale(scale_x, scale_y)` — exact for
    extent-preserving resizes (including rotated transforms). Read-only on the
    source; never fabricates a CRS.
    """
    try:
        import rasterio
    except ImportError:
        return {"status": "skipped_no_rasterio"}
    try:
        with rasterio.open(str(src_raster)) as ds:
            crs = ds.crs
            t = ds.transform
    except Exception:
        return {"status": "skipped_unreadable", "source": str(src_raster)}
    if t is None or t == rasterio.Affine.identity():
        return {"status": "skipped_no_transform", "source": str(src_raster)}
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    m = m.astype(np.uint8)
    t_used = t @ rasterio.Affine.scale(scale_x, scale_y)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out,
        "w",
        driver="GTiff",
        width=int(m.shape[1]),
        height=int(m.shape[0]),
        count=1,
        dtype="uint8",
        crs=crs,
        transform=t_used,
    ) as ds:
        ds.write(m, 1)
    return {
        "status": "written",
        "path": str(out),
        "crs": str(crs) if crs is not None else None,
        "source": str(src_raster),
        "transform_scaled": bool(scale_x != 1.0 or scale_y != 1.0),
        "provenance": "tools.export_mask_geotiff (copy of the "
        "source raster's crs+transform; deterministic tool measurement).",
    }


# --- cdvqa_map: deterministic second_semantic packet -> CDVQA answer -------
#
# Answer vocabulary is the 19-label CDVQA target set, read from the official
# annotation JSONs (Val/Test/Train answers): {yes, no}, 6 class tokens (gold
# casing: "NVG_surface"), and 11 change-ratio bins {"0", "0_to_10", ...,
# "90_to_100"}. Mapper rules were verified against TRAIN gold labels +
# answers (never eval): change_or_not/change_to_what/change_ratio/
# change_ratio_types/increase_or_not/decrease_or_not reproduce 100% of train
# answers; largest/smallest_change ~99% (residual is generator tie noise).
CDVQA_CLASS_TOKENS: tuple[str, ...] = (
    "water",
    "NVG_surface",
    "low_vegetation",
    "trees",
    "buildings",
    "playgrounds",
)
CDVQA_BIN_TOKENS: tuple[str, ...] = (
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
CDVQA_ANSWER_VOCAB: frozenset[str] = frozenset(
    ("yes", "no") + CDVQA_CLASS_TOKENS + CDVQA_BIN_TOKENS
)
# Generator default when a dominance question has no changed pixels at all
# (verified on train gold: all-zero hist -> "nvg_surface").
CDVQA_EMPTY_CHANGE_TOKEN = "NVG_surface"
# smallest_change ignores classes below this share of changed pixels — under
# a 0.417-mIoU backbone, sub-1% participation is mask noise, not a class.
CDVQA_MIN_SMALLEST_SHARE = 0.01
# Semantic-token -> answer-vocab casing (gold uses NVG_surface).
_CDVQA_TOKEN_CASE = {"nvg_surface": "NVG_surface"}

CDVQA_FAMILY_OF_TYPE: dict[str, str] = {
    "change_or_not": "binary",
    "change_to_what": "transition",
    "largest_change": "dominance",
    "smallest_change": "dominance",
    "increase_or_not": "direction",
    "decrease_or_not": "direction",
    "built_up_direction": "direction",
    "change_ratio": "ratio",
    "change_ratio_types": "ratio",
}
CDVQA_CLAIM_OF_TYPE: dict[str, str] = {
    "change_or_not": "change_detected",
    "change_to_what": "primary_transition",
    "largest_change": "largest_change_class",
    "smallest_change": "largest_change_class",
    "increase_or_not": "trend",
    "decrease_or_not": "trend",
    "built_up_direction": "trend",
    "change_ratio": "change_ratio",
    "change_ratio_types": "change_ratio",
}
_CDQA_RATIO_NEG = ("not changed", "non-change", "non-changed", "unchanged")


def _cdvqa_semantic_helpers() -> dict[str, Any]:
    """Lazy cf_ft.semantic import (demo keeps tools.py self-contained)."""
    sat = Path(__file__).resolve().parent.parent
    if str(sat) not in sys.path:
        sys.path.insert(0, str(sat))
    from cf_ft.semantic import (  # noqa: E402
        BUILT_UP_CLASS_ID,
        CLASS_ID_TO_TOKEN,
        RATIO_BINS,
        bin_ratio,
        classify_semantic_family,
        extract_class_id,
        extract_time_side,
    )

    return {
        "BUILT_UP_CLASS_ID": BUILT_UP_CLASS_ID,
        "CLASS_ID_TO_TOKEN": CLASS_ID_TO_TOKEN,
        "RATIO_BINS": RATIO_BINS,
        "bin_ratio": bin_ratio,
        "classify_semantic_family": classify_semantic_family,
        "extract_class_id": extract_class_id,
        "extract_time_side": extract_time_side,
    }


def _cdvqa_family(question: str | None, official_type: str | None = None) -> str | None:
    """Official type from annotations, else classify natural text.

    Mirrors cf_ft.semantic.classify_semantic_family and adds the two families
    it does not detect: per-class binary ("did the areas of X change?") and
    the global with/without-change ratio ("how much of the area changed?").
    """
    t = str(official_type or "").strip()
    if t:
        return t
    q = (question or "").strip().lower()
    if not q:
        return None
    h = _cdvqa_semantic_helpers()
    fam = h["classify_semantic_family"](q)
    if fam == "change_ratio_types" and h["extract_class_id"](q) is None:
        # "change ratio of the imagery" / "how much area changed" — no class.
        return "change_ratio"
    if fam is not None:
        return fam
    cid = h["extract_class_id"](q)
    # Official template variants the compiler's detector does not cover:
    # "what type of change is the largest/smallest" (dominance),
    # "change proportion/percentage of <C>" (per-class ratio), and
    # paraphrased "change(d) to" (transition).
    if "type of change" in q:
        if "largest" in q:
            return "largest_change"
        if "smallest" in q:
            return "smallest_change"
    if re.search(r"\b(changed|change|became|turned) to\b", q) and cid is not None:
        return "change_to_what"
    if re.search(r"(percent|ratio|proportion|how much)", q) and re.search(
        r"chang|unchanged|non-?change", q
    ):
        return "change_ratio_types" if cid is not None else "change_ratio"
    if cid is not None and re.search(
        r"\b(did|do|does|have|has|is|are|was|were)\b", q
    ) and re.search(r"\bchang", q):
        return "change_or_not"
    return None


def _cdvqa_argmax(vals: list[int]) -> int | None:
    """argmax with first-index ties (np.argmax order); None when all zero."""
    m = max(vals, default=0)
    if m <= 0:
        return None
    for i, v in enumerate(vals):
        if v == m:
            return i
    return None


def _cdvqa_argmin_pos(vals: list[int], floor: int = 1) -> int | None:
    """argmin over entries >= floor, first-index ties; None when none."""
    pos = [i for i, v in enumerate(vals) if v >= floor]
    if not pos:
        return None
    m = min(vals[i] for i in pos)
    for i in pos:
        if vals[i] == m:
            return i
    return None


def cdvqa_map(
    question: str | None,
    packet: dict[str, Any] | None = None,
    *,
    semantic: dict[str, Any] | None = None,
    official_type: str | None = None,
    vocab: frozenset[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Deterministic second_semantic packet -> CDVQA canonical answer string.

    No VLM in the scored path. Reads the ``semantic`` tool output's compiler
    ``features`` (from_to / hists / class counts over the predicted-change
    mask) either directly or inside a packet's ``tool_outputs.semantic``.

    Family -> claim -> rule (train-gold verified):
      binary      change_or_not               -> change_detected
                  yes iff the named class participates in predicted change
                  (t1->from_hist, t2->to_hist, unqualified->either)
      transition  change_to_what              -> primary_transition
                  dominant to-class among predicted transitions of the named
                  class (from_to row argmax); no predicted transition and the
                  class exists at t1 -> the class itself (it stayed); class
                  absent at t1 -> withhold.
      dominance   largest_change/smallest     -> largest_change_class
                  argmax / argmin-above-1%-share of the side hist
                  (t2->to, t1->from, unqualified->from+to); empty -> NVG_surface
      direction   increase_or_not/decrease    -> trend
                  count_b vs count_a of the named class -> yes/no
                  (3-way "increased, decreased, or unchanged" withholds: the
                  trend is real but not a yes/no CDVQA string)
      ratio       change_ratio/change_ratio_types -> change_ratio
                  bin(changed/total), negated phrasing -> bin(unchanged/total);
                  per-class type -> bin(count_side[class]/total)

    With/without-change ratio polarity is explicit in the question text;
    unparseable polarity or an unknown family withholds rather than guesses.
    Outputs are always members of CDVQA_ANSWER_VOCAB or withheld.
    """
    h = _cdvqa_semantic_helpers()
    vocab_set = vocab if vocab is not None else CDVQA_ANSWER_VOCAB
    family = _cdvqa_family(question, official_type)
    base: dict[str, Any] = {
        "answer": None,
        "answer_vocab_ok": False,
        "family": CDVQA_FAMILY_OF_TYPE.get(str(family or "")),
        "official_type": family,
        "claim": CDVQA_CLAIM_OF_TYPE.get(str(family or "")),
        "question": question,
        "withheld": True,
        "withheld_reason": None,
        "raw_value": None,
        "question_class": None,
        "question_side": None,
        "method": "cdvqa_map_v1 (deterministic packet->vocab map over second_semantic features)",
        "provenance": (
            "tools.cdvqa_map (deterministic tool measurement). "
            "source=second_semantic packet features over tools.change_detect "
            "mask. Not learned."
        ),
    }
    if family is None:
        base["withheld_reason"] = "unknown_family"
        return base
    if family not in CDVQA_FAMILY_OF_TYPE:
        base["withheld_reason"] = f"unsupported_family:{family}"
        return base

    sem_rec = semantic
    if sem_rec is None and packet:
        sem_rec = (packet.get("tool_outputs") or {}).get("semantic") or {}
    feat = (sem_rec or {}).get("features") or {}
    if not feat or "from_to" not in feat:
        base["withheld_reason"] = "no_semantic_evidence"
        return base

    n_cls = len(CDVQA_CLASS_TOKENS)
    from_to = np.asarray(feat.get("from_to"), dtype=np.int64).reshape(n_cls, n_cls)
    from_hist = [int(x) for x in (feat.get("from_hist") or [0] * n_cls)][:n_cls]
    to_hist = [int(x) for x in (feat.get("to_hist") or [0] * n_cls)][:n_cls]
    count_a = [int(x) for x in (feat.get("count_a") or [0] * n_cls)][:n_cls]
    count_b = [int(x) for x in (feat.get("count_b") or [0] * n_cls)][:n_cls]
    changed = int(feat.get("changed_pixels") or 0)
    total = int(feat.get("total_pixels") or 0)
    cid = h["extract_class_id"](question)
    side = h["extract_time_side"](question)
    base["question_side"] = side
    tok = h["CLASS_ID_TO_TOKEN"]

    def _vocab_tok(class_id_1based: int) -> str:
        return _CDVQA_TOKEN_CASE.get(tok[class_id_1based], tok[class_id_1based])

    def _emit(answer: str | None, raw: Any = None, reason: str | None = None) -> dict[str, Any]:
        out = dict(base)
        out["raw_value"] = raw if raw is not None else answer
        if answer is not None and answer in vocab_set:
            out.update({"answer": answer, "answer_vocab_ok": True,
                        "withheld": False, "withheld_reason": None})
        elif answer is not None:
            out["withheld_reason"] = f"vocab_mismatch:{answer}"
        else:
            out["withheld_reason"] = reason or "withheld"
        return out

    if family in ("change_or_not", "change_to_what", "change_ratio_types"):
        if cid is None:
            return _emit(None, reason="no_named_class")
        base["question_class"] = _vocab_tok(cid)

    if family == "change_or_not":
        if side == "t2":
            hit = to_hist[cid - 1] > 0
        elif side == "t1":
            hit = from_hist[cid - 1] > 0
        else:
            hit = from_hist[cid - 1] > 0 or to_hist[cid - 1] > 0
        return _emit("yes" if hit else "no", raw={"class_changed": bool(hit)})

    if family == "change_to_what":
        row = [int(x) for x in from_to[cid - 1]]
        i = _cdvqa_argmax(row)
        if i is not None:
            return _emit(_vocab_tok(i + 1), raw={"from_class": _vocab_tok(cid)})
        if count_a[cid - 1] > 0:
            # named class present at t1 with no predicted transition -> it stayed
            return _emit(_vocab_tok(cid), raw={"from_class": _vocab_tok(cid),
                                               "note": "no_predicted_transition"})
        return _emit(None, reason="no_evidence_for_class")

    if family in ("largest_change", "smallest_change"):
        hist = to_hist if side == "t2" else (
            from_hist if side == "t1"
            else [a + b for a, b in zip(from_hist, to_hist)]
        )
        if family == "smallest_change":
            floor = max(1, int(round(CDVQA_MIN_SMALLEST_SHARE * changed)))
            i = _cdvqa_argmin_pos(hist, floor)
            if i is None:
                i = _cdvqa_argmin_pos(hist, 1)
        else:
            i = _cdvqa_argmax(hist)
        if i is None:
            return _emit(CDVQA_EMPTY_CHANGE_TOKEN, raw="empty_change_default")
        return _emit(_vocab_tok(i + 1))

    if family in ("increase_or_not", "decrease_or_not"):
        if cid is None:
            cid = int(h["BUILT_UP_CLASS_ID"])
        base["question_class"] = _vocab_tok(cid)
        delta = count_b[cid - 1] - count_a[cid - 1]
        trend = "increase" if delta > 0 else ("decrease" if delta < 0 else "no_change")
        if family == "increase_or_not":
            return _emit("yes" if delta > 0 else "no", raw={"trend": trend})
        return _emit("yes" if delta < 0 else "no", raw={"trend": trend})

    if family == "built_up_direction":
        cid = cid or int(h["BUILT_UP_CLASS_ID"])
        base["question_class"] = _vocab_tok(cid)
        delta = count_b[cid - 1] - count_a[cid - 1]
        trend = "increase" if delta > 0 else ("decrease" if delta < 0 else "no_change")
        return _emit(None, raw={"trend": trend},
                     reason="three_way_direction_is_not_a_cdvqa_vocab_string")

    if family == "change_ratio":
        ql = (question or "").lower()
        if any(m in ql for m in _CDQA_RATIO_NEG):
            neg = True
        elif re.search(r"chang", ql):
            neg = False
        else:
            return _emit(None, reason="ambiguous_ratio_polarity")
        share = (changed / total) if total else 0.0
        if neg:
            share = 1.0 - share
        return _emit(h["bin_ratio"](share), raw={"share": round(share, 6), "negated": neg})

    if family == "change_ratio_types":
        cnt = count_b[cid - 1] if side == "t2" else count_a[cid - 1]
        share = (cnt / total) if total else 0.0
        return _emit(h["bin_ratio"](share), raw={"class_share": round(share, 6)})

    return _emit(None, reason=f"unsupported_family:{family}")


def vv_to_preview(vv: np.ndarray) -> Image.Image:
    x = np.asarray(vv, dtype=np.float32)
    if x.ndim == 3:
        x = x[..., 0]
    lo, hi = np.percentile(x[np.isfinite(x)], (2, 98)) if np.isfinite(x).any() else (0, 1)
    if hi <= lo:
        hi = lo + 1
    scaled = np.clip((x - lo) / (hi - lo), 0, 1)
    gray = (scaled * 255).astype(np.uint8)
    return Image.fromarray(gray, mode="L").convert("RGB")


def vqa(
    prompt: str,
    image_paths: list[str | Path],
    url: str = DEFAULT_VLM_URL,
    model: str = DEFAULT_VLM_MODEL,
    max_tokens: int = 220,
    timeout: float = 180.0,
) -> dict[str, Any]:
    """Call local llama-server. Returns text + latencies. No commercial APIs."""
    parts: list[dict] = [{"type": "text", "text": prompt}]
    used = []
    for p in image_paths:
        path = Path(p)
        raw = path.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        mime = _mime(path)
        parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        used.append(str(path))
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": parts}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    first_token_s = None
    chunks: list[str] = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        while True:
            line = resp.readline()
            if not line:
                break
            s = line.decode("utf-8", errors="replace").strip()
            if not s:
                continue
            if s.startswith("data: "):
                s = s[6:]
            if s == "[DONE]":
                break
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            delta = (((obj.get("choices") or [{}])[0]).get("delta") or {}).get("content")
            if not delta:
                # some builds send the full message on the last event
                delta = (((obj.get("choices") or [{}])[0]).get("message") or {}).get("content")
            if delta:
                if first_token_s is None:
                    first_token_s = time.perf_counter() - t0
                chunks.append(delta)
    complete_s = time.perf_counter() - t0
    text = "".join(chunks).strip()
    return {
        "text": text,
        "model": model,
        "url": url,
        "n_images": len(used),
        "image_paths": used,
        "first_token_s": None if first_token_s is None else round(first_token_s, 3),
        "complete_s": round(complete_s, 3),
        "provenance": "tools.vqa (local llama-server; no commercial vision API)",
    }


def vqa_nonstream(
    prompt: str,
    image_paths: list[str | Path],
    url: str = DEFAULT_VLM_URL,
    model: str = DEFAULT_VLM_MODEL,
    max_tokens: int = 220,
    timeout: float = 180.0,
) -> dict[str, Any]:
    """Non-stream fallback if the server ignores stream=true."""
    parts: list[dict] = [{"type": "text", "text": prompt}]
    used = []
    for p in image_paths:
        path = Path(p)
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        parts.append({"type": "image_url", "image_url": {"url": f"data:{_mime(path)};base64,{b64}"}})
        used.append(str(path))
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": parts}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    complete_s = time.perf_counter() - t0
    text = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    return {
        "text": text.strip(),
        "model": model,
        "url": url,
        "n_images": len(used),
        "image_paths": used,
        "first_token_s": None,
        "complete_s": round(complete_s, 3),
        "provenance": "tools.vqa_nonstream (local llama-server)",
    }


def ask_vlm(prompt: str, image_paths: list[str | Path], **kwargs) -> dict[str, Any]:
    try:
        out = vqa(prompt, image_paths, **kwargs)
        if out["text"]:
            return out
    except Exception:
        pass
    return vqa_nonstream(prompt, image_paths, **kwargs)


# Canonical answer seat: adapted Qwen3VL-8B LR-fold (LoRA+projector merged,
# HR-replay; quantized Q4_K_M) served by a second llama-server on
# 127.0.0.1:8091. Narrator stays on :8080 — never repoint it. SHAs from
# gates/qwen3vl/canonical/SHA256SUMS.txt (SWAP-8091; prior ATTACH-8091
# artifacts retained on disk as the rollback path).
CANONICAL_VLM_URL = "http://127.0.0.1:8091"
CANONICAL_VLM_SEAT = "127.0.0.1:8091"
CANONICAL_MODEL_LABEL = "canonical-lrfold"
CANONICAL_GGUF_SHA256 = (
    "24df79c4e181033b68aeafc4319d62da517a9277eabfaf377d662c6372a56a2e"
)
CANONICAL_MMPROJ_SHA256 = (
    "3197a0db6d988baea40cf5dc894e93cc09fcc02e1041740e944a6b59b0610b74"
)
# Decode contract mirrors the eval recipe (work/smoke_8091.py): greedy,
# repetition_penalty 1.08, 16 max tokens — canonical answers are 1-5 tokens.
CANONICAL_DECODE = {
    "temperature": 0.0,
    "repeat_penalty": 1.08,
    "repetition_penalty": 1.08,
    "max_tokens": 16,
}


def canonical_vqa(
    prompt: str,
    image_paths: list[str | Path],
    url: str = CANONICAL_VLM_URL,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Canonical short answer from the adapted RSVQA model served on :8091.

    Deterministic greedy decode (temperature 0, repeat_penalty 1.08,
    max_tokens 16) against a separate llama-server seat — NOT the :8080
    narrator. On any server error returns a structured unavailable record;
    there is never a silent fallback to the narrator (different model,
    different provenance).
    """
    parts: list[dict] = [{"type": "text", "text": prompt}]
    used = []
    for p in image_paths:
        path = Path(p)
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        parts.append(
            {"type": "image_url", "image_url": {"url": f"data:{_mime(path)};base64,{b64}"}}
        )
        used.append(str(path))
    out: dict[str, Any] = {
        "available": False,
        "answer": None,
        "text": "",
        "model": None,
        "model_label": CANONICAL_MODEL_LABEL,
        "gguf_sha256": CANONICAL_GGUF_SHA256,
        "mmproj_sha256": CANONICAL_MMPROJ_SHA256,
        "url": url,
        "seat": CANONICAL_VLM_SEAT,
        "decode": dict(CANONICAL_DECODE),
        "n_images": len(used),
        "image_paths": used,
        "latency_s": None,
        "error": None,
        "provenance": (
            "tools.canonical_vqa — 127.0.0.1:8091 unavailable; withheld "
            "(no narrator substitution: different model, different provenance)"
        ),
    }
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(
            url.rstrip("/") + "/v1/models", timeout=min(5.0, timeout)
        ) as resp:
            models = json.loads(resp.read().decode("utf-8"))
        served_id = ((models.get("data") or [{}])[0]).get("id")
        out["model"] = served_id
        payload = {
            "model": served_id or CANONICAL_MODEL_LABEL,
            "messages": [{"role": "user", "content": parts}],
            "max_tokens": CANONICAL_DECODE["max_tokens"],
            "temperature": CANONICAL_DECODE["temperature"],
            "repeat_penalty": CANONICAL_DECODE["repeat_penalty"],
            "repetition_penalty": CANONICAL_DECODE["repetition_penalty"],
            "stream": False,
        }
        req = urllib.request.Request(
            url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        text = (
            ((body.get("choices") or [{}])[0].get("message") or {}).get("content")
            or ""
        ).strip()
        out.update(
            {
                "available": True,
                "text": text,
                "answer": re.sub(r"\s+", " ", text.lower()).rstrip(".") or None,
                "latency_s": round(time.perf_counter() - t0, 3),
                "provenance": (
                    "tools.canonical_vqa (adapted RSVQA specialist on "
                    "127.0.0.1:8091; greedy decode; not the :8080 narrator)"
                ),
            }
        )
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["latency_s"] = round(time.perf_counter() - t0, 3)
    return out


def narration_prompt(query: str, tool_json: dict[str, Any], task: str) -> str:
    """Evidence-fusion prompt: VLM may only use numbers present in tool_json."""
    payload = json.dumps(tool_json, indent=2, default=str)
    return (
        "You are SatQuery AI, an offline satellite analyst. "
        "Use ONLY numbers that appear in the TOOL EVIDENCE JSON below. "
        "If a number is not in the JSON, say you do not have a measurement "
        "rather than inventing one. Cite the tool name when you quote a figure.\n\n"
        f"TASK: {task}\nQUERY: {query}\n\nTOOL EVIDENCE JSON:\n{payload}\n\n"
        "Write a short analyst answer (4–8 sentences). Mention where in the image "
        "(e.g. northern sector) only if the JSON has quadrant/region fields. "
        "Write the acronym SDWI as SDWI — do not expand it. "
        "radiometric_label is brightness inside the change mask only; "
        "do not treat it as construction or 'unchanged built-up'. "
        "built_up_direction is not_determined."
    )


class ToolTests(unittest.TestCase):
    def test_area_calc_known(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:10, :10] = 1  # 100 pixels
        out = area_calc(mask, gsd_m=0.5, label="built-up")
        self.assertEqual(out["changed_pixels"], 100)
        self.assertAlmostEqual(out["area_m2"], 100 * 0.25)
        self.assertAlmostEqual(out["area_km2"], 25 / 1_000_000)
        self.assertEqual(out["dominant_quadrant"], "NW")
        self.assertIn("provenance", out)

    def test_area_calc_withheld_without_gsd(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[:2, :2] = 1
        out = area_calc(mask, gsd_m=None, label="upload")
        self.assertEqual(out["changed_pixels"], 4)
        self.assertIsNone(out["area_m2"])
        self.assertIsNone(out["area_km2"])
        self.assertIn("withheld", out["formula"])

    def test_mask_metrics_perfect(self) -> None:
        m = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        met = mask_metrics(m, m)
        self.assertEqual(met["iou"], 1.0)
        self.assertEqual(met["f1"], 1.0)

    def test_mask_metrics_zero(self) -> None:
        pred = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        gt = np.array([[0, 1], [0, 0]], dtype=np.uint8)
        met = mask_metrics(pred, gt)
        self.assertEqual(met["tp"], 0)
        self.assertEqual(met["iou"], 0.0)

    def test_classical_change_detect(self) -> None:
        before = np.zeros((64, 64, 3), dtype=np.uint8)
        after = np.zeros((64, 64, 3), dtype=np.uint8)
        after[20:40, 20:40] = 255
        out = change_detect_classical(before, after)
        self.assertEqual(out["rung"], "classical_otsu")
        self.assertGreater(int(out["mask"].sum()), 50)
        self.assertLess(int(out["mask"].sum()), 64 * 64 * 0.5)

    def test_sar_read_threshold(self) -> None:
        # Patch 16×16 so 3×3 boxcar still leaves a dark core above 50 water px.
        vv = np.full((32, 32), -8.0, dtype=np.float32)
        vv[:16, :16] = -22.0
        vh = vv.copy()
        out = sar_read(vv, vh, water_db_threshold=-16.0)
        self.assertGreaterEqual(out["water_pixels"], 50)
        self.assertIn("vv_db_stats", out)
        self.assertIn("provenance", out)
        self.assertTrue(out["sdwi_finite"])
        self.assertGreater(out["sdwi_finite_n"], 0)
        self.assertTrue(math.isfinite(out["sdwi_stats"]["mean"]))
        self.assertEqual(out["sdwi_label"], "SDWI")
        self.assertTrue(out["water_calibrated"])
        self.assertIn("SDWI reported as letters SDWI", out["method"])

    def test_sar_read_sdwi_tiny_pair(self) -> None:
        vv = np.array([[-10.0, -20.0], [-12.0, -18.0]], dtype=np.float32)
        vh = np.array([[-11.0, -19.0], [-13.0, -17.0]], dtype=np.float32)
        out = sar_read(vv, vh)
        self.assertTrue(out["sdwi_finite"])
        self.assertEqual(out["sdwi_finite_n"], 4)
        self.assertTrue(math.isfinite(out["sdwi_stats"]["mean"]))

    def test_water_highlight_blue_blob(self) -> None:
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb[10:30, 10:30] = (20, 40, 200)
        out = water_highlight(rgb)
        mask = out["mask"]
        blob = mask[10:30, 10:30]
        outside = mask.copy()
        outside[10:30, 10:30] = 0
        self.assertGreaterEqual(int(blob.sum()), 20 * 20 * 0.8)
        self.assertLess(int(outside.sum()), 64)
        self.assertLess(out["water_pixels"], 64 * 64 * 0.5)
        self.assertIn("B-R", out["method"])

    def test_change_direction_brighter_after(self) -> None:
        before = np.full((32, 32, 3), 10, dtype=np.uint8)
        after = np.full((32, 32, 3), 10, dtype=np.uint8)
        after[8:24, 8:24] = 200
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 1
        out = change_direction_proxy(before, after, mask)
        self.assertEqual(out["radiometric_label"], "brighter_after")
        self.assertEqual(out["built_up_direction"], "not_determined")
        self.assertNotIn("direction", out)
        self.assertGreater(out["delta_mean"], 8.0)

    def test_change_direction_identical(self) -> None:
        img = np.full((16, 16, 3), 40, dtype=np.uint8)
        mask = np.ones((16, 16), dtype=np.uint8)
        out = change_direction_proxy(img, img, mask)
        self.assertEqual(out["radiometric_label"], "similar")
        self.assertEqual(out["built_up_direction"], "not_determined")
        self.assertAlmostEqual(out["delta_mean"], 0.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
