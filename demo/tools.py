"""Specialist tools for the SatQuery demo (DEMO-SPEC-05).

Each function returns a structured dict (values + provenance).
The VLM never computes these numbers.

Tools:
  vqa            — llama-server Qwen3-VL-8B (OpenAI-compatible)
  change_detect  — ChangeFormerV6 LEVIR-CD pretrained; classical OpenCV fallback
  area_calc      — pixel count * GSD^2
  sar_read       — VV/VH backscatter stats + water threshold (no learned fusion)
"""
from __future__ import annotations

import base64
import json
import mimetypes
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
LEVIR_DIR = SATQUERY / "gates" / "_cache" / "levir_cd"
DEFAULT_VLM_URL = "http://127.0.0.1:8080"
DEFAULT_VLM_MODEL = "qwen3vl"
LEVIR_GSD_M = 0.5
S2_GSD_M = 10.0
CHANGEFORMER_TILE = 256

# Rung recorded at import-of-prepare time; change_detect fills this per call.
LAST_CD_RUNG = "uninitialized"


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
    gsd_m: float,
    label: str = "changed",
) -> dict[str, Any]:
    """Pixel count × GSD². Also reports which image quadrant holds most of the class."""
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    binary = m > 0
    h, w = binary.shape
    n = int(binary.sum())
    total = int(binary.size)
    px_m2 = float(gsd_m) * float(gsd_m)
    area_m2 = n * px_m2
    area_km2 = area_m2 / 1_000_000.0
    pct = (100.0 * n / total) if total else 0.0
    # north-up assumption for "sector"
    mid_r, mid_c = h // 2, w // 2
    quads = {
        "NW": int(binary[:mid_r, :mid_c].sum()),
        "NE": int(binary[:mid_r, mid_c:].sum()),
        "SW": int(binary[mid_r:, :mid_c].sum()),
        "SE": int(binary[mid_r:, mid_c:].sum()),
    }
    dominant = max(quads, key=quads.get) if n else "none"
    north_frac = ((quads["NW"] + quads["NE"]) / n) if n else 0.0
    return {
        "label": label,
        "changed_pixels": n,
        "total_pixels": total,
        "gsd_m": float(gsd_m),
        "pixel_area_m2": px_m2,
        "area_m2": area_m2,
        "area_km2": area_km2,
        "percent_of_image": pct,
        "quadrants": quads,
        "dominant_quadrant": dominant,
        "north_fraction": north_frac,
        "formula": "area_m2 = count(mask>0) * gsd_m^2",
        "provenance": "tools.area_calc (raster math; VLM did not compute this)",
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
    return {
        "mask": mask,
        "rung": "classical_otsu",
        "rung_label": "classical baseline (abs-diff + Otsu + morphology)",
        "otsu_threshold": float(_thr),
        "method": "cv2.absdiff -> grayscale -> THRESH_OTSU -> morph open/close 5x5",
        "provenance": "tools.change_detect_classical",
    }


def _find_changeformer_ckpt() -> Path | None:
    for p in sorted(CF_CKPT_DIR.rglob("best_ckpt.pt")):
        if p.is_file() and p.stat().st_size > 1_000_000:
            return p
    for p in sorted(CF_CKPT_DIR.rglob("*.pt")):
        if p.is_file() and p.stat().st_size > 1_000_000:
            return p
    return None


_CF_NET = None
_CF_DEVICE = None
_CF_CKPT = None
_CF_ERROR = None


def _load_changeformer(device: str = "cpu"):
    """Lazy-load ChangeFormerV6. Returns (net, device, ckpt_path) or raises."""
    global _CF_NET, _CF_DEVICE, _CF_CKPT, _CF_ERROR
    if _CF_NET is not None and _CF_DEVICE == device:
        return _CF_NET, _CF_DEVICE, _CF_CKPT
    ckpt = _find_changeformer_ckpt()
    if ckpt is None:
        raise FileNotFoundError(f"No ChangeFormer .pt under {CF_CKPT_DIR}")
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
) -> dict[str, Any]:
    """Rung 1: pretrained ChangeFormerV6, 256 tiles, ImageNet-style [-1,1] norm."""
    import torch

    net, device, ckpt = _load_changeformer(device=device)
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
    return {
        "mask": mask,
        "rung": "changeformer_v6",
        "rung_label": "ChangeFormerV6 pretrained on LEVIR-CD (zero extra training)",
        "checkpoint": str(ckpt),
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
            result = change_detect_changeformer(before, after, device=device)
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
    if gt_mask is not None:
        out["vs_gt"] = mask_metrics(mask, gt_mask)
    return out


def sar_read(
    vv: np.ndarray,
    vh: np.ndarray,
    water_db_threshold: float = -16.0,
) -> dict[str, Any]:
    """Classical VV/VH stats + water threshold. No learned optical↔SAR fusion."""
    vv = np.asarray(vv, dtype=np.float32)
    vh = np.asarray(vh, dtype=np.float32)
    if vv.ndim == 3:
        vv = vv[..., 0]
    if vh.ndim == 3:
        vh = vh[..., 0]

    def _to_db(x: np.ndarray) -> np.ndarray:
        finite = x[np.isfinite(x)]
        if finite.size == 0:
            return x
        # reBEN S1 is typically already dB-ish (negative to low positive). If
        # values look linear (mostly > 5), convert.
        if float(np.nanmedian(finite)) > 5.0:
            return 10.0 * np.log10(np.clip(x, 1e-10, None))
        return x

    vv_db = _to_db(vv)
    vh_db = _to_db(vh)

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

    water = (vv_db < water_db_threshold).astype(np.uint8)
    try:
        import cv2

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        water = (cv2.morphologyEx(water * 255, cv2.MORPH_OPEN, k) > 0).astype(np.uint8)
    except Exception:
        pass
    n = int(water.sum())
    frac = n / float(water.size) if water.size else 0.0
    return {
        "vv_db_stats": _stats(vv_db),
        "vh_db_stats": _stats(vh_db),
        "water_db_threshold": water_db_threshold,
        "water_mask": water,
        "water_pixels": n,
        "water_fraction": frac,
        "method": (
            f"water = (VV_dB < {water_db_threshold}) + 3x3 morph open; "
            "late fusion only (structured stats, not learned pixel fusion)"
        ),
        "provenance": "tools.sar_read (classical backscatter; VLM did not compute this)",
    }


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
        "(e.g. northern sector) only if the JSON has quadrant/region fields."
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
        vv = np.full((32, 32), -8.0, dtype=np.float32)
        vv[:8, :8] = -22.0
        vh = vv.copy()
        out = sar_read(vv, vh, water_db_threshold=-16.0)
        self.assertGreaterEqual(out["water_pixels"], 50)
        self.assertIn("vv_db_stats", out)
        self.assertIn("provenance", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
