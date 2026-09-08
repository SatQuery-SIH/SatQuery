"""FINALE-HARDEN-10 helpers: downloadable markdown report, TIFF ingest, measurement card.

Inference still uses prepared scene_paths (test_45 / VRSBench / BEN Lithuania).
Uploads are ingest preview / validation — not the judge's Cartosat pair.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

DEMO = Path(__file__).resolve().parent
REPORTS = DEMO / "reports"
HONESTY = (
    "INGEST PREVIEW / VALIDATION — not your Cartosat pair. "
    "Inference still uses prepared demo assets via scene_paths "
    "(Scene 2 = LEVIR-CD test_45). Uploaded GeoTIFF/PNG is shown here only."
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def measurement_markdown(trace: dict[str, Any] | None) -> str:
    """Judge-pointable area_calc card. Whole-image pixels ≠ NE quadrant pixels."""
    if not trace:
        return "_No run yet. Run a scene to fill this card._"
    area = (trace.get("tool_outputs") or {}).get("area_calc")
    lines = [
        "### Measurement card (tool `area_calc` — VLM did not compute these)",
        "",
    ]
    if not area:
        lines.append("No `area_calc` on this run (caption/VQA/refusal).")
        return "\n".join(lines)
    quads = area.get("quadrants") or {}
    ne = int(quads.get("NE") or 0)
    whole = int(area.get("changed_pixels") or 0)
    gsd = float(area.get("gsd_m") or 0)
    px_m2 = float(area.get("pixel_area_m2") or (gsd * gsd))
    ne_m2 = ne * px_m2
    lines += [
        f"- **label:** `{area.get('label')}`",
        f"- **changed_pixels (WHOLE IMAGE):** {whole:,}",
        f"- **GSD:** {gsd} m/px",
        f"- **formula:** `{area.get('formula')}`",
        f"- **area_m2 (whole):** {area.get('area_m2')}",
        f"- **area_km2 (whole):** {area.get('area_km2')}",
        f"- **percent_of_image (whole):** {area.get('percent_of_image')}",
        f"- **quadrants (pixels):** NW={quads.get('NW')} NE={quads.get('NE')} "
        f"SW={quads.get('SW')} SE={quads.get('SE')}",
        f"- **dominant_quadrant:** {area.get('dominant_quadrant')}",
        f"- **provenance:** {area.get('provenance')}",
        "",
        "**Do not conflate:**",
        f"- Whole-image `{whole:,}` px × {px_m2} m²/px = **{area.get('area_m2')} m²** "
        f"({float(area.get('percent_of_image') or 0):.4f}% of image).",
        f"- NE quadrant `{ne:,}` px × {px_m2} m²/px = **{ne_m2} m²** "
        f"(not the whole-image km² / percent).",
        "- If the VLM paragraph attaches NE pixels to whole-image percent, **the JSON card wins.**",
    ]
    return "\n".join(lines)


def confidence_markdown(trace: dict[str, Any] | None) -> str:
    if not trace:
        return "_No run yet._"
    plan = trace.get("plan") or {}
    tools = plan.get("tools") or []
    vlm_role = plan.get("vlm_role")
    lines = [
        "### Confidence hierarchy (honesty, not a fake 0.99)",
        "",
        "**Tool measurements** — raster math / ChangeFormer / SAR threshold. "
        "These are the numbers a judge should cite.",
    ]
    tout = trace.get("tool_outputs") or {}
    if "area_calc" in tout:
        a = tout["area_calc"]
        lines.append(
            f"- `area_calc`: {a.get('changed_pixels')} px → {a.get('area_m2')} m² "
            f"@ {a.get('gsd_m')} m GSD ({a.get('provenance')})."
        )
    if "change_detect" in tout:
        c = tout["change_detect"]
        vs = c.get("vs_gt") or {}
        iou = vs.get("iou")
        lines.append(
            f"- `change_detect`: rung `{c.get('rung')}`; "
            + (f"IoU vs GT {iou} (mask_metrics)." if iou is not None else "no GT IoU this call.")
        )
    if "sar_read" in tout:
        s = tout["sar_read"]
        lines.append(
            f"- `sar_read`: water_pixels={s.get('water_pixels')} "
            f"({s.get('provenance', 'tools.sar_read')})."
        )
    if not tout:
        lines.append("- None this run (planner refused or caption-only).")
    lines += [
        "",
        f"**VLM interpretation (not a measurement)** — role `{vlm_role}`; tools selected: {tools}. "
        "The answer paragraph may mis-attach numbers. Do not treat prose km² as independent evidence.",
    ]
    if not plan.get("supported"):
        lines.append(f"- Refusal (not a count/area): {plan.get('refusal')}")
    return "\n".join(lines)


def write_report(trace: dict[str, Any], extra_note: str = "") -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = REPORTS / f"satquery_run_{stamp}.md"
    area = (trace.get("tool_outputs") or {}).get("area_calc") or {}
    body = [
        "# SatQuery run report",
        "",
        f"- UTC: `{_now()}`",
        f"- mode: `{'live' if trace.get('live') else 'cached'}`",
        f"- query: {trace.get('query')!r}",
        f"- input_mode: `{trace.get('input_mode')}` scene=`{trace.get('scene')}`",
        f"- {HONESTY}",
        "",
        extra_note,
        "",
        "## Plan",
        "",
        "```json",
        json.dumps(trace.get("plan"), indent=2, default=str),
        "```",
        "",
        measurement_markdown(trace),
        "",
        confidence_markdown(trace),
        "",
        "## Answer (VLM interpretation — not a measurement)",
        "",
        trace.get("answer") or "_empty_",
        "",
        "## Image / mask paths",
        "",
    ]
    for p in trace.get("images") or []:
        body.append(f"- image: `{p}`")
    if trace.get("overlay_path"):
        body.append(f"- overlay/mask: `{trace.get('overlay_path')}`")
    if area:
        body.append(
            f"- area_calc snapshot: changed_pixels={area.get('changed_pixels')} "
            f"GSD={area.get('gsd_m')} area_m2={area.get('area_m2')} "
            f"quadrants={area.get('quadrants')}"
        )
    body += [
        "",
        "## Full tool_outputs",
        "",
        "```json",
        json.dumps(trace.get("tool_outputs"), indent=2, default=str),
        "```",
        "",
        f"- first_token_s: {trace.get('first_token_s')} complete_s: {trace.get('complete_s')}",
    ]
    path.write_text("\n".join(body), encoding="utf-8")
    return path


def _percentile_stretch(arr) -> Image.Image:
    import numpy as np

    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    if a.ndim == 3 and a.shape[0] in (1, 3, 4) and a.shape[-1] not in (1, 3, 4):
        a = np.moveaxis(a, 0, -1)
    if a.ndim == 3 and a.shape[-1] > 3:
        a = a[..., :3]
    if a.ndim == 3 and a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    out = np.zeros_like(a, dtype=np.uint8)
    for c in range(min(3, a.shape[-1])):
        band = a[..., c]
        lo, hi = np.percentile(band, (2, 98))
        if hi <= lo:
            hi = lo + 1.0
        scaled = (band - lo) / (hi - lo)
        out[..., c] = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out[..., :3], mode="RGB")


def ingest_preview(src: str | Path) -> tuple[Image.Image | None, str]:
    """PNG/JPEG stay RGB. GeoTIFF/TIFF → RGB preview. Never claims Cartosat inference."""
    if src is None:
        return None, "no file"
    path = Path(str(src))
    if not path.is_file():
        return None, f"missing: {path}"
    suf = path.suffix.lower()
    note_prefix = HONESTY + " "
    pil_note = ""
    if suf in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        im = Image.open(path).convert("RGB")
        return im, note_prefix + f"PNG/JPEG preview `{path.name}` ({im.size[0]}×{im.size[1]})."
    if suf not in {".tif", ".tiff"}:
        return None, note_prefix + f"Unsupported suffix {suf!r}. Use PNG/JPEG/TIFF."

    try:
        import rasterio

        with rasterio.open(path) as ds:
            n = ds.count
            if n >= 3:
                arr = ds.read([1, 2, 3])
            else:
                arr = ds.read(1)
            im = _percentile_stretch(arr)
            crs = ds.crs
            return im, (
                note_prefix
                + f"GeoTIFF via rasterio `{path.name}` bands={n} size={im.size} crs={crs}. "
                "RGB stretch is preview only."
            )
    except ImportError:
        pass
    except Exception as e:
        pil_note = f"rasterio read failed ({type(e).__name__}: {e}); falling back to PIL. "
    else:
        pil_note = ""

    try:
        im = Image.open(path)
        im.seek(0)
        frame = im.convert("RGB") if im.mode != "RGB" else im.copy()
        return frame, (
            note_prefix
            + pil_note
            + "FALLBACK: rasterio/GDAL missing or failed. First TIFF page/band rendered to RGB. "
            f"`{path.name}` mode={im.mode} size={frame.size}. Not a georeferenced analysis."
        )
    except Exception as e:
        return None, note_prefix + f"TIFF ingest failed: {type(e).__name__}: {e}"
