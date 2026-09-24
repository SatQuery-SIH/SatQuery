"""Raster ingest for demo inference: GSD from GeoTIFF, RGB for tools/VLM.

Prepared PNG/JPEG scenes keep their benchmark GSD constants (caller passes them).
Uploads never silently inherit LEVIR 0.5 m / Sentinel-2 10 m.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

DEMO = Path(__file__).resolve().parent
UPLOAD_DIR = DEMO / "data" / "_uploads"
MAX_INFER_EDGE = 1024
M_PER_DEG = 111_320.0
ASSUME_METERS_MIN = 0.05
ASSUME_METERS_MAX = 100.0
DEGREE_SCALE_MAX = 0.01

TIFF_MODEL_PIXEL_SCALE = 33550
TIFF_MODEL_TIEPOINT = 33922
TIFF_MODEL_TRANSFORM = 34264
TIFF_GEO_KEY_DIRECTORY = 34735
GT_MODEL_TYPE_GEOKEY = 1024
MODEL_TYPE_PROJECTED = 1
MODEL_TYPE_GEOGRAPHIC = 2

IMAGE_SUFFIX = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
TIFF_SUFFIX = {".tif", ".tiff"}
NPZ_SUFFIX = {".npz"}


def _path(src: str | Path | None) -> Path | None:
    if src is None or src == "":
        return None
    if isinstance(src, dict):
        src = src.get("path") or src.get("name")
        if not src:
            return None
    return Path(str(src))


def percentile_stretch(arr: Any) -> Image.Image:
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


def _tag_value(tags: Any, code: int):
    if tags is None:
        return None
    if hasattr(tags, "get"):
        item = tags.get(code)
        if item is None:
            item = tags.get(str(code))
        if item is None:
            return None
        if hasattr(item, "value"):
            return item.value
        return item
    return None


def _as_floats(value) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, (bytes, bytearray)):
        arr = np.frombuffer(value, dtype="<f8")
        return tuple(float(x) for x in arr)
    if np.isscalar(value):
        return (float(value),)
    try:
        return tuple(float(x) for x in value)
    except TypeError:
        return ()


def _model_type_from_geokeys(raw) -> int | None:
    vals = []
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        vals = list(np.frombuffer(raw, dtype="<u2"))
    else:
        try:
            vals = [int(x) for x in raw]
        except TypeError:
            return None
    if len(vals) < 8:
        return None
    n_keys = vals[3]
    for i in range(n_keys):
        base = 4 + i * 4
        if base + 3 >= len(vals):
            break
        key_id, location, _count, offset = vals[base : base + 4]
        if key_id == GT_MODEL_TYPE_GEOKEY and location == 0:
            return int(offset)
    return None


def _read_geotiff_tags(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "scale": None,
        "tiepoint": None,
        "transform": None,
        "model_type": None,
        "backend": None,
    }
    try:
        import tifffile

        with tifffile.TiffFile(str(path)) as tf:
            page = tf.pages[0]
            tags = page.tags
            out["scale"] = _as_floats(_tag_value(tags, TIFF_MODEL_PIXEL_SCALE))
            out["tiepoint"] = _as_floats(_tag_value(tags, TIFF_MODEL_TIEPOINT))
            out["transform"] = _as_floats(_tag_value(tags, TIFF_MODEL_TRANSFORM))
            out["model_type"] = _model_type_from_geokeys(
                _tag_value(tags, TIFF_GEO_KEY_DIRECTORY)
            )
            out["backend"] = "tifffile"
            return out
    except Exception:
        pass
    try:
        im = Image.open(path)
        tags = getattr(im, "tag_v2", None)
        out["scale"] = _as_floats(_tag_value(tags, TIFF_MODEL_PIXEL_SCALE))
        out["tiepoint"] = _as_floats(_tag_value(tags, TIFF_MODEL_TIEPOINT))
        out["transform"] = _as_floats(_tag_value(tags, TIFF_MODEL_TRANSFORM))
        out["model_type"] = _model_type_from_geokeys(
            _tag_value(tags, TIFF_GEO_KEY_DIRECTORY)
        )
        out["backend"] = "pil_tiff_tags"
        im.close()
    except Exception:
        out["backend"] = None
    return out


def _gsd_from_scale(
    scale_x: float,
    scale_y: float,
    *,
    model_type: int | None,
    tie_y: float | None,
    crs_geographic: bool | None = None,
    real_transform: bool | None = None,
) -> tuple[float | None, str, str]:
    """Return (gsd_m, source, provenance). source is geotransform | assumed_meters | none.

    A non-identity affine is a real geotransform regardless of whether the
    CRS parses to a real EPSG (rasterio reads some projected GeoKeys as
    LOCAL_CS) — so real_transform=True keeps source="geotransform".
    "assumed_meters" is reserved for rasters with no real transform
    (identity/default), where the scale is just pixel-size-1 placeholders.
    """
    sx, sy = abs(float(scale_x)), abs(float(scale_y))
    if sx == 0.0 or sy == 0.0 or not math.isfinite(sx) or not math.isfinite(sy):
        return None, "none", "GeoTIFF scale is zero or non-finite; m² withheld."

    geographic = crs_geographic
    if geographic is None:
        if model_type == MODEL_TYPE_GEOGRAPHIC:
            geographic = True
        elif model_type == MODEL_TYPE_PROJECTED:
            geographic = False
        else:
            geographic = sx <= DEGREE_SCALE_MAX and sy <= DEGREE_SCALE_MAX

    if geographic:
        lat = float(tie_y) if tie_y is not None and math.isfinite(tie_y) else 0.0
        lat = max(-89.9, min(89.9, lat))
        gsd_x = sx * M_PER_DEG * math.cos(math.radians(lat))
        gsd_y = sy * M_PER_DEG
        gsd = (gsd_x + gsd_y) / 2.0
        return (
            gsd,
            "geotransform",
            f"geographic pixel scale ({sx}, {sy}) deg → meters at lat={lat:.4f} "
            f"(mean of x/y); backend tags.",
        )

    gsd = (sx + sy) / 2.0
    if model_type == MODEL_TYPE_PROJECTED or real_transform:
        return (
            gsd,
            "geotransform",
            f"projected/local pixel scale ({sx}, {sy}) treated as metres; "
            "mean used.",
        )
    if ASSUME_METERS_MIN <= gsd <= ASSUME_METERS_MAX:
        return (
            gsd,
            "assumed_meters",
            f"pixel scale ({sx}, {sy}) has no CRS; assumed metres because "
            f"{ASSUME_METERS_MIN}–{ASSUME_METERS_MAX} m is a typical RS GSD. "
            "Not a dataset constant.",
        )
    return (
        None,
        "none",
        f"pixel scale ({sx}, {sy}) is not a plausible metre GSD and CRS is missing; "
        "m² withheld.",
    )


def read_gsd(src: str | Path) -> dict[str, Any]:
    """GSD in metres for one file. Never invents LEVIR/S2 constants."""
    path = _path(src)
    empty = {
        "path": str(path) if path else None,
        "gsd_m": None,
        "gsd_x_m": None,
        "gsd_y_m": None,
        "source": "none",
        "crs": None,
        "crs_note": None,
        "native_width": None,
        "native_height": None,
        "provenance": "no file",
    }
    if path is None or not path.is_file():
        return empty
    suf = path.suffix.lower()
    native_w = native_h = None
    try:
        with Image.open(path) as im:
            native_w, native_h = im.size
    except Exception:
        pass

    if suf in IMAGE_SUFFIX:
        return {
            **empty,
            "path": str(path),
            "native_width": native_w,
            "native_height": native_h,
            "source": "none",
            "provenance": (
                f"PNG/JPEG `{path.name}` has no geotransform. "
                "area_m2 withheld unless the caller supplies a benchmark constant "
                "for a prepared scene."
            ),
        }

    gsd_m = None
    source = "none"
    provenance = f"`{path.name}`: no geotransform tags."
    crs = None
    crs_note = None
    backend = None

    try:
        import rasterio
        from affine import Affine

        with rasterio.open(path) as ds:
            native_w, native_h = int(ds.width), int(ds.height)
            t = ds.transform
            sx, sy = abs(float(t.a)), abs(float(t.e))
            # A non-identity affine carries a derivable GSD even when the CRS
            # does not parse to an EPSG authority (LOCAL_CS etc.).
            real_transform = t != Affine.identity()
            crs_obj = ds.crs
            crs = str(crs_obj) if crs_obj else None
            is_geo = None
            if crs_obj is not None:
                try:
                    is_geo = bool(crs_obj.is_geographic)
                except Exception:
                    is_geo = None
                try:
                    if crs_obj.to_epsg() is None:
                        crs_note = (
                            f"CRS has no EPSG authority ({crs}); "
                            "non-identity transform still yields GSD."
                        )
                except Exception:
                    crs_note = f"CRS unparseable ({crs}); transform yields GSD."
            else:
                crs_note = "no CRS in file; transform units treated as metres."
            tie_y = None
            try:
                tie_y = float((ds.bounds.top + ds.bounds.bottom) / 2.0) if is_geo else None
            except Exception:
                tie_y = None
            gsd_m, source, provenance = _gsd_from_scale(
                sx, sy, model_type=None, tie_y=tie_y, crs_geographic=is_geo,
                real_transform=real_transform,
            )
            backend = "rasterio"
            if gsd_m is not None:
                provenance = f"rasterio `{path.name}` crs={crs}. " + provenance
    except ImportError:
        pass
    except Exception as e:
        provenance = f"rasterio failed ({type(e).__name__}: {e}); trying TIFF tags."

    if gsd_m is None:
        tags = _read_geotiff_tags(path)
        backend = tags.get("backend") or backend
        scale = tags.get("scale") or ()
        tie = tags.get("tiepoint") or ()
        transform = tags.get("transform") or ()
        sx = sy = None
        if len(scale) >= 2:
            sx, sy = abs(scale[0]), abs(scale[1])
        elif len(transform) >= 6:
            # 4x4 row-major: pixel size ~ hypot of first two affine terms
            sx = math.hypot(transform[0], transform[1])
            sy = math.hypot(transform[4], transform[5])
        if sx and sy:
            tie_y = tie[4] if len(tie) >= 6 else None
            # scale/transform tags are themselves a real transform.
            gsd_m, source, provenance = _gsd_from_scale(
                sx, sy, model_type=tags.get("model_type"), tie_y=tie_y,
                real_transform=True,
            )
            if tags.get("model_type") is None and crs_note is None:
                crs_note = "no GeoKey CRS in file; transform units treated as metres."
            provenance = f"{backend or 'tags'} `{path.name}`. " + provenance

    return {
        "path": str(path),
        "gsd_m": None if gsd_m is None else float(gsd_m),
        "gsd_x_m": None,
        "gsd_y_m": None,
        "source": source,
        "crs": crs,
        "crs_note": crs_note,
        "native_width": native_w,
        "native_height": native_h,
        "backend": backend,
        "provenance": provenance,
    }


def scale_gsd_for_resize(
    gsd_info: dict[str, Any], native_w: int, used_w: int
) -> dict[str, Any]:
    """If tools run on a downsampled RGB, GSD must grow by native/used."""
    out = dict(gsd_info)
    if not native_w or not used_w or native_w == used_w:
        out["resize_scale"] = 1.0
        out["gsd_m_used"] = gsd_info.get("gsd_m")
        return out
    factor = float(native_w) / float(used_w)
    out["resize_scale"] = factor
    gsd = gsd_info.get("gsd_m")
    out["gsd_m_native"] = gsd
    out["gsd_m_used"] = None if gsd is None else float(gsd) * factor
    out["provenance"] = (
        f"{gsd_info.get('provenance')} Downsampled {native_w}→{used_w} px; "
        f"gsd_used = gsd_native × {factor:.6f}."
    )
    return out


# --- C1: explicit sensor profiles — band order is declared, never guessed ---
#
# A 4-band file is NOT automatically B,G,R,NIR (Cartosat-2S MX) — NAIP ships
# R,G,B,NIR — so band identity must come from a declared profile or from the
# file's own metadata (descriptions / colorinterp / channel_names). When
# neither exists the honest answer is "unidentified", and spectral claims
# withhold rather than guess.
SENSOR_PROFILES: dict[str, dict[str, int]] = {
    "cartosat2s_mx": {"blue": 1, "green": 2, "red": 3, "nir": 4},
    "cartosat_pan": {"pan": 1},
    "naip": {"red": 1, "green": 2, "blue": 3, "nir": 4},
    "sentinel2_10m": {"blue": 1, "green": 2, "red": 3, "nir": 4},
}

_BAND_TOKENS = {
    "red": {"red"},
    "green": {"green"},
    "blue": {"blue"},
    "nir": {"nir", "nir08", "b08", "b8", "b8a", "nearinfrared"},
    "pan": {"pan", "panchromatic"},
}

_SAR_POL_TOKENS = {"vv", "vh", "hh", "hv", "rh", "rv"}


def _band_tokens(text: str) -> set[str]:
    import re

    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _tiff_channel_names(path: Path) -> list[str]:
    """Per-band name strings for band-identity decisions.

    `channel_names=` (page description — our test/product convention and a
    common writer convention) wins. Otherwise per-band descriptions plus
    colorinterp — but colorinterp only counts when count <= 3: writers
    auto-tag 4-sample data as RGB+alpha, which is a display hint, not band
    identity (a B,G,R,NIR product would be misread as RGBA). [] when nothing
    names bands.
    """
    names: list[str] = []
    page_desc = ""
    try:
        import rasterio

        with rasterio.open(path) as ds:
            count = int(ds.count)
            for i in range(1, count + 1):
                desc = ds.descriptions[i - 1] or ""
                ci = ""
                if count <= 3:
                    try:
                        ci = getattr(ds.colorinterp[i - 1], "name", "") or ""
                    except Exception:
                        ci = ""
                names.append(" ".join(x for x in (desc, ci) if x))
            page_desc = str(ds.tags().get("TIFFTAG_IMAGEDESCRIPTION") or "")
    except Exception:
        names, page_desc = [], ""
    try:
        import tifffile

        with tifffile.TiffFile(str(path)) as tf:
            if tf.pages:
                d0 = str(getattr(tf.pages[0], "description", None) or "")
                if d0:
                    page_desc = (page_desc + " " + d0).strip()
    except Exception:
        pass
    for blob in names + [page_desc]:
        if "channel_names=" in blob:
            tail = blob.split("channel_names=", 1)[1]
            return [x.strip() for x in tail.split(",") if x.strip()]
    return names


def _map_named_bands(names: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, raw in enumerate(names, start=1):
        toks = _band_tokens(raw)
        for canon, keys in _BAND_TOKENS.items():
            if canon not in out and toks & keys:
                out[canon] = i
    return out


def _sar_pol_of(name: str) -> str | None:
    toks = _band_tokens(name)
    if toks & {"stokes", "mchi", "m-chi"}:
        return "stokes"
    hits = toks & _SAR_POL_TOKENS
    return sorted(hits)[0] if hits else None


def resolve_band_map(
    path: str | Path, sensor_profile: str | None = None
) -> tuple[dict[str, int] | None, str]:
    """({canonical_name: 1-based index} | None, provenance note).

    Evidence order: declared sensor profile → file metadata → None
    (unidentified — callers must withhold spectral claims, not guess).
    """
    p = _path(path)
    if p is None or not p.is_file():
        return None, f"missing: {path}"
    if p.suffix.lower() not in TIFF_SUFFIX:
        return {"red": 1, "green": 2, "blue": 3}, f"non-TIFF image `{p.name}` (implicit RGB)"
    if sensor_profile:
        prof = SENSOR_PROFILES.get(str(sensor_profile).strip().lower())
        if prof is None:
            return None, f"unknown sensor profile {sensor_profile!r}"
        try:
            import rasterio

            with rasterio.open(p) as ds:
                n = int(ds.count)
            if max(prof.values()) > n:
                return None, (
                    f"profile {sensor_profile!r} needs band {max(prof.values())}; "
                    f"`{p.name}` has {n}"
                )
        except ImportError:
            pass
        return dict(prof), f"declared sensor profile {sensor_profile}"
    names = _tiff_channel_names(p)
    m = _map_named_bands(names)
    if m:
        return m, f"band metadata {m}"
    return None, "bands unidentified (no declared profile, no band names/colorinterp)"


def _load_rgb_array(
    path: Path, sensor_profile: str | None = None
) -> tuple[np.ndarray | None, str]:
    suf = path.suffix.lower()
    if suf in IMAGE_SUFFIX:
        im = Image.open(path).convert("RGB")
        return np.asarray(im), f"RGB `{path.name}` {im.size[0]}×{im.size[1]}."
    if suf not in TIFF_SUFFIX:
        return None, f"unsupported suffix {suf!r}"

    rasterio_err = ""
    try:
        import rasterio

        band_map, bm_note = resolve_band_map(path, sensor_profile)
        with rasterio.open(path) as ds:
            n = ds.count
            if band_map and all(k in band_map for k in ("red", "green", "blue")):
                sel = [band_map["red"], band_map["green"], band_map["blue"]]
                sel_note = f" true-color bands={sel} ({bm_note})."
            else:
                sel = [1, 2, 3] if n >= 3 else [1]
                sel_note = (
                    " Bands 1-3 assumed for display only — band order "
                    "unidentified; spectral claims withheld."
                )
            arr = ds.read(sel) if n >= 3 else ds.read(1)
            im = percentile_stretch(arr)
            return np.asarray(im), (
                f"GeoTIFF via rasterio `{path.name}` bands={n} size={im.size} crs={ds.crs}."
                + sel_note
            )
    except ImportError:
        pass
    except Exception as e:
        rasterio_err = f"rasterio read failed ({type(e).__name__}: {e}). "

    try:
        import tifffile

        arr = tifffile.imread(str(path))
        im = percentile_stretch(arr)
        return np.asarray(im), (
            rasterio_err + f"GeoTIFF via tifffile `{path.name}` size={im.size}."
        )
    except Exception:
        pass
    try:
        im = Image.open(path)
        im.seek(0)
        frame = im.convert("RGB") if im.mode != "RGB" else im.copy()
        return np.asarray(frame), (
            rasterio_err
            + f"PIL TIFF page `{path.name}` mode={im.mode} size={frame.size}."
        )
    except Exception as e:
        return None, rasterio_err + f"TIFF ingest failed: {type(e).__name__}: {e}"


def maybe_downscale(rgb: np.ndarray, max_edge: int = MAX_INFER_EDGE) -> tuple[np.ndarray, int, int]:
    h, w = rgb.shape[:2]
    edge = max(h, w)
    if edge <= max_edge:
        return rgb, w, h
    scale = max_edge / float(edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    im = Image.fromarray(rgb).resize((new_w, new_h), Image.Resampling.BILINEAR)
    return np.asarray(im), new_w, new_h


def materialize_rgb(
    src: str | Path,
    dest: str | Path,
    max_edge: int = MAX_INFER_EDGE,
    sensor_profile: str | None = None,
) -> dict[str, Any]:
    """Write an 8-bit RGB PNG for ChangeFormer / llama.cpp. Returns size + note."""
    path = _path(src)
    dest_p = Path(dest)
    dest_p.parent.mkdir(parents=True, exist_ok=True)
    if path is None or not path.is_file():
        return {"ok": False, "error": f"missing: {src}", "path": None}
    rgb, note = _load_rgb_array(path, sensor_profile)
    if rgb is None:
        return {"ok": False, "error": note, "path": None}
    native_h, native_w = rgb.shape[:2]
    used, used_w, used_h = maybe_downscale(rgb, max_edge=max_edge)
    Image.fromarray(used).save(dest_p)
    return {
        "ok": True,
        "error": None,
        "path": dest_p,
        "native_width": native_w,
        "native_height": native_h,
        "used_width": used_w,
        "used_height": used_h,
        "note": note + ("" if (used_w, used_h) == (native_w, native_h) else f" Infer size {used_w}×{used_h}."),
    }


# --- MISREG-NOTE (additive): estimate rigid shift; do not warp or resize tool inputs ---
MISREG_DOWNSAMPLE_PX = 256
MISREG_SHIFT_BAR_PX = 5.0
MISREG_MIN_STD = 1e-3
MISREG_MIN_PEAK = 0.05
MISREG_NOTE_TEMPLATE = (
    "estimated rigid shift ≈ {n:.0f} px; change mask may include registration "
    "artifacts; no automatic realignment performed"
)


def _rgb_to_gray01(rgb: np.ndarray) -> np.ndarray:
    a = np.asarray(rgb, dtype=np.float32)
    if a.ndim == 3:
        a = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    if float(np.nanmax(a)) > 1.5:
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)


def _resize_square01(gray01: np.ndarray, size: int = MISREG_DOWNSAMPLE_PX) -> np.ndarray:
    u8 = np.clip(np.asarray(gray01) * 255.0, 0, 255).astype(np.uint8)
    im = Image.fromarray(u8, mode="L").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def _estimate_shift(before_g: np.ndarray, after_g: np.ndarray) -> dict[str, Any]:
    """Phase-correlation peak on already-downsampled grayscale (same HxW).

    Returns dx/dy in *that* grid (after relative to before). No warping.
    """
    a = np.asarray(before_g, dtype=np.float32)
    b = np.asarray(after_g, dtype=np.float32)
    if a.shape != b.shape or a.ndim != 2:
        return {"ok": False, "reason": "shape", "dx": None, "dy": None, "peak": None}
    std_a = float(np.std(a))
    std_b = float(np.std(b))
    if std_a < MISREG_MIN_STD or std_b < MISREG_MIN_STD:
        return {
            "ok": False,
            "reason": "uniform",
            "dx": None,
            "dy": None,
            "peak": None,
            "std_before": std_a,
            "std_after": std_b,
        }
    a = a - float(a.mean())
    b = b - float(b.mean())
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    cross = fa * np.conj(fb)
    mag = np.abs(cross)
    if float(np.median(mag)) < 1e-12:
        return {"ok": False, "reason": "degenerate_spectrum", "dx": None, "dy": None, "peak": None}
    ncc = cross / (mag + 1e-12)
    corr = np.fft.ifft2(ncc).real
    peak_idx = np.unravel_index(int(np.argmax(corr)), corr.shape)
    peak = float(corr[peak_idx])
    dy = int(peak_idx[0])
    dx = int(peak_idx[1])
    h, w = corr.shape
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w
    if peak < MISREG_MIN_PEAK:
        return {
            "ok": False,
            "reason": "weak_peak",
            "dx": float(dx),
            "dy": float(dy),
            "peak": peak,
        }
    return {
        "ok": True,
        "reason": None,
        "dx": float(dx),
        "dy": float(dy),
        "peak": peak,
    }


def pair_misreg_fields(before_src: str | Path, after_src: str | Path) -> dict[str, Any]:
    """Read-only shift check. Does not realign, resize, or alter tool rasters."""
    empty = {
        "misreg_check": "inconclusive",
        "misregistration_note": None,
        "misreg_shift_px": None,
        "misreg_dx_px": None,
        "misreg_dy_px": None,
        "misreg_peak": None,
        "misreg_downsample_px": MISREG_DOWNSAMPLE_PX,
        "misreg_bar_px": MISREG_SHIFT_BAR_PX,
    }
    bp, ap = _path(before_src), _path(after_src)
    if bp is None or ap is None or (not bp.is_file()) or (not ap.is_file()):
        return {**empty, "misreg_reason": "missing"}
    rgb_b, err_b = _load_rgb_array(bp)
    rgb_a, err_a = _load_rgb_array(ap)
    if rgb_b is None or rgb_a is None:
        return {**empty, "misreg_reason": err_b or err_a or "load"}
    orig_h, orig_w = rgb_b.shape[:2]
    g0 = _resize_square01(_rgb_to_gray01(rgb_b), MISREG_DOWNSAMPLE_PX)
    g1 = _resize_square01(_rgb_to_gray01(rgb_a), MISREG_DOWNSAMPLE_PX)
    est = _estimate_shift(g0, g1)
    if not est.get("ok"):
        return {
            **empty,
            "misreg_check": "inconclusive",
            "misreg_reason": est.get("reason"),
            "misreg_peak": est.get("peak"),
        }
    scale_x = float(orig_w) / float(MISREG_DOWNSAMPLE_PX)
    scale_y = float(orig_h) / float(MISREG_DOWNSAMPLE_PX)
    dx_px = float(est["dx"]) * scale_x
    dy_px = float(est["dy"]) * scale_y
    mag = float(math.hypot(dx_px, dy_px))
    note = None
    check = "ok"
    if mag > MISREG_SHIFT_BAR_PX:
        note = MISREG_NOTE_TEMPLATE.format(n=mag)
        check = "noted"
    return {
        "misreg_check": check,
        "misregistration_note": note,
        "misreg_shift_px": mag,
        "misreg_dx_px": dx_px,
        "misreg_dy_px": dy_px,
        "misreg_peak": est.get("peak"),
        "misreg_downsample_px": MISREG_DOWNSAMPLE_PX,
        "misreg_bar_px": MISREG_SHIFT_BAR_PX,
        "misreg_reason": None,
    }


def read_sar_arrays(src: str | Path) -> dict[str, Any]:
    """VV/VH arrays from npz, GeoTIFF, or 8-bit PNG (PNG is a last resort).

    Flag `calibrated` before any float32 cast. uint8 / PNG / JPEG are preview DN.
    """
    path = _path(src)
    if path is None or not path.is_file():
        return {"ok": False, "error": f"missing SAR: {src}"}
    suf = path.suffix.lower()
    if suf in NPZ_SUFFIX:
        blob = np.load(path)
        if "vv" not in blob:
            return {"ok": False, "error": f"{path.name} npz has no 'vv' array"}
        vv_raw = np.asarray(blob["vv"])
        vh_raw = np.asarray(blob["vh"]) if "vh" in blob else vv_raw.copy()
        calibrated = vv_raw.dtype != np.uint8 and vh_raw.dtype != np.uint8
        return {
            "ok": True,
            "vv": np.asarray(vv_raw, dtype=np.float32),
            "vh": np.asarray(vh_raw, dtype=np.float32),
            "calibrated": calibrated,
            "provenance": (
                f"npz `{path.name}` keys={blob.files} calibrated={calibrated}"
            ),
        }
    if suf in TIFF_SUFFIX:
        try:
            import tifffile

            names = _tiff_channel_names(path)
            pols = [_sar_pol_of(n) for n in names]
            bad = sorted({p for p in pols if p and p not in {"vv", "vh"}})
            if bad:
                return {
                    "ok": False,
                    "error": (
                        f"unsupported polarization {bad} in `{path.name}` — "
                        "sar_read needs VV/VH; RISAT hybrid-pol products are "
                        "Stokes-derived, not VV/VH"
                    ),
                }
            vv_i = next((i for i, p in enumerate(pols) if p == "vv"), None)
            vh_i = next((i for i, p in enumerate(pols) if p == "vh"), None)
            pol_verified = vv_i is not None and vh_i is not None

            arr_raw = np.asarray(tifffile.imread(str(path)))
            calibrated = arr_raw.dtype != np.uint8
            arr = np.asarray(arr_raw, dtype=np.float32)
            if arr.ndim == 3:
                bands_first = arr.shape[0] <= 4 and arr.shape[-1] > 4
                nb = arr.shape[0] if bands_first else arr.shape[-1]
                bi = vv_i if vv_i is not None else 0
                bj = vh_i if vh_i is not None else (1 if nb > 1 else 0)
                if bands_first:
                    vv = arr[bi]
                    vh = arr[bj]
                else:
                    vv = arr[..., bi]
                    vh = arr[..., bj]
            else:
                vv = arr
                vh = arr.copy()
            pol_note = (
                f"polarization vv=band{bi + 1},vh=band{bj + 1} verified from band names"
                if pol_verified
                else f"polarization ASSUMED vv=band{bi + 1},vh=band{bj + 1} (unnamed bands — not verified)"
            )
            return {
                "ok": True,
                "vv": vv,
                "vh": vh,
                "calibrated": calibrated,
                "pol_verified": pol_verified,
                "provenance": (
                    f"SAR raster `{path.name}` shape={tuple(np.asarray(vv).shape)} "
                    f"dtype={arr_raw.dtype} calibrated={calibrated}; {pol_note}"
                ),
            }
        except Exception:
            pass
    im = Image.open(path)
    arr_raw = np.asarray(im)
    calibrated = False if (suf in IMAGE_SUFFIX or arr_raw.dtype == np.uint8) else True
    arr = np.asarray(arr_raw, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return {
        "ok": True,
        "vv": arr,
        "vh": arr.copy(),
        "calibrated": calibrated,
        "provenance": (
            f"SAR from 8-bit `{path.name}` — not native backscatter. "
            "water_calibrated=false; -16 dB threshold not applied; preview DN."
        ),
    }


def preview_file(src: str | Path | None) -> tuple[Image.Image | None, str]:
    """RGB preview + note. Does not decide whether inference uses the file."""
    path = _path(src)
    if path is None:
        return None, "no file"
    if not path.is_file():
        return None, f"missing: {path}"
    rgb, note = _load_rgb_array(path)
    if rgb is None:
        return None, note
    gsd = read_gsd(path)
    gsd_s = (
        f"GSD={gsd['gsd_m']:.6f} m ({gsd['source']})"
        if gsd.get("gsd_m") is not None
        else f"GSD unknown ({gsd['source']})"
    )
    im = Image.fromarray(rgb)
    return im, f"{note} {gsd_s}. {gsd.get('provenance')}"


def write_test_geotiff(
    path: str | Path,
    array: np.ndarray,
    scale_x: float,
    scale_y: float,
    *,
    tie_x: float = 0.0,
    tie_y: float = 0.0,
    geographic: bool = False,
    channel_names: str | tuple[str, ...] | list[str] | None = None,
) -> Path:
    """Test helper: GeoTIFF with ModelPixelScale + optional geographic GeoKey."""
    import tifffile

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    extra = [
        (TIFF_MODEL_PIXEL_SCALE, "d", 3, (float(scale_x), float(scale_y), 0.0), True),
        (
            TIFF_MODEL_TIEPOINT,
            "d",
            6,
            (0.0, 0.0, 0.0, float(tie_x), float(tie_y), 0.0),
            True,
        ),
    ]
    if geographic:
        geokeys = (1, 1, 0, 1, GT_MODEL_TYPE_GEOKEY, 0, 1, MODEL_TYPE_GEOGRAPHIC)
        extra.append((TIFF_GEO_KEY_DIRECTORY, "H", len(geokeys), geokeys, True))
    else:
        geokeys = (1, 1, 0, 1, GT_MODEL_TYPE_GEOKEY, 0, 1, MODEL_TYPE_PROJECTED)
        extra.append((TIFF_GEO_KEY_DIRECTORY, "H", len(geokeys), geokeys, True))
    kw: dict[str, Any] = {"extratags": extra}
    if channel_names:
        if isinstance(channel_names, (list, tuple)):
            names = ",".join(str(x) for x in channel_names)
        else:
            names = str(channel_names)
        kw["description"] = f"channel_names={names}"
    tifffile.imwrite(str(dest), np.asarray(array), **kw)
    return dest
