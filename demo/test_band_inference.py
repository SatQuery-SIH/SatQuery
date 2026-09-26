"""BAND-INFER gate tests — the checked G=2/NIR=4 inference for unnamed
4-band TIFFs (2026-09-26, external review).

Evidence order in resolve_band_map: declared profile → band names →
inference → withhold. The inference only ever claims green/nir —
red-vs-blue stays "assumed" everywhere downstream.

Pre-registered thresholds (ingest._INFER_*):
  veg set V = top-5% band-4 px; accept iff
    median_V(b4) >= 1.5 * median_V(b3)   (NIR towers over band 3)
    median_V(b3) <= 0.9 * median_V(b2)   (band 3 absorbed — chlorophyll)
    p99(b4)     >= 1.3 * p50(b4)         (band 4 carries real variance)
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

DEMO = Path(__file__).resolve().parent
SAT = DEMO.parent
OPTICAL = SAT / "web" / "public" / "presets" / "sundarbans_optical.tiff"

H, W = 64, 64
HALF = np.index_exp[:, : W // 2]


def _two_zone(left: list[int], right: list[int]) -> np.ndarray:
    """(4,H,W) uint16 — left half gets `left` band values, right `right`."""
    arr = np.zeros((4, H, W), np.uint16)
    for i in range(4):
        arr[i, :, : W // 2] = left[i]
        arr[i, :, W // 2 :] = right[i]
    return arr


def _write(tmp: Path, name: str, arr: np.ndarray) -> Path:
    from ingest import write_test_geotiff

    return write_test_geotiff(tmp / name, arr, 1.0, 1.0, geographic=False)


class BandInferenceTests(unittest.TestCase):
    def test_bgrn_order_infers_and_water_measures(self) -> None:
        # B,G,R,NIR (Sentinel-2 / Cartosat-MX order): veg = left half.
        from ingest import resolve_band_map
        from tools import water_highlight

        tmp = Path(tempfile.mkdtemp())
        arr = _two_zone([700, 900, 450, 2600], [500, 800, 550, 150])
        tif = _write(tmp, "bgrn.tif", arr)
        m, note = resolve_band_map(tif)
        self.assertEqual(m, {"green": 2, "nir": 4}, note)
        self.assertIn("inferred", note)
        wh = water_highlight(np.zeros((H, W, 3), np.uint8), source_path=tif)
        self.assertFalse(wh.get("withheld"), wh.get("method"))
        frac = int(wh["water_pixels"]) / int(np.asarray(wh["mask"]).size)
        self.assertAlmostEqual(frac, 0.5, places=2)

    def test_rgbn_naip_order_also_infers(self) -> None:
        # R,G,B,NIR: green and NIR sit at the same positions — must pass.
        from ingest import resolve_band_map

        tmp = Path(tempfile.mkdtemp())
        arr = _two_zone([450, 900, 500, 2600], [550, 800, 520, 150])
        m, note = resolve_band_map(_write(tmp, "rgbn.tif", arr))
        self.assertEqual(m, {"green": 2, "nir": 4}, note)

    def test_liss_order_withholds(self) -> None:
        # G,R,NIR,SWIR (LISS-III/AWiFS): b3 is itself NIR — the two-part
        # check must reject rather than run NDWI on the wrong bands.
        from ingest import resolve_band_map
        from tools import water_highlight

        tmp = Path(tempfile.mkdtemp())
        arr = _two_zone([900, 500, 2500, 1100], [1400, 1500, 1600, 2000])
        tif = _write(tmp, "liss.tif", arr)
        m, note = resolve_band_map(tif)
        self.assertIsNone(m)
        self.assertIn("unidentified", note)
        wh = water_highlight(np.zeros((H, W, 3), np.uint8), source_path=tif)
        self.assertTrue(wh.get("withheld"))
        self.assertEqual(wh.get("withheld_reason"), "no_spectral_basis")

    def test_rgba_alpha_band_withholds(self) -> None:
        # 4-band uint8 photo with constant alpha — flat-band guard.
        from ingest import resolve_band_map

        tmp = Path(tempfile.mkdtemp())
        arr = _two_zone([120, 140, 100, 255], [160, 180, 130, 255])
        m, note = resolve_band_map(_write(tmp, "rgba.tif", arr))
        self.assertIsNone(m)
        self.assertIn("flat", note)

    def test_no_vegetation_scene_withholds(self) -> None:
        # nothing band-4-bright to test the hypothesis against — flat.
        from ingest import resolve_band_map

        tmp = Path(tempfile.mkdtemp())
        arr = _two_zone([500, 600, 550, 140], [520, 610, 560, 145])
        m, note = resolve_band_map(_write(tmp, "ocean.tif", arr))
        self.assertIsNone(m)

    def test_stripped_sundarbans_recovers_529(self) -> None:
        # the real regression case: the preset file with its band
        # descriptions stripped must infer and measure ≈52.9%.
        from ingest import resolve_band_map, write_test_geotiff, _load_rgb_array
        from tools import water_highlight

        tmp = Path(tempfile.mkdtemp())
        import rasterio

        with rasterio.open(OPTICAL) as ds:
            data = ds.read()  # pixels only — descriptions dropped
        tif = write_test_geotiff(tmp / "sb.tif", data, 10.0, 10.0,
                                 geographic=False)
        m, note = resolve_band_map(tif)
        self.assertEqual(m, {"green": 2, "nir": 4}, note)
        rgb, _ = _load_rgb_array(tif)
        wh = water_highlight(rgb, source_path=tif)
        self.assertFalse(wh.get("withheld"), wh.get("method"))
        frac = int(wh["water_pixels"]) / int(np.asarray(wh["mask"]).size)
        self.assertGreater(frac, 0.4)
        self.assertLess(frac, 0.65)


if __name__ == "__main__":
    unittest.main()
