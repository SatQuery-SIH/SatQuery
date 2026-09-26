"""Bundled-preset smoke tests — the presets a judge clicks must produce
their expected tool result deterministically (no model seats).

Catches the 2026-09-26 regression class: the C1 band-identity gate made
`water_highlight` withhold `no_spectral_basis` on the bundled Sundarbans
GeoTIFF because the file carried no band descriptions and the preset path
declares no sensor profile. The fix wrote semantic band descriptions into
the files themselves so every ingest route (web preset, manual upload,
judge kit) resolves identity identically.
"""
from __future__ import annotations

import base64
import unittest
from pathlib import Path

import numpy as np

DEMO = Path(__file__).resolve().parent
SAT = DEMO.parent
PRESETS = SAT / "web" / "public" / "presets"
ASSETS = SAT / "web" / "src" / "assets"

OPTICAL = PRESETS / "sundarbans_optical.tiff"
SAR = PRESETS / "sundarbans_sar.tiff"


class PresetAssetTests(unittest.TestCase):
    def test_optical_bands_self_describe_and_water_measures(self) -> None:
        from ingest import resolve_band_map, _load_rgb_array
        from tools import water_highlight

        band_map, note = resolve_band_map(OPTICAL)
        self.assertEqual(
            band_map, {"blue": 1, "green": 2, "red": 3, "nir": 4}, note
        )
        rgb, rgb_note = _load_rgb_array(OPTICAL)
        self.assertIsNotNone(rgb, rgb_note)
        out = water_highlight(rgb, source_path=OPTICAL)
        self.assertFalse(out.get("withheld"), out.get("method"))
        self.assertEqual(out.get("evidence_class"), "measured_index")
        total = int(np.prod(out["mask_shape"]))
        frac = out["water_pixels"] / total
        # estuary scene — water must be a large fraction, sanity band only
        self.assertGreater(frac, 0.3)
        self.assertLess(frac, 0.9)

    def test_sar_polarizations_self_describe(self) -> None:
        from ingest import read_sar_arrays

        sar = read_sar_arrays(SAR)
        self.assertTrue(sar["ok"], sar.get("error"))
        self.assertTrue(sar["sar_evidence"])
        self.assertTrue(sar["pol_verified"], sar.get("provenance"))
        self.assertEqual(sar["named_pols"], ["vv", "vh"])

    def test_bitemporal_pngs_load_as_rgb(self) -> None:
        from tools import load_rgb

        for name in ("scene2_before.png", "scene2_after.png"):
            rgb = load_rgb(PRESETS / name)
            self.assertEqual(rgb.ndim, 3, name)
            self.assertGreaterEqual(rgb.shape[2], 3, name)

    def test_b64_fallbacks_are_byte_exact(self) -> None:
        # the .b64 assets back the preset uploads when a browser can't fetch
        # the .tiff — a drifted fallback would silently run a different file
        for stem in ("sundarbans_optical", "sundarbans_sar"):
            raw = (PRESETS / f"{stem}.tiff").read_bytes()
            b64 = base64.b64decode(
                (ASSETS / f"{stem}.tiff.b64").read_text().strip()
            )
            self.assertEqual(raw, b64, stem)


if __name__ == "__main__":
    unittest.main()
