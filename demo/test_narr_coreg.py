"""NARR-PCT-FIX + COREG-NOTE CPU tests.

Part A: check_narration accepts only *derivable* percentages (tool value x100
at the token's displayed precision). Wrong percents and bare non-matching
numbers stay flagged.

Part B: coreg_check transform + pixel levels on synthetic GeoTIFFs / arrays;
the measured shift feeds sar_agreement's misreg gate.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

DEMO = Path(__file__).resolve().parent
SAT = DEMO.parent
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))

from evidence_packet import build_packet, validate_packet  # noqa: E402
from report import check_narration  # noqa: E402
from tools import AGREE_MISREG_PX, coreg_check, sar_agreement  # noqa: E402


def _tiff(path: Path, arr: np.ndarray, *, origin=(500000.0, 2500000.0)) -> Path:
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        raise unittest.SkipTest("rasterio not installed")
    a = np.asarray(arr)
    count = a.shape[0] if a.ndim == 3 else 1
    with rasterio.open(
        path, "w", driver="GTiff", width=a.shape[-1], height=a.shape[-2],
        count=count, dtype=str(a.dtype), crs="EPSG:32643",
        transform=from_origin(origin[0], origin[1], 10.0, 10.0),
    ) as ds:
        if a.ndim == 3:
            ds.write(a)
        else:
            ds.write(a, 1)
    return path


def _field(h: int = 64, w: int = 64, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.random((h // 8, w // 8)).astype(np.float32)
    return np.asarray(
        __import__("PIL.Image", fromlist=["Image"]).fromarray(base).resize((w, h))
    )


class TestNarrationPercent(unittest.TestCase):
    TOOL = {"sar_read": {"water_fraction": 0.1417083740234375,
                         "water_pixels": 9287}}

    def test_correct_derivation_passes(self) -> None:
        r = check_narration(
            "Water covers 14.17% of the scene (9287 px).", self.TOOL
        )
        self.assertEqual(r["issues"], [])

    def test_percent_word_passes(self) -> None:
        r = check_narration("Water covers 14.17 percent of the scene.", self.TOOL)
        self.assertEqual(r["issues"], [])

    def test_integer_percent_passes(self) -> None:
        r = check_narration("Water covers 14% of the scene.", self.TOOL)
        self.assertEqual(r["issues"], [])

    def test_wrong_percent_fails(self) -> None:
        for bad in ("1.42%", "28.3%", "50%"):
            r = check_narration(f"Water covers {bad} of the scene.", self.TOOL)
            self.assertTrue(
                any("invented number" in i for i in r["issues"]),
                f"{bad} should flag invented",
            )

    def test_bare_nonmatching_still_flagged(self) -> None:
        # The same digits without a % marker get no derivation pass.
        r = check_narration("Water covers 14.17 of the scene.", self.TOOL)
        self.assertTrue(any("invented number 14.17" in i for i in r["issues"]))

    def test_invented_number_still_fails(self) -> None:
        r = check_narration("Scene is 99.9% cloud and 7.5 units tall.", self.TOOL)
        self.assertIn("invented number 99.9", r["issues"])
        self.assertIn("invented number 7.5", r["issues"])


class TestCoregTransform(unittest.TestCase):
    def test_identical_transform_offset_zero(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        a = _tiff(tmp / "a.tif", np.zeros((8, 8), np.uint8))
        b = _tiff(tmp / "b.tif", np.zeros((8, 8), np.uint8))
        cg = coreg_check(a, b)
        tr = cg["transform"]
        self.assertEqual(tr["status"], "measured")
        self.assertEqual(tr["offset_m"], 0.0)
        self.assertTrue(tr["same_res"])

    def test_shifted_origin_offset(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        a = _tiff(tmp / "a.tif", np.zeros((8, 8), np.uint8))
        b = _tiff(tmp / "b.tif", np.zeros((8, 8), np.uint8),
                  origin=(500020.0, 2500000.0))
        cg = coreg_check(a, b)
        self.assertAlmostEqual(cg["transform"]["offset_m"], 20.0, places=3)

    def test_res_mismatch_flagged(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        a = _tiff(tmp / "a.tif", np.zeros((8, 8), np.uint8))
        p = tmp / "b.tif"
        try:
            import rasterio
            from rasterio.transform import from_origin
        except ImportError:
            self.skipTest("rasterio not installed")
        with rasterio.open(
            p, "w", driver="GTiff", width=8, height=8, count=1,
            dtype="uint8", crs="EPSG:32643",
            transform=from_origin(500000, 2500000, 30.0, 30.0),
        ) as ds:
            ds.write(np.zeros((8, 8), np.uint8), 1)
        tr = coreg_check(a, p)["transform"]
        self.assertFalse(tr["same_res"])

    def test_non_geotiff_unavailable(self) -> None:
        from PIL import Image

        tmp = Path(tempfile.mkdtemp())
        p1 = tmp / "a.png"; p2 = tmp / "b.png"
        Image.new("RGB", (8, 8)).save(p1)
        Image.new("RGB", (8, 8)).save(p2)
        tr = coreg_check(p1, p2)["transform"]
        self.assertEqual(tr["status"], "unavailable")


class TestCoregPixel(unittest.TestCase):
    def test_known_2px_shift_recovered(self) -> None:
        a = _field()
        b = np.roll(a, (0, 2))
        rgb = np.stack([a, a, a], axis=-1)
        px = coreg_check(None, None, optical_rgb=rgb, sar_vv=b)["pixel"]
        self.assertEqual(px["status"], "measured")
        self.assertAlmostEqual(px["shift_px"], 2.0, delta=0.3)
        self.assertAlmostEqual(abs(px["dx_px"]), 2.0, delta=0.3)

    def test_identical_arrays_zero(self) -> None:
        a = _field()
        rgb = np.stack([a, a, a], axis=-1)
        px = coreg_check(None, None, optical_rgb=rgb, sar_vv=a)["pixel"]
        self.assertEqual(px["status"], "measured")
        self.assertAlmostEqual(px["shift_px"], 0.0, delta=0.3)

    def test_uniform_inconclusive(self) -> None:
        a = np.zeros((32, 32), np.float32)
        px = coreg_check(None, None, optical_rgb=np.stack([a] * 3, -1),
                         sar_vv=a)["pixel"]
        self.assertEqual(px["status"], "inconclusive")

    def test_inputs_missing_inconclusive(self) -> None:
        px = coreg_check(None, None)["pixel"]
        self.assertEqual(px["status"], "inconclusive")
        self.assertEqual(px["reason"], "inputs_missing")

    def test_gate_goes_live_on_large_shift(self) -> None:
        a = _field()
        b = np.roll(a, (0, 6))
        rgb = np.stack([a, a, a], axis=-1)
        cg = coreg_check(None, None, optical_rgb=rgb, sar_vv=b)
        self.assertGreater(cg["shift_px"], AGREE_MISREG_PX)
        mask = np.zeros((8, 8), np.uint8)
        mask[:4] = 1
        ag = sar_agreement(mask, mask, misreg_shift_px=cg["shift_px"])
        self.assertEqual(ag["verdicts"]["water"], "withheld_misregistered")

    def test_gate_open_on_aligned(self) -> None:
        a = _field()
        cg = coreg_check(None, None, optical_rgb=np.stack([a] * 3, -1),
                         sar_vv=a)
        mask = np.zeros((8, 8), np.uint8)
        mask[:4] = 1
        ag = sar_agreement(mask, mask, misreg_shift_px=cg["shift_px"])
        self.assertEqual(ag["verdicts"]["water"], "both_support")


class TestCoregPipelineAndPacket(unittest.TestCase):
    def _pair(self, shift: int = 0):
        tmp = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(3)
        opt = rng.integers(0, 4000, (4, 64, 64), dtype=np.uint16)
        vv = _field(64, 64, seed=9) * 30.0 - 25.0
        if shift:
            vv = np.roll(vv, (0, shift))
        sar = np.stack([vv.astype(np.float64),
                        (vv - 12.0).astype(np.float64)])
        o = _tiff(tmp / "opt.tiff", opt)
        s = _tiff(tmp / "sar.tiff", sar)
        return o, s

    def test_bind_sets_coreg_and_gate_fields(self) -> None:
        from pipeline import bind_inputs

        o, s = self._pair()
        b = bind_inputs("optical+sar", uploads={"optical": o, "sar": s})
        self.assertTrue(b["ok"])
        self.assertIsNotNone(b.get("coreg"))
        self.assertEqual(b["coreg"]["transform"]["status"], "measured")
        self.assertEqual(b["coreg"]["transform"]["offset_m"], 0.0)
        self.assertIn("sar_original", b["paths"])
        self.assertIsNotNone(b.get("misreg_shift_px"))

    def test_bind_large_shift_feeds_gate(self) -> None:
        from pipeline import bind_inputs

        o, s = self._pair(shift=6)
        b = bind_inputs("optical+sar", uploads={"optical": o, "sar": s})
        self.assertTrue(b["ok"])
        self.assertIsNotNone(b.get("misreg_shift_px"))
        self.assertGreater(float(b["misreg_shift_px"]), AGREE_MISREG_PX)

    def test_packet_carries_coreg_claims(self) -> None:
        cg = {
            "transform": {"status": "measured", "offset_m": 0.0,
                          "same_res": True,
                          "basis": "affine transform comparison"},
            "pixel": {"status": "measured", "shift_px": 0.42, "peak": 0.31,
                      "method": "phase corr", "caveat": "noisy"},
            "shift_px": 0.42,
        }
        pkt = build_packet(
            task="water_identification",
            tool_outputs={"sar_agreement": {
                "verdicts": {"water": "both_support",
                             "built_up": "withheld_no_tool"},
                "iou": 0.5, "grid": "same_grid",
            }, "coreg_check": cg},
        )
        preds = {c["predicate"] for c in pkt["claims"]}
        self.assertIn("coreg_transform_offset_m", preds)
        self.assertIn("coreg_same_res", preds)
        self.assertIn("coreg_shift_px", preds)
        self.assertEqual(validate_packet(pkt), [])

    def test_packet_withheld_when_inconclusive(self) -> None:
        cg = {
            "transform": {"status": "unavailable",
                          "reason": "optical_transform=identity sar_transform=identity",
                          "offset_m": None, "same_res": None},
            "pixel": {"status": "inconclusive", "reason": "weak_peak",
                      "shift_px": None},
            "shift_px": None,
        }
        pkt = build_packet(task="water_identification",
                           tool_outputs={"coreg_check": cg})
        by_pred = {c["predicate"]: c for c in pkt["claims"]}
        self.assertEqual(by_pred["coreg_shift_px"]["confidence"]["level"],
                         "withheld")
        self.assertEqual(validate_packet(pkt), [])


if __name__ == "__main__":
    unittest.main()
