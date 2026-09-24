"""DEMO-P2-TRUST CPU tests. Does not start llama-server."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

DEMO = Path(__file__).resolve().parent

SCENE2_LIKE = {
    "change_detect": {
        "rung": "cached_mask",
        "changed_pixels": 270611,
        "vs_gt": {"iou": 0.0677},
        "radiometric_label": "similar",
        "radiometric_delta": -7.40,
        "delta_mean": -7.40,
        "direction_epsilon": 8.0,
        "built_up_direction": "not_determined",
    },
    "area_calc": {
        "changed_pixels": 270611,
        "area_m2": 67652.75,
        "gsd_m": 0.5,
    },
}


class TrustTests(unittest.TestCase):
    def test_check_narration_invented_iou_fails(self) -> None:
        from report import check_narration

        out = check_narration(
            "Pair IoU is 0.99 versus the ground-truth mask.",
            SCENE2_LIKE,
        )
        self.assertFalse(out["ok"])
        self.assertTrue(any("0.99" in i for i in out["issues"]), out)

    def test_check_narration_scene2_numbers_pass(self) -> None:
        from report import check_narration

        out = check_narration(
            "Changed pixels 270611 with IoU 0.0677 versus GT.",
            SCENE2_LIKE,
        )
        self.assertTrue(out["ok"], out)

    def test_check_narration_sdwi_expansion_fails(self) -> None:
        from report import check_narration

        tool = {"sar_read": {"sdwi_label": "SDWI", "water_pixels": 100}}
        out = check_narration(
            "The Synthetic Dual-Wideband Index indicates water.",
            tool,
        )
        self.assertFalse(out["ok"])
        self.assertTrue(any("banned" in i.lower() for i in out["issues"]), out)

    def test_compose_keeps_vlm_below_and_flags_unverified(self) -> None:
        from report import compose_visible_answer, findings_header

        header = findings_header({"tool_outputs": SCENE2_LIKE})
        self.assertIn("### Findings (from tools)", header)
        self.assertIn("built_up_direction=`not_determined`", header)
        self.assertIn("radiometric_label=`similar`", header)
        vis = compose_visible_answer(
            {
                "tool_outputs": SCENE2_LIKE,
                "answer": "IoU is 0.99 on this pair.",
                "vlm": {"text": "IoU is 0.99 on this pair."},
            }
        )
        self.assertIn("### Interpretation", vis)
        self.assertIn("cite the measurement card", vis)
        self.assertLess(vis.index("### Findings (from tools)"), vis.index("0.99"))
        self.assertTrue(vis.strip().endswith("IoU is 0.99 on this pair."))

    def test_findings_header_sar_uses_sdwi_letters(self) -> None:
        from report import findings_header

        hdr = findings_header(
            {
                "tool_outputs": {
                    "sar_read": {
                        "water_pixels": 255,
                        "water_calibrated": True,
                        "sdwi_label": "SDWI",
                        "sdwi_stats": {"mean": -9.1},
                    }
                }
            }
        )
        self.assertIn("SDWI", hdr)
        self.assertNotIn("Synthetic Dual-Wideband", hdr)
        self.assertNotIn("Dual-Wideband Index", hdr)

    def test_geotiff_source_path_ndwi(self) -> None:
        from ingest import write_test_geotiff
        from pipeline import bind_inputs
        from tools import water_highlight

        tmp = Path(tempfile.mkdtemp())
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[..., 0] = 40
        arr[..., 1] = 50
        arr[..., 2] = 30
        arr[..., 3] = 180
        arr[8:24, 8:24, 1] = 200
        arr[8:24, 8:24, 3] = 20
        tif = write_test_geotiff(
            tmp / "ms.tif",
            arr,
            10.0,
            10.0,
            geographic=False,
            channel_names=("red", "green", "blue", "nir"),
        )
        bound = bind_inputs("single", scene=1, uploads={"image": tif})
        self.assertTrue(bound["ok"], bound.get("error"))
        orig = Path(bound["paths"]["source_original"])
        png = Path(bound["paths"]["image"])
        self.assertEqual(orig.suffix.lower(), ".tif")
        self.assertEqual(png.suffix.lower(), ".png")
        self.assertTrue(orig.is_file())
        self.assertTrue(png.is_file())
        wh = water_highlight(png, source_path=orig)
        self.assertIn("NDWI", wh["method"])
        self.assertGreater(wh["water_pixels"], 50)

    def test_cpu_geotiff_gsd_and_highlight_smoke(self) -> None:
        from ingest import read_gsd, write_test_geotiff
        from tools import water_highlight

        tmp = Path(tempfile.mkdtemp())
        arr = np.zeros((24, 32, 4), dtype=np.uint8)
        arr[..., 1] = 40
        arr[..., 3] = 160
        arr[4:20, 4:28, 1] = 210
        arr[4:20, 4:28, 3] = 15
        tif = write_test_geotiff(
            tmp / "smoke.tif",
            arr,
            2.0,
            2.0,
            geographic=False,
            channel_names="red,green,blue,nir",
        )
        gsd = read_gsd(tif)
        self.assertAlmostEqual(gsd["gsd_m"], 2.0, places=4)
        rgb = np.stack([arr[..., 0], arr[..., 1], arr[..., 2]], axis=-1)
        wh = water_highlight(rgb, source_path=tif)
        self.assertIn("NDWI", wh["method"])
        self.assertGreater(wh["water_pixels"], 20)

    def test_prepared_scene1_highlight_still_rgb_otsu(self) -> None:
        from pipeline import run_query

        trace = run_query(
            "Highlight the water body referred to in the query",
            "single",
            scene=1,
            live=False,
        )
        wh = (trace.get("tool_outputs") or {}).get("water_highlight") or {}
        self.assertIn("B-R", wh.get("method", ""))
        self.assertNotIn("NDWI", wh.get("method", ""))

    def test_8bit_sar_not_calibrated_threshold(self) -> None:
        from ingest import read_sar_arrays
        from tools import sar_read

        tmp = Path(tempfile.mkdtemp()) / "sar.png"
        im = Image.new("L", (32, 32), 80)
        px = im.load()
        for y in range(16):
            for x in range(16):
                px[x, y] = 2
        im.save(tmp)
        pack = read_sar_arrays(tmp)
        self.assertTrue(pack["ok"])
        self.assertFalse(pack["calibrated"])
        self.assertEqual(pack["vv"].dtype, np.float32)
        out = sar_read(pack["vv"], pack["vh"], calibrated=pack["calibrated"])
        self.assertFalse(out["water_calibrated"])
        self.assertIsNone(out["water_pixels"])
        self.assertIsNone(out["water_db_threshold"])
        self.assertIn("preview DN", out["method"])
        self.assertIn("not applied", out["method"])

    def test_float_sar_still_finds_dark_patch(self) -> None:
        from tools import sar_read

        vv = np.full((32, 32), -8.0, dtype=np.float32)
        vv[:16, :16] = -22.0
        out = sar_read(vv, vv.copy(), water_db_threshold=-16.0, calibrated=True)
        self.assertTrue(out["water_calibrated"])
        self.assertGreaterEqual(out["water_pixels"], 50)
        self.assertEqual(out["sdwi_label"], "SDWI")

    def test_bi_temporal_hxw_mismatch_errors(self) -> None:
        from pipeline import bind_inputs

        tmp = Path(tempfile.mkdtemp())
        before = tmp / "b.png"
        after = tmp / "a.png"
        Image.new("RGB", (32, 32), (10, 10, 10)).save(before)
        Image.new("RGB", (48, 32), (10, 10, 10)).save(after)
        bound = bind_inputs(
            "bi-temporal",
            scene=2,
            uploads={"before": before, "after": after},
        )
        self.assertFalse(bound["ok"])
        err = (bound.get("error") or "").lower()
        self.assertIn("dimensions differ", err)
        self.assertIn("not co-registered", err)

    def test_prepared_scene2_radiometry_not_built_up(self) -> None:
        from pipeline import run_query

        pred = DEMO / "data" / "scene2" / "pred_mask.png"
        mtime_before = pred.stat().st_mtime
        trace = run_query(
            "Has built-up area increased, decreased, or remained unchanged?",
            "bi-temporal",
            scene=2,
            live=False,
            cd_prefer="classical",
        )
        self.assertEqual(pred.stat().st_mtime, mtime_before)
        cd = (trace.get("tool_outputs") or {}).get("change_detect") or {}
        self.assertEqual(cd.get("built_up_direction"), "not_determined")
        self.assertNotIn("direction", cd)
        self.assertIn(
            cd.get("radiometric_label"),
            {"brighter_after", "darker_after", "similar", "unknown"},
        )
        vis = trace.get("visible_answer") or ""
        self.assertIn("### Findings (from tools)", vis)
        self.assertIn("built_up_direction=`not_determined`", vis)

    def test_prepared_scene3_sdwi_overlay(self) -> None:
        from pipeline import run_query

        trace = run_query(
            "Identify water-covered regions.",
            "optical+sar",
            scene=3,
            live=False,
        )
        sr = (trace.get("tool_outputs") or {}).get("sar_read") or {}
        self.assertEqual(sr.get("sdwi_label"), "SDWI")
        self.assertTrue(sr.get("water_calibrated"))
        self.assertIsNotNone(sr.get("water_pixels"))
        self.assertTrue(Path(trace["overlay_path"]).is_file())


class RealScene2NarrationTests(unittest.TestCase):
    """P2B: invented-number check against actual prepared Scene 2 tool_outputs."""

    @classmethod
    def setUpClass(cls) -> None:
        from pipeline import run_query

        pred = DEMO / "data" / "scene2" / "pred_mask.png"
        cls._mtime_before = pred.stat().st_mtime
        cls.trace = run_query(
            "Has built-up area increased, decreased, or remained unchanged?",
            "bi-temporal",
            scene=2,
            live=False,
            cd_prefer="classical",
        )
        cls.tout = cls.trace.get("tool_outputs") or {}
        cls._mtime_after = pred.stat().st_mtime

    def test_pred_mask_not_rewritten(self) -> None:
        self.assertEqual(self._mtime_after, self._mtime_before)

    def test_real_scene2_invented_099_fails(self) -> None:
        from report import check_narration, compose_visible_answer

        out = check_narration(
            "Pair IoU is 0.99 versus the ground-truth mask.",
            self.tout,
        )
        self.assertFalse(out["ok"], out)
        self.assertTrue(any("0.99" in i for i in out["issues"]), out)
        vis = compose_visible_answer(
            {
                "tool_outputs": self.tout,
                "answer": "Pair IoU is 0.99 versus the ground-truth mask.",
                "vlm": {"text": "Pair IoU is 0.99 versus the ground-truth mask."},
            }
        )
        self.assertIn("### Findings (from tools)", vis)
        self.assertIn("### Interpretation", vis)
        self.assertIn("cite the measurement card", vis)
        self.assertLess(
            vis.index("### Findings (from tools)"),
            vis.index("### Interpretation"),
        )
        self.assertLess(
            vis.index("### Interpretation"),
            vis.index("Pair IoU is 0.99 versus the ground-truth mask."),
        )
        self.assertTrue(
            vis.strip().endswith("Pair IoU is 0.99 versus the ground-truth mask.")
        )

    def test_real_scene2_cite_pixels_and_area_pass(self) -> None:
        from report import check_narration

        out = check_narration(
            "Changed pixels 270611 with IoU 0.0677 versus GT.",
            self.tout,
        )
        self.assertTrue(out["ok"], out)

    def test_real_scene2_cite_iou_truncation_pass(self) -> None:
        from report import check_narration

        out = check_narration(
            "Pair IoU is 0.852 versus the ground-truth mask.",
            self.tout,
        )
        self.assertTrue(out["ok"], out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
