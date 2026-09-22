"""P0 ingest: GeoTIFF GSD + upload→infer without touching prepared Scene 2 mask."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

DEMO = Path(__file__).resolve().parent


class IngestPipelineTests(unittest.TestCase):
    def test_projected_geotiff_gsd(self) -> None:
        from ingest import read_gsd, write_test_geotiff

        tmp = Path(tempfile.mkdtemp()) / "proj.tif"
        arr = np.zeros((16, 20), dtype=np.uint8)
        write_test_geotiff(tmp, arr, 0.6, 0.6, geographic=False)
        gsd = read_gsd(tmp)
        self.assertAlmostEqual(gsd["gsd_m"], 0.6, places=5)
        self.assertEqual(gsd["source"], "geotransform")

    def test_projected_local_cs_records_crs_note(self) -> None:
        # rasterio reads the tifffile-written projected GeoKey as LOCAL_CS;
        # the caveat is recorded, not used to downgrade the source.
        from ingest import read_gsd, write_test_geotiff

        tmp = Path(tempfile.mkdtemp()) / "lcs.tif"
        arr = np.zeros((16, 20), dtype=np.uint8)
        write_test_geotiff(tmp, arr, 0.6, 0.6, geographic=False)
        gsd = read_gsd(tmp)
        self.assertEqual(gsd["source"], "geotransform")
        self.assertIsNotNone(gsd["crs"])
        self.assertIn("LOCAL_CS", gsd["crs"])
        self.assertIsNotNone(gsd["crs_note"])
        self.assertIn("EPSG", gsd["crs_note"])

    def test_epsg_projected_geotiff_gsd(self) -> None:
        # A real EPSG-rooted projected CRS (Cartosat-style) keeps working.
        from ingest import read_gsd

        try:
            import rasterio
            from rasterio.transform import from_origin
        except ImportError:
            self.skipTest("rasterio not installed")

        tmp = Path(tempfile.mkdtemp()) / "epsg.tif"
        arr = np.zeros((8, 8), dtype=np.uint8)
        with rasterio.open(
            tmp, "w", driver="GTiff", width=8, height=8, count=1,
            dtype="uint8", crs="EPSG:32643",
            transform=from_origin(500000, 2500000, 0.5, 0.5),
        ) as ds:
            ds.write(arr, 1)
        gsd = read_gsd(tmp)
        self.assertAlmostEqual(gsd["gsd_m"], 0.5, places=4)
        self.assertEqual(gsd["source"], "geotransform")
        self.assertIsNone(gsd["crs_note"])

    def test_no_transform_reports_assumed_meters(self) -> None:
        # Untagged TIFF: rasterio yields the identity transform — no real
        # geotransform, so the honest assumed-meters path stays.
        import tifffile
        from ingest import read_gsd

        tmp = Path(tempfile.mkdtemp()) / "plain.tif"
        tifffile.imwrite(str(tmp), np.zeros((16, 16), dtype=np.uint8))
        gsd = read_gsd(tmp)
        self.assertEqual(gsd["source"], "assumed_meters")
        self.assertAlmostEqual(gsd["gsd_m"], 1.0, places=5)

    def test_geographic_geotiff_gsd_at_equator(self) -> None:
        from ingest import M_PER_DEG, read_gsd, write_test_geotiff

        tmp = Path(tempfile.mkdtemp()) / "geo.tif"
        arr = np.zeros((8, 8), dtype=np.uint8)
        deg = 1e-5
        write_test_geotiff(tmp, arr, deg, deg, tie_x=0.0, tie_y=0.0, geographic=True)
        gsd = read_gsd(tmp)
        self.assertIsNotNone(gsd["gsd_m"])
        self.assertAlmostEqual(gsd["gsd_m"], deg * M_PER_DEG, places=3)
        self.assertEqual(gsd["source"], "geotransform")

    def test_bind_prepared_scene2_keeps_levir_constant(self) -> None:
        from pipeline import bind_inputs
        from tools import LEVIR_GSD_M

        bound = bind_inputs("bi-temporal", scene=2, uploads=None)
        self.assertTrue(bound["ok"])
        self.assertEqual(bound["source"], "prepared")
        self.assertTrue(bound["use_prepared_cd_cache"])
        self.assertEqual(bound["gsd"]["gsd_m"], LEVIR_GSD_M)
        self.assertEqual(bound["gsd"]["source"], "benchmark_constant")
        self.assertTrue(Path(bound["paths"]["before"]).is_file())

    def test_partial_upload_is_error_not_mix(self) -> None:
        from pipeline import bind_inputs

        tmp = Path(tempfile.mkdtemp()) / "only_before.png"
        Image.new("RGB", (32, 32), (1, 2, 3)).save(tmp)
        bound = bind_inputs("bi-temporal", scene=2, uploads={"before": tmp})
        self.assertFalse(bound["ok"])
        self.assertIn("incomplete", bound["error"].lower())
        self.assertFalse(bound["use_prepared_cd_cache"])

    def test_upload_png_pair_runs_classical_and_withholds_m2(self) -> None:
        from pipeline import run_query

        pred = DEMO / "data" / "scene2" / "pred_mask.png"
        mtime_before = pred.stat().st_mtime if pred.is_file() else None
        tmp = Path(tempfile.mkdtemp())
        before = tmp / "b.png"
        after = tmp / "a.png"
        Image.new("RGB", (64, 64), (10, 10, 10)).save(before)
        im = Image.new("RGB", (64, 64), (10, 10, 10))
        px = im.load()
        for y in range(20, 40):
            for x in range(20, 40):
                px[x, y] = (220, 220, 220)
        im.save(after)
        trace = run_query(
            "What changed between these two dates, and where?",
            "bi-temporal",
            scene=2,
            live=False,
            uploads={"before": before, "after": after},
            cd_prefer="classical",
        )
        self.assertEqual(trace["input_source"], "upload")
        self.assertIn("Analyzing your uploaded files", trace["ingest_note"])
        self.assertNotIn("scene2", Path(trace["images"][0]).as_posix())
        area = trace["tool_outputs"]["area_calc"]
        self.assertIsNone(area["area_m2"])
        self.assertGreater(area["changed_pixels"], 50)
        overlay = Path(trace["overlay_path"])
        self.assertTrue(overlay.is_file())
        self.assertIn("_uploads", overlay.as_posix())
        if mtime_before is not None:
            self.assertEqual(pred.stat().st_mtime, mtime_before)

    def test_upload_geotiff_pair_uses_tag_gsd(self) -> None:
        from ingest import write_test_geotiff
        from pipeline import run_query

        tmp = Path(tempfile.mkdtemp())
        b = np.zeros((48, 48, 3), dtype=np.uint8)
        a = b.copy()
        a[8:24, 8:24] = 255
        before = write_test_geotiff(tmp / "b.tif", b, 2.0, 2.0, geographic=False)
        after = write_test_geotiff(tmp / "a.tif", a, 2.0, 2.0, geographic=False)
        trace = run_query(
            "How much built-up area increased, in km2?",
            "bi-temporal",
            scene=2,
            live=False,
            uploads={"before": before, "after": after},
            cd_prefer="classical",
        )
        area = trace["tool_outputs"]["area_calc"]
        self.assertAlmostEqual(area["gsd_m"], 2.0, places=4)
        self.assertIsNotNone(area["area_m2"])
        self.assertAlmostEqual(area["area_m2"], area["changed_pixels"] * 4.0, places=3)

    def test_resize_scales_gsd(self) -> None:
        from ingest import scale_gsd_for_resize

        info = {"gsd_m": 0.5, "source": "geotransform", "provenance": "x"}
        out = scale_gsd_for_resize(info, native_w=2048, used_w=1024)
        self.assertAlmostEqual(out["gsd_m_used"], 1.0)
        self.assertAlmostEqual(out["resize_scale"], 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
