"""GEO-EXPORT CPU tests: GeoTIFF sidecars for mask artifacts.

Written files must round-trip (crs + scaled transform + identical pixels);
no-transform sources record skipped_no_transform — a recorded event, never
silence. Additive only: PNG flow untouched.
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
from tools import export_mask_geotiff  # noqa: E402


def _geotiff(path: Path, arr: np.ndarray, *, origin=(500000.0, 2500000.0),
             res: float = 10.0) -> Path:
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
        transform=from_origin(origin[0], origin[1], res, res),
    ) as ds:
        ds.write(a if a.ndim == 3 else a[None],)
    return path


class TestExportMaskGeotiff(unittest.TestCase):
    def test_round_trip_same_grid(self) -> None:
        import rasterio

        tmp = Path(tempfile.mkdtemp())
        src = _geotiff(tmp / "src.tif", np.zeros((4, 64, 64), np.uint16))
        mask = np.zeros((64, 64), np.uint8)
        mask[10:20, 5:9] = 1
        rec = export_mask_geotiff(mask, src, tmp / "m.tif")
        self.assertEqual(rec["status"], "written")
        with rasterio.open(rec["path"]) as ds:
            self.assertEqual(str(ds.crs), "EPSG:32643")
            self.assertEqual(ds.transform.a, 10.0)
            self.assertEqual(ds.transform.c, 500000.0)
            np.testing.assert_array_equal(ds.read(1), mask)

    def test_transform_scaled_to_mask_grid(self) -> None:
        # Mask on a 2x-downscaled preview grid: pixel size doubles, origin held.
        import rasterio

        tmp = Path(tempfile.mkdtemp())
        src = _geotiff(tmp / "src.tif", np.zeros((64, 64), np.uint8))
        mask = np.zeros((32, 32), np.uint8)
        mask[4:8] = 1
        rec = export_mask_geotiff(mask, src, tmp / "m.tif",
                                  scale_x=2.0, scale_y=2.0)
        self.assertTrue(rec["transform_scaled"])
        with rasterio.open(rec["path"]) as ds:
            self.assertEqual(ds.transform.a, 20.0)
            self.assertEqual(ds.transform.c, 500000.0)

    def test_no_transform_recorded(self) -> None:
        import tifffile

        tmp = Path(tempfile.mkdtemp())
        src = tmp / "plain.tif"
        tifffile.imwrite(src, np.zeros((16, 16), np.uint8))
        rec = export_mask_geotiff(np.zeros((16, 16), np.uint8), src, tmp / "m.tif")
        self.assertEqual(rec["status"], "skipped_no_transform")
        self.assertFalse((tmp / "m.tif").exists())

    def test_png_source_recorded(self) -> None:
        from PIL import Image

        tmp = Path(tempfile.mkdtemp())
        src = tmp / "img.png"
        Image.new("RGB", (16, 16)).save(src)
        rec = export_mask_geotiff(np.zeros((16, 16), np.uint8), src, tmp / "m.tif")
        self.assertEqual(rec["status"], "skipped_no_transform")

    def test_missing_source_recorded(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        rec = export_mask_geotiff(
            np.zeros((8, 8), np.uint8), tmp / "nope.tif", tmp / "m.tif"
        )
        self.assertIn(rec["status"], ("skipped_unreadable", "skipped_no_transform"))


class TestGeoExportPipeline(unittest.TestCase):
    def _pair(self):
        tmp = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(5)
        opt = rng.integers(0, 4000, (4, 64, 64), dtype=np.uint16)
        vv = (rng.random((64, 64)) * 30.0 - 25.0).astype(np.float64)
        sar = np.stack([vv, vv - 12.0])
        return _geotiff(tmp / "o.tiff", opt), _geotiff(tmp / "s.tiff", sar)

    def test_run_query_writes_sidecars_and_records(self) -> None:
        import rasterio

        from pipeline import run_query

        o, s = self._pair()
        # C1: the synthetic optical is an unnamed 4-band TIFF — without a
        # declared profile the water mask withholds (no spectral basis).
        t = run_query(
            "Is there water in this scene?", "optical+sar",
            uploads={"optical": o, "sar": s}, live=False,
            sensor_profile="cartosat2s_mx",
        )
        geo = t.get("geo_exports") or {}
        self.assertIn("water_sar_mask.tif", geo)
        self.assertIn("water_optical_mask.tif", geo)
        self.assertIn("agreement_map_live.tif", geo)
        for name in ("water_sar_mask.tif", "agreement_map_live.tif"):
            rec = geo[name]
            self.assertEqual(rec["status"], "written", rec)
            with rasterio.open(rec["path"]) as ds:
                self.assertEqual(str(ds.crs), "EPSG:32643")
                self.assertEqual(ds.transform.a, 10.0)
                self.assertEqual(ds.transform.c, 500000.0)
        pkt = t.get("evidence_packet") or {}
        self.assertIn("geo_exports", pkt)
        self.assertEqual(validate_packet(pkt), [])
        types = {a["type"] for a in pkt.get("artifacts") or []}
        self.assertIn("geotiff", types)

    def test_prepared_scene_records_skip(self) -> None:
        from pipeline import run_query

        if not (DEMO / "data" / "scene3" / "sar_arrays.npz").is_file():
            self.skipTest("scene3 assets absent")
        t = run_query(
            "Is there water in this scene?", "optical+sar", live=False
        )
        geo = t.get("geo_exports") or {}
        self.assertTrue(geo)
        for rec in geo.values():
            self.assertTrue(str(rec.get("status", "")).startswith("skipped"), rec)
        self.assertEqual(
            validate_packet(t.get("evidence_packet") or {}), []
        )


class TestGeoExportPacket(unittest.TestCase):
    def test_written_and_skipped_both_listed(self) -> None:
        pkt = build_packet(
            task="water_identification",
            tool_outputs={},
            geo_exports={
                "water_sar_mask.tif": {
                    "status": "written",
                    "path": "C:/x/water_sar_mask.tif",
                    "crs": "EPSG:32643",
                    "source": "C:/x/sar.tiff",
                },
                "water_optical_mask.tif": {
                    "status": "skipped_no_transform",
                    "source": "C:/x/opt.png",
                },
            },
        )
        ge = pkt.get("geo_exports") or {}
        self.assertEqual(ge["water_optical_mask.tif"]["status"],
                         "skipped_no_transform")
        tifs = [a for a in pkt["artifacts"] if a["type"] == "geotiff"]
        self.assertEqual(len(tifs), 1)
        self.assertEqual(tifs[0]["crs"], "EPSG:32643")
        self.assertEqual(validate_packet(pkt), [])


if __name__ == "__main__":
    unittest.main()
