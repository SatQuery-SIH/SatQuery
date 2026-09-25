"""FINALE-HARDEN-10 A–E evidence (CPU). Does not start llama-server."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

DEMO = Path(__file__).resolve().parent
SAT = DEMO.parent


class HardenTests(unittest.TestCase):
    def test_measurement_card_splits_whole_vs_ne(self) -> None:
        from report import measurement_markdown, write_report

        trace = json.loads((DEMO / "traces" / "scene2.json").read_text(encoding="utf-8"))
        card = measurement_markdown(trace)
        self.assertIn("270,611", card)
        self.assertIn("81,882", card)
        self.assertIn("Reading these numbers", card)
        self.assertIn("20470.5", card)
        path = write_report(trace)
        text = path.read_text(encoding="utf-8")
        self.assertIn("Built-in examples below", text)
        self.assertIn("changed_pixels", text)
        self.assertIn(trace["query"], text)
        self.assertTrue(path.suffix == ".md")
        import re

        m = re.search(r"trace_sha256: `([0-9a-f]{64})`", text)
        self.assertIsNotNone(m, text[:500])
        from report import tool_outputs_sha256

        self.assertEqual(m.group(1), tool_outputs_sha256(trace.get("tool_outputs") or {}))

    def test_tiff_gsd_and_preview(self) -> None:
        from ingest import write_test_geotiff, read_gsd
        from report import ingest_preview
        import numpy as np

        tmp = Path(tempfile.mkdtemp())
        rgb = np.zeros((24, 32, 3), dtype=np.uint8)
        rgb[..., 1] = 80
        tif = write_test_geotiff(tmp / "tiny.tif", rgb, 0.5, 0.5, geographic=False)
        gsd = read_gsd(tif)
        self.assertAlmostEqual(gsd["gsd_m"], 0.5, places=4)
        self.assertEqual(gsd["source"], "geotransform")
        im, note = ingest_preview(tif)
        self.assertIsNotNone(im)
        self.assertEqual(im.size, (32, 24))
        self.assertIn("GSD", note)

    def test_png_upload_does_not_invent_gsd(self) -> None:
        from ingest import read_gsd

        tmp = Path(tempfile.mkdtemp()) / "x.png"
        Image.new("RGB", (16, 16), (10, 80, 30)).save(tmp)
        gsd = read_gsd(tmp)
        self.assertIsNone(gsd["gsd_m"])
        self.assertEqual(gsd["source"], "none")

    def test_banner_and_serve_base(self) -> None:
        import app as demo_app

        demo_app.MODE = "live"
        b = demo_app._banner()
        self.assertIn("Live demo", b)
        self.assertIn("measurement tools", b)
        ps1 = (DEMO / "serve.ps1").read_text(encoding="utf-8")
        self.assertIn("Qwen3VL-8B-Instruct-Q4_K_M.gguf", ps1)
        self.assertIn("mmproj-Qwen3VL-8B-Instruct-F16.gguf", ps1)
        self.assertNotIn("adapted08", ps1)
        self.assertNotIn("8091", ps1)
        self.assertIn("--port 8080", ps1)

    def test_buildings_refused(self) -> None:
        from planner import plan

        # WIRE-8091: single-image counts route to the canonical_vqa seat
        # (RSVQA count family); the refusal now applies to modes with no
        # counting seat, e.g. bi-temporal.
        p = plan("How many buildings are in this image?", "single")
        self.assertTrue(p["supported"])
        self.assertIn("canonical_vqa", p["tools"])
        bt = plan("How many buildings are in this image?", "bi-temporal")
        self.assertFalse(bt["supported"])
        self.assertIn("count", (bt["refusal"] or "").lower())

    def test_prepared_scene2_does_not_rewrite_pred_mask(self) -> None:
        from pipeline import run_query

        pred = DEMO / "data" / "scene2" / "pred_mask.png"
        mtime_before = pred.stat().st_mtime
        size_before = pred.stat().st_size
        trace = run_query(
            "Has built-up area increased, decreased, or remained unchanged?",
            "bi-temporal",
            scene=2,
            live=False,
            cd_prefer="classical",
        )
        self.assertEqual(pred.stat().st_mtime, mtime_before)
        self.assertEqual(pred.stat().st_size, size_before)
        cd = (trace.get("tool_outputs") or {}).get("change_detect") or {}
        self.assertEqual(cd.get("built_up_direction"), "not_determined")
        self.assertIn(
            cd.get("radiometric_label"),
            {"brighter_after", "darker_after", "similar", "unknown"},
        )
        self.assertNotIn("direction", cd)
        self.assertIn("direction_provenance", cd)

    def test_model_fallback_never_fires_on_cached_runs(self) -> None:
        # HYBRID-PLAN: live=False runs are fully deterministic — the
        # second-chance router must not touch the seat at all.
        import pipeline
        from unittest.mock import patch

        calls: list = []
        with patch.object(
            pipeline, "plan_model", side_effect=lambda *a, **k: calls.append(1)
        ):
            trace = pipeline.run_query(
                "what do you see", "single", scene=1, live=False
            )
        self.assertEqual(calls, [])
        self.assertEqual(trace["plan"].get("router"), "regex")

    def test_model_first_routes_even_when_regex_covers(self) -> None:
        # MODEL-FIRST-ROUTER: on live runs the seat is the primary router
        # even when the regex plan has full coverage — regex is fallback.
        import pipeline
        from unittest.mock import patch

        calls: list = []
        model_plan = {
            "task": "water_highlight",
            "input_mode": "single",
            "tools": ["water_highlight", "area_calc"],
            "vlm_role": "none",
            "supported": True,
            "refusal": None,
            "query": "highlight the water",
            "router": "model-fallback",
            "model_plan_raw": '{"tools":["water_highlight"]}',
        }

        def fake(*a, **k):
            calls.append(1)
            return dict(model_plan)

        with patch.object(pipeline, "MODEL_FIRST", True), patch.object(
            pipeline, "plan_model", side_effect=fake
        ):
            trace = pipeline.run_query(
                "highlight the water", "single", scene=1, live=True
            )
        self.assertEqual(calls, [1])
        self.assertEqual(trace["plan"].get("router"), "model")
        self.assertEqual(trace["plan"]["tools"], ["water_highlight", "area_calc"])

    def test_model_first_regex_fallback_when_model_fails(self) -> None:
        # A failed/malformed model plan must not strand the query — the
        # regex plan runs and the trace records the fallback.
        import pipeline
        from unittest.mock import patch

        with patch.object(pipeline, "MODEL_FIRST", True), patch.object(
            pipeline, "plan_model", side_effect=lambda *a, **k: None
        ), patch.object(
            pipeline,
            "bind_inputs",
            return_value={"ok": False, "error": "stubbed"},
        ):
            trace = pipeline.run_query(
                "highlight the water", "single", scene=1, live=True
            )
        self.assertEqual(trace["plan"].get("router"), "regex-fallback")
        self.assertEqual(
            trace["plan"]["tools"], ["water_highlight", "area_calc", "vqa"]
        )

    def test_model_first_never_calls_seat_on_refusal(self) -> None:
        # Refusals are deterministic safety — the model never sees them,
        # even under model-primary routing.
        import pipeline
        from unittest.mock import patch

        calls: list = []
        with patch.object(pipeline, "MODEL_FIRST", True), patch.object(
            pipeline, "plan_model", side_effect=lambda *a, **k: calls.append(1)
        ):
            trace = pipeline.run_query(
                "What changed between the two dates?",
                "single",
                scene=1,
                live=True,
            )
        self.assertEqual(calls, [])
        self.assertFalse(trace["plan"].get("supported"))

    def test_validator_couples_cdvqa_map_to_change_detect(self) -> None:
        # ROUTE-PROBE finding: the model proposed cdvqa_map alone for
        # "map which land-cover classes transitioned" — but cdvqa_map is
        # nested inside the pipeline's change_detect block, so without
        # change_detect it silently never runs. Validator must couple it.
        from planner import validate_model_plan

        out = validate_model_plan('{"tools": ["cdvqa_map"]}', "bi-temporal")
        self.assertIsNotNone(out)
        self.assertIn("cdvqa_map", out)
        self.assertIn("change_detect", out)
        self.assertLess(out.index("change_detect"), out.index("cdvqa_map"))

    def test_validator_mask_only_bitemporal_drops_vqa(self) -> None:
        # M6/D-021: mask-only bi-temporal phrasing (routing-suite cases
        # 94-96 semantics) must not get narration forced onto it — the
        # validator mirrors the regex mask-only branch.
        from planner import validate_model_plan

        out = validate_model_plan(
            '{"tools": ["change_detect"]}',
            "bi-temporal",
            "show me the change mask only",
        )
        self.assertEqual(out, ["change_detect"])

    def test_validator_bitemporal_question_keeps_vqa(self) -> None:
        from planner import validate_model_plan

        out = validate_model_plan(
            '{"tools": ["change_detect"]}',
            "bi-temporal",
            "what changed between the two dates?",
        )
        self.assertIsNotNone(out)
        self.assertIn("vqa", out)

    def test_validator_single_question_canonical_floor(self) -> None:
        # D-018: the adapted answer seat is not optional on question-shaped
        # single-image queries — mirrors regex is_question semantics.
        from planner import validate_model_plan

        out = validate_model_plan(
            '{"tools": ["water_highlight"]}',
            "single",
            "is there any water in this image?",
        )
        self.assertIsNotNone(out)
        self.assertIn("canonical_vqa", out)
        self.assertIn("vqa", out)
        self.assertIn("area_calc", out)

    def test_validator_single_imperative_no_canonical(self) -> None:
        # Mirrors regex exactly: imperative mask requests get vqa narration
        # but not the canonical seat.
        from planner import validate_model_plan

        out = validate_model_plan(
            '{"tools": ["water_highlight"]}', "single", "highlight the water"
        )
        self.assertEqual(out, ["water_highlight", "area_calc", "vqa"])

    def test_model_first_fallback_reason_recorded(self) -> None:
        # D-021/M3: a model-routing failure must record why — seat down,
        # malformed JSON, or validator reject are different diagnoses.
        import pipeline
        from unittest.mock import patch

        def _boom(*a, **k):
            if isinstance(k.get("_reason"), dict):
                k["_reason"]["why"] = "seat_unreachable_or_timeout"
            return None

        with patch.object(pipeline, "MODEL_FIRST", True), patch.object(
            pipeline, "plan_model", side_effect=_boom
        ), patch.object(
            pipeline,
            "bind_inputs",
            return_value={"ok": False, "error": "stubbed"},
        ):
            trace = pipeline.run_query(
                "highlight the water", "single", scene=1, live=True
            )
        self.assertEqual(trace["plan"].get("router"), "regex-fallback")
        self.assertEqual(
            trace["plan"].get("model_fallback_reason"),
            "seat_unreachable_or_timeout",
        )

    def test_scene1_highlight_writes_overlay_not_primary(self) -> None:
        from pipeline import run_query

        man = json.loads((DEMO / "data" / "scene1" / "manifest.json").read_text(encoding="utf-8"))
        primary = DEMO / "data" / "scene1" / man["primary"]
        mtime_before = primary.stat().st_mtime
        size_before = primary.stat().st_size
        trace = run_query(
            "Highlight the water body referred to in the query",
            "single",
            scene=1,
            live=False,
        )
        self.assertEqual(trace["plan"]["tools"], ["water_highlight", "area_calc", "vqa"])
        self.assertIn("water_highlight", trace["tool_outputs"])
        overlay = Path(trace["overlay_path"])
        self.assertTrue(overlay.is_file())
        self.assertEqual(overlay.name, "water_overlay_live.png")
        self.assertEqual(primary.stat().st_mtime, mtime_before)
        self.assertEqual(primary.stat().st_size, size_before)


class SensorProfileC1Tests(unittest.TestCase):
    """C1: band order comes from a declared profile or band metadata, never
    from band count. Unidentified multispectral/PAN must withhold instead of
    producing a confident wrong answer on the hidden set."""

    @staticmethod
    def _mx_tiff(tmp: Path, names=None):
        """4-band uint16 TIFF in Cartosat-2S MX order B,G,R,NIR.
        Water = left 12 cols (30%): green high, NIR low -> NDWI>0."""
        import numpy as np
        from ingest import write_test_geotiff

        h, w = 40, 40
        water = np.zeros((h, w), dtype=bool)
        water[:, :12] = True
        blue = np.full((h, w), 300, np.uint16)
        green = np.where(water, 900, 200).astype(np.uint16)
        red = np.where(water, 150, 500).astype(np.uint16)
        nir = np.where(water, 40, 700).astype(np.uint16)
        arr = np.stack([blue, green, red, nir])
        tif = write_test_geotiff(
            tmp / "mx.tif", arr, 1.0, 1.0,
            geographic=False, channel_names=names,
        )
        return tif, water

    def test_unnamed_multispectral_withholds(self) -> None:
        # Mechanism evidence (synthetic, assumed B,G,R,NIR): today the tool
        # runs the B-R heuristic on swapped bands — the honest answer is
        # withheld, not a confident number.
        import numpy as np
        from tools import water_highlight

        tmp = Path(tempfile.mkdtemp())
        tif, _water = self._mx_tiff(tmp)
        rgb = np.zeros((40, 40, 3), np.uint8)
        wh = water_highlight(rgb, source_path=tif)
        self.assertTrue(wh.get("withheld"))
        self.assertEqual(wh.get("withheld_reason"), "no_spectral_basis")
        self.assertIn(int(wh.get("water_pixels") or 0), (0,))

    def test_declared_profile_drives_ndwi(self) -> None:
        import numpy as np
        from tools import water_highlight

        tmp = Path(tempfile.mkdtemp())
        tif, _water = self._mx_tiff(tmp)
        rgb = np.zeros((40, 40, 3), np.uint8)
        wh = water_highlight(rgb, source_path=tif, sensor_profile="cartosat2s_mx")
        self.assertFalse(wh.get("withheld"))
        self.assertEqual(wh.get("evidence_class"), "measured_index")
        self.assertIn("NDWI", wh.get("method", ""))
        frac = int(wh["water_pixels"]) / int(np.asarray(wh["mask"]).size)
        self.assertAlmostEqual(frac, 0.30, places=2)

    def test_named_bands_still_ndwi(self) -> None:
        import numpy as np
        from tools import water_highlight

        tmp = Path(tempfile.mkdtemp())
        tif, _water = self._mx_tiff(tmp, names=("blue", "green", "red", "nir"))
        rgb = np.zeros((40, 40, 3), np.uint8)
        wh = water_highlight(rgb, source_path=tif)
        self.assertFalse(wh.get("withheld"))
        self.assertEqual(wh.get("evidence_class"), "measured_index")
        frac = int(wh["water_pixels"]) / int(np.asarray(wh["mask"]).size)
        self.assertAlmostEqual(frac, 0.30, places=2)

    def test_pan_single_band_withholds(self) -> None:
        # Old behavior: PAN returned a confident 0.0% water.
        import numpy as np
        from ingest import write_test_geotiff
        from tools import water_highlight

        tmp = Path(tempfile.mkdtemp())
        pan = np.full((40, 40), 500, np.uint16)
        tif = write_test_geotiff(tmp / "pan.tif", pan, 1.0, 1.0, geographic=False)
        rgb = np.zeros((40, 40, 3), np.uint8)
        wh = water_highlight(rgb, source_path=tif)
        self.assertTrue(wh.get("withheld"))
        self.assertEqual(wh.get("withheld_reason"), "no_spectral_basis")

    def test_unknown_profile_withholds(self) -> None:
        import numpy as np
        from tools import water_highlight

        tmp = Path(tempfile.mkdtemp())
        tif, _water = self._mx_tiff(tmp)
        rgb = np.zeros((40, 40, 3), np.uint8)
        wh = water_highlight(rgb, source_path=tif, sensor_profile="bogus_sat")
        self.assertTrue(wh.get("withheld"))
        self.assertEqual(wh.get("withheld_reason"), "unknown_sensor_profile")

    def test_naip_profile_rgbn_order(self) -> None:
        # NAIP-style R,G,B,NIR ordering must also resolve — the profile, not
        # the band count, carries the order.
        import numpy as np
        from ingest import write_test_geotiff
        from tools import water_highlight

        tmp = Path(tempfile.mkdtemp())
        h, w = 40, 40
        water = np.zeros((h, w), dtype=bool)
        water[:, :12] = True
        red = np.where(water, 150, 500).astype(np.uint16)
        green = np.where(water, 900, 200).astype(np.uint16)
        blue = np.full((h, w), 300, np.uint16)
        nir = np.where(water, 40, 700).astype(np.uint16)
        arr = np.stack([red, green, blue, nir])
        tif = write_test_geotiff(tmp / "naip.tif", arr, 1.0, 1.0, geographic=False)
        rgb = np.zeros((40, 40, 3), np.uint8)
        wh = water_highlight(rgb, source_path=tif, sensor_profile="naip")
        self.assertFalse(wh.get("withheld"))
        self.assertEqual(wh.get("evidence_class"), "measured_index")
        frac = int(wh["water_pixels"]) / int(np.asarray(wh["mask"]).size)
        self.assertAlmostEqual(frac, 0.30, places=2)

    def test_profile_fixes_display_composite(self) -> None:
        # The composite that feeds both VLM seats must use profile indices —
        # on B,G,R,NIR input, channel 0 must be the red band, not band 1.
        from ingest import _load_rgb_array

        tmp = Path(tempfile.mkdtemp())
        tif, _water = self._mx_tiff(tmp)
        prof, _n1 = _load_rgb_array(tif, sensor_profile="cartosat2s_mx")
        naive, _n2 = _load_rgb_array(tif)
        self.assertIsNotNone(prof)
        self.assertIsNotNone(naive)
        self.assertGreater(prof[..., 0].mean(), prof[..., 2].mean())
        self.assertLess(naive[..., 0].mean(), naive[..., 2].mean())

    def test_rgb_png_heuristic_unchanged(self) -> None:
        # Plain RGB imagery keeps the heuristic path, now labelled.
        import numpy as np
        from tools import water_highlight

        tmp = Path(tempfile.mkdtemp())
        rgb = np.zeros((20, 20, 3), np.uint8)
        rgb[..., 2] = 60
        png = tmp / "x.png"
        Image.fromarray(rgb).save(png)
        wh = water_highlight(png)
        self.assertFalse(wh.get("withheld"))
        self.assertEqual(wh.get("evidence_class"), "heuristic_estimate")

    def test_bind_threads_sensor_profile(self) -> None:
        from pipeline import bind_inputs

        tmp = Path(tempfile.mkdtemp())
        tif, _water = self._mx_tiff(tmp)
        bound = bind_inputs(
            "single", uploads={"image": str(tif)}, sensor_profile="cartosat2s_mx"
        )
        self.assertTrue(bound["ok"], bound.get("error"))
        self.assertEqual(bound.get("sensor_profile"), "cartosat2s_mx")

    def test_sar_named_non_vvvh_polarization_refused(self) -> None:
        # RISAT hybrid-pol delivers RH/RV (Stokes-derived), not VV/VH —
        # labelling them VV/VH is an invented claim; refuse instead.
        import numpy as np
        from ingest import read_sar_arrays, write_test_geotiff

        tmp = Path(tempfile.mkdtemp())
        arr = np.stack(
            [np.full((20, 20), 0.2, np.float32), np.full((20, 20), 0.1, np.float32)]
        )
        tif = write_test_geotiff(
            tmp / "risat.tif", arr, 1.0, 1.0,
            geographic=False, channel_names=("rh", "rv"),
        )
        out = read_sar_arrays(tif)
        self.assertFalse(out.get("ok"))
        self.assertIn("polariz", (out.get("error") or "").lower())

    def test_sar_named_vv_vh_verified(self) -> None:
        import numpy as np
        from ingest import read_sar_arrays, write_test_geotiff

        tmp = Path(tempfile.mkdtemp())
        arr = np.stack(
            [np.full((20, 20), 0.2, np.float32), np.full((20, 20), 0.1, np.float32)]
        )
        tif = write_test_geotiff(
            tmp / "s1.tif", arr, 1.0, 1.0,
            geographic=False, channel_names=("vv", "vh"),
        )
        out = read_sar_arrays(tif)
        self.assertTrue(out.get("ok"), out.get("error"))
        self.assertTrue(out.get("pol_verified"))


class SingleSarAndAliasTests(unittest.TestCase):
    """D-019 part A (single-SAR un-refusal) + sensor-profile alias
    normalization (review 2026-09-25 item 1)."""

    def test_profile_alias_normalization(self) -> None:
        import numpy as np
        from ingest import resolve_band_map, write_test_geotiff

        tmp = Path(tempfile.mkdtemp())
        arr = np.stack([np.full((8, 8), 100, np.uint16)] * 4)
        tif = write_test_geotiff(tmp / "x.tif", arr, 1.0, 1.0, geographic=False)
        for alias in (
            "cartosat2s_mx",
            "Cartosat-2S MX",
            "cartosat-2s-mx",
            " CARTOSAT_2S_MX ",
        ):
            m, _note = resolve_band_map(tif, sensor_profile=alias)
            self.assertIsNotNone(m, alias)
            self.assertEqual(m.get("nir"), 4, alias)
        # Semantic aliases are NOT resolved — "cartosat2s" alone is ambiguous
        # between PAN and MX products; guessing it is the C1 bug shape.
        m, note = resolve_band_map(tif, sensor_profile="cartosat2s")
        self.assertIsNone(m)
        self.assertIn("unknown sensor profile", note)

    def test_single_sar_plan_shape(self) -> None:
        from planner import plan

        p = plan("What does the SAR layer show here?", "single")
        self.assertTrue(p["supported"])
        self.assertIsNone(p["refusal"])
        self.assertEqual(p["tools"], ["vqa", "sar_read", "canonical_vqa"])

        p2 = plan("Give me the VV backscatter stats.", "single")
        self.assertTrue(p2["supported"])
        self.assertEqual(p2["tools"], ["vqa", "sar_read"])

        p3 = plan("How much water area does this SAR image hold?", "single")
        self.assertIn("sar_read", p3["tools"])
        self.assertIn("area_calc", p3["tools"])

        # Caption wording stays a caption — no forced SAR stats.
        p4 = plan("describe this SAR scene", "single")
        self.assertEqual(p4["tools"], ["vqa"])

    @staticmethod
    def _sar_tiff(tmp: Path, names=("vv", "vh")):
        import numpy as np
        from ingest import write_test_geotiff

        vv = np.full((24, 24), -10.0, np.float32)
        vv[:, :6] = -30.0  # dark patch -> below the -16 dB water threshold
        vh = np.full((24, 24), -20.0, np.float32)
        return write_test_geotiff(
            tmp / "sar.tif", np.stack([vv, vh]), 1.0, 1.0,
            geographic=False, channel_names=names,
        )

    def test_single_sar_pipeline_runs(self) -> None:
        from pipeline import run_query

        tmp = Path(tempfile.mkdtemp())
        tif = self._sar_tiff(tmp)
        trace = run_query(
            "Give me the VV backscatter stats.",
            "single",
            live=False,
            uploads={"image": str(tif)},
        )
        self.assertTrue(trace["plan"]["supported"])
        sr = trace["tool_outputs"].get("sar_read") or {}
        self.assertFalse(sr.get("withheld"), sr)
        self.assertTrue(sr.get("pol_verified"))
        self.assertIsNotNone(sr.get("vv_db_stats"))
        self.assertTrue(sr.get("water_calibrated"))
        # 6/24 cols below the -16 dB threshold -> ~25% water (boxcar+morph
        # shifts the edge slightly — assert the band, not an exact figure).
        frac = sr.get("water_fraction")
        self.assertIsNotNone(frac)
        self.assertGreater(frac, 0.15)
        self.assertLess(frac, 0.40)

    def test_single_sar_pol_refusal_withholds(self) -> None:
        # Named non-VV/VH (RISAT hybrid RH/RV): ingest refuses, pipeline
        # records a withheld sar_read instead of invented stats.
        from pipeline import run_query

        tmp = Path(tempfile.mkdtemp())
        tif = self._sar_tiff(tmp, names=("rh", "rv"))
        trace = run_query(
            "Give me the VV backscatter stats.",
            "single",
            live=False,
            uploads={"image": str(tif)},
        )
        sr = trace["tool_outputs"].get("sar_read") or {}
        self.assertTrue(sr.get("withheld"), sr)
        self.assertIn("polariz", (sr.get("withheld_reason") or "").lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
