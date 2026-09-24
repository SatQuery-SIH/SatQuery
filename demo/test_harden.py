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


if __name__ == "__main__":
    unittest.main(verbosity=2)
