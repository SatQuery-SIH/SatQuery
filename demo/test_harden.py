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
        self.assertIn("Do not conflate", card)
        self.assertIn("20470.5", card)
        path = write_report(trace)
        text = path.read_text(encoding="utf-8")
        self.assertIn("INGEST PREVIEW / VALIDATION", text)
        self.assertIn("changed_pixels", text)
        self.assertIn(trace["query"], text)
        self.assertTrue(path.suffix == ".md")

    def test_tiff_pil_fallback(self) -> None:
        from report import ingest_preview

        tmp = Path(tempfile.mkdtemp()) / "tiny.tif"
        Image.new("RGB", (32, 24), (10, 80, 30)).save(tmp, format="TIFF")
        im, note = ingest_preview(tmp)
        self.assertIsNotNone(im)
        self.assertEqual(im.size, (32, 24))
        self.assertIn("INGEST PREVIEW / VALIDATION", note)
        self.assertIn("cartosat", note.lower())

    def test_banner_and_serve_base(self) -> None:
        import app as demo_app

        demo_app.MODE = "live"
        b = demo_app._banner()
        self.assertIn("LIVE = Qwen3-VL-8B zero-shot, adapter off", b)
        ps1 = (DEMO / "serve.ps1").read_text(encoding="utf-8")
        self.assertIn("Qwen3VL-8B-Instruct-Q4_K_M.gguf", ps1)
        self.assertIn("mmproj-Qwen3VL-8B-Instruct-F16.gguf", ps1)
        self.assertNotIn("adapted08", ps1)
        self.assertNotIn("8091", ps1)
        self.assertIn("--port 8080", ps1)

    def test_buildings_refused(self) -> None:
        from planner import plan

        p = plan("How many buildings are in this image?", "single")
        self.assertFalse(p["supported"])
        self.assertIn("count", (p["refusal"] or "").lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
