"""EVIDENCE-PACKET CPU tests. Does not start llama-server. No Modal."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

DEMO = Path(__file__).resolve().parent
SAT = DEMO.parent
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))

from evidence_packet import (  # noqa: E402
    BENCHMARK_KEYS,
    CACHE,
    CANONICAL_NOT_DETERMINED,
    CLAIM_FIELDS,
    build_packet,
    check_packet_narration,
    dump_prepared_scene2,
    render_benchmark,
    render_product,
    validate_packet,
)
from report import check_narration  # noqa: E402

SCENE2_TOOLS = {
    "change_detect": {
        "rung": "cached_mask",
        "changed_pixels": 270611,
        "vs_gt": {"iou": 0.852207},
        "radiometric_label": "similar",
        "radiometric_delta": -7.40,
        "delta_mean": -7.40,
        "direction_epsilon": 8.0,
        "built_up_direction": "not_determined",
        "provenance": "demo/data/scene2/pred_mask.png",
    },
    "area_calc": {
        "changed_pixels": 270611,
        "area_m2": 67652.75,
        "area_km2": 0.06765275,
        "gsd_m": 0.5,
        "formula": "area_m2 = count(mask>0) * gsd_m^2",
        "dominant_quadrant": "NE",
    },
}


class PacketBuilderTests(unittest.TestCase):
    def test_scene2_canonical_is_not_determined(self) -> None:
        packet = build_packet(
            SCENE2_TOOLS,
            "change_description",
            mask_path="demo/data/scene2/pred_mask.png",
        )
        self.assertEqual(packet["canonical_answer"], CANONICAL_NOT_DETERMINED)
        self.assertEqual(packet["task"], "change_description")
        for key in CLAIM_FIELDS:
            self.assertTrue(all(key in c for c in packet["claims"]), key)
        preds = {c["predicate"]: c["value"] for c in packet["claims"]}
        self.assertEqual(preds["built_up_direction"], CANONICAL_NOT_DETERMINED)
        self.assertEqual(preds["radiometric_label"], "similar")
        self.assertEqual(preds["radiometric_delta"], -7.40)
        self.assertTrue(preds["change_detected"])
        self.assertEqual(validate_packet(packet), [])
        blob = json.dumps(packet["canonical_answer"])
        self.assertEqual(json.loads(blob), "not_determined")

    def test_built_up_guess_cannot_own_canonical(self) -> None:
        tout = json.loads(json.dumps(SCENE2_TOOLS))
        tout["change_detect"]["built_up_direction"] = "class_map_guess"
        packet = build_packet(tout, "change_description", mask_path="pred_mask.png")
        self.assertEqual(packet["canonical_answer"], CANONICAL_NOT_DETERMINED)
        bu = [c for c in packet["claims"] if c["predicate"] == "built_up_direction"]
        self.assertEqual(bu[0]["value"], CANONICAL_NOT_DETERMINED)

    def test_radiometric_similar_is_a_claim_not_canonical(self) -> None:
        packet = build_packet(SCENE2_TOOLS, "change_description", mask_path="pred_mask.png")
        self.assertNotEqual(packet["canonical_answer"], "similar")
        radio = [c for c in packet["claims"] if c["predicate"] == "radiometric_label"]
        self.assertEqual(radio[0]["value"], "similar")


class RendererTests(unittest.TestCase):
    def setUp(self) -> None:
        self.packet = build_packet(
            SCENE2_TOOLS,
            "change_description",
            mask_path="demo/data/scene2/pred_mask.png",
        )

    def test_benchmark_is_answer_mask_box_only(self) -> None:
        bench = render_benchmark(self.packet)
        self.assertEqual(set(bench.keys()), set(BENCHMARK_KEYS))
        self.assertEqual(bench["answer"], CANONICAL_NOT_DETERMINED)
        self.assertEqual(bench["mask"], "demo/data/scene2/pred_mask.png")

    def test_product_findings_header_first(self) -> None:
        text = render_product(self.packet, qwen_text="Claim E1 is a mask measurement.")
        self.assertTrue(text.startswith("### Findings (from tools)"), text[:80])
        self.assertIn("built_up_direction=`not_determined`", text)
        self.assertIn("radiometric_label=`similar`", text)
        self.assertLess(text.index("### Findings (from tools)"), text.index("Claim E1"))

    def test_invented_099_still_fails_check_narration(self) -> None:
        invented = "Pair IoU is 0.99 versus the ground-truth mask."
        out = check_narration(invented, SCENE2_TOOLS)
        self.assertFalse(out["ok"], out)
        self.assertTrue(any("0.99" in i for i in out["issues"]), out)
        pkt = check_packet_narration(invented, self.packet)
        self.assertFalse(pkt["ok"], pkt)
        vis = render_product(self.packet, qwen_text=invented)
        self.assertTrue(vis.startswith("### Findings (from tools)"))
        self.assertIn("UNVERIFIED INTERPRETATION — JSON card wins.", vis)
        self.assertLess(
            vis.index("### Findings (from tools)"),
            vis.index("UNVERIFIED INTERPRETATION — JSON card wins."),
        )
        self.assertTrue(vis.strip().endswith(invented))

    def test_cited_tool_numbers_pass_gate(self) -> None:
        ok_text = "Claim E1: changed pixels 270611, IoU 0.852 versus GT, delta -7.40."
        out = check_narration(ok_text, SCENE2_TOOLS)
        self.assertTrue(out["ok"], out)
        pkt = check_packet_narration(ok_text, self.packet)
        self.assertTrue(pkt["ok"], pkt)
        vis = render_product(self.packet, qwen_text=ok_text)
        self.assertNotIn("UNVERIFIED INTERPRETATION", vis)
        self.assertTrue(vis.startswith("### Findings (from tools)"))

    def test_unknown_claim_id_fails(self) -> None:
        out = check_packet_narration("See claim E99 for the mask.", self.packet)
        self.assertFalse(out["ok"], out)
        self.assertTrue(any("E99" in i for i in out["issues"]), out)


class PreparedScene2DumpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pred = DEMO / "data" / "scene2" / "pred_mask.png"
        cls._mtime_before = pred.stat().st_mtime
        cls.report = dump_prepared_scene2(CACHE)
        cls._mtime_after = pred.stat().st_mtime
        cls.packet = json.loads((CACHE / "scene2_packet.json").read_text(encoding="utf-8"))
        cls.bench = json.loads((CACHE / "scene2_benchmark.json").read_text(encoding="utf-8"))

    def test_pred_mask_not_rewritten(self) -> None:
        self.assertEqual(self._mtime_after, self._mtime_before)

    def test_dumped_canonical_and_renderers(self) -> None:
        self.assertEqual(self.packet["canonical_answer"], CANONICAL_NOT_DETERMINED)
        self.assertEqual(self.packet["task"], "change_description")
        self.assertEqual(self.bench["answer"], CANONICAL_NOT_DETERMINED)
        self.assertEqual(set(self.bench.keys()), set(BENCHMARK_KEYS))
        self.assertTrue(Path(self.bench["mask"]).is_file(), self.bench["mask"])
        self.assertIsInstance(self.bench["box"], list)
        self.assertEqual(len(self.bench["box"]), 4)
        product = (CACHE / "scene2_product.md").read_text(encoding="utf-8")
        self.assertTrue(product.startswith("### Findings (from tools)"))
        self.assertIn("### Claims (from tools)", product)
        self.assertEqual(self.report["invented_0.99_check_narration"]["ok"], False)
        self.assertTrue(
            any("0.99" in i for i in self.report["invented_0.99_check_narration"]["issues"])
        )
        bad = (CACHE / "scene2_product_invented_099.md").read_text(encoding="utf-8")
        self.assertIn("UNVERIFIED INTERPRETATION — JSON card wins.", bad)
        self.assertEqual(validate_packet(self.packet), [])
        radio = [c for c in self.packet["claims"] if c["predicate"] == "radiometric_label"]
        self.assertEqual(radio[0]["value"], "similar")


if __name__ == "__main__":
    unittest.main(verbosity=2)
