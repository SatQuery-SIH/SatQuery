"""PACKET-PRODUCT-WIRE CPU tests. No llama-server. No Gradio launch. No Modal."""
from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

DEMO = Path(__file__).resolve().parent
SAT = DEMO.parent
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))

from report import tool_outputs_sha256, write_report, write_report_bundle  # noqa: E402

S2_QUERY = "What changed between these two dates, and where?"
S1_QUERY = "Describe the land cover and major objects."
S3_QUERY = "Identify water-covered regions."
# WIRE-8091: the old count refusal retired — single-image counts now route
# to canonical_vqa. Refusal-trace safety now exercised via an OOS query.
REFUSAL_QUERY = "What will the weather be tomorrow over this scene?"
INVENTED = "Pair IoU is 0.99 versus the ground-truth mask."
CACHE = SAT / "gates" / "_cache" / "evidence_packet"
PRED = DEMO / "data" / "scene2" / "pred_mask.png"


def _s2() -> dict:
    from pipeline import run_query

    return run_query(S2_QUERY, "bi-temporal", scene=2, live=False, cd_prefer="classical")


class PacketWireTests(unittest.TestCase):
    def test_scene2_packet_in_trace_no_drift(self) -> None:
        from pipeline import run_query
        import pipeline as pl

        mtime_before = PRED.stat().st_mtime

        def skip(trace, the_plan, bound):
            return None

        orig = pl._attach_packet
        pl._attach_packet = skip
        try:
            off = run_query(
                S2_QUERY, "bi-temporal", scene=2, live=False, cd_prefer="classical"
            )
        finally:
            pl._attach_packet = orig
        h_off = tool_outputs_sha256(off.get("tool_outputs") or {})
        on = _s2()
        h_on = tool_outputs_sha256(on.get("tool_outputs") or {})
        self.assertEqual(h_off, h_on)
        self.assertNotIn("evidence_packet", off)
        pkt = on.get("evidence_packet") or {}
        self.assertEqual(pkt.get("canonical_answer"), "not_determined")
        self.assertEqual(pkt.get("task"), "change_description")
        self.assertTrue(pkt.get("claims"))
        self.assertEqual(PRED.stat().st_mtime, mtime_before)
        CACHE.mkdir(parents=True, exist_ok=True)
        (CACHE / "wire_scene2_hash.txt").write_text(h_on + "\n", encoding="utf-8")

    def test_failure_isolation_monkeypatch(self) -> None:
        from pipeline import run_query

        mtime_before = PRED.stat().st_mtime
        with patch(
            "evidence_packet.build_packet_for_run",
            side_effect=RuntimeError("forced packet failure"),
        ):
            trace = run_query(
                S2_QUERY, "bi-temporal", scene=2, live=False, cd_prefer="classical"
            )
        self.assertIn("evidence_packet_error", trace)
        self.assertIn("forced packet failure", trace["evidence_packet_error"])
        self.assertNotIn("evidence_packet", trace)
        cd = (trace.get("tool_outputs") or {}).get("change_detect") or {}
        self.assertEqual(cd.get("changed_pixels"), 270611)
        self.assertEqual(cd.get("built_up_direction"), "not_determined")
        vis = trace.get("visible_answer") or ""
        self.assertIn("### Findings (from tools)", vis)
        self.assertEqual(PRED.stat().st_mtime, mtime_before)

    def test_bundle_keys_and_findings_first(self) -> None:
        trace = _s2()
        paths = write_report_bundle(trace)
        for key in ("md", "packet", "benchmark", "product"):
            self.assertIn(key, paths)
            self.assertTrue(Path(paths[key]).is_file(), key)
        stamp = paths["md"].stem.replace("satquery_run_", "", 1)
        self.assertTrue(paths["packet"].name.endswith(f"{stamp}_packet.json"))
        self.assertTrue(paths["benchmark"].name.endswith(f"{stamp}_benchmark.json"))
        self.assertTrue(paths["product"].name.endswith(f"{stamp}_product.md"))
        bench = json.loads(paths["benchmark"].read_text(encoding="utf-8"))
        self.assertEqual(set(bench.keys()), {"answer", "mask", "box"})
        self.assertEqual(bench["answer"], "not_determined")
        product = paths["product"].read_text(encoding="utf-8")
        self.assertTrue(product.startswith("### Findings (from tools)"))
        md = paths["md"].read_text(encoding="utf-8")
        self.assertIn("## Evidence packet", md)
        self.assertIn("canonical_answer: `not_determined`", md)
        CACHE.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(paths["benchmark"], CACHE / "wire_scene2_benchmark.json")
        (CACHE / "wire_scene2_product_first_line.txt").write_text(
            product.splitlines()[0] + "\n", encoding="utf-8"
        )
        (CACHE / "wire_bundle_paths.json").write_text(
            json.dumps({k: str(v) for k, v in paths.items()}, indent=2) + "\n",
            encoding="utf-8",
        )
        self._bundle = paths

    def test_invented_099_unverified_in_product(self) -> None:
        trace = _s2()
        trace = dict(trace)
        trace["answer"] = INVENTED
        paths = write_report_bundle(trace)
        product = paths["product"].read_text(encoding="utf-8")
        self.assertTrue(product.startswith("### Findings (from tools)"))
        self.assertIn("UNVERIFIED INTERPRETATION — JSON card wins.", product)
        self.assertLess(
            product.index("### Findings (from tools)"),
            product.index("UNVERIFIED INTERPRETATION — JSON card wins."),
        )
        vis = trace.get("visible_answer") or ""
        self.assertIn("### Findings (from tools)", vis)

    def test_refusal_trace_safe(self) -> None:
        from pipeline import run_query

        trace = run_query(REFUSAL_QUERY, "single", scene=1, live=False)
        self.assertFalse((trace.get("plan") or {}).get("supported"))
        md = write_report(trace)
        self.assertTrue(md.is_file())
        bundle = write_report_bundle(trace)
        self.assertTrue(bundle["md"].is_file())
        self.assertTrue(bundle["packet"].is_file())
        pkt = trace.get("evidence_packet") or json.loads(
            bundle["packet"].read_text(encoding="utf-8")
        )
        self.assertIn("claims", pkt)
        self.assertIn("limitations", pkt)
        self.assertTrue(pkt.get("limitations"))

    def test_scene1_and_scene3_packets(self) -> None:
        from pipeline import run_query

        t1 = run_query(S1_QUERY, "single", scene=1, live=False)
        self.assertIn("evidence_packet", t1)
        p1 = t1["evidence_packet"]
        self.assertIn("claims", p1)
        self.assertIn("limitations", p1)
        self.assertIsNone(p1.get("canonical_answer"))
        t3 = run_query(S3_QUERY, "optical+sar", scene=3, live=False)
        self.assertIn("evidence_packet", t3)
        p3 = t3["evidence_packet"]
        self.assertTrue(any(c.get("predicate") == "water_calibrated" for c in p3["claims"]))
        write_report_bundle(t1)
        write_report_bundle(t3)

    def test_cached_trapdoor_bundle_safe(self) -> None:
        dummy = {
            "plan": {"supported": False, "tools": [], "refusal": "cached+upload", "task": "unsupported"},
            "tool_outputs": {},
            "answer": "Cached trapdoor uses prepared scenes only.",
            "live": False,
            "query": S1_QUERY,
            "input_mode": "single",
            "scene": 1,
            "first_token_s": 0.0,
            "complete_s": 0.0,
            "overlay_path": None,
            "images": [],
        }
        paths = write_report_bundle(dummy)
        self.assertTrue(paths["md"].is_file())
        self.assertTrue(paths["product"].is_file())
        product = paths["product"].read_text(encoding="utf-8")
        self.assertTrue(product.startswith("### Findings (from tools)"))

    def test_pack_returns_bundle_paths_no_launch(self) -> None:
        import app as demo_app

        trace = _s2()
        out = demo_app._pack(trace, None)
        self.assertEqual(len(out), 8)
        md_path = Path(out[6])
        extras = out[7]
        self.assertTrue(md_path.is_file())
        self.assertEqual(len(extras), 3)
        for p in extras:
            self.assertTrue(Path(p).is_file(), p)


if __name__ == "__main__":
    unittest.main(verbosity=2)
