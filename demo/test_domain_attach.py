"""CF-DOMAIN-ATTACH CPU tests. No llama-server. No Gradio launch. No Modal train."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

DEMO = Path(__file__).resolve().parent
SAT = DEMO.parent
if str(SAT) not in sys.path:
    sys.path.insert(0, str(SAT))
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))

TEAM_SHA = "cbfefe6564613571e824cb2cee0f971b4793ca0dda6c1a5c5741cbcdf4c76fe6"
IMPORTED_SHA = "db0dd783c3c3f27d02f55f24fa8199d2b2b47c422c7f7bd2e4d96e2fa65d69f2"
SCENE2_TOOL_SHA = "11ccb8da1bb7f9467c15dd0a62c8d42560feaac8da9f959d0bde3957f16a9964"  # 2026-09-24 authorized WEB-POLISH-V3
PRED = DEMO / "data" / "scene2" / "pred_mask.png"
PRED_SHA = "d531400303e1c76c47ed76885808b72be1ab6fffbc5dcf683876d5d67d665b14"
APP_PY = DEMO / "app.py"
DISCLOSURE = (
    "Two change-finding models run here, picked automatically:"
)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Scene4AssetTests(unittest.TestCase):
    def test_manifest_val_pair_from_frozen_val(self) -> None:
        man = json.loads((DEMO / "data" / "scene4" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(man["id"], "02524.png")
        self.assertEqual(man["domain"], "second")
        self.assertGreaterEqual(float(man["val_iou"]), 0.5)
        self.assertIn("never train", man["source"])
        if not (SAT / "gates" / "_cache" / "cf_ft" / "second_val_ids.json").is_file():
            self.skipTest("second_val_ids.json not present")
        val = json.loads((SAT / "gates" / "_cache" / "cf_ft" / "second_val_ids.json").read_text(encoding="utf-8"))
        self.assertIn("02524.png", val["ids"])
        self.assertEqual(int(val["seed"]), 42)
        for name in ("before.png", "after.png", "gt_mask.png"):
            self.assertTrue((DEMO / "data" / "scene4" / name).is_file())


class RouterTests(unittest.TestCase):
    def test_manifest_domain_only(self) -> None:
        from pipeline import _scene_manifest_domain, bind_inputs

        self.assertEqual(_scene_manifest_domain(4), "second")
        self.assertEqual(_scene_manifest_domain(2), "levir")
        b4 = bind_inputs("bi-temporal", scene=4, uploads=None)
        self.assertTrue(b4["ok"])
        self.assertEqual(b4["cd_domain"], "second")
        self.assertFalse(b4["use_prepared_cd_cache"])
        b2 = bind_inputs("bi-temporal", scene=2, uploads=None)
        self.assertEqual(b2["cd_domain"], "levir")
        self.assertTrue(b2["use_prepared_cd_cache"])

    def test_upload_defaults_imported_with_limitation(self) -> None:
        from pipeline import bind_inputs
        from tools import UPLOAD_DOMAIN_LIMITATION

        tmp = Path(tempfile.mkdtemp())
        b = tmp / "b.png"
        a = tmp / "a.png"
        Image.new("RGB", (32, 32), (1, 2, 3)).save(b)
        Image.new("RGB", (32, 32), (4, 5, 6)).save(a)
        bound = bind_inputs("bi-temporal", scene=2, uploads={"before": b, "after": a})
        self.assertTrue(bound["ok"])
        self.assertEqual(bound["source"], "upload")
        self.assertEqual(bound["cd_domain"], "levir")
        self.assertIn(UPLOAD_DOMAIN_LIMITATION, bound["ingest_note"])

    def test_finder_imported_for_levir(self) -> None:
        from tools import _find_changeformer_ckpt, sha256_file

        p = _find_changeformer_ckpt(domain="levir")
        self.assertIsNotNone(p)
        self.assertEqual(sha256_file(p), IMPORTED_SHA)

    def test_finder_team_for_second_when_approved(self) -> None:
        from tools import _find_changeformer_ckpt, sha256_file

        p = _find_changeformer_ckpt(domain="second")
        self.assertIsNotNone(p)
        self.assertEqual(sha256_file(p), TEAM_SHA)


class NoRegressionTests(unittest.TestCase):
    def test_scene2_pred_mask_bytes_and_tool_hash(self) -> None:
        from pipeline import run_query
        from report import tool_outputs_sha256

        self.assertEqual(_sha(PRED), PRED_SHA)
        mtime = PRED.stat().st_mtime
        trace = run_query(
            "What changed between these two dates, and where?",
            "bi-temporal",
            scene=2,
            live=False,
        )
        self.assertEqual(PRED.stat().st_mtime, mtime)
        self.assertEqual(_sha(PRED), PRED_SHA)
        cd = (trace.get("tool_outputs") or {}).get("change_detect") or {}
        self.assertEqual(cd.get("built_up_direction"), "not_determined")
        pkt = trace.get("evidence_packet") or {}
        self.assertEqual(pkt.get("canonical_answer"), "not_determined")
        self.assertEqual(tool_outputs_sha256(trace.get("tool_outputs") or {}), SCENE2_TOOL_SHA)
        # live=False prepared Scene 2 loads pred_mask.png (rung cached_mask; no live ckpt).
        # Imported LEVIR still owns that domain; team_second is not in this trace.
        self.assertEqual(cd.get("rung"), "cached_mask")
        self.assertNotEqual(cd.get("checkpoint_sha256"), TEAM_SHA)
        from tools import _find_changeformer_ckpt, sha256_file

        self.assertEqual(sha256_file(_find_changeformer_ckpt(domain="levir")), IMPORTED_SHA)

    def test_levir_eval_paths_unchanged(self) -> None:
        if importlib.util.find_spec("cf_ft.eval_levir") is None:
            self.skipTest("cf_ft.eval_levir module not present")
        from cf_ft.eval_levir import LEVIR_CD, SCENE2, SCORE_JSON

        self.assertEqual(SCENE2, SAT / "demo" / "data" / "scene2")
        self.assertEqual(LEVIR_CD, SAT / "gates" / "_cache" / "levir_cd")
        self.assertTrue(SCORE_JSON.is_file())


class LiveTeamScene4Tests(unittest.TestCase):
    def test_pipeline_team_second_live_beats_imported(self) -> None:
        from pipeline import run_query

        mtime = PRED.stat().st_mtime
        sha_before = _sha(PRED)
        trace = run_query(
            "What changed between these two dates, and where?",
            "bi-temporal",
            scene=4,
            live=False,
        )
        self.assertEqual(PRED.stat().st_mtime, mtime)
        self.assertEqual(_sha(PRED), sha_before)
        cd = (trace.get("tool_outputs") or {}).get("change_detect") or {}
        self.assertEqual(cd.get("rung"), "team_second_live")
        self.assertEqual(cd.get("checkpoint_sha256"), TEAM_SHA)
        self.assertEqual(cd.get("built_up_direction"), "not_determined")
        iou = float((cd.get("vs_gt") or {}).get("iou") or 0)
        if not (SAT / "gates" / "_cache" / "cf_ft" / "scene4_live.json").is_file():
            self.skipTest("scene4_live.json not present")
        live = json.loads((SAT / "gates" / "_cache" / "cf_ft" / "scene4_live.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(iou, float(live["imported_iou"]) + 0.10)
        ov = Path(trace["overlay_path"])
        self.assertTrue(ov.is_file())
        self.assertIn("scene4", ov.as_posix())
        self.assertNotIn("scene2", ov.as_posix())


class DisclosureAndRegistryTests(unittest.TestCase):
    def test_ui_copy_states_win_and_regression(self) -> None:
        text = APP_PY.read_text(encoding="utf-8")
        self.assertIn(DISCLOSURE, text)
        self.assertIn("Housing construction (high-res)", text)
        self.assertIn("Mixed land change (general purpose)", text)

    def test_registry_team_second_routed_retention_untouched(self) -> None:
        rec = json.loads((SAT / "gates" / "_cache" / "cf_ft" / "ckpt_registry.json").read_text(encoding="utf-8"))
        by = {c["role"]: c for c in rec["checkpoints"]}
        team = by["team_second"]
        self.assertEqual(team["sha256"], TEAM_SHA)
        self.assertEqual(team["domains"], ["second_like"])
        self.assertEqual(team["route_scope"], "second_like")
        self.assertTrue(team["approved_for_demo"])
        self.assertTrue(rec["approved_for_demo_any"])
        ret = by["team_second_retention"]
        self.assertTrue(ret["retention_failed"])
        self.assertFalse(ret["approved_for_demo"])
        self.assertIsNone(ret["path"])
        self.assertEqual(by["imported_levir"]["sha256"], IMPORTED_SHA)
        self.assertFalse(by["imported_levir"].get("approved_for_demo"))


if __name__ == "__main__":
    unittest.main()
