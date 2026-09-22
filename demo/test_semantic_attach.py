"""SEMANTIC-ATTACH CPU tests. No llama-server. No Gradio launch. No Modal train."""
from __future__ import annotations

import hashlib
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
SEM_SHA = "438cac09be7c630254a12278550b64f86254ecc131ee0cdc723fd526210764d8"
SCENE2_TOOL_SHA = "cfe8f6ec85df0a53d643c716de351697161c54b841d70bd6d69b95d7718c8f27"  # additive rung_display; core 7a83e510… unchanged
PRED = DEMO / "data" / "scene2" / "pred_mask.png"
PRED_SHA = "d531400303e1c76c47ed76885808b72be1ab6fffbc5dcf683876d5d67d665b14"
APP_PY = DEMO / "app.py"
LIVE_JSON = SAT / "gates" / "_cache" / "cf_ft" / "scene4_semantic_live.json"
V1_SCORES = SAT / "gates" / "_cache" / "cf_ft" / "cdvqa_scores_sem.json"
V1_PREDS = SAT / "gates" / "_cache" / "cf_ft" / "cdvqa_preds_sem.jsonl"
V1_CDVQA = SAT / "gates" / "_cache" / "scoreboard" / "cdvqa_preds.jsonl"
V1_SCORES_SHA = "bd562b2dc63b92786f6de54716f06917d876614cebc47c273792d94b97079f60"
V1_PREDS_SHA = "77b731bbf6d1d86c0d5e3d52f12ece3460eba967ce97fd392dab0953224e114c"
V1_CDVQA_SHA = "881d27d21e9a4c7b96cb8565058c83e33a2dd5e5f3b1bb71699160be0cb6ae26"
DISCLOSURE_KEEP = "Two change-finding models run here, picked automatically:"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Scene4NeverTrainTests(unittest.TestCase):
    def test_scene4_is_frozen_val_not_train(self) -> None:
        man = json.loads((DEMO / "data" / "scene4" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(man["id"], "02524.png")
        self.assertEqual(man["domain"], "second")
        self.assertIn("never train", man["source"])
        if not (SAT / "gates" / "_cache" / "cf_ft" / "second_val_ids.json").is_file():
            self.skipTest("second_val_ids.json not present")
        val = json.loads((SAT / "gates" / "_cache" / "cf_ft" / "second_val_ids.json").read_text(encoding="utf-8"))
        self.assertIn("02524.png", val["ids"])
        if not (SAT / "gates" / "_cache" / "scoreboard" / "cdvqa_threshold_fitlist.json").is_file():
            self.skipTest("cdvqa_threshold_fitlist.json not present")
        fit = json.loads(
            (SAT / "gates" / "_cache" / "scoreboard" / "cdvqa_threshold_fitlist.json").read_text(encoding="utf-8")
        )
        self.assertNotIn("02524.png", fit["ids"])


class FinderAndRouterTests(unittest.TestCase):
    def test_finder_requires_approval_for_demo(self) -> None:
        from tools import _find_second_semantic_ckpt, sha256_file

        rec = json.loads((SAT / "gates" / "_cache" / "cf_ft" / "ckpt_registry.json").read_text(encoding="utf-8"))
        by = {c["role"]: c for c in rec["checkpoints"]}
        sem = by["second_semantic"]
        p_meas = _find_second_semantic_ckpt(require_approved=False)
        self.assertIsNotNone(p_meas)
        self.assertEqual(sha256_file(p_meas), SEM_SHA)
        if sem.get("approved_for_demo") is True and sem.get("route_scope") == "second_like_type":
            p_demo = _find_second_semantic_ckpt(require_approved=True)
            self.assertIsNotNone(p_demo)
            self.assertEqual(sha256_file(p_demo), SEM_SHA)
        else:
            self.assertIsNone(_find_second_semantic_ckpt(require_approved=True))

    def test_upload_defaults_without_semantic(self) -> None:
        from pipeline import bind_inputs
        from tools import SEMANTIC_UPLOAD_LIMITATION, UPLOAD_DOMAIN_LIMITATION

        tmp = Path(tempfile.mkdtemp())
        b = tmp / "b.png"
        a = tmp / "a.png"
        Image.new("RGB", (32, 32), (1, 2, 3)).save(b)
        Image.new("RGB", (32, 32), (4, 5, 6)).save(a)
        bound = bind_inputs("bi-temporal", scene=4, uploads={"before": b, "after": a})
        self.assertTrue(bound["ok"])
        self.assertEqual(bound["source"], "upload")
        self.assertEqual(bound["cd_domain"], "levir")
        self.assertIn(UPLOAD_DOMAIN_LIMITATION, bound["ingest_note"])
        self.assertIn(SEMANTIC_UPLOAD_LIMITATION, bound["ingest_note"])

    def test_generic_scene4_query_is_not_type_family(self) -> None:
        from cf_ft.semantic import classify_semantic_family

        self.assertIsNone(
            classify_semantic_family("What changed between these two dates, and where?")
        )


class NoRegressionTests(unittest.TestCase):
    def test_scene2_byte_stable_and_not_determined(self) -> None:
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
        self.assertNotIn("semantic", trace.get("tool_outputs") or {})
        from tools import _find_changeformer_ckpt, sha256_file

        self.assertEqual(sha256_file(_find_changeformer_ckpt(domain="levir")), IMPORTED_SHA)

    def test_v1_cdvqa_files_untouched(self) -> None:
        if not (V1_SCORES.is_file() and V1_PREDS.is_file() and V1_CDVQA.is_file()):
            self.skipTest("v1 cdvqa artifacts not present")
        self.assertEqual(_sha(V1_SCORES), V1_SCORES_SHA)
        self.assertEqual(_sha(V1_PREDS), V1_PREDS_SHA)
        self.assertEqual(_sha(V1_CDVQA), V1_CDVQA_SHA)
        self.assertEqual(V1_CDVQA.stat().st_size, 30355272)

    def test_team_second_routing_unchanged(self) -> None:
        from tools import _find_changeformer_ckpt, sha256_file

        self.assertEqual(sha256_file(_find_changeformer_ckpt(domain="second")), TEAM_SHA)
        rec = json.loads((SAT / "gates" / "_cache" / "cf_ft" / "ckpt_registry.json").read_text(encoding="utf-8"))
        by = {c["role"]: c for c in rec["checkpoints"]}
        team = by["team_second"]
        self.assertEqual(team["route_scope"], "second_like")
        self.assertTrue(team["approved_for_demo"])
        self.assertTrue(by["team_second_retention"]["retention_failed"])
        self.assertFalse(by["imported_levir"].get("approved_for_demo"))


class LiveSemanticGateTests(unittest.TestCase):
    def test_scene4_live_json_if_present(self) -> None:
        if not LIVE_JSON.is_file():
            self.skipTest("scene4_semantic_live.json not written yet")
        rec = json.loads(LIVE_JSON.read_text(encoding="utf-8"))
        self.assertEqual(rec.get("rung"), "semantic_live")
        self.assertEqual(rec.get("checkpoint_sha256"), SEM_SHA)
        self.assertTrue(rec.get("valid_6class_maps"))
        self.assertEqual(rec.get("id"), "02524.png")
        self.assertTrue(rec.get("never_train"))
        self.assertIn(rec.get("live_built_up_direction"), {"increase", "decrease", "no_change"})
        gold = rec.get("gold") or {}
        self.assertTrue(gold.get("scorer_only"))
        self.assertEqual(gold.get("built_up_direction"), rec.get("live_built_up_direction"))

    def test_pipeline_generic_scene4_stays_binary(self) -> None:
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
        self.assertNotIn("semantic", trace.get("tool_outputs") or {})

    def test_pipeline_type_family_scene4_after_approval(self) -> None:
        rec = json.loads((SAT / "gates" / "_cache" / "cf_ft" / "ckpt_registry.json").read_text(encoding="utf-8"))
        by = {c["role"]: c for c in rec["checkpoints"]}
        sem = by["second_semantic"]
        if not (sem.get("approved_for_demo") is True and sem.get("route_scope") == "second_like_type"):
            self.skipTest("second_semantic not approved — attach withheld")
        from pipeline import run_query

        trace = run_query(
            "Has built-up area increased, decreased, or remained unchanged?",
            "bi-temporal",
            scene=4,
            live=False,
        )
        tout = trace.get("tool_outputs") or {}
        sem_out = tout.get("semantic") or {}
        cd = tout.get("change_detect") or {}
        self.assertEqual(sem_out.get("rung"), "semantic_live")
        self.assertEqual(sem_out.get("checkpoint_sha256"), SEM_SHA)
        self.assertTrue(sem_out.get("valid_6class_maps"))
        self.assertIn(cd.get("built_up_direction"), {"increase", "decrease", "no_change"})
        self.assertEqual(cd.get("built_up_direction"), sem_out.get("built_up_direction"))


class DisclosureTests(unittest.TestCase):
    def test_tab2_keeps_prior_copy_and_states_semantic_if_attached(self) -> None:
        text = APP_PY.read_text(encoding="utf-8")
        self.assertIn(DISCLOSURE_KEEP, text)
        self.assertIn("Housing construction (high-res)", text)
        self.assertIn("Mixed land change (general purpose)", text)
        rec = json.loads((SAT / "gates" / "_cache" / "cf_ft" / "ckpt_registry.json").read_text(encoding="utf-8"))
        by = {c["role"]: c for c in rec["checkpoints"]}
        sem = by["second_semantic"]
        if sem.get("approved_for_demo") is True:
            self.assertIn("0.417", text)
            self.assertIn("0.057", text)
            self.assertIn("SECOND-domain", text)
            self.assertIn("0.005", text)


if __name__ == "__main__":
    unittest.main()
