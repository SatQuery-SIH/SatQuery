"""DISPLAY-NAMES CPU tests. No llama-server. No Gradio launch. No Modal."""
from __future__ import annotations

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
SCENE2_TOOL_SHA_CORE = "7a83e510934be5d190325e6a575437474bbb473a19af743f8cd9637a4c36f4d1"
APP_PY = DEMO / "app.py"
TOOLS_PY = DEMO / "tools.py"
PIPE_PY = DEMO / "pipeline.py"
REPORT_PY = DEMO / "report.py"
REGISTRY = SAT / "gates" / "_cache" / "cf_ft" / "ckpt_registry.json"
TAB2_KEEP = (
    "Two change-finding models run here, picked automatically:",
    "Housing construction (high-res)",
    "Mixed land change (general purpose)",
    "SECOND-domain",
    "0.417",
    "0.057",
    "0.005",
)
DISPLAY_ROLES = {
    "team_second": "Housing Change Specialist (SECOND-tuned)",
    "second_semantic": "Land-Cover Change Describer (6-class)",
    "imported_levir": "General Change Model (imported)",
    "second_like": "SECOND-style detailed pairs",
    "second": "SECOND-style detailed pairs",
}
DISPLAY_RUNGS = {
    "team_second_live": "housing-change (live)",
    "semantic_live": "land-cover (live)",
    "changeformer_v6": "general change (live)",
    "cached_mask": "cached mask",
}
REGISTRY_ROLES = (
    "imported_levir",
    "team_second",
    "team_second_retention",
    "second_semantic",
)
TRACE_KEYS = (
    "checkpoint_sha256",
    "semantic_checkpoint_sha256",
)
RUNG_KEYS = (
    "team_second_live",
    "semantic_live",
    "changeformer_v6",
    "cached_mask",
    "classical_otsu",
)
DISPLAY_FIELD_KEYS = frozenset({"rung_display", "semantic_rung_display"})


def _strip_display(obj):
    if isinstance(obj, dict):
        return {
            k: _strip_display(v)
            for k, v in obj.items()
            if k not in DISPLAY_FIELD_KEYS
        }
    if isinstance(obj, list):
        return [_strip_display(x) for x in obj]
    return obj


class MappingTests(unittest.TestCase):
    def test_mapping_completeness(self) -> None:
        from tools import MODEL_DISPLAY, RUNG_DISPLAY, display_name_for_role, display_name_for_rung

        for role, label in DISPLAY_ROLES.items():
            self.assertEqual(MODEL_DISPLAY[role], label)
            self.assertEqual(display_name_for_role(role), label)
        for rung, label in DISPLAY_RUNGS.items():
            self.assertEqual(RUNG_DISPLAY[rung], label)
            self.assertEqual(display_name_for_rung(rung), label)
        self.assertEqual(display_name_for_role("unknown_role"), "unknown_role")
        self.assertEqual(display_name_for_rung("unknown_rung"), "unknown_rung")
        self.assertEqual(display_name_for_role(""), "")
        self.assertEqual(display_name_for_rung(None), "")


class InvarianceTests(unittest.TestCase):
    def test_sha_constants_unchanged(self) -> None:
        from tools import IMPORTED_LEVIR_SHA, SECOND_SEMANTIC_SHA, TEAM_SECOND_SHA

        self.assertEqual(TEAM_SECOND_SHA, TEAM_SHA)
        self.assertEqual(IMPORTED_LEVIR_SHA, IMPORTED_SHA)
        self.assertEqual(SECOND_SEMANTIC_SHA, SEM_SHA)
        tools_src = TOOLS_PY.read_text(encoding="utf-8")
        self.assertIn(f'TEAM_SECOND_SHA = "{TEAM_SHA}"', tools_src)
        self.assertIn(f'IMPORTED_LEVIR_SHA = "{IMPORTED_SHA}"', tools_src)
        self.assertIn(f'SECOND_SEMANTIC_SHA = "{SEM_SHA}"', tools_src)

    def test_registry_roles_unchanged_on_disk(self) -> None:
        rec = json.loads(REGISTRY.read_text(encoding="utf-8"))
        roles = [c["role"] for c in rec["checkpoints"]]
        self.assertEqual(tuple(roles), REGISTRY_ROLES)
        by = {c["role"]: c for c in rec["checkpoints"]}
        self.assertEqual(by["team_second"]["sha256"], TEAM_SHA)
        self.assertEqual(by["imported_levir"]["sha256"], IMPORTED_SHA)
        self.assertEqual(by["second_semantic"]["sha256"], SEM_SHA)
        self.assertEqual(by["team_second"]["route_scope"], "second_like")
        self.assertEqual(by["second_semantic"]["route_scope"], "second_like_type")
        self.assertEqual(by["team_second"]["domains"], ["second_like"])
        self.assertEqual(by["second_semantic"]["domains"], ["second_like_type"])
        tools_src = TOOLS_PY.read_text(encoding="utf-8")
        self.assertIn('_registry_entry("team_second")', tools_src)
        self.assertIn('_registry_entry("second_semantic")', tools_src)
        self.assertIn('_registry_ckpt_path("imported_levir")', tools_src)
        self.assertIn('_registry_ckpt_path("team_second")', tools_src)

    def test_rung_and_trace_keys_unchanged(self) -> None:
        tools_src = TOOLS_PY.read_text(encoding="utf-8")
        pipe_src = PIPE_PY.read_text(encoding="utf-8")
        report_src = REPORT_PY.read_text(encoding="utf-8")
        blob = tools_src + pipe_src + report_src
        for key in RUNG_KEYS:
            self.assertIn(f'"{key}"', blob, key)
        for key in TRACE_KEYS:
            self.assertIn(f'"{key}"', blob, key)
        self.assertIn('"rung": "team_second_live" if team else "changeformer_v6"', tools_src)
        self.assertIn('"rung": "classical_otsu"', tools_src)
        self.assertIn('"rung": "cached_mask"', pipe_src)
        self.assertIn('cd["semantic_rung"] = "semantic_live"', pipe_src)
        self.assertIn('cd["semantic_checkpoint_sha256"]', pipe_src)
        self.assertIn('"checkpoint_sha256": sha', tools_src)
        self.assertNotIn("housing-change_live", blob)
        self.assertNotIn("land-cover_live", blob)


class Tab2CopyTests(unittest.TestCase):
    def test_required_substrings_and_display_names(self) -> None:
        text = APP_PY.read_text(encoding="utf-8")
        for needle in TAB2_KEEP:
            self.assertIn(needle, text)
        self.assertIn("Housing Change Specialist (SECOND-tuned)", text)
        self.assertIn("General Change Model (imported)", text)
        self.assertIn("Land-Cover Change Describer (6-class)", text)
        self.assertIn("SECOND-style detailed pairs", text)
        self.assertIn("SECOND = Semantic Change Detection dataset", text)
        self.assertIn('("Housing construction (high-res)", 2)', text)
        self.assertIn('("Mixed land change (general purpose)", 4)', text)


class LimitationWiringTests(unittest.TestCase):
    def test_limitation_constants_human_readable_and_wired(self) -> None:
        from pipeline import bind_inputs
        from tools import SEMANTIC_UPLOAD_LIMITATION, UPLOAD_DOMAIN_LIMITATION

        self.assertIn("Housing Change Specialist (SECOND-tuned)", UPLOAD_DOMAIN_LIMITATION)
        self.assertIn("General Change Model (imported)", UPLOAD_DOMAIN_LIMITATION)
        self.assertIn("SECOND-style detailed pairs", UPLOAD_DOMAIN_LIMITATION)
        self.assertIn("SECOND = Semantic Change Detection dataset", UPLOAD_DOMAIN_LIMITATION)
        self.assertIn("disclosed", UPLOAD_DOMAIN_LIMITATION)
        self.assertIn("Land-Cover Change Describer (6-class)", SEMANTIC_UPLOAD_LIMITATION)
        self.assertIn("disclosed", SEMANTIC_UPLOAD_LIMITATION)
        tmp = Path(tempfile.mkdtemp())
        b = tmp / "b.png"
        a = tmp / "a.png"
        Image.new("RGB", (32, 32), (1, 2, 3)).save(b)
        Image.new("RGB", (32, 32), (4, 5, 6)).save(a)
        bound = bind_inputs("bi-temporal", scene=4, uploads={"before": b, "after": a})
        self.assertTrue(bound["ok"])
        self.assertIn(UPLOAD_DOMAIN_LIMITATION, bound["ingest_note"])
        self.assertIn(SEMANTIC_UPLOAD_LIMITATION, bound["ingest_note"])


class PipelineDisplayFieldTests(unittest.TestCase):
    def test_scene4_live_false_has_rung_display_alongside_rung(self) -> None:
        from pipeline import run_query

        trace = run_query(
            "What changed between these two dates, and where?",
            "bi-temporal",
            scene=4,
            live=False,
        )
        cd = (trace.get("tool_outputs") or {}).get("change_detect") or {}
        self.assertEqual(cd.get("rung"), "team_second_live")
        self.assertEqual(cd.get("rung_display"), "housing-change (live)")
        self.assertEqual(cd.get("checkpoint_sha256"), TEAM_SHA)
        self.assertIn("rung", cd)
        self.assertIn("rung_display", cd)
        self.assertNotIn("housing-change_live", cd)

    def test_scene2_core_tool_hash_stable_after_stripping_display(self) -> None:
        from pipeline import run_query
        from report import findings_header, tool_outputs_sha256

        trace = run_query(
            "What changed between these two dates, and where?",
            "bi-temporal",
            scene=2,
            live=False,
        )
        tout = trace.get("tool_outputs") or {}
        cd = tout.get("change_detect") or {}
        self.assertEqual(cd.get("rung"), "cached_mask")
        self.assertEqual(cd.get("rung_display"), "cached mask")
        self.assertEqual(tool_outputs_sha256(_strip_display(tout)), SCENE2_TOOL_SHA_CORE)
        self.assertNotEqual(tool_outputs_sha256(tout), SCENE2_TOOL_SHA_CORE)
        header = findings_header(trace)
        self.assertIn("rung `cached_mask` (cached mask)", header)
        self.assertIn("built_up_direction=`not_determined`", header)


if __name__ == "__main__":
    unittest.main()
