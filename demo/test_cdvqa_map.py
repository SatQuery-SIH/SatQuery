"""CDVQA-MAP CPU tests. Synthetic semantic features only — no files/models."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

DEMO = Path(__file__).resolve().parent
SAT = DEMO.parent
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))

from evidence_packet import build_packet, validate_packet  # noqa: E402
from planner import plan  # noqa: E402
from tools import CDVQA_ANSWER_VOCAB, cdvqa_map  # noqa: E402


def _feat(
    from_to=None,
    from_hist=None,
    to_hist=None,
    count_a=None,
    count_b=None,
    changed=0,
    total=1000,
):
    return {
        "from_to": from_to if from_to is not None else [[0] * 6 for _ in range(6)],
        "from_hist": from_hist or [0] * 6,
        "to_hist": to_hist or [0] * 6,
        "count_a": count_a or [0] * 6,
        "count_b": count_b or [0] * 6,
        "changed_pixels": changed,
        "total_pixels": total,
    }


def _sem(feat=None, **extra):
    rec = {
        "rung": "semantic_live",
        "checkpoint_sha256": "abc",
        "features": feat if feat is not None else _feat(),
    }
    rec.update(extra)
    return rec


class CdvqaMapFamilyTests(unittest.TestCase):
    def test_binary_change_or_not(self) -> None:
        # water participates in change at t1 -> yes
        f = _feat(from_hist=[5, 0, 0, 0, 0, 0], changed=5)
        out = cdvqa_map(
            "Have the areas of water changed in the first image?", semantic=_sem(f)
        )
        self.assertEqual(out["official_type"], "change_or_not")
        self.assertEqual(out["family"], "binary")
        self.assertEqual(out["claim"], "change_detected")
        self.assertEqual(out["answer"], "yes")
        # t2 side reads to_hist; water absent there -> no
        out = cdvqa_map(
            "Have the areas of water changed in the second image?",
            semantic=_sem(f),
        )
        self.assertEqual(out["answer"], "no")
        # no participation at all -> no
        out = cdvqa_map(
            "Did the areas of water change?", semantic=_sem(_feat(changed=9))
        )
        self.assertEqual(out["answer"], "no")

    def test_transition_change_to_what(self) -> None:
        # water(1): 8 changed px went to buildings(5); 2 unchanged stay water.
        ft = [[0] * 6 for _ in range(6)]
        ft[0][4] = 8
        f = _feat(
            from_to=ft,
            from_hist=[8, 0, 0, 0, 0, 0],
            to_hist=[0, 0, 0, 0, 8, 0],
            count_a=[10, 0, 0, 0, 0, 0],
            count_b=[2, 0, 0, 0, 8, 0],
            changed=8,
        )
        out = cdvqa_map(
            "What have the areas of water mainly changed to?", semantic=_sem(f)
        )
        self.assertEqual(out["official_type"], "change_to_what")
        self.assertEqual(out["family"], "transition")
        self.assertEqual(out["claim"], "primary_transition")
        self.assertEqual(out["answer"], "buildings")

    def test_transition_nvg_casing(self) -> None:
        # water -> nvg_surface: answer must be the gold-cased token.
        ft = [[0] * 6 for _ in range(6)]
        ft[0][1] = 7
        f = _feat(
            from_to=ft,
            from_hist=[7, 0, 0, 0, 0, 0],
            to_hist=[0, 7, 0, 0, 0, 0],
            count_a=[7, 0, 0, 0, 0, 0],
            count_b=[0, 7, 0, 0, 0, 0],
            changed=7,
        )
        out = cdvqa_map(
            "What have the areas of water mainly changed to?", semantic=_sem(f)
        )
        self.assertEqual(out["answer"], "NVG_surface")
        self.assertIn(out["answer"], CDVQA_ANSWER_VOCAB)

    def test_dominance_largest_and_smallest(self) -> None:
        f = _feat(
            from_hist=[4, 30, 0, 0, 2, 0],
            to_hist=[0, 3, 0, 12, 1, 0],
            changed=36,
        )
        out = cdvqa_map(
            "What is the largest change between the two images?",
            semantic=_sem(f),
        )
        self.assertEqual(out["official_type"], "largest_change")
        self.assertEqual(out["family"], "dominance")
        self.assertEqual(out["claim"], "largest_change_class")
        # unqualified -> from+to: nvg_surface 33 wins
        self.assertEqual(out["answer"], "NVG_surface")
        # t2 side -> to_hist: trees 12 wins
        out = cdvqa_map(
            "What is the largest change in the second image?", semantic=_sem(f)
        )
        self.assertEqual(out["answer"], "trees")
        # smallest positive on unqualified hist (from+to = 4,33,0,12,3,0)
        out = cdvqa_map(
            "What is the smallest change between the two images?",
            semantic=_sem(f),
        )
        self.assertEqual(out["answer"], "buildings")
        # "what type of change is the largest" template also routes
        out = cdvqa_map(
            "What type of change is the largest in the second image?",
            semantic=_sem(f),
        )
        self.assertEqual(out["official_type"], "largest_change")
        self.assertEqual(out["answer"], "trees")

    def test_dominance_empty_defaults_nvg(self) -> None:
        out = cdvqa_map(
            "What is the largest change between the two images?",
            semantic=_sem(_feat()),
        )
        self.assertEqual(out["answer"], "NVG_surface")
        self.assertFalse(out["withheld"])

    def test_direction_increase_decrease(self) -> None:
        f = _feat(count_a=[0, 0, 0, 4, 0, 0], count_b=[0, 0, 0, 9, 0, 0])
        out = cdvqa_map(
            "Did the areas of trees increase?", semantic=_sem(f)
        )
        self.assertEqual(out["official_type"], "increase_or_not")
        self.assertEqual(out["family"], "direction")
        self.assertEqual(out["claim"], "trend")
        self.assertEqual(out["answer"], "yes")
        out = cdvqa_map(
            "Did the areas of trees decrease?", semantic=_sem(f)
        )
        self.assertEqual(out["official_type"], "decrease_or_not")
        self.assertEqual(out["answer"], "no")
        # decrease present -> yes
        f2 = _feat(count_a=[0, 0, 0, 9, 0, 0], count_b=[0, 0, 0, 4, 0, 0])
        out = cdvqa_map("Did the areas of trees decrease?", semantic=_sem(f2))
        self.assertEqual(out["answer"], "yes")

    def test_three_way_direction_withholds(self) -> None:
        f = _feat(count_a=[0, 0, 0, 0, 4, 0], count_b=[0, 0, 0, 0, 9, 0])
        out = cdvqa_map(
            "Has built-up area increased, decreased, or remained unchanged?",
            semantic=_sem(f),
        )
        self.assertEqual(out["official_type"], "built_up_direction")
        self.assertIsNone(out["answer"])
        self.assertTrue(out["withheld"])
        self.assertEqual((out["raw_value"] or {}).get("trend"), "increase")

    def test_ratio_global_and_per_class(self) -> None:
        f = _feat(changed=250, total=1000)
        out = cdvqa_map(
            "What is the change ratio of the imagery?", semantic=_sem(f)
        )
        self.assertEqual(out["official_type"], "change_ratio")
        self.assertEqual(out["family"], "ratio")
        self.assertEqual(out["answer"], "20_to_30")
        out = cdvqa_map(
            "What percentage of the area has not changed?", semantic=_sem(f)
        )
        self.assertEqual(out["answer"], "70_to_80")
        # per-class share: water count_a 120/1000 -> 10_to_20
        f2 = _feat(count_a=[120, 0, 0, 0, 0, 0], changed=250, total=1000)
        out = cdvqa_map(
            "What is the change ratio of the areas of water?", semantic=_sem(f2)
        )
        self.assertEqual(out["official_type"], "change_ratio_types")
        self.assertEqual(out["answer"], "10_to_20")


class CdvqaMapWithholdTests(unittest.TestCase):
    def test_unknown_family_withholds(self) -> None:
        out = cdvqa_map("Describe the land cover in this scene.", semantic=_sem(_feat()))
        self.assertTrue(out["withheld"])
        self.assertEqual(out["withheld_reason"], "unknown_family")
        self.assertIsNone(out["answer"])

    def test_unsupported_official_type_withholds(self) -> None:
        out = cdvqa_map(
            "q?", semantic=_sem(_feat()), official_type="some_other_type"
        )
        self.assertTrue(out["withheld"])
        self.assertIn("unsupported_family", out["withheld_reason"])

    def test_missing_evidence_withholds(self) -> None:
        out = cdvqa_map("What is the largest change?")
        self.assertTrue(out["withheld"])
        self.assertEqual(out["withheld_reason"], "no_semantic_evidence")
        out = cdvqa_map("What is the largest change?", semantic={"features": {}})
        self.assertTrue(out["withheld"])

    def test_missing_class_withholds(self) -> None:
        out = cdvqa_map(
            "What have the regions mainly changed to?",
            semantic=_sem(_feat(changed=5)),
        )
        # change_to_what phrasing but no named class -> withhold
        self.assertTrue(out["withheld"])
        self.assertEqual(out["withheld_reason"], "no_named_class")

    def test_vocab_mismatch_withholds(self) -> None:
        ft = [[0] * 6 for _ in range(6)]
        ft[0][4] = 8
        f = _feat(
            from_to=ft,
            from_hist=[8, 0, 0, 0, 0, 0],
            to_hist=[0, 0, 0, 0, 8, 0],
            count_a=[8, 0, 0, 0, 0, 0],
            count_b=[0, 0, 0, 0, 8, 0],
            changed=8,
        )
        out = cdvqa_map(
            "What have the areas of water mainly changed to?",
            semantic=_sem(f),
            vocab={"yes", "no"},
        )
        self.assertTrue(out["withheld"])
        self.assertIn("vocab_mismatch", out["withheld_reason"])
        self.assertIsNone(out["answer"])

    def test_all_answers_in_vocab(self) -> None:
        ft = [[0] * 6 for _ in range(6)]
        ft[0][4] = 8
        f = _feat(
            from_to=ft,
            from_hist=[8, 0, 0, 0, 0, 0],
            to_hist=[0, 0, 0, 0, 8, 0],
            count_a=[8, 0, 0, 0, 0, 0],
            count_b=[0, 0, 0, 0, 8, 0],
            changed=100,
            total=1000,
        )
        qs = [
            "Have the areas of water changed in the first image?",
            "What have the areas of water mainly changed to?",
            "What is the largest change between the two images?",
            "What is the smallest change between the two images?",
            "Did the areas of buildings increase?",
            "What is the change ratio of the imagery?",
            "What is the change ratio of the areas of water?",
        ]
        for q in qs:
            out = cdvqa_map(q, semantic=_sem(f))
            if out["answer"] is not None:
                self.assertIn(out["answer"], CDVQA_ANSWER_VOCAB, q)

    def test_packet_input_path(self) -> None:
        f = _feat(from_hist=[3, 0, 0, 0, 0, 0], changed=3)
        packet = {"tool_outputs": {"semantic": _sem(f)}}
        out = cdvqa_map(
            "Did the areas of water change?", packet=packet
        )
        self.assertEqual(out["answer"], "yes")


class CdvqaMapPacketTests(unittest.TestCase):
    def test_packet_claims_and_validation(self) -> None:
        ft = [[0] * 6 for _ in range(6)]
        ft[0][4] = 8
        f = _feat(
            from_to=ft,
            from_hist=[8, 0, 0, 0, 0, 0],
            to_hist=[0, 0, 0, 0, 9, 0],
            count_a=[8, 0, 0, 0, 0, 0],
            count_b=[0, 0, 0, 0, 9, 0],
            changed=9,
        )
        sem = _sem(
            f,
            dominant_transition={
                "token": "water->buildings",
                "from_token": "water",
                "to_token": "buildings",
                "count": 8,
            },
            built_up_direction="increase",
        )
        cm = cdvqa_map(
            "What have the areas of water mainly changed to?", semantic=sem
        )
        tout = {
            "change_detect": {"changed_pixels": 8, "rung": "synthetic"},
            "semantic": sem,
            "cdvqa_map": cm,
        }
        packet = build_packet(tout, "change_description", mask_path="m.png")
        preds = {c.get("predicate"): c for c in packet.get("claims") or []}
        self.assertIn("cdvqa_answer", preds)
        self.assertEqual(preds["cdvqa_answer"]["value"], "buildings")
        self.assertEqual(
            preds["cdvqa_answer"]["provenance"]["tool"], "cdvqa_map"
        )
        self.assertIn("cdvqa_family", preds)
        self.assertIn("primary_transition", preds)
        self.assertEqual(preds["primary_transition"]["value"], "water->buildings")
        self.assertIn("largest_change_class", preds)
        self.assertEqual(preds["largest_change_class"]["value"], "buildings")
        self.assertIn("trend", preds)
        self.assertEqual(packet.get("canonical_answer"), "buildings")
        self.assertEqual(validate_packet(packet), [])
        self.assertTrue(
            any("mIoU 0.417" in lim for lim in packet.get("limitations") or [])
        )

    def test_withheld_packet_still_validates(self) -> None:
        cm = cdvqa_map("Describe the scene.", semantic=_sem(_feat()))
        tout = {"change_detect": {"changed_pixels": 0}, "cdvqa_map": cm}
        packet = build_packet(tout, "change_description", mask_path="m.png")
        preds = {c.get("predicate"): c for c in packet.get("claims") or []}
        self.assertIsNone(preds["cdvqa_answer"]["value"])
        self.assertEqual(
            preds["cdvqa_answer"]["confidence"]["level"], "withheld"
        )
        self.assertEqual(validate_packet(packet), [])
        self.assertTrue(
            any("cdvqa_map withheld" in lim for lim in packet.get("limitations") or [])
        )

    def test_planner_emits_tool(self) -> None:
        for q in (
            "What is the largest change between the two dates?",
            "Did the areas of trees increase?",
            "What did the water regions mainly change to?",
            "What percentage of the area has not changed?",
            "Have the areas of water changed?",
        ):
            p = plan(q, "bi-temporal")
            self.assertIn("cdvqa_map", p["tools"], q)
        # generic change questions do not gain the tool
        p = plan("What changed between these two dates, and where?", "bi-temporal")
        self.assertNotIn("cdvqa_map", p["tools"])


if __name__ == "__main__":
    unittest.main()
