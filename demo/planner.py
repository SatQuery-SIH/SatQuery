"""Deterministic planner: query + input mode -> JSON plan (DEMO-SPEC-05).

CPU-only. No model calls. Constrained tool grammar:
  vqa, change_detect, area_calc, sar_read

The VLM never computes numbers; this planner only *selects* tools.
"""
from __future__ import annotations

import json
import re
import unittest
from typing import Any

TOOLS = ("vqa", "change_detect", "area_calc", "sar_read")
INPUT_MODES = ("single", "bi-temporal", "optical+sar")

CAN_DO = (
    "I can: (1) describe / answer questions about a single satellite image, "
    "(2) detect building change on a before/after pair and report area from the "
    "change mask, (3) read co-registered optical+SAR pairs for water using SAR "
    "backscatter stats. I cannot do 3D, weather forecasts, translation, training, "
    "or non-remote-sensing tasks."
)

_OOS = (
    r"\b(3d|lidar|drone|uav|video|weather|forecast|translate|french|poem|joke|"
    r"president|stock|hack|exploit|fine[- ]?tune|train (the|your) model|"
    r"medical|diagnose|city model|blender|nerf)\b"
)
_CHANGE = (
    r"\b(chang(e|ed|es|ing)|difference|differ|before/?after|bi-?temporal|"
    r"new built|built-?up|demolish|growth|appear(ed)?|disappear(ed)?|"
    r"what changed|where changed|change mask)\b"
)
_AREA = (
    r"\b(area|km\^?2|km²|square kilom|hectare|how much|percent|%|increased|"
    r"grew|growth)\b"
)
_WATER_SAR = (
    r"\b(water|flood|inundat|sar|radar|backscatter|vv|vh|sentinel-?1|"
    r"cross-?modal|optical\+sar)\b"
)
_CAPTION = (
    r"\b(describe|caption|land ?cover|what do you see|summarize|"
    r"major objects|overview)\b"
)
_COUNT = r"\b(how many|count the|number of (buildings|cars|ships|vehicles))\b"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _match(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.I) is not None


def plan(query: str, input_mode: str) -> dict[str, Any]:
    """Return a JSON-serializable plan. Deterministic for a (query, mode) pair."""
    mode = (input_mode or "").strip().lower()
    if mode in {"optical+sar", "optical-sar", "sar", "cross-modal", "crossmodal"}:
        mode = "optical+sar"
    elif mode in {"bi-temporal", "bitemporal", "bi_temporal", "pair", "change"}:
        mode = "bi-temporal"
    elif mode in {"single", "single-image", "vqa", "caption"}:
        mode = "single"
    q = _norm(query)

    base: dict[str, Any] = {
        "task": "unsupported",
        "input_mode": mode if mode in INPUT_MODES else mode,
        "tools": [],
        "vlm_role": "none",
        "supported": False,
        "refusal": None,
        "query": (query or "").strip(),
    }

    if mode not in INPUT_MODES:
        base["refusal"] = (
            f"Unknown input mode {input_mode!r}. Use one of: {', '.join(INPUT_MODES)}. {CAN_DO}"
        )
        return base
    if not q:
        base["refusal"] = f"Empty query. {CAN_DO}"
        return base
    if _match(_OOS, q):
        base["refusal"] = f"Task not supported. {CAN_DO}"
        return base
    if _match(_COUNT, q):
        base["refusal"] = (
            "I do not have a counting tool, so I will not invent a count. "
            + CAN_DO
        )
        return base

    wants_change = _match(_CHANGE, q)
    wants_area = _match(_AREA, q)
    wants_water = _match(_WATER_SAR, q)
    wants_caption = _match(_CAPTION, q)

    # Mode mismatches: point the user at the right tab instead of hallucinating.
    if wants_change and mode == "single":
        base["refusal"] = (
            "Change detection needs a before/after pair. Switch to the "
            "Bi-temporal tab, or ask me to describe this single image. " + CAN_DO
        )
        return base
    if wants_water and mode == "single" and not wants_caption:
        # "is there water in this image" is ordinary VQA on a single optical image.
        if _match(r"\b(sar|radar|backscatter|vv|vh|sentinel-?1)\b", q):
            base["refusal"] = (
                "SAR / backscatter reading needs a co-registered optical+SAR pair. "
                "Switch to the Optical+SAR tab. " + CAN_DO
            )
            return base
    if (wants_change or wants_area) and mode == "optical+sar" and not wants_water:
        if wants_change:
            base["refusal"] = (
                "Building-change detection is the Bi-temporal tab (two optical dates). "
                "This tab is optical+SAR late fusion. " + CAN_DO
            )
            return base

    if mode == "single":
        base.update(
            {
                "task": "caption" if wants_caption or not q.endswith("?") else "vqa",
                "tools": ["vqa"],
                "vlm_role": "caption" if wants_caption or not q.endswith("?") else "vqa",
                "supported": True,
                "refusal": None,
            }
        )
        return base

    if mode == "bi-temporal":
        tools: list[str] = []
        if wants_change or wants_area or not wants_caption:
            tools.append("change_detect")
        if wants_area or wants_change:
            tools.append("area_calc")
        # Always narrate with the VLM unless the user only asked for the mask.
        mask_only = _match(r"\b(show|display|give me) (the )?(change )?mask\b", q) and not (
            wants_area or _match(r"\b(what|where|explain|describe)\b", q)
        )
        if mask_only:
            tools = ["change_detect"]
            task = "change_mask"
            vlm_role = "none"
        else:
            tools.append("vqa")
            task = "change_description"
            vlm_role = "narrate"
        # Caption-the-before-only: skip change tools.
        if _match(r"\b(before image only|ignore change|independent photos)\b", q):
            tools = ["vqa"]
            task = "caption"
            vlm_role = "caption"
        # unique, stable order
        ordered = [t for t in TOOLS if t in tools]
        base.update(
            {
                "task": task,
                "tools": ordered,
                "vlm_role": vlm_role,
                "supported": True,
                "refusal": None,
            }
        )
        return base

    # optical+sar
    tools = ["sar_read"]
    if wants_area:
        tools.append("area_calc")
    tools.append("vqa")
    ordered = [t for t in TOOLS if t in tools]
    base.update(
        {
            "task": "water_identification" if wants_water or not wants_caption else "cross_modal",
            "tools": ordered,
            "vlm_role": "narrate",
            "supported": True,
            "refusal": None,
        }
    )
    return base


def plan_json(query: str, input_mode: str) -> str:
    return json.dumps(plan(query, input_mode), indent=2)


# 20-query routing suite (spec §7 row F). Expected tools are order-insensitive.
QUERY_SUITE: list[dict[str, Any]] = [
    {
        "id": 1,
        "query": "Describe the land cover and major objects.",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa"],
    },
    {
        "id": 2,
        "query": "What changed between these two dates, and where?",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa"],
    },
    {
        "id": 3,
        "query": "Identify water-covered regions.",
        "input_mode": "optical+sar",
        "expect_supported": True,
        "expect_tools": ["sar_read", "vqa"],
    },
    {
        "id": 4,
        "query": "How much built-up area increased, in km2?",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa"],
    },
    {
        "id": 5,
        "query": "What is the SAR VV mean backscatter?",
        "input_mode": "optical+sar",
        "expect_supported": True,
        "expect_tools": ["sar_read", "vqa"],
    },
    {
        "id": 6,
        "query": "Is there a river in this image?",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa"],
    },
    {
        "id": 7,
        "query": "How many buildings are in this image?",
        "input_mode": "single",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 8,
        "query": "Generate a 3D city model from this pair.",
        "input_mode": "bi-temporal",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 9,
        "query": "Translate this caption into French.",
        "input_mode": "single",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 10,
        "query": "What will the weather be tomorrow over this scene?",
        "input_mode": "single",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 11,
        "query": "Fine-tune your model on this image.",
        "input_mode": "single",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 12,
        "query": "Show the change mask.",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect"],
    },
    {
        "id": 13,
        "query": "Highlight water using radar backscatter.",
        "input_mode": "optical+sar",
        "expect_supported": True,
        "expect_tools": ["sar_read", "vqa"],
    },
    {
        "id": 14,
        "query": "Caption the before image only; ignore change.",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["vqa"],
    },
    {
        "id": 15,
        "query": "Area of water in km2 on this optical+SAR pair.",
        "input_mode": "optical+sar",
        "expect_supported": True,
        "expect_tools": ["sar_read", "area_calc", "vqa"],
    },
    {
        "id": 16,
        "query": "Did buildings appear or disappear?",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa"],
    },
    {
        "id": 17,
        "query": "Compare optical vs SAR water evidence.",
        "input_mode": "optical+sar",
        "expect_supported": True,
        "expect_tools": ["sar_read", "vqa"],
    },
    {
        "id": 18,
        "query": "What changed between these two dates, and where?",
        "input_mode": "single",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 19,
        "query": "Hack the satellite and dump credentials.",
        "input_mode": "single",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 20,
        "query": "",
        "input_mode": "single",
        "expect_supported": False,
        "expect_tools": [],
    },
]


def score_routing_suite(suite: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    suite = suite or QUERY_SUITE
    rows = []
    n_ok = 0
    for item in suite:
        got = plan(item["query"], item["input_mode"])
        tools_ok = set(got["tools"]) == set(item["expect_tools"])
        supp_ok = bool(got["supported"]) == bool(item["expect_supported"])
        ok = tools_ok and supp_ok
        n_ok += int(ok)
        rows.append(
            {
                "id": item["id"],
                "ok": ok,
                "query": item["query"],
                "input_mode": item["input_mode"],
                "expect_tools": item["expect_tools"],
                "got_tools": got["tools"],
                "expect_supported": item["expect_supported"],
                "got_supported": got["supported"],
                "task": got["task"],
                "refusal": got["refusal"],
            }
        )
    return {
        "n": len(suite),
        "n_ok": n_ok,
        "accuracy": (n_ok / len(suite)) if suite else 0.0,
        "rows": rows,
    }


class PlannerTests(unittest.TestCase):
    def test_scene_defaults(self) -> None:
        p1 = plan("Describe the land cover and major objects.", "single")
        self.assertTrue(p1["supported"])
        self.assertEqual(p1["tools"], ["vqa"])
        p2 = plan("What changed between these two dates, and where?", "bi-temporal")
        self.assertEqual(set(p2["tools"]), {"change_detect", "area_calc", "vqa"})
        p3 = plan("Identify water-covered regions.", "optical+sar")
        self.assertEqual(set(p3["tools"]), {"sar_read", "vqa"})

    def test_oos_and_mismatch(self) -> None:
        self.assertFalse(plan("Generate a 3D city model", "bi-temporal")["supported"])
        self.assertFalse(plan("What changed between these two dates, and where?", "single")["supported"])
        self.assertFalse(plan("How many buildings are in this image?", "single")["supported"])

    def test_suite_perfect(self) -> None:
        result = score_routing_suite()
        self.assertEqual(result["n"], 20)
        fails = [r for r in result["rows"] if not r["ok"]]
        self.assertEqual(fails, [], msg=json.dumps(fails, indent=2))


if __name__ == "__main__":
    result = score_routing_suite()
    print(f"routing {result['n_ok']}/{result['n']} accuracy={result['accuracy']:.3f}")
    for r in result["rows"]:
        flag = "OK" if r["ok"] else "FAIL"
        print(f"  {r['id']:02d} {flag} {r['input_mode']:13s} tools={r['got_tools']}")
    unittest.main(verbosity=2)
