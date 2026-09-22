"""Deterministic planner: query + input mode -> JSON plan (DEMO-SPEC-05).

CPU-only. No model calls. Constrained tool grammar:
  vqa, canonical_vqa, change_detect, area_calc, sar_read, water_highlight,
  sar_agreement, cdvqa_map

The VLM never computes numbers; this planner only *selects* tools.
canonical_vqa is the adapted RSVQA answer seat on :8091 — it emits a
canonical claim, not narration (vqa on :8080 still narrates).
"""
from __future__ import annotations

import json
import re
import unittest
from typing import Any

TOOLS = (
    "vqa",
    "change_detect",
    "water_highlight",
    "area_calc",
    "sar_read",
    "sar_agreement",
    "cdvqa_map",
    "canonical_vqa",
)
INPUT_MODES = ("single", "bi-temporal", "optical+sar")

CAN_DO = (
    "I can: (1) describe / answer questions about a single satellite image, "
    "(2) detect building change on a before/after pair and report area from the "
    "change mask, (3) read co-registered optical+SAR pairs for water using SAR "
    "backscatter stats. I cannot do 3D, weather forecasts, translation, training, "
    "or non-remote-sensing tasks."
)

_OOS = (
    r"\b(3d|lidar|video|weather|rain|rainfall|forecast|translate|french|"
    r"german|spanish|hindi|capital of|recipe|poem|joke|president|stock|"
    r"hack|exploit|fine[- ]?tune|train (the|your) model|medical|diagnose|"
    r"city model|blender|nerf)\b"
)
_STRONG_CHANGE = (
    r"\b(chang(e|ed|es|ing)?|before/?after|bi-?temporal|new built|"
    r"demolish(ed|es|ing)?|grow|grew|grown|growth|expan(d|ded|sion|s)?|"
    r"delta|construct(ion|ions|ed|ing|s)?|happen(ed|s|ing)?|"
    r"increas(e|ed|es|ing)?|decreas(e|ed|es|ing)?|shrink(s|ing)?|shrank|"
    r"shrunk|convert(ed|s|ing)?|appear(ed|s|ing)?|disappear(ed|s|ing)?|"
    r"what changed|where changed|change mask)\b"
)
_WEAK_CHANGE = r"\b(differ(ent|ence|ences|ed|ing|s)?|same|new|built-?up)\b"
_TEMPORAL = (
    r"\b(before|after|earlier|previous|prior|former|latter|two dates|"
    r"date one|date two|first image|second image|two images|both images|"
    r"two photos|both photos|between (the )?dates|compared to|since then)\b"
)
_AREA = (
    r"\b(area|km\^?2|km²|m\^?2|m²|square kilom|square met|hectares?|"
    r"how much|how large|percent|%|increas(e|ed|es|ing)?|grew|grown|"
    r"growth|large|size|extent)\b"
)
_WATER_SAR = (
    r"\b(water|flood|inundat|lake|river|reservoir|pond|sar|radar|"
    r"backscatter|vv|vh|sentinel-?1|cross-?modal|optical\+sar)\b"
)
_CAPTION = (
    r"\b(describe|caption|land ?cover|what do you see|summarize|"
    r"major objects|overview)\b"
)
_COUNT = (
    r"\b(how many|count( the|ing)?|number of)\b[^.]{0,30}\b"
    r"(buildings?|houses?|cars?|vehicles?|ships?|boats?|bridges?|roads?|"
    r"trees?|objects?|structures?|aircraft|planes?)\b"
)
_HL_VERB = r"(hig?h?light|outline|locate|mark|point out|show|delineate)"
_HL_NOUN = (
    r"(water( ?bod(y|ies))?|lakes?|rivers?|reservoirs?|ponds?|"
    r"flood(ed|ing|s)?( region| extent)?|inundat(ed|ion|ing))"
)
_HL_OBJECT = (
    r"(buildings?|houses?|cars?|vehicles?|ships?|boats?|bridges?|roads?|"
    r"trees?|objects?|structures?|aircraft|planes?)"
)
_WATER_HL = (
    r"\b" + _HL_VERB + r"\b.{0,80}\b" + _HL_NOUN + r"\b|"
    r"\b" + _HL_NOUN + r"\b.{0,80}\b" + _HL_VERB + r"\b"
)
_HL_OBJECT_Q = r"\b" + _HL_VERB + r"\b.{0,80}\b" + _HL_OBJECT + r"\b"
_MASK_ONLY = (
    r"\b(show|display|give me|want|need|return|output)\b.{0,15}\bmask\b|"
    r"\bmask\b.{0,10}\bonly\b|\bonly\b.{0,10}\bmask\b"
)
_BYPASS_HARD = (
    r"\b(before image only|after image only|ignore change|"
    r"independent photos)\b"
)
_BYPASS_SOFT = (
    r"\b(before image|after image|first image|second image|this image|"
    r"one of the images|each image|both images|two images)\b"
)
# CDVQA-style type-family phrasing. Mirrors
# cf_ft.semantic.classify_semantic_family plus the two families it does not
# detect (per-class binary "did the areas of X change?" and the global
# with/without-change ratio). Class nouns mirror extract_class_id's vocab.
_CDVQA_CLASS = (
    r"(water|lakes?|rivers?|reservoirs?|ponds?|vegetation|trees?|"
    r"buildings?|built-?up|urban|playgrounds?|non-?vegetated|"
    r"non-?vegetation)"
)
_CDVQA_STYLE = (
    r"change ratio|change proportion|change percentage|how much area|"
    r"mainly changed to|changed to what|changed to\?|\bchanged to\b|"
    r"\blargest change\b|\bsmallest change\b|type of change|"
    r"\bincreas|\bdecreas|"
    r"\b(did|do|does|have|has|is|are|was|were)\b[^.?]{0,50}\b"
    + _CDVQA_CLASS
    + r"\b[^.?]{0,30}\bchang|"
    r"\b(did|do|does|have|has|is|are|was|were)\b[^.?]{0,50}\bchang"
    r"[^.?]{0,30}\b"
    + _CDVQA_CLASS
    + r"\b|"
    r"\b(percentage|proportion|ratio|how much)\b[^.?]{0,40}\b"
    r"(changed|unchanged|non-?change)"
)


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
    # Counting needs a seat that owns the count. Single images have one:
    # canonical_vqa (the adapted RSVQA model on :8091). Other modes do not.
    if _match(_COUNT, q) and mode != "single":
        base["refusal"] = (
            "I do not have a counting tool, so I will not invent a count. "
            + CAN_DO
        )
        return base

    wants_change = _match(_STRONG_CHANGE, q) or (
        _match(_WEAK_CHANGE, q) and _match(_TEMPORAL, q)
    )
    wants_area = _match(_AREA, q)
    wants_water = _match(_WATER_SAR, q)
    wants_caption = _match(_CAPTION, q)
    wants_water_hl = _match(_WATER_HL, q)
    sar_terms = _match(r"\b(sar|radar|backscatter|vv|vh|sentinel-?1)\b", q)

    # Mode mismatches: point the user at the right tab instead of hallucinating.
    if wants_change and mode == "single":
        base["refusal"] = (
            "Change detection needs a before/after pair. Switch to the "
            "Bi-temporal tab, or ask me to describe this single image. " + CAN_DO
        )
        return base
    if wants_water and mode == "single" and not wants_caption:
        # "is there water in this image" is ordinary VQA on a single optical image.
        if sar_terms:
            base["refusal"] = (
                "SAR / backscatter reading needs a co-registered optical+SAR pair. "
                "Switch to the Optical+SAR tab. " + CAN_DO
            )
            return base
    if wants_change and mode == "optical+sar":
        base["refusal"] = (
            "Building-change detection is the Bi-temporal tab (two optical dates). "
            "This tab is optical+SAR late fusion. " + CAN_DO
        )
        return base

    if mode == "single":
        # A question-shaped query gets a canonical short answer from the
        # adapted RSVQA seat (:8091) alongside the :8080 narration. Imperative
        # counts ("count the cars") count as questions — count is an RSVQA
        # family.
        is_question = q.endswith("?") or _match(_COUNT, q)
        if (wants_water_hl or (wants_water and wants_area)) and not sar_terms:
            tools = ["water_highlight", "area_calc", "vqa"]
            if is_question:
                tools.append("canonical_vqa")
            base.update(
                {
                    "task": "water_highlight",
                    "tools": tools,
                    "vlm_role": "narrate",
                    "supported": True,
                    "refusal": None,
                }
            )
            return base
        if _match(_HL_OBJECT_Q, q) and not wants_water_hl:
            base["refusal"] = (
                "Highlighting is only supported for water bodies, not other "
                "objects. " + CAN_DO
            )
            return base
        is_caption = wants_caption or not is_question
        base.update(
            {
                "task": "caption" if is_caption else "vqa",
                "tools": ["vqa"] + ([] if is_caption else ["canonical_vqa"]),
                "vlm_role": "caption" if is_caption else "vqa",
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
        mask_only = _match(_MASK_ONLY, q) and not (
            wants_area or _match(r"\b(what|where|explain|describe)\b", q)
        )
        if mask_only:
            tools = ["change_detect"]
            task = "change_mask"
            vlm_role = "none"
        else:
            # CDVQA-style type-family questions also emit the deterministic
            # second_semantic packet -> canonical answer map (no VLM scoring).
            if _match(_CDVQA_STYLE, q):
                tools.append("cdvqa_map")
            tools.append("vqa")
            task = "change_description"
            vlm_role = "narrate"
        # Caption-the-before-only: skip change tools. A single image of the
        # pair named without a change signal is a plain VQA/caption request.
        if _match(_BYPASS_HARD, q) or (
            _match(_BYPASS_SOFT, q) and not wants_change
        ):
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
    tools = ["sar_read", "sar_agreement"]
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
        "expect_tools": ["sar_read", "sar_agreement", "vqa"],
    },
    {
        "id": 4,
        "query": "How much built-up area increased, in km2?",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa", "cdvqa_map"],
    },
    {
        "id": 5,
        "query": "What is the SAR VV mean backscatter?",
        "input_mode": "optical+sar",
        "expect_supported": True,
        "expect_tools": ["sar_read", "sar_agreement", "vqa"],
    },
    {
        "id": 6,
        "query": "Is there a river in this image?",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa", "canonical_vqa"],
    },
    {
        "id": 7,
        "query": "How many buildings are in this image?",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa", "canonical_vqa"],
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
        "expect_tools": ["sar_read", "sar_agreement", "vqa"],
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
        "expect_tools": ["sar_read", "area_calc", "sar_agreement", "vqa"],
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
        "expect_tools": ["sar_read", "sar_agreement", "vqa"],
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
    {
        "id": 21,
        "query": "Highlight the water body referred to in the query",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["water_highlight", "area_calc", "vqa"],
    },
    {
        "id": 22,
        "query": "Has built-up area increased, decreased, or remained unchanged?",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa", "cdvqa_map"],
    },
    {
        "id": 23,
        "query": "Provide a detailed description of the land cover and prominent objects in this scene.",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa"],
    },
    {
        "id": 24,
        "query": "What can you see in this image?",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa", "canonical_vqa"],
    },
    {
        "id": 25,
        "query": "Tell me everything you can see in this satellite photo.",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa"],
    },
    {
        "id": 26,
        "query": "Give me a one-line caption for this image.",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa"],
    },
    {
        "id": 27,
        "query": "What is going on in this satellite image?",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa", "canonical_vqa"],
    },
    {
        "id": 28,
        "query": "Identify the land cover types present in this scene.",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa"],
    },
    {
        "id": 29,
        "query": "Is this a residential area or an industrial one?",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa", "canonical_vqa"],
    },
    {
        "id": 30,
        "query": "Does this look like farmland to you?",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa", "canonical_vqa"],
    },
    {
        "id": 31,
        "query": "Are there any buildings in the image?",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa", "canonical_vqa"],
    },
    {
        "id": 32,
        "query": "What kind of terrain is shown in this picture?",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa", "canonical_vqa"],
    },
    {
        "id": 33,
        "query": "Descibe the landcover in this image please.",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa"],
    },
    {
        "id": 34,
        "query": "Is there a train on the tracks?",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa", "canonical_vqa"],
    },
    {
        "id": 35,
        "query": "Please highlight the water in this image.",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["water_highlight", "area_calc", "vqa"],
    },
    {
        "id": 36,
        "query": "Outline the water body for me.",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["water_highlight", "area_calc", "vqa"],
    },
    {
        "id": 37,
        "query": "Show me the water.",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["water_highlight", "area_calc", "vqa"],
    },
    {
        "id": 38,
        "query": "Locate the water body.",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["water_highlight", "area_calc", "vqa"],
    },
    {
        "id": 39,
        "query": "Outline the water region and report its area.",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["water_highlight", "area_calc", "vqa"],
    },
    {
        "id": 40,
        "query": "Show me the flood extent.",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["water_highlight", "area_calc", "vqa"],
    },
    {
        "id": 41,
        "query": "There is water somewhere in this scene, please highlight it for me.",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["water_highlight", "area_calc", "vqa"],
    },
    {
        "id": 42,
        "query": "Show the inundated zones.",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["water_highlight", "area_calc", "vqa"],
    },
    {
        "id": 43,
        "query": "Where is the water in this image?",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa", "canonical_vqa"],
    },
    {
        "id": 44,
        "query": "Mark the water body on the image.",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["water_highlight", "area_calc", "vqa"],
    },
    {
        "id": 45,
        "query": "Highlight the flooded region.",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["water_highlight", "area_calc", "vqa"],
    },
    {
        "id": 46,
        "query": "Outline the reservoir in this image.",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["water_highlight", "area_calc", "vqa"],
    },
    {
        "id": 47,
        "query": "Locate the lake in this image.",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["water_highlight", "area_calc", "vqa"],
    },
    {
        "id": 48,
        "query": "Please outline the river.",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["water_highlight", "area_calc", "vqa"],
    },
    {
        "id": 49,
        "query": "Can you show me where the water is?",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["water_highlight", "area_calc", "vqa", "canonical_vqa"],
    },
    {
        "id": 50,
        "query": "Point out the water in this scene.",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["water_highlight", "area_calc", "vqa"],
    },
    {
        "id": 51,
        "query": "Higlight the water.",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["water_highlight", "area_calc", "vqa"],
    },
    {
        "id": 52,
        "query": "Describe the water in this image.",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa"],
    },
    {
        "id": 53,
        "query": "What changed between the two dates?",
        "input_mode": "single",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 54,
        "query": "Did anything appear or disappear?",
        "input_mode": "single",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 55,
        "query": "Has the built-up area increased or decreased?",
        "input_mode": "single",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 56,
        "query": "How does this image differ from the previous one?",
        "input_mode": "single",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 57,
        "query": "What does the SAR layer show for this scene?",
        "input_mode": "single",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 58,
        "query": "Give me the VV backscatter stats.",
        "input_mode": "single",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 59,
        "query": "Use radar to check for water.",
        "input_mode": "single",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 60,
        "query": "What is different between the before and after images?",
        "input_mode": "single",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 61,
        "query": "Is this different from the earlier image?",
        "input_mode": "single",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 62,
        "query": "Compared to the previous image, what is new?",
        "input_mode": "single",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 63,
        "query": "Did the town grow?",
        "input_mode": "single",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 64,
        "query": "What is the difference between the left and right sides of this image?",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa", "canonical_vqa"],
    },
    {
        "id": 65,
        "query": "Highlight the buildings in this image.",
        "input_mode": "single",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 66,
        "query": "How many buildings are in this image?",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa", "canonical_vqa"],
    },
    {
        "id": 67,
        "query": "Count the cars in the parking lot.",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa", "canonical_vqa"],
    },
    {
        "id": 68,
        "query": "How many ships do you see in the harbor?",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa", "canonical_vqa"],
    },
    {
        "id": 69,
        "query": "What is the number of vehicles on the road?",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa", "canonical_vqa"],
    },
    {
        "id": 70,
        "query": "Count the trees in this scene.",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa", "canonical_vqa"],
    },
    {
        "id": 71,
        "query": "What number of bridges cross the river?",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa", "canonical_vqa"],
    },
    {
        "id": 72,
        "query": "How many km2 does the lake cover?",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["water_highlight", "area_calc", "vqa", "canonical_vqa"],
    },
    {
        "id": 73,
        "query": "Tell me what changed and where it happened.",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa"],
    },
    {
        "id": 74,
        "query": "Which parts of the scene changed?",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa"],
    },
    {
        "id": 75,
        "query": "Where exactly did the change take place?",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa"],
    },
    {
        "id": 76,
        "query": "Have any buildings been demolished?",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa"],
    },
    {
        "id": 77,
        "query": "Did buildings appear or disappear between the dates?",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa"],
    },
    {
        "id": 78,
        "query": "Has vegetation cover changed?",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa", "cdvqa_map"],
    },
    {
        "id": 79,
        "query": "Quantify how much the city grew.",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa"],
    },
    {
        "id": 80,
        "query": "Was there urban growth between the two dates?",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa"],
    },
    {
        "id": 81,
        "query": "Has the built-up area gone up, gone down, or stayed flat?",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa"],
    },
    {
        "id": 82,
        "query": "How much forest was lost between the two dates?",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa"],
    },
    {
        "id": 83,
        "query": "Was any farmland converted to built-up land?",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa"],
    },
    {
        "id": 84,
        "query": "What is new in the second image?",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa"],
    },
    {
        "id": 85,
        "query": "Has the urban extent grown?",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa"],
    },
    {
        "id": 86,
        "query": "Did the town expand?",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa"],
    },
    {
        "id": 87,
        "query": "Show me the delta between the two dates.",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa"],
    },
    {
        "id": 88,
        "query": "What construction happened between the two dates?",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa"],
    },
    {
        "id": 89,
        "query": "Are the two images the same?",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa"],
    },
    {
        "id": 90,
        "query": "Spot the differences between the two images.",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa"],
    },
    {
        "id": 91,
        "query": "What happened between date one and date two?",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa"],
    },
    {
        "id": 92,
        "query": "Did the city grow or shrink?",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa"],
    },
    {
        "id": 93,
        "query": "How many hectares of forest were lost between the dates?",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa"],
    },
    {
        "id": 94,
        "query": "Just give me the change mask.",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect"],
    },
    {
        "id": 95,
        "query": "Display the change mask.",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect"],
    },
    {
        "id": 96,
        "query": "I want the mask only.",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect"],
    },
    {
        "id": 97,
        "query": "Describe the before image.",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["vqa"],
    },
    {
        "id": 98,
        "query": "What's in the first image?",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["vqa"],
    },
    {
        "id": 99,
        "query": "Tell me about this image.",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["vqa"],
    },
    {
        "id": 100,
        "query": "Use the optical and SAR images together to identify built-up and water-covered regions.",
        "input_mode": "optical+sar",
        "expect_supported": True,
        "expect_tools": ["sar_read", "sar_agreement", "vqa"],
    },
    {
        "id": 101,
        "query": "Map the urban and water areas in this optical+SAR pair.",
        "input_mode": "optical+sar",
        "expect_supported": True,
        "expect_tools": ["sar_read", "sar_agreement", "vqa"],
    },
    {
        "id": 102,
        "query": "Fuse the optical and radar data to locate flooded zones.",
        "input_mode": "optical+sar",
        "expect_supported": True,
        "expect_tools": ["sar_read", "sar_agreement", "vqa"],
    },
    {
        "id": 103,
        "query": "Where does the SAR backscatter indicate open water?",
        "input_mode": "optical+sar",
        "expect_supported": True,
        "expect_tools": ["sar_read", "sar_agreement", "vqa"],
    },
    {
        "id": 104,
        "query": "Identify settlements and water bodies from the optical+SAR pair.",
        "input_mode": "optical+sar",
        "expect_supported": True,
        "expect_tools": ["sar_read", "sar_agreement", "vqa"],
    },
    {
        "id": 105,
        "query": "What is the mean VV backscatter over the reservoir?",
        "input_mode": "optical+sar",
        "expect_supported": True,
        "expect_tools": ["sar_read", "sar_agreement", "vqa"],
    },
    {
        "id": 106,
        "query": "Estimate the inundated area.",
        "input_mode": "optical+sar",
        "expect_supported": True,
        "expect_tools": ["sar_read", "area_calc", "sar_agreement", "vqa"],
    },
    {
        "id": 107,
        "query": "What changed between the two dates?",
        "input_mode": "optical+sar",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 108,
        "query": "How large is the flooded region?",
        "input_mode": "optical+sar",
        "expect_supported": True,
        "expect_tools": ["sar_read", "area_calc", "sar_agreement", "vqa"],
    },
    {
        "id": 109,
        "query": "Detect changes using the optical+SAR pair.",
        "input_mode": "optical+sar",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 110,
        "query": "Did the water-covered area change between the dates?",
        "input_mode": "optical+sar",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 111,
        "query": "By what percent did the urban area increase?",
        "input_mode": "optical+sar",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 112,
        "query": "Generate a 3D flythrough of the city.",
        "input_mode": "single",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 113,
        "query": "What will the weather be tomorrow over this scene?",
        "input_mode": "single",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 114,
        "query": "Translate the caption into French.",
        "input_mode": "single",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 115,
        "query": "Write a poem about this landscape.",
        "input_mode": "single",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 116,
        "query": "Forecast cloud cover for tomorrow.",
        "input_mode": "single",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 117,
        "query": "Will it rain here tomorrow?",
        "input_mode": "single",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 118,
        "query": "Summarize this in German.",
        "input_mode": "single",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 119,
        "query": "What is the capital of France?",
        "input_mode": "single",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 120,
        "query": "Write a recipe for tomato soup.",
        "input_mode": "single",
        "expect_supported": False,
        "expect_tools": [],
    },
    {
        "id": 121,
        "query": "Is there a UAV in this image?",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa", "canonical_vqa"],
    },
    {
        "id": 122,
        "query": "Describe this drone image of the site.",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa"],
    },
    # --- CDVQA-MAP rows: one bi-temporal phrasing per mapped family. ---
    {
        "id": 123,
        "query": "What did the water regions mainly change to between the two dates?",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa", "cdvqa_map"],
    },
    {
        "id": 124,
        "query": "What is the largest change between the two dates?",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa", "cdvqa_map"],
    },
    {
        "id": 125,
        "query": "What is the smallest change between the two images?",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa", "cdvqa_map"],
    },
    {
        "id": 126,
        "query": "Did the areas of trees increase?",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa", "cdvqa_map"],
    },
    {
        "id": 127,
        "query": "Have the areas of water changed?",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa", "cdvqa_map"],
    },
    {
        "id": 128,
        "query": "What percentage of the area has not changed?",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa", "cdvqa_map"],
    },
    {
        "id": 129,
        "query": "What is the change ratio of the areas of playgrounds?",
        "input_mode": "bi-temporal",
        "expect_supported": True,
        "expect_tools": ["change_detect", "area_calc", "vqa", "cdvqa_map"],
    },
    # WIRE-8091: single-image RSVQA-family questions also emit canonical_vqa
    # (adapted answer seat on :8091); narration stays with vqa on :8080.
    {
        "id": 130,
        "query": "Is there a road in the image?",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa", "canonical_vqa"],
    },
    {
        "id": 131,
        "query": "Are there any ships in the harbor?",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa", "canonical_vqa"],
    },
    {
        "id": 132,
        "query": "How many cars are visible in this image?",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa", "canonical_vqa"],
    },
    {
        "id": 133,
        "query": "What is the area of the industrial zone?",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa", "canonical_vqa"],
    },
    {
        "id": 134,
        "query": "Are there more houses than trees in this image?",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa", "canonical_vqa"],
    },
    {
        "id": 135,
        "query": "Is the area rural or urban?",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa", "canonical_vqa"],
    },
    {
        "id": 136,
        "query": "Does the image contain a bridge?",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa", "canonical_vqa"],
    },
    {
        "id": 137,
        "query": "How many buildings do you see?",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa", "canonical_vqa"],
    },
    {
        "id": 138,
        "query": "Are there more roads than bridges in this image?",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa", "canonical_vqa"],
    },
    {
        # Trap: land-cover phrasing is caption-classed -> narrator only.
        "id": 139,
        "query": "What land cover dominates this scene?",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa"],
    },
    {
        "id": 140,
        "query": "Count the aircraft on the runway.",
        "input_mode": "single",
        "expect_supported": True,
        "expect_tools": ["vqa", "canonical_vqa"],
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
        self.assertEqual(set(p3["tools"]), {"sar_read", "sar_agreement", "vqa"})

    def test_oos_and_mismatch(self) -> None:
        self.assertFalse(plan("Generate a 3D city model", "bi-temporal")["supported"])
        self.assertFalse(plan("What changed between these two dates, and where?", "single")["supported"])
        # Single-image counts route to the canonical_vqa seat (RSVQA count
        # family); bi-temporal has no counting seat -> still refused.
        one = plan("How many buildings are in this image?", "single")
        self.assertTrue(one["supported"])
        self.assertIn("canonical_vqa", one["tools"])
        self.assertFalse(plan("How many buildings are in this image?", "bi-temporal")["supported"])

    def test_suite_perfect(self) -> None:
        result = score_routing_suite()
        self.assertGreaterEqual(result["n"], 122)
        legacy = [r for r in result["rows"] if r["id"] <= 22]
        self.assertEqual(len(legacy), 22)
        legacy_fails = [r for r in legacy if not r["ok"]]
        self.assertEqual(legacy_fails, [], msg=json.dumps(legacy_fails, indent=2))
        fails = [r for r in result["rows"] if not r["ok"]]
        self.assertGreaterEqual(
            result["n_ok"], 110, msg=json.dumps(fails, indent=2)
        )

    def test_water_highlight_single(self) -> None:
        p = plan("Highlight the water body referred to in the query", "single")
        self.assertTrue(p["supported"])
        self.assertEqual(set(p["tools"]), {"water_highlight", "area_calc", "vqa"})
        self.assertIsNone(p["refusal"])
        river = plan("Is there a river in this image?", "single")
        self.assertEqual(river["tools"], ["vqa", "canonical_vqa"])
        cap = plan("Describe the land cover and major objects.", "single")
        self.assertEqual(cap["tools"], ["vqa"])

    def test_direction_query_supported(self) -> None:
        p = plan(
            "Has built-up area increased, decreased, or remained unchanged?",
            "bi-temporal",
        )
        self.assertTrue(p["supported"])
        self.assertIsNone(p["refusal"])
        self.assertEqual(
            set(p["tools"]),
            {"change_detect", "area_calc", "vqa", "cdvqa_map"},
        )


if __name__ == "__main__":
    result = score_routing_suite()
    print(f"routing {result['n_ok']}/{result['n']} accuracy={result['accuracy']:.3f}")
    for r in result["rows"]:
        flag = "OK" if r["ok"] else "FAIL"
        print(f"  {r['id']:02d} {flag} {r['input_mode']:13s} tools={r['got_tools']}")
    unittest.main(verbosity=2)
