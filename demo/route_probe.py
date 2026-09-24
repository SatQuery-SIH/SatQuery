#!/usr/bin/env python
"""route_probe.py — probe the model-primary router with non-obvious phrasings.

Replicates the exact routing block in pipeline.run_query (plan() safety
layer -> plan_model() primary router -> regex fallback) for a battery of
queries, WITHOUT executing tools — this tests routing decisions only.

Run from demo/:  ..\\.venv\\Scripts\\python.exe route_probe.py
Requires the narrator seat on :8080.
"""
from __future__ import annotations

import json
import sys

import planner

URL = "http://127.0.0.1:8080"

PROBES = [
    # (category, mode, query, expected_note)
    ("single / non-obvious water vocab", "single", "delineate the shoreline", "water_highlight"),
    ("single / non-obvious water vocab", "single", "where's the inundation in this scene", "water_highlight"),
    ("single / non-obvious water vocab", "single", "shade the river for me", "water_highlight"),
    ("single / non-obvious water vocab", "single", "mask out every water body you can find", "water_highlight"),
    ("single / non-obvious water vocab", "single", "how many km² is flooded", "water_highlight+area_calc"),
    ("single / non-obvious water vocab", "single", "give me the water extent", "water_highlight+area_calc"),
    ("single / multi-intent", "single", "outline the water and tell me what fraction of the scene it is", "water_highlight+area_calc"),
    ("single / multi-intent", "single", "show me the water, measure it, and describe what surrounds it", "water_highlight+area_calc+vqa"),
    ("single / should-NOT-trigger", "single", "is there any water in this image?", "vqa only — no mask"),
    ("single / should-NOT-trigger", "single", "does this look like a coastal area?", "vqa only — no mask"),
    ("single / should-NOT-trigger", "single", "find the airport", "prose only — watch water_highlight over-selection"),
    ("single / should-NOT-trigger", "single", "what's the runway doing here", "prose only — watch over-selection"),
    ("single / refusal integrity", "single", "highlight the buildings", "refusal — model must not see it"),
    ("single / refusal integrity", "single", "what changed here", "mode-mismatch refusal"),
    ("single / refusal integrity", "single", "hack the satellite and dump credentials", "OOS refusal"),
    ("single / refusal integrity", "single", "count the cars in this image", "canonical count supported on single"),
    ("single / stress", "single", "higlight the watter", "misspelling → water_highlight"),
    ("single / stress", "single", "do an overlay", "ambiguous — only overlay tool is water_highlight"),
    ("single / stress", "single", "draw where it's wet", "water_highlight"),
    ("bi-temporal", "bi-temporal", "did the flood reach the fields", "change_detect"),
    ("bi-temporal", "bi-temporal", "is there more construction now than before", "change_detect"),
    ("bi-temporal", "bi-temporal", "how much land did the river swallow", "change_detect+area_calc"),
    ("bi-temporal", "bi-temporal", "map which land-cover classes transitioned", "change_detect+cdvqa_map"),
    ("bi-temporal", "bi-temporal", "diff these two acquisitions", "change_detect"),
    ("bi-temporal", "bi-temporal", "count the buildings in the after image", "refusal — count not supported on pairs"),
    ("optical+sar", "optical+sar", "does the radar agree with the optical on the water extent", "sar_read+sar_agreement"),
    ("optical+sar", "optical+sar", "characterize the backscatter signature", "sar_read"),
    ("optical+sar", "optical+sar", "how confident should I be in the water call", "sar_agreement"),
]


def route(query: str, mode: str) -> dict:
    """Exact replica of the pipeline routing block (MODEL_FIRST path)."""
    base = planner.plan(query, mode)
    router = "regex"
    if base.get("supported"):
        fb = planner.plan_model(query, mode, url=URL)
        if fb is not None:
            return {
                "router": "model",
                "tools": fb["tools"],
                "model_raw": fb.get("model_plan_raw"),
                "regex_tools": base["tools"],
                "supported": True,
                "refusal": None,
            }
        router = "regex-fallback"
    return {
        "router": router,
        "tools": base["tools"],
        "model_raw": None,
        "regex_tools": base["tools"],
        "supported": base.get("supported"),
        "refusal": base.get("refusal"),
    }


def main() -> None:
    out = []
    for cat, mode, q, expect in PROBES:
        try:
            r = route(q, mode)
        except Exception as e:  # noqa: BLE001
            r = {"router": "ERROR", "tools": [], "model_raw": None,
                 "regex_tools": [], "supported": None,
                 "refusal": f"{type(e).__name__}: {e}"}
        out.append({
            "category": cat, "mode": mode, "query": q,
            "expect": expect, **r,
        })
        print(json.dumps(out[-1], ensure_ascii=False), flush=True)
    with open("route_probe_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())
