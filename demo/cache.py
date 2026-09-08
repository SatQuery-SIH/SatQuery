"""Pre-compute traces for the 3 scenes + likely judge questions (DEMO-SPEC-05).

Usage:
  python cache.py                 # write traces (needs llama-server for live VLM)
  python cache.py --mode cached   # print that traces exist
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO))

from pipeline import run_query  # noqa: E402

TRACES = DEMO / "traces"
TRACES.mkdir(parents=True, exist_ok=True)

CANONICAL = [
    {
        "id": "scene1",
        "scene": 1,
        "input_mode": "single",
        "query": "Describe the land cover and major objects.",
    },
    {
        "id": "scene2",
        "scene": 2,
        "input_mode": "bi-temporal",
        "query": "What changed between these two dates, and where?",
    },
    {
        "id": "scene3",
        "scene": 3,
        "input_mode": "optical+sar",
        "query": "Identify water-covered regions.",
    },
]

# 6–8 likely judge questions (trapdoor).
JUDGE = [
    {"id": "j1", "scene": 1, "input_mode": "single", "query": "Is this scene predominantly urban or rural?"},
    {"id": "j2", "scene": 1, "input_mode": "single", "query": "Are there any visible water bodies?"},
    {"id": "j3", "scene": 2, "input_mode": "bi-temporal", "query": "How much built-up area increased, in km2?"},
    {"id": "j4", "scene": 2, "input_mode": "bi-temporal", "query": "Show the change mask."},
    {"id": "j5", "scene": 2, "input_mode": "bi-temporal", "query": "Which quadrant has the most new built-up?"},
    {"id": "j6", "scene": 3, "input_mode": "optical+sar", "query": "What is the SAR VV mean backscatter?"},
    {"id": "j7", "scene": 3, "input_mode": "optical+sar", "query": "Compare optical vs SAR water evidence."},
    {"id": "j8", "scene": 1, "input_mode": "single", "query": "Generate a 3D city model from this image."},
]


def _strip_arrays(obj):
    if isinstance(obj, dict):
        return {k: _strip_arrays(v) for k, v in obj.items() if k not in {"mask", "water_mask"}}
    if isinstance(obj, list):
        return [_strip_arrays(x) for x in obj]
    return obj


def trace_path(item_id: str) -> Path:
    return TRACES / f"{item_id}.json"


def load_trace(item_id: str) -> dict | None:
    p = trace_path(item_id)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def match_cached(query: str, input_mode: str) -> dict | None:
    q = (query or "").strip().lower()
    mode = (input_mode or "").strip().lower()
    for item in CANONICAL + JUDGE:
        if item["query"].strip().lower() == q and item["input_mode"] == mode:
            return load_trace(item["id"])
    return None


def write_all(live: bool = True, vlm_url: str = "http://127.0.0.1:8080") -> dict:
    written = []
    for item in CANONICAL + JUDGE:
        print(f"cache {item['id']} ...", flush=True)
        trace = run_query(
            item["query"],
            item["input_mode"],
            scene=item["scene"],
            live=live,
            vlm_url=vlm_url,
        )
        trace["cache_id"] = item["id"]
        trace["live"] = False  # stored traces are the trapdoor
        trace["cached_label"] = True
        slim = _strip_arrays(trace)
        trace_path(item["id"]).write_text(json.dumps(slim, indent=2, default=str), encoding="utf-8")
        written.append(
            {
                "id": item["id"],
                "supported": slim["plan"]["supported"],
                "has_answer": bool(slim.get("answer")),
                "complete_s": slim.get("complete_s"),
                "first_token_s": slim.get("first_token_s"),
            }
        )
    index = {"n": len(written), "items": written}
    (TRACES / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["live", "cached"], default="live")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    args = ap.parse_args()
    if args.mode == "cached":
        n = len(list(TRACES.glob("scene*.json")))
        print(f"traces dir={TRACES} scene_files={n}")
        idx = TRACES / "index.json"
        if idx.is_file():
            print(idx.read_text(encoding="utf-8"))
        return
    index = write_all(live=True, vlm_url=args.url)
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
