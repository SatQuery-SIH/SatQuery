"""VRSBench-like dumps -> frozen schema (mode=single). Drops gold if present."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from judge_kit.adapters._cli import run_convert


def _rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("questions", "data", "rows"):
            if isinstance(payload.get(key), list):
                return [x for x in payload[key] if isinstance(x, dict)]
        return [payload]
    return []


def convert(src: Path, images_dir: Path | None = None) -> list[dict[str, Any]]:
    src = Path(src)
    payload = json.loads(src.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for i, obj in enumerate(_rows_from_payload(payload)):
        image = (
            obj.get("image")
            or obj.get("image_id")
            or obj.get("image_path")
            or obj.get("filename")
            or ""
        )
        image_s = str(image).strip()
        if images_dir is not None and image_s:
            name = Path(image_s).name
            cand = Path(images_dir) / name
            if cand.is_file():
                image_s = str(cand)
        qid = str(obj.get("id") or obj.get("example_id") or obj.get("question_id") or f"vrs_{i:04d}")
        question = str(obj.get("question") or obj.get("query") or "").strip()
        out.append(
            {
                "id": qid,
                "mode": "single",
                "question": question,
                "image": image_s,
            }
        )
    return out


if __name__ == "__main__":
    run_convert(convert, description="Convert a VRSBench-like JSON dump to questions.jsonl")
