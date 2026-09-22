"""Pass-through: source is already frozen-schema JSONL (or a JSON list of those objects)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from judge_kit.adapters._cli import run_convert
from judge_kit.schema import validate_row


def convert(src: Path, images_dir: Path | None = None) -> list[dict[str, Any]]:
    text = Path(src).read_text(encoding="utf-8")
    raw: list[Any]
    stripped = text.lstrip()
    if stripped.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("generic adapter: JSON must be a list of row objects")
        raw = payload
    else:
        raw = []
        for i, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            raw.append(json.loads(line))
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for obj in raw:
        row, err = validate_row(obj, seen_ids=seen)
        if err or row is None:
            raise ValueError(f"generic adapter: {err}")
        out.append(row)
    return out


if __name__ == "__main__":
    run_convert(convert, description="Copy already-schema rows into questions.jsonl")
