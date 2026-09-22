"""CDVQA-like pair dumps -> frozen schema (mode=change). Drops gold if present."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from judge_kit.adapters._cli import run_convert


def _load_rows(src: Path) -> list[dict[str, Any]]:
    text = src.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        payload = json.loads(text)
        return [x for x in payload if isinstance(x, dict)]
    if text.lstrip().startswith("{"):
        payload = json.loads(text)
        if isinstance(payload, dict):
            for key in ("questions", "pairs", "data", "rows"):
                if isinstance(payload.get(key), list):
                    return [x for x in payload[key] if isinstance(x, dict)]
            return [payload]
    rows = []
    for line in text.splitlines():
        if line.strip():
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def convert(src: Path, images_dir: Path | None = None) -> list[dict[str, Any]]:
    src = Path(src)
    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in {".json", ".jsonl"})
        rows: list[dict[str, Any]] = []
        for f in files:
            rows.extend(_load_rows(f))
        if not rows:
            raise ValueError("cdvqa_like adapter: folder has no JSON pair rows")
    else:
        rows = _load_rows(src)
    out: list[dict[str, Any]] = []
    for i, obj in enumerate(rows):
        qid = str(obj.get("id") or obj.get("question_id") or obj.get("pair_id") or f"cdvqa_{i:04d}")
        question = str(obj.get("question") or obj.get("query") or "").strip()
        a = obj.get("image_a") or obj.get("im1") or obj.get("A") or obj.get("before") or obj.get("image1") or ""
        b = obj.get("image_b") or obj.get("im2") or obj.get("B") or obj.get("after") or obj.get("image2") or ""
        a_s, b_s = str(a).strip(), str(b).strip()
        if images_dir is not None:
            if a_s:
                ca = Path(images_dir) / Path(a_s).name
                if ca.is_file():
                    a_s = str(ca)
            if b_s:
                cb = Path(images_dir) / Path(b_s).name
                if cb.is_file():
                    b_s = str(cb)
        out.append(
            {
                "id": qid,
                "mode": "change",
                "question": question,
                "image_a": a_s,
                "image_b": b_s,
            }
        )
    return out


if __name__ == "__main__":
    run_convert(convert, description="Convert a CDVQA-like pair dump to questions.jsonl")
