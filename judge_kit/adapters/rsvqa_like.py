"""RSVQA-like dumps -> frozen schema (mode=single). Drops gold if present.

Accepts a JSON file, a JSONL file, or a folder containing USGS-style
`*_questions.json` plus optional `*_images.json`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from judge_kit.adapters._cli import run_convert


def _load(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        return json.loads(text)
    if text.lstrip().startswith("{"):
        return json.loads(text)
    rows = []
    for line in text.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _as_list(payload: Any, *keys: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for k in keys:
            if isinstance(payload.get(k), list):
                return [x for x in payload[k] if isinstance(x, dict)]
        if any(x in payload for x in ("question", "ques")):
            return [payload]
    return []


def _from_folder(folder: Path, images_dir: Path | None) -> list[dict[str, Any]]:
    qfiles = sorted(folder.glob("*questions*.json")) + sorted(folder.glob("*question*.json"))
    ifiles = sorted(folder.glob("*images*.json")) + sorted(folder.glob("*image*.json"))
    questions: list[dict[str, Any]] = []
    for qf in qfiles:
        questions.extend(_as_list(_load(qf), "questions", "question", "data"))
    images: dict[str, str] = {}
    for imf in ifiles:
        for im in _as_list(_load(imf), "images", "image", "data"):
            iid = im.get("id", im.get("img_id", im.get("image_id")))
            fn = im.get("filename") or im.get("file_name") or im.get("path") or im.get("name")
            if iid is not None and fn:
                images[str(iid)] = str(fn)
    if not questions:
        # folder of jsonl
        for p in sorted(folder.glob("*.jsonl")):
            questions.extend(_as_list(_load(p), "questions"))
    return _rows_from_questions(questions, images, images_dir)


def _rows_from_questions(
    questions: list[dict[str, Any]],
    images: dict[str, str],
    images_dir: Path | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, obj in enumerate(questions):
        qid = str(obj.get("id") or obj.get("question_id") or obj.get("qid") or f"rsvqa_{i:04d}")
        question = str(obj.get("question") or obj.get("ques") or obj.get("query") or "").strip()
        img_id = obj.get("img_id", obj.get("image_id", obj.get("image")))
        image = str(obj.get("filename") or obj.get("path") or "")
        if img_id is not None and str(img_id) in images:
            image = images[str(img_id)]
        if not image and img_id is not None:
            image = str(img_id)
        if images_dir is not None and image:
            name = Path(image).name
            for cand in (
                Path(images_dir) / name,
                Path(images_dir) / f"{img_id}.tif" if img_id is not None else None,
                Path(images_dir) / f"{img_id}.png" if img_id is not None else None,
            ):
                if cand is not None and cand.is_file():
                    image = str(cand)
                    break
        out.append({"id": qid, "mode": "single", "question": question, "image": image})
    return out


def convert(src: Path, images_dir: Path | None = None) -> list[dict[str, Any]]:
    src = Path(src)
    if src.is_dir():
        return _from_folder(src, images_dir)
    payload = _load(src)
    questions = _as_list(payload, "questions", "question", "data", "rows")
    images: dict[str, str] = {}
    if isinstance(payload, dict):
        for im in _as_list(payload.get("images"), "images"):
            iid = im.get("id", im.get("img_id"))
            fn = im.get("filename") or im.get("path")
            if iid is not None and fn:
                images[str(iid)] = str(fn)
    return _rows_from_questions(questions, images, images_dir)


if __name__ == "__main__":
    run_convert(convert, description="Convert an RSVQA-like dump or folder to questions.jsonl")
