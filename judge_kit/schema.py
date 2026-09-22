"""Frozen questions.jsonl schema for the eval kit.

Declared here, never guessed at runtime. Violations are row-level errors;
this module does not repair missing fields.

Required keys per row:
  id (str, unique) · mode (single | change | sar) · question (str)
  image (str) for single
  image_a + image_b (str) for change and sar

Image paths are relative to the images directory, or absolute.
"""
from __future__ import annotations

from typing import Any

MODES = ("single", "change", "sar")

# Read-only from demo/planner.py INPUT_MODES + aliases, and demo/ingest.py bind_inputs:
#   single      -> input_mode "single",        uploads={"image": path}
#   change      -> input_mode "bi-temporal",   uploads={"before": path, "after": path}
#   sar         -> input_mode "optical+sar",   uploads={"optical": path, "sar": path}
#                  bind_inputs also accepts aliases sar_vv / sar_npz for the SAR file.
KIT_MODE_TO_INPUT_MODE = {
    "single": "single",
    "change": "bi-temporal",
    "sar": "optical+sar",
}


def _as_str(val: Any) -> str:
    if val is None:
        return ""
    return str(val).strip()


def validate_row(obj: Any, *, seen_ids: set[str] | None = None) -> tuple[dict[str, Any] | None, str | None]:
    """Return (normalized_row, None) or (None, error). Never mutates a bad row into a good one."""
    if not isinstance(obj, dict):
        return None, "schema: row is not a JSON object"
    rid = _as_str(obj.get("id"))
    if not rid:
        return None, "schema: missing id"
    if seen_ids is not None:
        if rid in seen_ids:
            return None, f"schema: duplicate id {rid!r}"
        seen_ids.add(rid)
    mode = _as_str(obj.get("mode")).lower()
    if mode not in MODES:
        return None, f"schema: mode must be one of {MODES}, got {obj.get('mode')!r}"
    question = obj.get("question")
    if not isinstance(question, str) or not question.strip():
        return None, "schema: missing question"
    row: dict[str, Any] = {
        "id": rid,
        "mode": mode,
        "question": question.strip(),
    }
    if mode == "single":
        image = _as_str(obj.get("image"))
        if not image:
            return None, "schema: mode=single requires image"
        row["image"] = image
    else:
        a = _as_str(obj.get("image_a"))
        b = _as_str(obj.get("image_b"))
        if not a or not b:
            return None, f"schema: mode={mode} requires image_a and image_b"
        row["image_a"] = a
        row["image_b"] = b
    return row, None
