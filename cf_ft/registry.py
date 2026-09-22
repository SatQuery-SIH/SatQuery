"""Checkpoint registry helper. Not the live demo picker.

Do **not** use first-sorted ``best_ckpt.pt`` as the live demo selector.
``demo/tools.py`` ``_find_changeformer_ckpt`` is unchanged this cycle.
``approved_for_demo`` stays false for every role.
"""
from __future__ import annotations

import json
from pathlib import Path

from cf_ft.dataset import CF_CACHE, SATQUERY, imported_ckpt_path, sha256_file

EXAMPLE_PATH = CF_CACHE / "ckpt_registry.example.json"
REGISTRY_PATH = CF_CACHE / "ckpt_registry.json"


def example_registry(imported: Path | None = None) -> dict:
    ckpt = imported or imported_ckpt_path()
    rel = ckpt
    try:
        rel = ckpt.relative_to(SATQUERY)
    except ValueError:
        rel = ckpt
    return {
        "note": (
            "Example registry only. Not a live picker. "
            "Do not use first-sorted best_ckpt.pt. "
            "Do not set approved_for_demo true this cycle."
        ),
        "approved_for_demo_any": False,
        "checkpoints": [
            {
                "role": "imported_levir",
                "path": str(rel).replace("\\", "/"),
                "sha256": sha256_file(ckpt),
                "approved_for_demo": False,
            },
            {
                "role": "team_second",
                "path": None,
                "sha256": None,
                "approved_for_demo": False,
                "note": "No team SECOND ckpt this cycle. smoke.pt is not a demo candidate.",
            },
        ],
    }


def write_example(path: Path = EXAMPLE_PATH, imported: Path | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = example_registry(imported)
    path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    return path


def write_registry(
    team_ckpt: Path,
    val_mean_iou: float,
    train_config_hash: str,
    imported: Path | None = None,
    path: Path = REGISTRY_PATH,
) -> dict:
    """Real registry file. Example json is left untouched. approved_for_demo stays false."""
    ckpt = imported or imported_ckpt_path()
    try:
        imp_rel = ckpt.relative_to(SATQUERY)
    except ValueError:
        imp_rel = ckpt
    try:
        team_rel = Path(team_ckpt).resolve().relative_to(SATQUERY)
    except ValueError:
        team_rel = Path(team_ckpt)
    rec = {
        "note": (
            "Real registry. Not a live picker. "
            "Do not use first-sorted best_ckpt.pt. "
            "approved_for_demo stays false this cycle."
        ),
        "approved_for_demo_any": False,
        "checkpoints": [
            {
                "role": "imported_levir",
                "path": str(imp_rel).replace("\\", "/"),
                "sha256": sha256_file(ckpt),
                "approved_for_demo": False,
            },
            {
                "role": "team_second",
                "path": str(team_rel).replace("\\", "/"),
                "sha256": sha256_file(team_ckpt),
                "val_mean_iou": val_mean_iou,
                "train_config_hash": train_config_hash,
                "approved_for_demo": False,
            },
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    return rec


def append_retention_entry(
    team_ckpt: Path | None,
    val_mean_iou: float | None,
    train_config_hash: str,
    path: Path = REGISTRY_PATH,
    retention_failed: bool = False,
) -> dict:
    """Append/replace ``team_second_retention`` only. Prior roles stay byte-stable in values."""
    path = Path(path)
    rec = json.loads(path.read_text(encoding="utf-8"))
    cps = list(rec.get("checkpoints") or [])
    roles = [c.get("role") for c in cps]
    if "imported_levir" not in roles or "team_second" not in roles:
        raise RuntimeError(
            f"refusing retention append: prior roles missing (have {roles})"
        )
    ckpt_path = None
    ckpt_sha = None
    if (
        not retention_failed
        and team_ckpt is not None
        and Path(team_ckpt).is_file()
    ):
        try:
            ckpt_path = str(Path(team_ckpt).resolve().relative_to(SATQUERY)).replace("\\", "/")
        except ValueError:
            ckpt_path = str(Path(team_ckpt)).replace("\\", "/")
        ckpt_sha = sha256_file(Path(team_ckpt))
    entry = {
        "role": "team_second_retention",
        "path": ckpt_path,
        "sha256": ckpt_sha,
        "val_mean_iou": val_mean_iou,
        "train_config_hash": train_config_hash,
        "retention_failed": bool(retention_failed),
        "approved_for_demo": False,
    }
    rec["checkpoints"] = [c for c in cps if c.get("role") != "team_second_retention"] + [entry]
    rec["approved_for_demo_any"] = False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    return rec


def append_second_semantic_entry(
    ckpt: Path,
    val_miou: float,
    train_config_hash: str,
    n_params: int | None = None,
    path: Path = REGISTRY_PATH,
) -> dict:
    """Append/replace ``second_semantic`` only. Prior roles stay value-stable."""
    path = Path(path)
    rec = json.loads(path.read_text(encoding="utf-8"))
    cps = list(rec.get("checkpoints") or [])
    roles = [c.get("role") for c in cps]
    if "imported_levir" not in roles or "team_second" not in roles:
        raise RuntimeError(f"refusing semantic append: prior roles missing (have {roles})")
    prior = [c for c in cps if c.get("role") != "second_semantic"]
    try:
        rel = str(Path(ckpt).resolve().relative_to(SATQUERY)).replace("\\", "/")
    except ValueError:
        rel = str(Path(ckpt)).replace("\\", "/")
    entry = {
        "role": "second_semantic",
        "path": rel,
        "sha256": sha256_file(Path(ckpt)),
        "val_mean_iou": float(val_miou),
        "train_config_hash": train_config_hash,
        "n_params": n_params,
        "approved_for_demo": False,
        "note": (
            "Per-timestamp 6-class SECOND semantic segmenter. Column only. "
            "WHERE remains team_second. built_up_direction stays not_determined."
        ),
    }
    rec["checkpoints"] = prior + [entry]
    rec["approved_for_demo_any"] = any(bool(c.get("approved_for_demo")) for c in rec["checkpoints"])
    path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    return rec


def mark_second_semantic_demo_route(
    path: Path = REGISTRY_PATH,
    *,
    ratio_outcome: str,
    note: str | None = None,
) -> dict:
    """Gate 5: route second_semantic for second_like_type AFTER gates 1–4. Priors untouched."""
    path = Path(path)
    rec = json.loads(path.read_text(encoding="utf-8"))
    found = False
    for ent in rec.get("checkpoints") or []:
        if ent.get("role") != "second_semantic":
            continue
        found = True
        ent["domains"] = ["second_like_type"]
        ent["route_scope"] = "second_like_type"
        ent["approved_for_demo"] = True
        ent["note"] = note or (
            "SECOND frozen-val mIoU 0.417 vs majority 0.057. "
            f"Ratio re-bin: {ratio_outcome}. "
            "SECOND-domain type-family only. "
            "built_up_direction from class counts with 0.005*total rule. "
            "WHERE remains team_second."
        )
    if not found:
        raise RuntimeError("second_semantic entry missing")
    rec["approved_for_demo_any"] = any(bool(c.get("approved_for_demo")) for c in rec.get("checkpoints") or [])
    path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    return rec


def mark_team_second_demo_route(path: Path = REGISTRY_PATH) -> dict:
    """Gate 5: route team_second for second_like after gates 1–4. Retention entry untouched."""
    path = Path(path)
    rec = json.loads(path.read_text(encoding="utf-8"))
    found = False
    for ent in rec.get("checkpoints") or []:
        if ent.get("role") != "team_second":
            continue
        found = True
        ent["domains"] = ["second_like"]
        ent["route_scope"] = "second_like"
        ent["approved_for_demo"] = True
        ent["note"] = (
            "SECOND frozen-val n=127 mean 0.371 vs start 0.015; "
            "LEVIR Scene 2 0.513 < 0.852 (disclosed). Binary change only."
        )
    if not found:
        raise RuntimeError("team_second entry missing")
    rec["approved_for_demo_any"] = True
    rec["note"] = (
        "Real registry. Domain-routed picker: second_like → team_second; "
        "else imported_levir. approved_for_demo_any is any(entry)."
    )
    path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    return rec


def main() -> None:
    out = write_example()
    print(json.dumps(json.loads(out.read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
