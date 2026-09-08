"""CPU: unique pack_images + X1 subset from prod10 samples.jsonl. No download. No train."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # SatQuery/
OUT = ROOT / "gates" / "_cache" / "prod10"
SAMPLES = OUT / "samples.jsonl"
SEED = 42
N_X1_VQA = 3200
N_X1_CAPTION = 4800


def prefix_of(path: str) -> str:
    p = path.replace("\\", "/")
    if p.startswith("png/vrs/"):
        return "vrs"
    if p.startswith("png/levir/"):
        return "levir"
    if p.startswith("png/ben_s1/"):
        return "ben_s1"
    if p.startswith("png/ben_s2/"):
        return "ben_s2"
    if p.startswith("png/second/"):
        return "second"
    return "other"


def main() -> None:
    rows = []
    with SAMPLES.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    by_slice = Counter(r.get("slice") for r in rows)
    by_prefix: dict[str, set[str]] = defaultdict(set)
    n_images = Counter()
    missing_pack = 0
    for r in rows:
        imgs = r.get("pack_images") or []
        if not imgs:
            missing_pack += 1
        n_images[len(imgs)] += 1
        for p in imgs:
            by_prefix[prefix_of(str(p))].add(str(p).replace("\\", "/"))

    unique = {k: sorted(v) for k, v in by_prefix.items()}
    unique_counts = {k: len(v) for k, v in unique.items()}

    vqa = [r for r in rows if r.get("slice") == "vqa"]
    cap = [r for r in rows if r.get("slice") == "caption"]
    rng = np.random.RandomState(SEED)
    if len(vqa) < N_X1_VQA or len(cap) < N_X1_CAPTION:
        raise RuntimeError(f"STOP: vqa={len(vqa)} caption={len(cap)}")
    vqa_idx = [int(i) for i in rng.choice(len(vqa), N_X1_VQA, replace=False)]
    cap_idx = [int(i) for i in rng.choice(len(cap), N_X1_CAPTION, replace=False)]
    x1 = [vqa[i] for i in sorted(vqa_idx)] + [cap[i] for i in sorted(cap_idx)]
    x1_ids = [str(r.get("id")) for r in x1]
    x1_vrs = sorted(
        {
            str(p).replace("\\", "/")
            for r in x1
            for p in (r.get("pack_images") or [])
            if str(p).replace("\\", "/").startswith("png/vrs/")
        }
    )

    hunt = {
        "n_rows": len(rows),
        "by_slice": dict(by_slice),
        "n_images_per_row": dict(n_images),
        "missing_pack_images": missing_pack,
        "unique_path_counts": unique_counts,
        "x1": {
            "seed": SEED,
            "n_vqa": N_X1_VQA,
            "n_caption": N_X1_CAPTION,
            "n": len(x1),
            "n_unique_vrs_png": len(x1_vrs),
            "note": "Subset of already-quarantined prod10 rows. Not Run 10. Tags unchanged.",
        },
        "sources": {
            "vrs": "HF xiang709/VRSBench Images_train.zip (~8.4 GB). Paths png/vrs/<image_id>.",
            "levir": "https://justchenhao.github.io/LEVIR/ — academic GE terms. 20 names in pack_log brief.pairs; ban test_45.",
            "ben": "hackelle/BigEarthNetV2-Lithuania-Summer-LMDB or full LMDB + BigEarthNet.txt. png/ben_s2 + png/ben_s1.",
            "second": "CDVQA QA = GitHub YZHJessica/CDVQA JSON. Pixels = SECOND public 2968 pairs from https://captain-whu.github.io/SCD/ (Google Drive on that page). IEEE DataPort 10.21227/nsbb-ar76 has no files. HF YZHJessica/CDVQA 401. Do not invent QA from masks.",
            "rsvqa": "Official Zenodo: RSVQA-LR 10.5281/zenodo.6344334 ; RSVQA-HR 10.5281/zenodo.6344367. Site https://rsvqa.sylvainlobry.com/#dataset. Not in Run 10 mix.",
        },
    }
    (OUT / "unique_images.json").write_text(
        json.dumps({k: unique[k] for k in sorted(unique)}, indent=2),
        encoding="utf-8",
    )
    (OUT / "image_manifest.json").write_text(json.dumps(hunt, indent=2), encoding="utf-8")
    (OUT / "x1_ids.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "n": len(x1_ids),
                "n_vqa": N_X1_VQA,
                "n_caption": N_X1_CAPTION,
                "ids": x1_ids,
                "n_unique_vrs_png": len(x1_vrs),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    x1_path = OUT / "x1_samples.jsonl"
    with x1_path.open("w", encoding="utf-8") as f:
        for r in x1:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({**hunt, "x1_bytes": x1_path.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
