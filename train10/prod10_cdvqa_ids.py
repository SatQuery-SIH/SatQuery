"""Vendor official CDVQA test pair/QA ids. Does not invent SECOND QA. No train."""
from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # SatQuery/
GATES = ROOT / "gates"
OUT_IDS = GATES / "cdvqa_eval_ids.json"
CACHE = GATES / "_cache" / "prod10" / "cdvqa_raw"
BASE = "https://raw.githubusercontent.com/YZHJessica/CDVQA/main/"
FILES = (
    "Test_images.json",
    "Test_questions.json",
    "Test_answers.json",
    "Test2_images.json",
    "Test2_questions.json",
    "Test2_answers.json",
    "Train_images.json",
    "Val_images.json",
)
SEED = 42
N_HOLDOUT = 100


def _download(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / name
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    url = BASE + name
    print(f"GET {url}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as r:
        dest.write_bytes(r.read())
    return dest


def _pair_ids(images_obj: dict) -> list[str]:
    rows = images_obj.get("images") or images_obj
    if isinstance(rows, dict):
        rows = list(rows.values())
    names = []
    for row in rows:
        fn = str(row.get("file_name") or row.get("filename") or "")
        if fn:
            names.append(fn)
    return sorted(set(names))


def _qa_ids(questions_obj: dict) -> list[int]:
    qs = questions_obj.get("questions") or questions_obj
    if isinstance(qs, dict):
        qs = list(qs.values())
    ids = []
    for q in qs:
        if isinstance(q, dict) and "question_id" in q:
            ids.append(int(q["question_id"]))
        elif isinstance(q, dict) and "id" in q:
            ids.append(int(q["id"]))
    return ids


def main() -> None:
    loaded = {}
    for name in FILES:
        p = _download(name)
        loaded[name] = json.loads(p.read_text(encoding="utf-8"))
        print(f"{name} bytes={p.stat().st_size} keys={list(loaded[name])[:8] if isinstance(loaded[name], dict) else type(loaded[name])}")

    test1_pairs = _pair_ids(loaded["Test_images.json"])
    test2_pairs = _pair_ids(loaded["Test2_images.json"])
    train_pairs = _pair_ids(loaded["Train_images.json"])
    val_pairs = _pair_ids(loaded["Val_images.json"])
    union = sorted(set(test1_pairs) | set(test2_pairs))
    overlap_train = sorted(set(union) & set(train_pairs))
    qa1 = _qa_ids(loaded["Test_questions.json"])
    qa2 = _qa_ids(loaded["Test2_questions.json"])
    # Test1 and Test2 both number questions from id=0; integer union collapses.
    # Namespace so holdout ids are unique across the two official test files.
    qa_ns = [f"test1:{i}" for i in qa1] + [f"test2:{i}" for i in qa2]
    rng = random.Random(SEED)
    holdout = qa_ns[:]
    rng.shuffle(holdout)
    holdout = sorted(holdout[:N_HOLDOUT])
    payload = {
        "source_repo": "https://github.com/YZHJessica/CDVQA",
        "license": "Apache-2.0",
        "paper": "https://arxiv.org/pdf/2112.06343",
        "ieee_doi": "10.1109/tgrs.2022.3203314",
        "seed": SEED,
        "note": (
            "Official JSON splits from GitHub YZHJessica/CDVQA (Apache-2.0). "
            "No image rasters in that repo (pixels are SECOND). "
            "Pair id = file_name from *_images.json. Tests share the same 968 pairs; "
            "QA ids are namespaced test1:<id> / test2:<id> because both files restart at 0. "
            "Holdout n=100 sampled now; do not score in PROD-PACK-10."
        ),
        "n_test1_pairs": len(test1_pairs),
        "n_test2_pairs": len(test2_pairs),
        "n_test_union_pairs": len(union),
        "n_train_pairs": len(train_pairs),
        "n_val_pairs": len(val_pairs),
        "n_overlap_test_vs_train": len(overlap_train),
        "n_test1_qa": len(qa1),
        "n_test2_qa": len(qa2),
        "n_test_union_qa_namespaced": len(qa_ns),
        "qa_id_format": "test1:<id> | test2:<id>",
        "pair_ids_test_union": union,
        "qa_ids_holdout_n100": holdout,
        "rasters_in_repo": False,
    }
    OUT_IDS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    hunt = {
        "github": "https://github.com/YZHJessica/CDVQA",
        "github_raw": BASE,
        "github_files": FILES,
        "hf_YZHJessica_CDVQA": "401 Unauthorized / dataset not public on the Hub",
        "license": "Apache-2.0",
        "rasters": "not in GitHub tree (JSON only). Pixels would be SECOND 512x512 pairs.",
        "paper_splits": {
            "train_pairs": 1600,
            "val_pairs": 400,
            "test_pairs": 968,
            "train_qa": 65967,
            "val_qa": 16441,
            "test1_qa": 39686,
            "test2_qa": 31036,
        },
        "observed": {
            "n_train_pairs": len(train_pairs),
            "n_val_pairs": len(val_pairs),
            "n_test_union_pairs": len(union),
            "n_test1_qa": len(qa1),
            "n_test2_qa": len(qa2),
        },
        "overlap_test_train_pairs": overlap_train[:20],
        "out": str(OUT_IDS),
    }
    (CACHE / "hunt_log.json").write_text(json.dumps(hunt, indent=2), encoding="utf-8")
    (CACHE / "train_pair_ids.json").write_text(
        json.dumps({"n": len(train_pairs), "pair_ids": train_pairs}, indent=2),
        encoding="utf-8",
    )
    print("CDVQA_IDS_OK", json.dumps({
        "n_test_union_pairs": len(union),
        "n_train_pairs": len(train_pairs),
        "n_holdout_qa": len(holdout),
        "n_test_union_qa_namespaced": len(qa_ns),
        "overlap_test_train": len(overlap_train),
        "out": str(OUT_IDS),
    }))


if __name__ == "__main__":
    main()
