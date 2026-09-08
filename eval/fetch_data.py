"""Download + cache VRSBench VQA eval split (BASELINE-SPEC-02).

Loads the RAW JSON (`VRSBench_EVAL_vqa.json`) from HF `xiang709/VRSBench`
(CC-BY-4.0). Does NOT use the Hugging Face dataset viewer / auto-cast
(the documented 'Q6' int-cast crash). All fields are coerced to strings.

Eval subset: test/EVAL split, n=300, seed=42, stratified across the 9
spec categories (Presence, Quantity, Color, Shape, Size, Position,
Direction, Scene, Reasoning). Official VRSBench also has a 10th column
(Category ← `object category`); those rows are counted and excluded from
the 300 per the spec's 9-category list. Ids are written to
`gates/_cache/vrsbench/subset_ids.json` and must stay quarantined from
all future training.

Usage (cwd SatQuery/):
    python eval/fetch_data.py
    python eval/fetch_data.py --n 300 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

from score import SPEC_CATEGORIES

HF_REPO = "xiang709/VRSBench"
HF_JSON = "VRSBench_EVAL_vqa.json"
HF_IMAGES_ZIP = "Images_val.zip"
SEED = 42
N_EVAL = 300

ROOT = Path(__file__).resolve().parent.parent  # SatQuery/
CACHE = ROOT / "gates" / "_cache" / "vrsbench"
HF_CACHE = ROOT / "gates" / "_cache" / "hf"
IDS_PATH = CACHE / "subset_ids.json"
SUBSET_PATH = CACHE / "subset.json"
ANN_PATH = CACHE / "VRSBench_EVAL_vqa.json"
IMAGES_DIR = CACHE / "images"

# Official VRSBench type strings (prepare_eval_all.ipynb + eval_vqa_gpt.ipynb).
# `image` and `rural or urban` collapse into Scene, matching the authors' metric notebook.
TYPE_TO_CATEGORY = {
    "object category": "Category",
    "object existence": "Presence",
    "object quantity": "Quantity",
    "object color": "Color",
    "object shape": "Shape",
    "object size": "Size",
    "object position": "Position",
    "object direction": "Direction",
    "scene type": "Scene",
    "image": "Scene",
    "rural or urban": "Scene",
    "reasoning": "Reasoning",
}


def _hf_download(filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=HF_REPO,
        filename=filename,
        repo_type="dataset",
        cache_dir=str(HF_CACHE),
    )
    return Path(path)


def coerce_record(raw: dict, idx: int) -> dict:
    """Coerce mixed-type HF/JSON fields. question_id may be int or 'Q6'-style str."""
    qid = raw.get("question_id", idx)
    image_id = raw.get("image_id", raw.get("image", ""))
    question = raw.get("question", "")
    gold = raw.get("ground_truth", raw.get("answer", ""))
    raw_type = raw.get("type", raw.get("category", ""))
    qid_s = str(qid).strip()
    image_s = str(image_id).strip()
    question_s = str(question).strip()
    gold_s = str(gold).strip()
    type_s = str(raw_type).strip().lower()
    category = TYPE_TO_CATEGORY.get(type_s, "UNKNOWN")
    return {
        "example_id": f"{image_s}::{qid_s}",
        "question_id": qid_s,
        "image_id": image_s,
        "question": question_s,
        "gold": gold_s,
        "type_raw": type_s,
        "category": category,
        "dataset": str(raw.get("dataset", "VRSBench")),
    }


def load_annotations(json_path: Path) -> list[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"Expected a JSON list in {json_path}, got {type(data)}")
    recs = [coerce_record(row, i) for i, row in enumerate(data)]
    return recs


def probe_q6_quirk(json_path: Path) -> dict:
    """Document the mixed-type / Q6 hazard without using the HF viewer.

    We inspect raw Python types of question_id and try a typed Arrow cast
    that reproduces the viewer's int64 failure if mixed strings exist.
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    qid_types = Counter(type(row.get("question_id")).__name__ for row in data)
    qid_str_samples = [
        row.get("question_id")
        for row in data
        if isinstance(row.get("question_id"), str)
    ][:8]
    q6_hits = [
        row.get("question_id")
        for row in data
        if str(row.get("question_id")).strip().upper() in {"Q6", "6"}
        or str(row.get("type", "")).strip().upper() == "Q6"
    ]
    arrow_error = None
    try:
        import pyarrow as pa

        # This is the class of failure the HF viewer hits: force int64 on mixed ids.
        pa.array([row.get("question_id") for row in data], type=pa.int64())
        arrow_cast = "int64-cast-succeeded (all question_id values were int-compatible)"
    except Exception as e:
        arrow_error = f"{type(e).__name__}: {e}"
        arrow_cast = "int64-cast-FAILED (mixed types; this is the Q6-class hazard)"
    return {
        "n_raw": len(data),
        "question_id_python_types": dict(qid_types),
        "string_question_id_samples": qid_str_samples,
        "q6_like_values_count": len(q6_hits),
        "q6_like_samples": [str(x) for x in q6_hits[:8]],
        "forced_int64_cast": arrow_cast,
        "forced_int64_error": arrow_error,
        "loader": "json.load + explicit str() coercion — HF viewer / load_dataset auto-schema NOT used",
    }


def stratified_sample(records: list[dict], n: int, seed: int, categories: list[str]) -> list[dict]:
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = {c: [] for c in categories}
    for rec in records:
        if rec["category"] in by_cat:
            by_cat[rec["category"]].append(rec)
    for c in categories:
        by_cat[c].sort(key=lambda r: (r["image_id"], r["question_id"], r["question"]))
        rng.shuffle(by_cat[c])

    present = [c for c in categories if by_cat[c]]
    if not present:
        raise SystemExit("No records matched the spec categories after type mapping.")
    base = n // len(present)
    rem = n % len(present)
    selected: list[dict] = []
    shortfalls = []
    for i, c in enumerate(present):
        k = base + (1 if i < rem else 0)
        pool = by_cat[c]
        if len(pool) < k:
            shortfalls.append({"category": c, "available": len(pool), "wanted": k})
            selected.extend(pool)
        else:
            selected.extend(pool[:k])

    if len(selected) < n:
        taken = {(r["image_id"], r["question_id"]) for r in selected}
        leftovers = []
        for c in present:
            for r in by_cat[c]:
                key = (r["image_id"], r["question_id"])
                if key not in taken:
                    leftovers.append(r)
        rng_fill = random.Random(seed + 1)
        rng_fill.shuffle(leftovers)
        selected.extend(leftovers[: n - len(selected)])

    if len(selected) != n:
        raise SystemExit(
            f"Could not draw n={n} from the 9 spec categories "
            f"(got {len(selected)}; shortfalls={shortfalls})"
        )
    selected.sort(key=lambda r: (r["image_id"], r["question_id"]))
    return selected


def extract_images(image_ids: list[str], zip_path: Path, dest: Path) -> dict[str, Path]:
    dest.mkdir(parents=True, exist_ok=True)
    wanted = set(image_ids)
    wanted_base = {Path(x).name for x in image_ids}
    mapping: dict[str, Path] = {}
    already = {}
    for img_id in image_ids:
        for cand in (dest / img_id, dest / Path(img_id).name):
            if cand.is_file():
                already[img_id] = cand
                mapping[img_id] = cand
                break
    missing = [i for i in image_ids if i not in mapping]
    if not missing:
        return mapping

    print(f"Opening {zip_path} ({zip_path.stat().st_size} bytes) to extract {len(missing)} images...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        by_name = {}
        by_base = {}
        for n in names:
            if n.endswith("/"):
                continue
            by_name[n] = n
            by_name[n.replace("\\", "/")] = n
            by_base[Path(n).name] = n
        print(f"zip members: {len(names)}  sample: {names[:5]}")
        extracted = 0
        for img_id in missing:
            member = (
                by_name.get(img_id)
                or by_name.get(img_id.replace("\\", "/"))
                or by_base.get(Path(img_id).name)
                or by_base.get(img_id)
            )
            if member is None:
                continue
            out_name = Path(img_id).name
            out_path = dest / out_name
            if not out_path.exists():
                with zf.open(member) as src, out_path.open("wb") as dst:
                    dst.write(src.read())
                extracted += 1
            mapping[img_id] = out_path
        print(f"extracted {extracted} new files into {dest}")

    still = [i for i in image_ids if i not in mapping]
    if still:
        raise SystemExit(
            f"{len(still)} image_ids not found in {zip_path.name}. "
            f"examples: {still[:10]}"
        )
    return mapping


def dump_examples(subset: list[dict], n: int = 3) -> None:
    print("\n=== 3-example dump (question / gold / category / image size) ===")
    for rec in subset[:n]:
        img_path = Path(rec["image_path"])
        with Image.open(img_path) as im:
            w, h = im.size
            mode = im.mode
        print(
            json.dumps(
                {
                    "example_id": rec["example_id"],
                    "question": rec["question"],
                    "gold": rec["gold"],
                    "category": rec["category"],
                    "type_raw": rec["type_raw"],
                    "image_id": rec["image_id"],
                    "image_path": rec["image_path"],
                    "image_size": [w, h],
                    "image_mode": mode,
                    "image_bytes": img_path.stat().st_size,
                },
                ensure_ascii=False,
            )
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_EVAL)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--skip-images",
        action="store_true",
        help="annotations + subset only (no Images_val.zip)",
    )
    args = ap.parse_args(argv)

    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {HF_JSON} from {HF_REPO} ...")
    src_json = _hf_download(HF_JSON)
    if not ANN_PATH.exists() or ANN_PATH.stat().st_size != src_json.stat().st_size:
        ANN_PATH.write_bytes(src_json.read_bytes())
    print(f"cached annotations → {ANN_PATH} ({ANN_PATH.stat().st_size} bytes)")

    quirk = probe_q6_quirk(ANN_PATH)
    print("\n=== schema / Q6 quirk probe ===")
    print(json.dumps(quirk, indent=2, default=str))

    recs = load_annotations(ANN_PATH)
    type_hist = Counter(r["type_raw"] for r in recs)
    cat_hist = Counter(r["category"] for r in recs)
    print("\n=== split sizes ===")
    print(f"VRSBench_EVAL_vqa.json records: {len(recs)}  (paper test VQA = 37408/37409)")
    print("raw type histogram:", json.dumps(dict(sorted(type_hist.items())), indent=2))
    print("mapped category histogram:", json.dumps(dict(cat_hist), indent=2))
    eligible = [r for r in recs if r["category"] in SPEC_CATEGORIES]
    excluded_cat = cat_hist.get("Category", 0)
    unknown = cat_hist.get("UNKNOWN", 0)
    print(f"eligible for 9-category subset: {len(eligible)}")
    print(f"excluded object-category (official 10th col 'Category'): {excluded_cat}")
    print(f"unmapped types: {unknown}")
    if unknown:
        unk = Counter(r["type_raw"] for r in recs if r["category"] == "UNKNOWN")
        print("unmapped type histogram:", dict(unk))

    subset_a = stratified_sample(eligible, args.n, args.seed, SPEC_CATEGORIES)
    subset_b = stratified_sample(eligible, args.n, args.seed, SPEC_CATEGORIES)
    ids_a = [r["example_id"] for r in subset_a]
    ids_b = [r["example_id"] for r in subset_b]
    if ids_a != ids_b:
        raise SystemExit("REPRODUCIBILITY FAIL: seed=42 resample produced different ids")
    print(f"reproducibility: seed={args.seed} resample MATCH ({len(ids_a)} ids)")

    subset_counts = Counter(r["category"] for r in subset_a)
    print("subset per-category counts:", json.dumps(dict(subset_counts), indent=2))

    if args.skip_images:
        for rec in subset_a:
            rec["image_path"] = ""
        image_map = {}
    else:
        print(f"\nDownloading {HF_IMAGES_ZIP} from {HF_REPO} (≈3.98 GB, eval images only)...")
        zip_path = _hf_download(HF_IMAGES_ZIP)
        print(f"zip at {zip_path} ({zip_path.stat().st_size} bytes)")
        unique_ids = list(dict.fromkeys(r["image_id"] for r in subset_a))
        image_map = extract_images(unique_ids, zip_path, IMAGES_DIR)
        for rec in subset_a:
            rec["image_path"] = str(image_map[rec["image_id"]].resolve())

    SUBSET_PATH.write_text(json.dumps(subset_a, indent=2, ensure_ascii=False), encoding="utf-8")
    payload = {
        "dataset": HF_REPO,
        "license": "CC-BY-4.0 (text); DOTA images academic-use note applies",
        "split": "test",
        "source_file": HF_JSON,
        "n": len(subset_a),
        "seed": args.seed,
        "categories": SPEC_CATEGORIES,
        "per_category_counts": dict(subset_counts),
        "full_split_n": len(recs),
        "eligible_n": len(eligible),
        "excluded_object_category_n": excluded_cat,
        "example_ids": ids_a,
        "first_id": ids_a[0],
        "last_id": ids_a[-1],
        "quarantine": "these example_ids MUST NOT appear in any future training set",
        "q6_quirk": quirk,
    }
    IDS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {SUBSET_PATH}")
    print(f"wrote {IDS_PATH}  first={ids_a[0]}  last={ids_a[-1]}")
    print(
        "QUARANTINE: log these example_ids; do not leak them into LoRA/training data."
    )

    if not args.skip_images:
        dump_examples(subset_a, 3)
        missing_px = [r["example_id"] for r in subset_a if not Path(r["image_path"]).is_file()]
        if missing_px:
            raise SystemExit(f"{len(missing_px)} subset rows missing image files")
        print(f"\nimages on disk: {len(list(IMAGES_DIR.iterdir()))} files in {IMAGES_DIR}")
    print("fetch_data.py: evidence ready (no PASS claim)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
