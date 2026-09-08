"""CPU-only BigEarthNet.txt -> Unsloth vision chat-format converter (G2-SPEC-01).

Source of truth for TEXT: Hugging Face `BIFOLD-BigEarthNetv2-0/BigEarthNet.txt`
(the official BigEarthNet.txt parquet: instruction/answer pairs).

Source of IMAGES: Sentinel-2 B04/B03/B02 from a rico-hdl LMDB of official
BigEarthNet v2.0 patches. The full official image archive is 118–155 GB; for
the 500-sample smoke run we join against the downloadable official-format
subset `hackelle/BigEarthNetV2-Lithuania-Summer-LMDB` (2.48 GB, 8,775 patches)
UNLESS a local Encoded-BigEarthNet LMDB covering more patches is provided.
This is NOT a different dataset: annotations are official BigEarthNet.txt;
pixels are official BigEarthNet-S2. Sampling uses seed=42 inside the
(image-metadata split='train' ∩ text split='train' ∩ type in
{binary, mcq, captioning} ∩ patches present in the LMDB) pool.
Bounding-box rows are excluded (orchestrator ruling G2-RESUME-01).

Unit-testable without GPU. `python gates/data_prep.py` runs the §7 pre-flight.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HF_TXT_REPO = "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt"
HF_TXT_FILE = "BigEarthNet.txt.parquet"
HF_IMG_REPO = "hackelle/BigEarthNetV2-Lithuania-Summer-LMDB"
SEED = 42
N_SMOKE = 500
N_TRAIN = 480
N_VAL = 20
MAX_EDGE = 512
RGB_BANDS = ("B04", "B03", "B02")  # Sentinel-2 optical R,G,B
ALLOWED_TYPES = frozenset({"binary", "mcq", "captioning"})
IMAGE_META_NAME = "metadata_lithuania_summer.parquet"

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "_cache"
INDICES_PATH = CACHE / "gate2_indices.json"


def _cache_dir() -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE


def ensure_assets(cache_dir: Path | None = None) -> dict:
    """Download parquet + image LMDB if missing. CPU/network only — no GPU."""
    from huggingface_hub import hf_hub_download, snapshot_download

    cache_dir = Path(cache_dir) if cache_dir else _cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = Path(
        hf_hub_download(
            repo_id=HF_TXT_REPO,
            filename=HF_TXT_FILE,
            repo_type="dataset",
            cache_dir=str(cache_dir / "hf"),
        )
    )

    override = os.environ.get("BEN_LMDB_PATH")
    if override:
        lmdb_path = Path(override)
        if not lmdb_path.exists():
            raise FileNotFoundError(f"BEN_LMDB_PATH does not exist: {lmdb_path}")
    else:
        img_root = Path(
            snapshot_download(
                repo_id=HF_IMG_REPO,
                repo_type="dataset",
                cache_dir=str(cache_dir / "hf"),
            )
        )
        candidates = list(img_root.glob("*.lmdb")) + list(img_root.glob("**/*.lmdb"))
        # HF stores LMDB as a directory (data.mdb inside)
        if not candidates:
            # directory itself may be the lmdb folder
            if (img_root / "data.mdb").exists():
                candidates = [img_root]
            else:
                nested = list(img_root.rglob("data.mdb"))
                candidates = [p.parent for p in nested]
        if not candidates:
            raise FileNotFoundError(
                f"No LMDB found under {img_root}. STOP: cannot substitute another image source."
            )
        lmdb_path = candidates[0]

    meta_path = _find_image_metadata(lmdb_path, cache_dir)
    return {
        "parquet": parquet_path,
        "lmdb": lmdb_path,
        "image_metadata": meta_path,
        "txt_repo": HF_TXT_REPO,
        "img_repo": HF_IMG_REPO if not override else f"local:{lmdb_path}",
    }


def _find_image_metadata(lmdb_path: Path, cache_dir: Path) -> Path:
    """Resolve metadata_lithuania_summer.parquet (may be a Windows LFS symlink)."""
    direct = lmdb_path.parent / IMAGE_META_NAME
    real = Path(os.path.realpath(direct))
    if real.exists() and real.stat().st_size > 0:
        return real
    blobs = cache_dir / "hf" / "datasets--hackelle--BigEarthNetV2-Lithuania-Summer-LMDB" / "blobs"
    if blobs.is_dir():
        for b in blobs.iterdir():
            if b.is_file() and 200_000 < b.stat().st_size < 400_000:
                return b
    raise FileNotFoundError(
        f"Image metadata parquet not found next to {lmdb_path} or in {blobs}"
    )


def load_train_patch_ids(metadata_path: Path) -> set[str]:
    """Patch IDs whose LMDB/image-metadata split is 'train'."""
    import pandas as pd

    df = pd.read_parquet(metadata_path)
    if "split" not in df.columns or "patch_id" not in df.columns:
        raise RuntimeError(f"Image metadata missing split/patch_id; columns={list(df.columns)}")
    train = df.loc[df["split"].astype(str).eq("train"), "patch_id"].astype(str)
    return set(train.tolist())


def _open_lmdb(lmdb_path: Path):
    import lmdb

    return lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        meminit=False,
        readahead=True,
        map_size=8 * 1024**3,
    )


def list_s2_patch_ids(lmdb_path: Path) -> set[str]:
    """Sentinel-2 keys look like S2*_MSIL2A_* ; S1 keys look like S1*_IW_*."""
    env = _open_lmdb(lmdb_path)
    keys: set[str] = set()
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for k, _ in cursor:
            key = k.decode("utf-8") if isinstance(k, (bytes, bytearray)) else str(k)
            if key.startswith("S2"):
                keys.add(key)
    env.close()
    return keys


def _to_2d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"Expected 2-D band, got shape {a.shape}")
    return a.astype(np.float32)


def _match_hw(bands: list[np.ndarray]) -> list[np.ndarray]:
    h = max(b.shape[0] for b in bands)
    w = max(b.shape[1] for b in bands)
    out = []
    for b in bands:
        if b.shape == (h, w):
            out.append(b)
            continue
        im = Image.fromarray(b, mode="F").resize((w, h), Image.Resampling.NEAREST)
        out.append(np.array(im, dtype=np.float32))
    return out


def s2_rgb_to_pil(b04, b03, b02) -> Image.Image:
    """Percentile-stretch Sentinel-2 B04/B03/B02 to 8-bit RGB."""
    bands = _match_hw([_to_2d(b04), _to_2d(b03), _to_2d(b02)])
    stacked = np.stack(bands, axis=-1)
    out = np.empty(stacked.shape, dtype=np.uint8)
    for c in range(3):
        ch = stacked[..., c]
        lo, hi = np.percentile(ch, (2.0, 98.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.min(ch)), float(np.max(ch))
            if hi <= lo:
                hi = lo + 1.0
        scaled = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
        out[..., c] = (scaled * 255.0).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def clamp_long_edge(im: Image.Image, max_edge: int = MAX_EDGE) -> Image.Image:
    w, h = im.size
    long = max(w, h)
    if long <= max_edge:
        return im
    scale = max_edge / float(long)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return im.resize((nw, nh), Image.Resampling.BILINEAR)


def load_s2_rgb(env, patch_id: str) -> Image.Image:
    from safetensors.numpy import load as safetensor_load

    with env.begin(write=False, buffers=True) as txn:
        raw = txn.get(patch_id.encode("utf-8"))
    if raw is None:
        raise KeyError(f"patch_id not in LMDB: {patch_id}")
    tensor = safetensor_load(bytes(raw))
    missing = [b for b in RGB_BANDS if b not in tensor]
    if missing:
        raise KeyError(f"{patch_id} missing bands {missing}; have {list(tensor)}")
    im = s2_rgb_to_pil(tensor["B04"], tensor["B03"], tensor["B02"])
    return clamp_long_edge(im.convert("RGB"), MAX_EDGE)


def format_sample(raw: dict) -> dict:
    """Convert one raw row into the Unsloth vision conversational sample.

    Required keys: instruction (or input), answer (or output), image (PIL.Image).
    """
    instruction = raw.get("instruction", raw.get("input", ""))
    answer = raw.get("answer", raw.get("output", ""))
    if instruction is None:
        instruction = ""
    if answer is None:
        answer = ""
    instruction = str(instruction).strip()
    answer = str(answer).strip()
    image = raw.get("image")
    if not isinstance(image, Image.Image):
        raise TypeError(f"raw['image'] must be PIL.Image, got {type(image)}")
    image = clamp_long_edge(image.convert("RGB"), MAX_EDGE)
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": image},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ],
            },
        ]
    }


def _eligible_frame(parquet_path: Path, train_patch_ids: set[str]):
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    need = {"ID", "input", "output", "patch_id", "split", "type"}
    missing = need - set(df.columns)
    if missing:
        raise RuntimeError(f"Parquet missing columns {missing}; have {list(df.columns)}")

    inp = df["input"].astype(str).str.strip()
    out = df["output"].astype(str).str.strip()
    pid = df["patch_id"].astype(str)
    split = df["split"].astype(str)
    typ = df["type"].astype(str).str.strip().str.lower()

    mask = (
        pid.isin(train_patch_ids)
        & split.eq("train")
        & typ.isin(ALLOWED_TYPES)
        & inp.ne("")
        & out.ne("")
        & inp.ne("None")
        & out.ne("None")
        & inp.ne("nan")
        & out.ne("nan")
    )
    elig = df.loc[mask].copy()
    elig["_orig_index"] = elig.index.astype(int)
    elig = elig.sort_values("ID").reset_index(drop=True)
    return elig


def select_indices(n: int = N_SMOKE, seed: int = SEED, assets: dict | None = None) -> dict:
    """Deterministic sample of n eligible parquet rows. Returns metadata + ID list."""
    assets = assets or ensure_assets()
    lmdb_s2 = list_s2_patch_ids(assets["lmdb"])
    meta_train = load_train_patch_ids(assets["image_metadata"])
    train_patch_ids = lmdb_s2 & meta_train
    elig = _eligible_frame(assets["parquet"], train_patch_ids)
    type_counts = {t: int((elig["type"].astype(str).str.lower() == t).sum()) for t in sorted(ALLOWED_TYPES)}
    if len(elig) < n:
        raise RuntimeError(
            f"STOP: eligible-train count after split='train' + bbox-drop = {len(elig)} (< {n}). "
            "Do not pad with test rows. "
            f"image-metadata train patches={len(meta_train)}; "
            f"LMDB S2 keys={len(lmdb_s2)}; intersection={len(train_patch_ids)}; "
            f"type_counts={type_counts}"
        )
    rng = np.random.RandomState(seed)
    chosen = rng.choice(len(elig), size=n, replace=False)
    chosen.sort()  # stable order for later 480/20 split
    rows = elig.iloc[chosen]
    ids = [str(x) for x in rows["ID"].tolist()]
    orig = [int(x) for x in rows["_orig_index"].tolist()]
    sampled_types = {}
    for t in rows["type"].astype(str).str.lower().tolist():
        sampled_types[t] = sampled_types.get(t, 0) + 1
    return {
        "n": n,
        "seed": seed,
        "txt_repo": assets["txt_repo"],
        "img_repo": assets["img_repo"],
        "allowed_types": sorted(ALLOWED_TYPES),
        "n_eligible": int(len(elig)),
        "n_s2_patches_in_lmdb": int(len(lmdb_s2)),
        "n_image_metadata_train_patches": int(len(meta_train)),
        "n_train_patches_with_images": int(len(train_patch_ids)),
        "eligible_type_counts": type_counts,
        "sampled_type_counts": sampled_types,
        "ids": ids,
        "orig_parquet_indices": orig,
        "chosen_positions_in_sorted_eligible": [int(x) for x in chosen.tolist()],
        "patch_ids": [str(x) for x in rows["patch_id"].tolist()],
        "train_ids": ids[:N_TRAIN],
        "val_ids": ids[N_TRAIN : N_TRAIN + N_VAL],
        "image_resize": f"clamp long edge <= {MAX_EDGE}px (native S2 10m typically 120px)",
        "geographic_restriction": (
            "500 samples from BigEarthNet v2.0 Lithuania-Summer S2 patches "
            "(image-metadata split=train only; bbox dropped)"
        ),
    }


def build_dataset(n: int = N_SMOKE, seed: int = SEED, assets: dict | None = None) -> list:
    """Return n Unsloth-format samples. Also writes indices JSON for reproducibility."""
    assets = assets or ensure_assets()
    meta = select_indices(n=n, seed=seed, assets=assets)
    import pandas as pd

    df = pd.read_parquet(assets["parquet"])
    by_id = df.set_index(df["ID"].astype(str), drop=False)
    env = _open_lmdb(assets["lmdb"])
    samples = []
    try:
        for sid in meta["ids"]:
            row = by_id.loc[sid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            image = load_s2_rgb(env, str(row["patch_id"]))
            samples.append(
                format_sample(
                    {
                        "input": row["input"],
                        "output": row["output"],
                        "image": image,
                    }
                )
            )
    finally:
        env.close()

    _cache_dir()
    INDICES_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return samples


def save_smoke_pack(samples: list, meta: dict, dest: Path | None = None) -> Path:
    """Write 500 PNG + jsonl for the Modal mount (no 2.5 GB LMDB upload)."""
    dest = dest or (_cache_dir() / "smoke500")
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    for i, s in enumerate(samples):
        im = s["messages"][0]["content"][1]["image"]
        png_name = f"{i:03d}.png"
        im.save(dest / png_name, format="PNG")
        records.append(
            {
                "i": i,
                "id": meta["ids"][i] if i < len(meta.get("ids", [])) else None,
                "split": "train" if i < N_TRAIN else "val",
                "instruction": s["messages"][0]["content"][0]["text"],
                "answer": s["messages"][1]["content"][0]["text"],
                "image": png_name,
            }
        )
    (dest / "samples.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    (dest / "indices.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return dest


def load_smoke_pack(dest: Path | None = None) -> tuple[list, dict]:
    """Reload smoke pack into Unsloth `messages` samples + meta."""
    dest = dest or (_cache_dir() / "smoke500")
    meta = json.loads((dest / "indices.json").read_text(encoding="utf-8"))
    samples = []
    with (dest / "samples.jsonl").open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            im = Image.open(dest / rec["image"]).convert("RGB")
            samples.append(
                format_sample({"input": rec["instruction"], "output": rec["answer"], "image": im})
            )
    return samples, meta


def _validate_sample(sample: dict, idx: int) -> None:
    msgs = sample.get("messages")
    if not isinstance(msgs, list) or len(msgs) != 2:
        raise AssertionError(f"[{idx}] messages must be length-2 list")
    user, asst = msgs
    if user.get("role") != "user" or asst.get("role") != "assistant":
        raise AssertionError(f"[{idx}] roles must be user, assistant")
    uc = user.get("content")
    ac = asst.get("content")
    if not isinstance(uc, list) or len(uc) != 2:
        raise AssertionError(f"[{idx}] user content must be [text, image]")
    if not isinstance(ac, list) or len(ac) != 1:
        raise AssertionError(f"[{idx}] assistant content must be [text]")
    if uc[0].get("type") != "text" or not str(uc[0].get("text", "")).strip():
        raise AssertionError(f"[{idx}] empty instruction")
    if uc[1].get("type") != "image":
        raise AssertionError(f"[{idx}] second user part must be image")
    im = uc[1].get("image")
    if not isinstance(im, Image.Image) or im.mode != "RGB":
        raise AssertionError(f"[{idx}] image must be PIL RGB, got {type(im)} mode={getattr(im,'mode',None)}")
    if max(im.size) > MAX_EDGE:
        raise AssertionError(f"[{idx}] long edge {max(im.size)} > {MAX_EDGE}")
    if ac[0].get("type") != "text" or not str(ac[0].get("text", "")).strip():
        raise AssertionError(f"[{idx}] empty answer")


def run_cpu_preflight() -> int:
    """G2-SPEC-01 §7. Prints evidence. Returns 0 if all checks hold."""
    print("=== GATE 2 §7 CPU PRE-FLIGHT ===")
    print(f"txt repo: {HF_TXT_REPO}")
    print(f"img repo (smoke image store): {HF_IMG_REPO}")
    print(
        "FLAG: full BigEarthNet-S2 archive is 118–155 GB. Smoke run joins "
        "official BigEarthNet.txt rows to the 2.48 GB Lithuania-Summer official-format "
        "S2 LMDB. Not a different dataset; geographic subset of the same patches."
    )
    print("Downloading/locating assets (CPU/network only)...")
    assets = ensure_assets()
    print(f"parquet: {assets['parquet']}")
    print(f"image metadata: {assets['image_metadata']}")

    meta_a = select_indices(n=N_SMOKE, seed=SEED, assets=assets)
    meta_b = select_indices(n=N_SMOKE, seed=SEED, assets=assets)
    same = meta_a["ids"] == meta_b["ids"]
    print(f"image-metadata train patches: {meta_a['n_image_metadata_train_patches']}")
    print(f"S2 patches in LMDB: {meta_a['n_s2_patches_in_lmdb']}")
    print(f"train patches with images: {meta_a['n_train_patches_with_images']}")
    print(f"eligible-train after bbox-drop: {meta_a['n_eligible']}")
    print(f"eligible type counts: {meta_a['eligible_type_counts']}")
    print(f"sampled type counts: {meta_a['sampled_type_counts']}")
    print(f"sampled n={len(meta_a['ids'])} seed={SEED}")
    print(f"reproducibility (same seed -> identical IDs): {same}")
    if not same:
        print("FAIL: seed=42 did not reproduce identical indices")
        return 1
    if any(t not in ALLOWED_TYPES for t in meta_a["sampled_type_counts"]):
        print("FAIL: sampled types include a disallowed type")
        return 1

    print("build_dataset(500) ...")
    ds = build_dataset(n=N_SMOKE, seed=SEED, assets=assets)
    print(f"len(dataset) = {len(ds)}")
    if len(ds) != N_SMOKE:
        print(f"FAIL: expected {N_SMOKE} samples, got {len(ds)}")
        return 1

    for i, s in enumerate(ds):
        _validate_sample(s, i)
    print("structure / RGB / <=512 / non-empty instruction+answer: OK for all 500")
    pack = save_smoke_pack(ds, meta_a)
    print(f"smoke pack written: {pack}")

    print("\n--- build_dataset(3) dump (criterion A preview) ---")
    preview = build_dataset(n=3, seed=SEED, assets=assets)
    # restore 500-index file after the n=3 overwrite
    INDICES_PATH.write_text(json.dumps(meta_a, indent=2), encoding="utf-8")
    for i, s in enumerate(preview):
        im = s["messages"][0]["content"][1]["image"]
        instr = s["messages"][0]["content"][0]["text"]
        ans = s["messages"][1]["content"][0]["text"]
        print(f"[{i}] image size={im.size} mode={im.mode}")
        print(f"    instruction[:160]={instr[:160]!r}")
        print(f"    answer[:160]={ans[:160]!r}")
        print(f"    keys={list(s.keys())} roles={[m['role'] for m in s['messages']]}")
        print(f"    user content types={[c['type'] for c in s['messages'][0]['content']]}")
        print(f"    asst content types={[c['type'] for c in s['messages'][1]['content']]}")

    print(f"\nindices written: {INDICES_PATH}")
    print(f"ID range (sorted chosen positions): first={meta_a['ids'][0]} last={meta_a['ids'][-1]}")
    print("§7 CHECKS: all assertions held. Evidence ready for orchestrator verification.")
    print("GPU/Modal NOT touched.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run_cpu_preflight())
    except Exception as e:
        print(f"STOP: {type(e).__name__}: {e}", file=sys.stderr)
        raise
