"""RASTER-HUNT-13 Modal rasters. No LoRA.

Owner GPU job. Do not re-run from a GitHub clone.

Stages (cwd SatQuery/):
  python -m modal run hunt/raster_hunt_modal.py --stage vrs
  python -m modal run hunt/raster_hunt_modal.py --stage second
  python -m modal run hunt/raster_hunt_modal.py --stage ben
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # SatQuery/
INPUTS_LOCAL = ROOT / "gates" / "_cache" / "prod10" / "raster_hunt_inputs.json"
BEN_MAP_LOCAL = ROOT / "gates" / "_cache" / "prod10" / "ben_map.json"
DATA_PREP_LOCAL = ROOT / "gates" / "data_prep.py"

VRS_VOLUME = "satquery-vrsbench"
PNG_VOLUME = "satquery-prod10-png"
HF_CACHE_VOLUME = "satquery-hf-cache"
DRIVE_HREFS = [
    "https://drive.google.com/file/d/1QlAdzrHpfBIOZ6SK78yHF2i1u6tikmBc/view?usp=sharing",
    "https://drive.google.com/file/d/1mN8jzCKKK27p3ODGoDgepjiRYGQpB34u/view?usp=sharing",
]
DRIVE_IDS = [
    "1QlAdzrHpfBIOZ6SK78yHF2i1u6tikmBc",
    "1mN8jzCKKK27p3ODGoDgepjiRYGQpB34u",
]

vrs_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub", "hf-transfer", "pillow")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/vrs_cache"})
)

second_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("unzip")
    .pip_install("gdown", "pillow")
)

unpack_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("unzip", "unar")
    .pip_install("pillow")
)

ben_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "lmdb",
        "safetensors",
        "pillow",
        "numpy",
        "pandas",
        "pyarrow",
        "huggingface_hub",
        "hf-transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/ben_cache"})
    .add_local_file(str(DATA_PREP_LOCAL), remote_path="/root/data_prep.py")
)

app = modal.App("satquery-raster-hunt-13")
vrs_vol = modal.Volume.from_name(VRS_VOLUME, create_if_missing=True)
png_vol = modal.Volume.from_name(PNG_VOLUME, create_if_missing=True)
hf_vol = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


def _zip_index(zf) -> dict[str, str]:
    out = {}
    for name in zf.namelist():
        if name.endswith("/") or "/__MACOSX/" in name or name.startswith("__MACOSX"):
            continue
        base = name.replace("\\", "/").split("/")[-1]
        if base.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            out[base] = name
    return out


@app.function(
    image=vrs_image,
    timeout=4 * 60 * 60,
    memory=8192,
    cpu=4,
    volumes={"/vrs_cache": vrs_vol},
)
def fetch_vrs(spot_basenames: list[str]) -> dict:
    import zipfile
    from huggingface_hub import hf_hub_download

    print("hf_hub_download xiang709/VRSBench Images_train.zip ...", flush=True)
    zip_path = Path(
        hf_hub_download(
            repo_id="xiang709/VRSBench",
            filename="Images_train.zip",
            repo_type="dataset",
            cache_dir="/vrs_cache",
        )
    )
    zip_bytes = zip_path.stat().st_size
    print(f"zip={zip_path} bytes={zip_bytes}", flush=True)
    vrs_vol.commit()

    missing = []
    hits = []
    n_members = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        idx = _zip_index(zf)
        n_members = len(idx)
        print(f"zip_png_members={n_members}", flush=True)
        for base in spot_basenames:
            member = idx.get(base)
            if member is None:
                missing.append(base)
            else:
                hits.append({"basename": base, "member": member})
    vrs_vol.commit()
    return {
        "ok": len(missing) == 0,
        "filename": "Images_train.zip",
        "repo": "xiang709/VRSBench",
        "license": "CC-BY-4.0",
        "zip_path": str(zip_path),
        "zip_bytes": zip_bytes,
        "n_members": n_members,
        "volume": VRS_VOLUME,
        "spot_n": len(spot_basenames),
        "spot_hits": hits,
        "spot_missing": missing,
        "spot_check": "pass" if not missing else "fail",
        "did_not_download_to_laptop": True,
    }


@app.function(
    image=second_image,
    timeout=4 * 60 * 60,
    memory=8192,
    cpu=2,
    volumes={"/png": png_vol},
)
def fetch_second(wanted_basenames: list[str]) -> dict:
    import zipfile

    import gdown

    out_dir = Path("/png/second_dl")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    any_match = 0
    blocked_reason = None
    for did in DRIVE_IDS:
        dest = out_dir / f"{did}.bin"
        url = f"https://drive.google.com/uc?id={did}"
        row = {"id": did, "url": url, "ok": False}
        try:
            print(f"gdown {did} ...", flush=True)
            gdown.download(id=did, output=str(dest), quiet=False)
            if not dest.exists() or dest.stat().st_size < 1000:
                row["error"] = "empty_or_html_confirm_page"
                row["bytes"] = dest.stat().st_size if dest.exists() else 0
                results.append(row)
                continue
            row["bytes"] = dest.stat().st_size
            row["ok"] = True
            names = []
            try:
                with zipfile.ZipFile(dest, "r") as zf:
                    idx = _zip_index(zf)
                    names = sorted(idx)
                    hit = sorted(set(wanted_basenames) & set(idx))
                    row["n_zip_images"] = len(idx)
                    row["n_exact_basename_hit"] = len(hit)
                    row["sample_zip"] = names[:8]
                    row["sample_hit"] = hit[:8]
                    any_match = max(any_match, len(hit))
            except zipfile.BadZipFile:
                row["not_zip"] = True
                row["head_hex"] = dest.read_bytes()[:32].hex()
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
        results.append(row)
        png_vol.commit()

    wanted = set(wanted_basenames)
    if any_match == 0:
        blocked_reason = (
            "Drive download failed, not a zip, or zip stems do not exactly match "
            "CDVQA file_name (e.g. 10589.png / 00003.png). No invented mapping."
        )
    return {
        "drive_hrefs": DRIVE_HREFS,
        "drive_ids": DRIVE_IDS,
        "files": results,
        "wanted_n": len(wanted),
        "best_exact_hit": any_match,
        "blocked": any_match == 0,
        "blocked_reason": blocked_reason,
        "volume": PNG_VOLUME,
        "cdvqa_eval_overlap_not_copied": True,
    }


def _find_lmdb(root: Path) -> Path:
    nested = list(root.rglob("data.mdb"))
    if not nested:
        raise FileNotFoundError(f"No data.mdb under {root}")
    for p in nested:
        if "BENv2" in p.parent.name or p.parent.name.endswith(".lmdb"):
            return p.parent
    return nested[0].parent


@app.function(
    image=ben_image,
    timeout=12 * 60 * 60,
    memory=32768,
    cpu=4,
    volumes={"/png": png_vol, "/ben_cache": hf_vol},
)
def rematerialize_ben(ben_rows: list[dict]) -> dict:
    import sys

    import lmdb
    import numpy as np
    from huggingface_hub import snapshot_download
    from PIL import Image
    from safetensors.numpy import load as safetensor_load

    sys.path.insert(0, "/root")
    from data_prep import load_s2_rgb, clamp_long_edge  # noqa: E402

    print("snapshot_download hackelle/BigEarthNetV2-LMDB ...", flush=True)
    img_root = Path(
        snapshot_download(
            repo_id="hackelle/BigEarthNetV2-LMDB",
            repo_type="dataset",
            cache_dir="/ben_cache",
        )
    )
    lmdb_path = _find_lmdb(img_root)
    data_mdb = lmdb_path / "data.mdb"
    print(f"lmdb={lmdb_path} data.mdb={data_mdb.stat().st_size if data_mdb.exists() else None}", flush=True)
    hf_vol.commit()

    s1_by_pid: dict[str, str] = {}
    try:
        import pandas as pd

        for pq in img_root.rglob("*.parquet"):
            try:
                df = pd.read_parquet(pq)
            except Exception as e:
                print(f"skip parquet {pq}: {type(e).__name__}: {e}", flush=True)
                continue
            cols = {str(c).lower(): c for c in df.columns}
            pid_col = cols.get("patch_id")
            s1_col = cols.get("s1_name") or cols.get("s1_id") or cols.get("s1")
            if not pid_col or not s1_col:
                continue
            for a, b in zip(df[pid_col].astype(str), df[s1_col].astype(str)):
                if b and b not in {"nan", "None"}:
                    s1_by_pid[str(a)] = str(b)
            print(f"s1_map from {pq.name} n={len(s1_by_pid)}", flush=True)
            if s1_by_pid:
                break
    except Exception as e:
        print(f"s1_map_failed {type(e).__name__}: {e}", flush=True)

    s2_dir = Path("/png/png/ben_s2")
    s1_dir = Path("/png/png/ben_s1")
    s2_dir.mkdir(parents=True, exist_ok=True)
    s1_dir.mkdir(parents=True, exist_ok=True)

    map_size = max((data_mdb.stat().st_size if data_mdb.exists() else 0) + 1024**3, 200 * 1024**3)
    env = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        meminit=False,
        readahead=True,
        map_size=map_size,
    )

    def vv_to_preview(vv) -> Image.Image:
        x = np.asarray(vv, dtype=np.float32)
        if x.ndim == 3:
            x = x[0] if x.shape[0] == 1 else x[..., 0]
        finite = x[np.isfinite(x)]
        lo, hi = (np.percentile(finite, (2, 98)) if finite.size else (0.0, 1.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0
        gray = (np.clip((x - lo) / (hi - lo), 0, 1) * 255.0).astype(np.uint8)
        return clamp_long_edge(Image.fromarray(gray, mode="L").convert("RGB"), 512)

    n_s2 = 0
    n_s1 = 0
    missing_s2 = []
    missing_s1 = []
    try:
        for i, rec in enumerate(ben_rows, start=1):
            idx = int(rec["i"])
            pid = str(rec["patch_id"])
            dest2 = s2_dir / f"{idx:05d}.png"
            dest1 = s1_dir / f"{idx:05d}.png"
            try:
                if not (dest2.exists() and dest2.stat().st_size > 100):
                    im = load_s2_rgb(env, pid)
                    im.save(dest2, format="PNG")
                n_s2 += 1
            except Exception as e:
                missing_s2.append({"i": idx, "patch_id": pid, "err": f"{type(e).__name__}: {e}"})
            # S1: exact LMDB key from rec if provided; else skip (no invented S2→S1 rename).
            s1_key = rec.get("s1_name") or s1_by_pid.get(pid)
            if not s1_key:
                missing_s1.append({"i": idx, "patch_id": pid, "err": "no s1_name in pack map"})
            else:
                try:
                    with env.begin(write=False, buffers=True) as txn:
                        raw = txn.get(str(s1_key).encode("utf-8"))
                    if raw is None:
                        raise KeyError(f"s1_name not in LMDB: {s1_key}")
                    tensor = safetensor_load(bytes(raw))
                    if "VV" not in tensor:
                        raise KeyError(f"no VV in {s1_key} keys={list(tensor)}")
                    vv_to_preview(tensor["VV"]).save(dest1, format="PNG")
                    n_s1 += 1
                except Exception as e:
                    missing_s1.append({"i": idx, "patch_id": pid, "s1_name": s1_key, "err": f"{type(e).__name__}: {e}"})
            if i % 200 == 0:
                print(f"ben {i}/{len(ben_rows)} s2={n_s2} s1={n_s1}", flush=True)
                png_vol.commit()
    finally:
        env.close()
    png_vol.commit()
    hf_vol.commit()
    s1_blocked = n_s1 == 0
    return {
        "img_repo": "hackelle/BigEarthNetV2-LMDB",
        "license": "CDLA-Permissive",
        "lmdb": str(lmdb_path),
        "data_mdb_bytes": data_mdb.stat().st_size if data_mdb.exists() else None,
        "n_requested": len(ben_rows),
        "n_s2": n_s2,
        "n_s1": n_s1,
        "missing_s2_n": len(missing_s2),
        "missing_s1_n": len(missing_s1),
        "missing_s2_head": missing_s2[:5],
        "missing_s1_head": missing_s1[:5],
        "ben_s1_blocked": s1_blocked,
        "four_band_cube": False,
        "volume": PNG_VOLUME,
        "where_s2": "/png/png/ben_s2",
        "where_s1": "/png/png/ben_s1",
    }


@app.function(
    image=second_image,
    timeout=30 * 60,
    memory=4096,
    volumes={"/png": png_vol},
)
def inspect_second(wanted_basenames: list[str]) -> dict:
    from collections import Counter

    root = Path("/png/second_dl")
    wanted = set(wanted_basenames)
    files = []
    if not root.is_dir():
        return {"error": "second_dl missing", "volume": PNG_VOLUME}
    for p in sorted(root.iterdir()):
        if not p.is_file():
            continue
        head = p.read_bytes()[:16]
        rec = {
            "name": p.name,
            "bytes": p.stat().st_size,
            "head_hex": head.hex(),
            "kind": (
                "rar" if head[:4] == b"Rar!"
                else "zip" if head[:2] == b"PK"
                else "other"
            ),
        }
        if rec["kind"] == "zip":
            import zipfile

            with zipfile.ZipFile(p, "r") as zf:
                names = zf.namelist()
            rec["n_members"] = len(names)
            rec["sample"] = names[:40]
            rec["ext"] = dict(Counter(Path(n).suffix.lower() or "[dir]" for n in names))
            bases = {Path(n).name for n in names if not n.endswith("/")}
            rec["n_basenames"] = len(bases)
            rec["exact_hit"] = len(bases & wanted)
            rec["has_10589_png"] = "10589.png" in bases
            rec["has_00003_png"] = "00003.png" in bases
            rec["nested_zip_rar"] = [
                n for n in names if n.lower().endswith((".zip", ".rar", ".7z", ".tar", ".gz"))
            ][:20]
        files.append(rec)
    return {"volume": PNG_VOLUME, "files": files, "wanted_n": len(wanted)}


@app.function(
    image=unpack_image,
    timeout=2 * 60 * 60,
    memory=8192,
    cpu=2,
    volumes={"/png": png_vol},
)
def unpack_second(wanted_basenames: list[str], eval_basenames: list[str]) -> dict:
    import shutil
    import subprocess
    import zipfile
    from collections import Counter

    wanted = set(wanted_basenames)
    eval_set = set(eval_basenames)
    work = Path("/png/second_unpack")
    work.mkdir(parents=True, exist_ok=True)
    outer = Path("/png/second_dl/1mN8jzCKKK27p3ODGoDgepjiRYGQpB34u.bin")
    rar_direct = Path("/png/second_dl/1QlAdzrHpfBIOZ6SK78yHF2i1u6tikmBc.bin")
    if not outer.is_file():
        return {"blocked": True, "blocked_reason": f"outer zip missing {outer}"}

    nested_dir = work / "nested"
    nested_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(outer, "r") as zf:
        zf.extractall(nested_dir)
    nested_names = sorted(p.name for p in nested_dir.rglob("*") if p.is_file())

    train_rar = next((p for p in nested_dir.rglob("*") if p.name == "SECOND_train_set.rar"), None)
    test_zip = next((p for p in nested_dir.rglob("*") if p.name == "SECOND_total_test.zip"), None)
    if train_rar is None and rar_direct.is_file():
        train_rar = rar_direct

    train_out = work / "train"
    train_out.mkdir(parents=True, exist_ok=True)
    unar_cmd = None
    unar_rc = None
    unar_tail = None
    already = list(train_out.rglob("*.png"))
    if len(already) >= 100:
        unar_tail = f"skip unar; already {len(already)} png under {train_out}"
    elif train_rar is not None:
        unar_cmd = ["unar", "-force-overwrite", "-o", str(train_out), str(train_rar)]
        proc = subprocess.run(unar_cmd, capture_output=True, text=True, timeout=3600)
        unar_rc = proc.returncode
        unar_tail = ((proc.stdout or "") + "\n" + (proc.stderr or ""))[-2000:]
        # rc!=0 can still mean partial extract (corrupt members). Scan whatever landed.

    pngs = [p for p in train_out.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}]
    bases = Counter(p.name for p in pngs)
    parents = Counter(p.parent.name for p in pngs)
    hit = sorted(set(bases) & wanted)
    miss = sorted(wanted - set(bases))
    eval_in_train = sorted(set(bases) & eval_set)

    dest_root = Path("/png/png/second")
    copied = 0
    skipped_eval = 0
    layout_note = "im1->A, im2->B, else parent name A/B"
    if hit:
        dest_root.mkdir(parents=True, exist_ok=True)
        for p in pngs:
            base = p.name
            if base not in wanted:
                continue
            if base in eval_set:
                skipped_eval += 1
                continue
            parent = p.parent.name.lower()
            if parent in {"im1", "a", "t1", "before"}:
                side = "A"
            elif parent in {"im2", "b", "t2", "after"}:
                side = "B"
            else:
                side = p.parent.name
            d = dest_root / side / base
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, d)
            copied += 1

    test_info = None
    if test_zip is not None:
        with zipfile.ZipFile(test_zip, "r") as zf:
            tnames = zf.namelist()
        tbases = {Path(n).name for n in tnames if not n.endswith("/")}
        test_info = {
            "n_members": len(tnames),
            "sample": tnames[:20],
            "exact_hit_vs_train_wanted": len(tbases & wanted),
            "exact_hit_vs_eval": len(tbases & eval_set),
            "has_10589_png": "10589.png" in tbases,
            "has_00003_png": "00003.png" in tbases,
            "copied_into_pack": False,
        }

    nA = len(list((dest_root / "A").glob("*"))) if (dest_root / "A").is_dir() else 0
    nB = len(list((dest_root / "B").glob("*"))) if (dest_root / "B").is_dir() else 0
    png_vol.commit()
    blocked = len(hit) == 0
    return {
        "nested_names": nested_names,
        "unar_rc": unar_rc,
        "unar_tail": unar_tail,
        "n_train_pngs": len(pngs),
        "train_parent_counts": dict(parents),
        "n_unique_train_basenames": len(bases),
        "exact_hit": len(hit),
        "missing_n": len(miss),
        "missing_head": miss[:20],
        "eval_names_in_train_tree": eval_in_train[:20],
        "eval_in_train_n": len(eval_in_train),
        "copied": copied,
        "skipped_eval": skipped_eval,
        "n_dest_A": nA,
        "n_dest_B": nB,
        "layout_note": layout_note,
        "test": test_info,
        "blocked": blocked,
        "blocked_reason": None if not blocked else "train rar/zip stems do not exactly match CDVQA file_name. No invented mapping.",
        "cdvqa_eval_overlap_copied": skipped_eval,
        "volume": PNG_VOLUME,
        "where": "/png/png/second",
    }


@app.local_entrypoint()
def main(stage: str = "vrs"):
    inp = json.loads(INPUTS_LOCAL.read_text(encoding="utf-8"))
    if stage == "vrs":
        out = fetch_vrs.remote(inp["vrs_spot_basenames"])
        print(json.dumps(out, indent=2))
        return
    if stage == "second":
        out = fetch_second.remote(inp["second_basenames"])
        print(json.dumps(out, indent=2))
        return
    if stage == "inspect_second":
        out = inspect_second.remote(inp["second_basenames"])
        print(json.dumps(out, indent=2))
        return
    if stage == "unpack_second":
        eval_path = ROOT / "gates" / "cdvqa_eval_ids.json"
        eval_ids = json.loads(eval_path.read_text(encoding="utf-8"))
        eval_basenames = list(eval_ids.get("pair_ids_test_union") or [])
        out = unpack_second.remote(inp["second_basenames"], eval_basenames)
        print(json.dumps(out, indent=2))
        return
    if stage == "ben":
        if not BEN_MAP_LOCAL.is_file():
            raise SystemExit(f"STOP: {BEN_MAP_LOCAL} missing — build ben_map.json first")
        rows = json.loads(BEN_MAP_LOCAL.read_text(encoding="utf-8"))
        out = rematerialize_ben.remote(rows)
        print(json.dumps(out, indent=2))
        return
    raise SystemExit(f"unknown stage {stage}")
