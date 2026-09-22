"""DATA-0M — fetch bulk dataset archives onto Modal Volume `satquery-data`.

Pixels live on Modal because evals and training run on Modal (MASTER_PLAN_V2 §5).
Laptop keeps only eval-id manifests + annotation JSONs.

Run per dataset (sequential; ops rule: no duplicate parallel runs of the same job):

    python -m modal run modal_fetch.py::fetch_rsvqa
    python -m modal run modal_fetch.py::fetch_vrsbench
    python -m modal run modal_fetch.py::fetch_cdvqa
    python -m modal run modal_fetch.py::fetch_ben
    python -m modal run modal_fetch.py::verify

CPU-only. No GPU functions, no model runs, no training.
Modal rules: profile `harsha-610vmg`, workspace `harsha-610vmg` only.
Every function logs wall-clock seconds + list-price USD estimate to stdout
(RUNSTATS line) and into /data/<name>/MANIFEST.json.

Layout produced on the volume:

    /data/rsvqa/hr/        RSVQA-HR (Zenodo 10.5281/zenodo.6344367)
    /data/rsvqa/lr/        RSVQA-LR (Zenodo 10.5281/zenodo.6344334)
    /data/rsvqa/MANIFEST.json
    /data/vrsbench/        HF xiang709/VRSBench full snapshot
    /data/vrsbench/MANIFEST.json
    /data/cdvqa/           CDVQA annotations (GitHub) + NEEDS_MANUAL.txt for images
    /data/cdvqa/MANIFEST.json
    /data/ben/             Copernicus-Bench BigEarthNet S1+S2 co-registered subset
    /data/ben/MANIFEST.json
"""

import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import modal

VOLUME_NAME = "satquery-data"
DATA_ROOT = "/data"
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

app = modal.App("satquery-data-fetch")

fetch_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "unzip", "ca-certificates", "aria2")
    .pip_install("huggingface_hub", "kaggle")
)

# CPU-only allocation. Modal on-demand list price (modal.com/pricing):
#   CPU    $0.0000131 / physical core / s   (cpu=N requests N physical cores)
#   Memory $0.00000222 / GiB / s
CPU_CORES = 2.0
MEM_GIB = 4.0
USD_PER_CORE_S = 0.0000131
USD_PER_GIB_S = 0.00000222


def _est_usd(seconds: float) -> float:
    rate = CPU_CORES * USD_PER_CORE_S + MEM_GIB * USD_PER_GIB_S
    return round(seconds * rate, 6)


def _runstats(t0: float) -> dict:
    secs = round(time.time() - t0, 1)
    stats = {
        "wall_seconds": secs,
        "est_usd": _est_usd(secs),
        "cpu_cores": CPU_CORES,
        "memory_gib": MEM_GIB,
        "pricing": "Modal on-demand list: $0.0000131/core-s + $0.00000222/GiB-s",
    }
    print("RUNSTATS " + json.dumps(stats), flush=True)
    return stats


def _sha256(path: Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_stats(root: Path) -> tuple[int, int]:
    count, total = 0, 0
    for dirpath, _, files in os.walk(root):
        for fn in files:
            count += 1
            total += (Path(dirpath) / fn).stat().st_size
    return count, total


def _curl(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "-fL", "--retry", "5", "--retry-delay", "5", "-o", str(dest), url],
        check=True,
    )


def _aria2(url: str, dest: Path) -> None:
    """Parallel-chunked download (Zenodo throttles single connections)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "aria2c", "-x", "16", "-s", "16", "-k", "16M", "-c",
            "--file-allocation=none", "--summary-interval=30",
            "--console-log-level=notice", "-d", str(dest.parent),
            "-o", dest.name, url,
        ],
        check=True,
    )


def _unzip(zip_path: Path, dest_dir: Path, delete: bool = True) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["unzip", "-o", "-q", str(zip_path), "-d", str(dest_dir)], check=True)
    if delete:
        zip_path.unlink()


def _retry(fn, what: str, attempts: int = 8):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - re-raised as RuntimeError below
            last = e
            wait = min(30 * (2**i), 300)
            print(f"[retry {i + 1}/{attempts}] {what}: {e} -> sleep {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"{what} failed after {attempts} attempts: {last}")


def _fetch(url: str, dest: Path, size: int | None = None) -> None:
    def once():
        if size is not None and dest.exists() and dest.stat().st_size == size:
            return  # already complete
        _aria2(url, dest)
        if size is not None and dest.stat().st_size != size:
            raise RuntimeError(
                f"size mismatch {dest.name}: {dest.stat().st_size} != {size}"
            )

    _retry(once, f"download {dest.name}")


def _zenodo_files(record_id: str) -> list[dict]:
    def once():
        with urllib.request.urlopen(
            f"https://zenodo.org/api/records/{record_id}", timeout=60
        ) as r:
            return json.load(r)

    rec = _retry(once, f"zenodo api {record_id}")
    return [
        {
            "key": f["key"],
            "size": f["size"],
            "url": f["links"]["self"],
            "license": rec["metadata"]["license"]["id"],
        }
        for f in rec["files"]
    ]


def _write_manifest(dirpath: Path, manifest: dict) -> Path:
    count, total = _tree_stats(dirpath)
    manifest["file_count"] = count
    manifest["total_bytes"] = total
    out = dirpath / "MANIFEST.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"MANIFEST {out} files={count} bytes={total}", flush=True)
    return out


def _fetch_rsvqa_cloud(part: str, cloud_url: str, dest_dir: Path, sources: list) -> None:
    """Fallback: author's Nextcloud whole-dataset zip (torchrs download URLs).

    Extracts, flattens the single RSVQA_* top dir into dest_dir, and unpacks
    any nested archives (LR keeps Images_LR.zip inside).
    """
    name = f"RSVQA_{part.upper()}.zip"
    marker = dest_dir / f".done_{name}"
    if marker.exists():
        print(f"[rsvqa/{part}] {name} already extracted, skip", flush=True)
    else:
        stage = Path("/tmp") / name
        _fetch(cloud_url, stage, None)
        extract_dir = Path("/tmp") / f"extract_{part}"
        subprocess.run(
            ["unzip", "-o", "-q", str(stage), "-d", str(extract_dir)], check=True
        )
        stage.unlink(missing_ok=True)
        tops = list(extract_dir.iterdir())
        src_root = tops[0] if len(tops) == 1 and tops[0].is_dir() else extract_dir
        subprocess.run(
            ["cp", "-a", str(src_root) + "/.", str(dest_dir) + "/"], check=True
        )
        for inner in dest_dir.rglob("*.zip"):
            _unzip(inner, inner.parent)
        marker.touch()
    sources.append(
        {
            "url": cloud_url,
            "file": name,
            "license": "cc-by-4.0",
            "part": part,
            "note": "author mirror (cloud.sylvainlobry.com, via torchrs scripts)",
        }
    )


@app.function(
    image=fetch_image,
    volumes={DATA_ROOT: vol},
    cpu=CPU_CORES,
    memory=int(MEM_GIB * 1024),
    ephemeral_disk=512 * 1024,
    timeout=4 * 3600,
    retries=2,
)
def fetch_rsvqa() -> dict:
    """RSVQA-HR (Zenodo 6344367) + RSVQA-LR (Zenodo 6344334) -> /data/rsvqa/."""
    t0 = time.time()
    out = Path(DATA_ROOT) / "rsvqa"
    # Per part: primary = Zenodo record; fallback = author's Nextcloud zip
    # (used by torchrs download scripts; independent of Zenodo).
    parts = {
        "hr": {
            "record": "6344367",
            "cloud": "https://cloud.sylvainlobry.com/s/f7NpYQKqx4bZStx/download",
        },
        "lr": {
            "record": "6344334",
            "cloud": "https://cloud.sylvainlobry.com/s/4Qg5AXX8YfCswmX/download",
        },
    }
    sources = []

    for part, cfg in parts.items():
        dest_dir = out / part
        dest_dir.mkdir(parents=True, exist_ok=True)
        if (dest_dir / ".done_kaggle").exists():
            print(f"[rsvqa/{part}] already populated via Kaggle mirror", flush=True)
            sources.append(
                {
                    "url": "https://www.kaggle.com/datasets/vishalravichandran/rsvqa-dataset",
                    "part": part,
                    "license": "cc-by-4.0",
                    "note": "fetched via Kaggle mirror — see lr/SOURCE_KAGGLE.txt",
                }
            )
            continue
        try:
            files = _zenodo_files(cfg["record"])
        except RuntimeError as e:
            files = None
            print(f"[rsvqa/{part}] zenodo listing failed: {e}", flush=True)

        if files is not None:
            try:
                for f in files:
                    key, url, size = f["key"], f["url"], f["size"]
                    print(f"[rsvqa/{part}] {key} ({size} B)", flush=True)
                    if key.lower().endswith((".tar", ".zip")):
                        marker = dest_dir / f".done_{key}"
                        if marker.exists():
                            print(
                                f"[rsvqa/{part}] {key} already extracted, skip",
                                flush=True,
                            )
                            continue
                        stage = Path("/tmp") / key
                        _fetch(url, stage, size)
                        if key.lower().endswith(".tar"):
                            subprocess.run(
                                ["tar", "-x", "-f", str(stage), "-C", str(dest_dir)],
                                check=True,
                            )
                        else:
                            _unzip(stage, dest_dir)
                        stage.unlink(missing_ok=True)
                        marker.touch()
                    else:
                        _fetch(url, dest_dir / key, size)
                    sources.append(
                        {
                            "url": f"https://doi.org/10.5281/zenodo.{cfg['record']}",
                            "file": key,
                            "license": f["license"],
                            "part": part,
                        }
                    )
            except RuntimeError as e:
                print(
                    f"[rsvqa/{part}] zenodo fetch failed ({e}); "
                    "falling back to author cloud",
                    flush=True,
                )
                _fetch_rsvqa_cloud(part, cfg["cloud"], dest_dir, sources)
        else:
            _fetch_rsvqa_cloud(part, cfg["cloud"], dest_dir, sources)

    primary = out / "hr" / "USGSanswers.json"
    manifest = {
        "dataset": "rsvqa",
        "path": "/data/rsvqa",
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "license": "CC-BY-4.0",
        "primary_index": {
            "path": str(primary.relative_to(out)),
            "sha256": _sha256(primary),
        },
        "notes": "hr/ = RSVQA-HR (USGS, Images.tar extracted); "
        "lr/ = RSVQA-LR (Sentinel-2, Images_LR.zip extracted).",
        "run": _runstats(t0),
    }
    mpath = _write_manifest(out, manifest)
    vol.commit()
    return {"manifest": str(mpath), **manifest["run"]}


@app.function(
    image=fetch_image,
    volumes={DATA_ROOT: vol},
    cpu=CPU_CORES,
    memory=int(MEM_GIB * 1024),
    ephemeral_disk=512 * 1024,
    timeout=1800,
    retries=2,
)
def fetch_rsvqa_lr_kaggle() -> dict:
    """RSVQA-LR via Kaggle mirror -> /data/rsvqa/lr/ (Zenodo-independent).

    Credentials: upload your kaggle.json once —
      python -m modal volume put satquery-data ~/.kaggle/kaggle.json _secrets/
    The Kaggle mirror carries LR only (Images_LR + LR_train/val/test splits);
    RSVQA-HR remains Zenodo-only.
    """
    t0 = time.time()
    kjson = Path(DATA_ROOT) / "_secrets" / "kaggle.json"
    if kjson.exists():
        creds = json.loads(kjson.read_text())
        os.environ["KAGGLE_USERNAME"] = creds["username"]
        os.environ["KAGGLE_KEY"] = creds["key"]
    elif not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        raise RuntimeError(
            "Kaggle creds missing: volume-put kaggle.json to "
            "/data/_secrets/ first (see docstring)"
        )
    out = Path(DATA_ROOT) / "rsvqa" / "lr"
    out.mkdir(parents=True, exist_ok=True)
    stage_dir = Path("/tmp/kdl")
    stage_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "kaggle", "datasets", "download", "-d",
            "vishalravichandran/rsvqa-dataset", "-p", str(stage_dir),
        ],
        check=True,
    )
    zips = list(stage_dir.glob("*.zip"))
    for z in zips:
        _unzip(z, out)
    (out / "SOURCE_KAGGLE.txt").write_text(
        "RSVQA-LR fetched via Kaggle mirror (Zenodo was unreachable).\n"
        "Source: https://www.kaggle.com/datasets/vishalravichandran/rsvqa-dataset\n"
        "Upstream: Zenodo 10.5281/zenodo.6344334 (Lobry et al., CC-BY-4.0)\n"
        f"Fetched: {datetime.now(timezone.utc).isoformat()}\n"
        "Layout: community-repackaged (Images_LR/ + LR_train|val|test dirs);\n"
        "official Zenodo file names may differ. Re-run fetch_rsvqa after\n"
        "Zenodo recovery to backfill canonical files if needed.\n"
    )
    (out / ".done_kaggle").touch()
    stats = _runstats(t0)
    vol.commit()
    return {"path": "/data/rsvqa/lr", **stats}


@app.function(
    image=fetch_image,
    volumes={DATA_ROOT: vol},
    cpu=CPU_CORES,
    memory=int(MEM_GIB * 1024),
    ephemeral_disk=512 * 1024,
    timeout=4 * 3600,
    retries=2,
)
def fetch_vrsbench() -> dict:
    """HF xiang709/VRSBench full snapshot -> /data/vrsbench/."""
    from huggingface_hub import hf_hub_download

    t0 = time.time()
    out = Path(DATA_ROOT) / "vrsbench"
    out.mkdir(parents=True, exist_ok=True)
    repo_files = [
        "Annotations_train.zip",
        "Annotations_val.zip",
        "Images_train.zip",
        "Images_val.zip",
        "VRSBench_train.json",
        "VRSBench_EVAL_vqa.json",
        "VRSBench_EVAL_Cap.json",
        "VRSBench_EVAL_referring.json",
    ]
    for name in repo_files:
        print(f"[vrsbench] {name}", flush=True)
        local = Path(
            hf_hub_download(
                repo_id="xiang709/VRSBench", repo_type="dataset", filename=name
            )
        )
        if name.endswith(".zip"):
            _unzip(local, out)
        else:
            subprocess.run(["cp", str(local), str(out / name)], check=True)

    primary = out / "VRSBench_train.json"
    manifest = {
        "dataset": "vrsbench",
        "path": "/data/vrsbench",
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [
            {
                "url": "https://huggingface.co/datasets/xiang709/VRSBench",
                "file": name,
                "license": "cc-by-4.0",
            }
            for name in repo_files
        ],
        "license": "CC-BY-4.0",
        "primary_index": {
            "path": primary.name,
            "sha256": _sha256(primary),
        },
        "notes": "Full HF snapshot: Images_*/Annotations_* zips extracted, "
        "train + eval JSONs at root.",
        "run": _runstats(t0),
    }
    mpath = _write_manifest(out, manifest)
    vol.commit()
    return {"manifest": str(mpath), **manifest["run"]}


@app.function(
    image=fetch_image,
    volumes={DATA_ROOT: vol},
    cpu=CPU_CORES,
    memory=int(MEM_GIB * 1024),
    ephemeral_disk=512 * 1024,
    timeout=3600,
    retries=2,
)
def fetch_vrsbench_eval() -> dict:
    """Scoped re-stage of the VRSBench eval split on a fresh workspace:
    Images_val.zip + VRSBench_EVAL_Cap.json only (the caption/VQA/referring
    eval images all live under Images_val). Eval JSONs are sha-pinned by the
    eval harness, so a drifted upstream file fails loudly there."""
    from huggingface_hub import hf_hub_download

    t0 = time.time()
    out = Path(DATA_ROOT) / "vrsbench"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("Images_val.zip", "VRSBench_EVAL_Cap.json"):
        print(f"[vrsbench-eval] {name}", flush=True)
        local = Path(
            hf_hub_download(
                repo_id="xiang709/VRSBench", repo_type="dataset", filename=name
            )
        )
        if name.endswith(".zip"):
            _unzip(local, out)
        else:
            subprocess.run(["cp", str(local), str(out / name)], check=True)

    n_img = sum(1 for _ in (out / "Images_val").rglob("*") if _.is_file())
    manifest = {
        "dataset": "vrsbench-eval-only",
        "path": "/data/vrsbench",
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [
            {
                "url": "https://huggingface.co/datasets/xiang709/VRSBench",
                "file": name,
                "license": "cc-by-4.0",
            }
            for name in ("Images_val.zip", "VRSBench_EVAL_Cap.json")
        ],
        "license": "CC-BY-4.0",
        "primary_index": {
            "path": "VRSBench_EVAL_Cap.json",
            "sha256": _sha256(out / "VRSBench_EVAL_Cap.json"),
        },
        "images_val_files": n_img,
        "notes": "Eval-split-only restage for the adapted-caption column; "
        "train zips intentionally skipped (not needed).",
        "run": _runstats(t0),
    }
    mpath = _write_manifest(out, manifest)
    vol.commit()
    return {"manifest": str(mpath), "images_val_files": n_img,
            **manifest["run"]}


@app.function(
    image=fetch_image,
    volumes={DATA_ROOT: vol},
    cpu=CPU_CORES,
    memory=int(MEM_GIB * 1024),
    timeout=1800,
    retries=2,
)
def fetch_cdvqa() -> dict:
    """CDVQA annotations (GitHub, Apache-2.0) -> /data/cdvqa/.

    CDVQA images are the SECOND dataset, released only via Google Drive /
    Baidu with interactive confirmation -> NEEDS_MANUAL.txt, skipped here.
    """
    t0 = time.time()
    out = Path(DATA_ROOT) / "cdvqa"
    ann = out / "annotations"
    ann.mkdir(parents=True, exist_ok=True)
    base = "https://raw.githubusercontent.com/YZHJessica/CDVQA/main"
    jsons = [
        f"{split}_{kind}.json"
        for split in ("Train", "Val", "Test", "Test2")
        for kind in ("questions", "answers", "images")
    ]
    for name in jsons:
        print(f"[cdvqa] {name}", flush=True)
        _curl(f"{base}/{name}", ann / name)

    (out / "NEEDS_MANUAL.txt").write_text(
        "CDVQA IMAGES — manual step required\n"
        "===================================\n\n"
        "The annotation JSONs (Train/Val/Test/Test2 questions/answers/images)\n"
        "are already downloaded to /data/cdvqa/annotations/ from the official\n"
        "release: https://github.com/YZHJessica/CDVQA (Apache-2.0).\n\n"
        "The CDVQA image pairs are the SECOND dataset images (2,968 bi-temporal\n"
        "pairs, 512x512). The authors distribute images only via interactive\n"
        "file hosts; automated fetch is not possible without auth:\n\n"
        "  1. Google Drive (official, linked from https://captain-whu.github.io/SCD/):\n"
        "       https://drive.google.com/file/d/1QlAdzrHpfBIOZ6SK78yHF2i1u6tikmBc/view\n"
        "     Browser-download the archive (Drive shows an interstitial confirm\n"
        "     page for large files), or use gdown locally:\n"
        "       pip install gdown\n"
        "       gdown 1QlAdzrHpfBIOZ6SK78yHF2i1u6tikmBc\n\n"
        "  2. Baidu Netdisk mirror (requires Baidu account):\n"
        "       https://pan.baidu.com/s/1-zTu1TJhf3gjBmmPbcvk7A  (pwd: rsai)\n\n"
        "After download: extract so that /data/cdvqa/images/ holds the SECOND\n"
        "train pairs (im1/im2 layout as released), then re-run\n"
        "  python -m modal run modal_fetch.py::verify\n"
        "to refresh counts. SECOND is released for research use; the CDVQA\n"
        "annotations already fetched here are Apache-2.0.\n"
    )

    primary = ann / "Train_questions.json"
    manifest = {
        "dataset": "cdvqa",
        "path": "/data/cdvqa",
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [
            {
                "url": f"{base}/{name}",
                "file": name,
                "license": "apache-2.0",
            }
            for name in jsons
        ],
        "license": "Apache-2.0 (annotations); SECOND images: research use, see NEEDS_MANUAL.txt",
        "primary_index": {
            "path": str(primary.relative_to(out)),
            "sha256": _sha256(primary),
        },
        "status": "partial — annotations fetched; images pending manual Drive/Baidu step (NEEDS_MANUAL.txt)",
        "run": _runstats(t0),
    }
    mpath = _write_manifest(out, manifest)
    vol.commit()
    return {"manifest": str(mpath), **manifest["run"]}


@app.function(
    image=fetch_image,
    volumes={DATA_ROOT: vol},
    cpu=CPU_CORES,
    memory=int(MEM_GIB * 1024),
    ephemeral_disk=512 * 1024,
    timeout=3600,
    retries=2,
)
def fetch_ben() -> dict:
    """BigEarthNet S1+S2 co-registered subset (Copernicus-Bench, ~24k pairs,
    ~4 GB) -> /data/ben/. NOT the full 118 GB archive."""
    from huggingface_hub import hf_hub_download

    t0 = time.time()
    out = Path(DATA_ROOT) / "ben"
    out.mkdir(parents=True, exist_ok=True)
    repo = "wangyi111/Copernicus-Bench"
    prefix = "l2_bigearthnet_s1s2/"

    print("[ben] bigearthnetv2.zip", flush=True)
    zpath = Path(
        hf_hub_download(
            repo_id=repo, repo_type="dataset", filename=prefix + "bigearthnetv2.zip"
        )
    )
    _unzip(zpath, out)
    for name in ("metadata-5%.parquet", "metadata-10%.parquet"):
        print(f"[ben] {name}", flush=True)
        local = Path(
            hf_hub_download(repo_id=repo, repo_type="dataset", filename=prefix + name)
        )
        subprocess.run(["cp", str(local), str(out / name)], check=True)

    primary = out / "metadata-5%.parquet"
    manifest = {
        "dataset": "bigearthnet-s1s2-subset",
        "path": "/data/ben",
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [
            {
                "url": f"https://huggingface.co/datasets/{repo}/blob/main/{prefix}{n}",
                "file": n,
            }
            for n in ("bigearthnetv2.zip", "metadata-5%.parquet", "metadata-10%.parquet")
        ],
        "license": "CDLA-Permissive-1.0",
        "primary_index": {
            "path": primary.name,
            "sha256": _sha256(primary),
        },
        "notes": "Copernicus-Bench L2 BigEarthNet subset: 24,002 co-registered "
        "S1 GRD + S2 SR patch pairs (120x120, 19-class multilabel), "
        "train/val/test = 11894/6117/5991. Subset of BigEarthNet v2.0.",
        "run": _runstats(t0),
    }
    mpath = _write_manifest(out, manifest)
    vol.commit()
    return {"manifest": str(mpath), **manifest["run"]}


def _check_ids(root: Path, ids: list, patterns: list[str]) -> dict:
    """Check each id resolves under root via any pattern (first match wins)."""
    ok, missing = 0, []
    for i in ids:
        if any((root / p.format(i)).exists() for p in patterns):
            ok += 1
        elif len(missing) < 10:
            missing.append(str(i))
    return {"referenced": len(ids), "resolved": ok, "pct": round(100 * ok / max(len(ids), 1), 2),
            "missing_sample": missing}


@app.function(image=fetch_image, volumes={DATA_ROOT: vol}, cpu=1.0, memory=2048, timeout=1800)
def verify_deep() -> dict:
    """Integrity check: every annotation-referenced image exists on the volume."""
    report = {}

    # RSVQA-HR: USGSimages.json ids -> Data/<id>.tif and .png
    hr = Path(DATA_ROOT) / "rsvqa" / "hr"
    imgs = json.loads((hr / "USGSimages.json").read_text())["images"]
    ids = [im["id"] for im in imgs]
    report["rsvqa_hr_tif"] = _check_ids(hr / "Data", ids, ["{}.tif"])
    report["rsvqa_hr_png"] = _check_ids(hr / "Data", ids, ["{}.png"])

    # RSVQA-LR: image ids across split files -> Images_LR/<id>.png
    lr = Path(DATA_ROOT) / "rsvqa" / "lr"
    lr_ids = set()
    for jf in lr.glob("LR_split_*_images.json"):
        for im in json.loads(jf.read_text())["images"]:
            lr_ids.add(str(im["id"]))
    report["rsvqa_lr"] = _check_ids(lr / "Images_LR", sorted(lr_ids), ["{}.png", "{}.tif"])

    # VRSBench: image refs in train + eval jsons -> Images_train|Images_val
    vb = Path(DATA_ROOT) / "vrsbench"
    vb_refs = set()

    def _collect_refs(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                _collect_refs(v)
        elif isinstance(obj, list):
            for v in obj:
                _collect_refs(v)
        elif isinstance(obj, str):
            if obj.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                vb_refs.add(obj.split("/")[-1])

    for jf in vb.glob("*.json"):
        _collect_refs(json.loads(jf.read_text()))
    report["vrsbench_images"] = _check_ids(
        vb, sorted(vb_refs), ["Images_train/{}", "Images_val/{}"]
    )
    report["vrsbench_counts"] = {
        "Images_train_files": sum(1 for _ in (vb / "Images_train").rglob("*") if _.is_file()),
        "Images_val_files": sum(1 for _ in (vb / "Images_val").rglob("*") if _.is_file()),
    }

    # BEN subset: split csvs -> per-modality path columns under S1-5%/S2-5%
    ben = Path(DATA_ROOT) / "ben" / "bigearthnet_s1s2"
    import csv as _csv
    s1_ids, s2_ids = set(), set()
    for cf in ben.glob("multilabel-*.csv"):
        with open(cf) as fh:
            reader = _csv.DictReader(fh)
            cols = reader.fieldnames or []
            s1_col = next((c for c in cols if "s1" in c.lower()), None)
            s2_col = next((c for c in cols if "s2" in c.lower()), None)
            fallback = cols[0] if cols else None
            for row in reader:
                s1_ids.add((row.get(s1_col) or row.get(fallback) or "").strip())
                s2_ids.add((row.get(s2_col) or row.get(fallback) or "").strip())
    report["ben_columns"] = {"s1_col": s1_col, "s2_col": s2_col, "fallback": fallback}
    report["ben_s1"] = _check_ids(ben / "BigEarthNet-S1-5%", sorted(s1_ids), ["{}"])
    report["ben_s2"] = _check_ids(ben / "BigEarthNet-S2-5%", sorted(s2_ids), ["{}"])

    # CDVQA: annotation file_names -> images/{im1,im2}/<file>
    cd = Path(DATA_ROOT) / "cdvqa"
    cd_ids = set()
    for jf in (cd / "annotations").glob("*_images.json"):
        data = json.loads(jf.read_text())
        for im in (data["images"] if isinstance(data, dict) and "images" in data else data):
            cd_ids.add(str(im.get("file_name") or im.get("filename") or im.get("id")))
    report["cdvqa_im1"] = _check_ids(cd / "images" / "im1", sorted(cd_ids), ["{}"])
    report["cdvqa_im2"] = _check_ids(cd / "images" / "im2", sorted(cd_ids), ["{}"])

    for k, v in report.items():
        print("DEEP", k, json.dumps(v), flush=True)
    return report


@app.function(image=fetch_image, volumes={DATA_ROOT: vol}, cpu=1.0, memory=1024, timeout=900)
def verify() -> dict:
    """Read-only: print per-dataset stats and every MANIFEST.json."""
    root = Path(DATA_ROOT)
    summary = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        count, total = _tree_stats(child)
        summary[child.name] = {"files": count, "bytes": total}
        mpath = child / "MANIFEST.json"
        if mpath.exists():
            print(f"===== {mpath} =====", flush=True)
            print(mpath.read_text(), flush=True)
    print("VERIFY " + json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    print("Run via: python -m modal run modal_fetch.py::<function>", file=sys.stderr)
    sys.exit(2)
