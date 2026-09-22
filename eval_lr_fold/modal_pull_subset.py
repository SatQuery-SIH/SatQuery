"""LR-FOLD staging pull — tar a file-subset of satquery-data into one tarball.

CPU-only helper on account proxynanmaga (the spec's eval-side account; its
volume holds every dataset tree). Per-file `modal volume get` over ~21k
images is unworkable; a single tarball is one transfer.

Usage:
    python -m modal run eval_lr_fold/modal_pull_subset.py::e_tar \
        (env LIST=/data/runs/lr_fold/pull/lists/<name>.txt
              OUT=/data/runs/lr_fold/pull/<name>.tar)

The list file is uploaded beforehand via `modal volume put`. Its lines are
volume-absolute paths (/data/...). Missing entries are reported, never
invented. Writes a small JSON manifest next to the tarball.
"""

import json
import os
import tarfile
import time
from pathlib import Path

import modal

VOLUME_NAME = "satquery-data"
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
DATA = "/data"

app = modal.App("satquery-lr-fold-pull")
img = modal.Image.debian_slim(python_version="3.12")

USD_CPU_CORE_S = 0.0000131
USD_MEM_GIB_S = 0.00000222


@app.function(image=img, volumes={DATA: vol}, cpu=2.0, memory=8 * 1024,
              timeout=3600)
def tar_subset(list_remote: str, out_remote: str) -> str:
    t0 = time.time()
    vol.reload()
    list_path = Path(list_remote)
    out_path = Path(out_remote)
    rel = [l.strip() for l in list_path.read_text().splitlines() if l.strip()]

    # map /data/x -> Path(DATA)/x ; verify membership
    missing, members = [], []
    for p in rel:
        lp = Path(p) if not p.startswith("/data") else Path(DATA) / p[len("/data"):].lstrip("/")
        if lp.is_file():
            members.append(lp)
        else:
            missing.append(p)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_bytes = 0
    with tarfile.open(out_path, "w") as tf:
        for lp in members:
            # arcname keeps the tree under data/ so untar on the studio lands
            # the same relative layout the mix uses
            arc = str(lp).lstrip("/")
            tf.add(lp, arcname=arc)
            n_bytes += lp.stat().st_size

    # VERIFY: re-open the tar on the volume and count members + payload.
    # A closed tar with fewer members than requested is corrupt (observed
    # once: 3,833/14,999 members readable, trailing 4.7GB garbage — likely a
    # zombie writer from a killed client racing the same out path). Never
    # commit a manifest that claims more than the archive actually holds.
    n_members = 0
    payload_read = 0
    with tarfile.open(out_path, "r") as tf:
        for m in tf:
            n_members += 1
            payload_read += m.size

    man = {
        "list": str(list_remote), "out": str(out_remote),
        "listed": len(rel), "tarred": len(members), "missing": len(missing),
        "missing_sample": missing[:20],
        "payload_bytes": n_bytes,
        "members_in_archive": n_members,
        "payload_in_archive": payload_read,
        "tar_bytes": out_path.stat().st_size,
        "wall_seconds": round(time.time() - t0, 1),
        "est_usd": round((time.time() - t0) * (2 * USD_CPU_CORE_S + 8 * USD_MEM_GIB_S), 6),
    }
    man["verified"] = (n_members == len(members)
                       and payload_read == n_bytes)
    mpath = out_path.with_suffix(".manifest.json")
    mpath.write_text(json.dumps(man, indent=2) + "\n")
    vol.commit()
    print("TAR_STATS " + json.dumps(man), flush=True)
    if not man["verified"]:
        raise RuntimeError(
            f"tar verification failed: {n_members}/{len(members)} members, "
            f"{payload_read}/{n_bytes} bytes — archive corrupt, see {mpath}")
    return json.dumps(man)


@app.local_entrypoint()
def e_tar():
    list_remote = os.environ["LIST"]
    out_remote = os.environ["OUT"]
    print(tar_subset.remote(list_remote, out_remote))
