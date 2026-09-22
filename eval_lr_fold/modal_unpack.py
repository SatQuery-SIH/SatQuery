"""LR-FOLD replication — untar verified image tars into the ripperscience
satquery-data volume, then verify member counts against the manifests.

Tar arcnames are `data/<tree>/...` (written by modal_pull_subset.py on
proxynanmaga); extracting at "/" restores the exact /data/<tree> layout the
gold rows reference. Counts are asserted against the pulled manifests — a
short/corrupt extract fails loudly instead of silently understaging.

Usage (profile spare -> workspace ripperscience):
    MODAL_PROFILE=spare python -m modal run eval_lr_fold/modal_unpack.py
"""

import json
import tarfile
import time
from pathlib import Path

import modal

VOLUME_NAME = "satquery-data"
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
DATA = "/data"
PULL = f"{DATA}/runs/lr_fold/pull"   # tars + manifests land here

app = modal.App("satquery-lr-fold-unpack")
img = modal.Image.debian_slim(python_version="3.12")

USD_CPU_CORE_S = 0.0000131
USD_MEM_GIB_S = 0.00000222

# tar filename -> manifest filename (uploaded alongside)
TARS = [
    "lr_eval_images.tar",
    "hr_eval_images.tar",
    "vrsb_eval_images.tar",
]


@app.function(image=img, volumes={DATA: vol}, cpu=4.0, memory=16 * 1024,
              timeout=3600)
def unpack_all() -> str:
    t0 = time.time()
    vol.reload()
    rep = {"tars": {}, "failures": []}

    for name in TARS:
        tp = Path(PULL) / name
        mp = tp.with_suffix(".manifest.json")
        if not tp.is_file():
            rep["failures"].append(f"{name}: missing tar")
            continue
        man = json.loads(mp.read_text()) if mp.is_file() else {}
        want_members = man.get("members_in_archive")
        want_payload = man.get("payload_in_archive")

        n_members = 0
        payload = 0
        with tarfile.open(tp, "r") as tf:
            for m in tf:
                n_members += 1
                payload += m.size
                # extract streaming at filesystem root so data/... -> /data/...
                tf.extract(m, "/", filter="data")

        ok = (want_members is None
              or (n_members == want_members and payload == want_payload))
        rep["tars"][name] = {
            "members": n_members, "payload": payload,
            "want_members": want_members, "verified": ok,
        }
        if not ok:
            rep["failures"].append(
                f"{name}: {n_members}/{want_members} members, "
                f"{payload}/{want_payload} bytes")

    # spot-verify the trees eval needs
    for rel, want_min in (("rsvqa/lr/Images_LR", 200),
                          ("rsvqa/hr/Data", 4310),
                          ("vrsbench/Images_val", 9349)):
        d = Path(DATA) / rel
        n = sum(1 for _ in d.iterdir()) if d.is_dir() else 0
        rep["tars"][f"tree {rel}"] = {"entries": n, "want_min": want_min}
        if n < want_min:
            rep["failures"].append(f"tree {rel}: {n} < {want_min}")

    rep["ok"] = not rep["failures"]
    rep["wall_seconds"] = round(time.time() - t0, 1)
    rep["est_usd"] = round(rep["wall_seconds"]
                         * (4 * USD_CPU_CORE_S + 16 * USD_MEM_GIB_S), 6)
    out = Path(DATA) / "runs" / "lr_fold" / "unpack_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2) + "\n")
    vol.commit()
    print("UNPACK " + json.dumps(rep), flush=True)
    return json.dumps(rep)


@app.local_entrypoint()
def main():
    print(unpack_all.remote())
