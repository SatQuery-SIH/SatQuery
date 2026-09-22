"""LR-FOLD resume on Modal — Lightning died at ~step 3,400 (credits); local
ckpt_last is opt_step 3,000. This app resumes the SAME run bit-continuatively:

  * same trainer file (eval_lr_fold/lightning_lr_fold.py — env-driven, no
    lightning_sdk imports) uploaded to the volume
  * same ckpt_last bundle (adapter + merger.pt + opt.pt + step.json) on the
    volume at runs/lr_fold/ckpt_last — resume auto-detects step.json and
    restores weights + optimizer + scheduler position (proven machinery)
  * same mix (runs/lr_fold/mix_lr_fold.jsonl) — deterministic SEED=42 order,
    so the resume lands exactly mid-epoch-2 at the right data offset
  * same image trees, all already on satquery-data: rsvqa/lr/Images_LR,
    rsvqa/hr/Data, vrsbench/Images_train — bridged into the mix's
    rsvqa_lr/rsvqa_hr/vrsbench layout via symlinks in the container
  * same deps pinned to the studio's installed versions (torch 2.8.0+cu128,
    transformers 4.57.6, peft 0.18.1, accelerate 1.15.0, safetensors 0.8.0)
  * GPU L40S (same device class the run trained on)

RUN_DIR = /data/runs/lr_fold (ON the volume): STATUS.json / train_log.jsonl /
ckpt_last / ckpt_final land durably by construction — no laptop sync needed
for crash safety; a commit thread flushes every 5 min. The evaluator already
expects ckpt_final at runs/lr_fold/ckpt_final.

Usage (profile proxynanmaga):
    modal run eval_lr_fold/modal_train_resume.py::preflight   # verify world
    modal run -d eval_lr_fold/modal_train_resume.py::train    # detached run
"""

import json
import os
import subprocess
import threading
import time
from pathlib import Path

import modal

VOLUME_NAME = "satquery-data"
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
DATA = "/data"
RUN_DIR = f"{DATA}/runs/lr_fold"
CODE = f"{RUN_DIR}/code/lightning_lr_fold.py"
MIX = f"{RUN_DIR}/mix_lr_fold.jsonl"
INIT_CKPT = f"{RUN_DIR}/ckpt_init"
CONT_DATA = "/root/data"          # symlink farm -> volume trees
HF_DIR = "/root/hf"               # container-local HF cache (16GB base)

app = modal.App("satquery-lr-fold-resume")

# Deps pinned to the studio's installed set — numerics continuity for a
# mid-run resume. torch 2.8.0 PyPI wheel is cu128 on linux.
img = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0", "torchvision==0.23.0",
        "transformers==4.57.6", "peft==0.18.1", "accelerate==1.15.0",
        "pillow", "numpy<2.4", "safetensors==0.8.0", "hf_transfer",
    )
)


def _sha256(p: Path, bufsize: int = 1 << 22) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


# expected shas of the ckpt_last bundle uploaded from the laptop (computed
# pre-upload into eval_lr_fold/run/ckpt_last_sha256.json and pasted here)
CKPT_LAST_SHA256 = {
    "adapter/adapter_config.json": "99cf22c96c9210c2435f3a2aeb9cdf168b642ef98de5d374710da3c90d9cfaad",
    "adapter/adapter_model.safetensors": "735c0abde45a9ad3070156ac7fd26da5e3f217ae46a7f3100b0c15d627bc8036",
    "adapter/README.md": "a0dbbd0ebc7a83d2f2283101289575cd7df0ce0192eaefdba745f13bd74259fa",
    "merger.pt": "eeae1978ac611cd64cd2ef96deb94beb53e7fbd904149f8c1d07a42b9cf189df",
    "opt.pt": "e916dafae00893edcc73cca1eddcaa294dfd1ca6ce0370cb1a41d99e4623978c",
    "step.json": "0670b6e8f4e58a52da2ec183729f33be625fe12b87c7551638a3905a5f14f4e5",
}


@app.function(image=img, volumes={DATA: vol}, cpu=2.0, memory=8 * 1024,
              timeout=3600)
def preflight() -> str:
    """Verify the whole world before spending GPU: volume trees, mix, code,
    ckpt_init, ckpt_last (sha-verified), and eval-side dirs."""
    vol.reload()
    rep = {"checks": {}, "failures": []}

    def chk(name, ok, detail=""):
        rep["checks"][name] = {"ok": bool(ok), "detail": str(detail)}
        if not ok:
            rep["failures"].append(f"{name}: {detail}")

    # dataset trees (supersets of the staged subsets — mix only touches ours)
    for rel, want_min in (("rsvqa/lr/Images_LR", 572),
                          ("rsvqa/hr/Data", 6009),
                          ("vrsbench/Images_train", 14999)):
        d = Path(DATA) / rel
        n = sum(1 for _ in d.iterdir()) if d.is_dir() else 0
        chk(f"tree {rel}", n >= want_min, f"{n} entries (need >= {want_min})")

    chk("mix", Path(MIX).is_file(), MIX)
    chk("code", Path(CODE).is_file(), CODE)
    chk("ckpt_init adapter",
        (Path(INIT_CKPT) / "adapter" / "adapter_model.safetensors").is_file())
    chk("ckpt_init merger", (Path(INIT_CKPT) / "merger.pt").is_file())

    # ckpt_last: every file present AND sha-matched to the laptop bundle
    cl = Path(RUN_DIR) / "ckpt_last"
    got_step = None
    for rel, want in CKPT_LAST_SHA256.items():
        p = cl / rel
        if not p.is_file():
            chk(f"ckpt_last {rel}", False, "MISSING")
            continue
        got = _sha256(p)
        chk(f"ckpt_last {rel}", got == want, f"{got[:16]} vs {want[:16]}")
    sp = cl / "step.json"
    if sp.is_file():
        got_step = json.loads(sp.read_text()).get("opt_step")
    chk("ckpt_last step", got_step == 3000, f"opt_step={got_step}")

    # mix spot-check: every image referenced resolves under the symlink farm
    import random as _r
    rows = [json.loads(l) for l in Path(MIX).open()]
    rng = _r.Random(0)
    sample = rng.sample(rows, min(2000, len(rows)))
    missing = [r["image"] for r in sample
               if not (Path(CONT_DATA) / r["image"]).is_file()
               and not _resolve_vol(r["image"])]
    chk("mix image sample", not missing,
        f"{len(missing)} missing of {len(sample)}: {missing[:3]}")

    rep["ok"] = not rep["failures"]
    print("PREFLIGHT " + json.dumps(rep), flush=True)
    return json.dumps(rep)


def _resolve_vol(rel: str):
    """map mix-relative image path -> real volume path (no symlink needed)."""
    m = {"rsvqa_lr/": "rsvqa/lr/", "rsvqa_hr/": "rsvqa/hr/"}
    for a, b in m.items():
        if rel.startswith(a):
            rel = b + rel[len(a):]
            break
    p = Path(DATA) / rel
    return p if p.is_file() else None


@app.function(image=img, volumes={DATA: vol}, gpu="L40S", cpu=8.0,
              memory=96 * 1024, timeout=6 * 3600)
def train() -> str:
    """Run the ported trainer as a subprocess; commit the volume every 5 min
    so ckpt_last/STATUS/train_log persist mid-run."""
    t0 = time.time()
    vol.reload()

    # symlink farm: mix-relative roots -> real volume trees
    d = Path(CONT_DATA)
    d.mkdir(parents=True, exist_ok=True)
    for link, target in (("rsvqa_lr", f"{DATA}/rsvqa/lr"),
                         ("rsvqa_hr", f"{DATA}/rsvqa/hr"),
                         ("vrsbench", f"{DATA}/vrsbench")):
        lp = d / link
        if not lp.exists():
            os.symlink(target, lp)

    env = dict(os.environ)
    env.update({
        "DATA_ROOT": CONT_DATA,
        "MIX": MIX,
        "RUN_DIR": RUN_DIR,
        "INIT_CKPT": INIT_CKPT,
        "MODE": "train",
        "HF_HOME": HF_DIR,
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUTF8": "1",
        "MODEL_ID": "Qwen/Qwen3-VL-8B-Instruct",
        "BASE_REVISION": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "CKPT_EVERY": "500",
        "LOG_EVERY": "50",
        "TIME_CAP_S": str(int(5.5 * 3600)),   # ~1,910 steps x 8.5s ≈ 4.5h
    })

    stop = threading.Event()

    def _committer():
        while not stop.wait(300):
            try:
                vol.commit()
                print("VOL_COMMIT flush", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"VOL_COMMIT fail {e}", flush=True)

    threading.Thread(target=_committer, daemon=True).start()

    proc = subprocess.Popen(
        ["python", CODE], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:      # stream trainer stdout into app logs
        print(line, end="", flush=True)
    rc = proc.wait()
    stop.set()
    try:
        vol.commit()
    except Exception as e:  # noqa: BLE001
        print(f"final VOL_COMMIT fail {e}", flush=True)

    out = {"rc": rc, "wall_seconds": round(time.time() - t0, 1)}
    rp = Path(RUN_DIR) / "report.json"
    if rp.is_file():
        out["report"] = json.loads(rp.read_text())
    print("TRAIN_DONE " + json.dumps(out)[:4000], flush=True)
    return json.dumps(out)
