"""SWAP-8091 helper — split the GGUF outputs on the volume into ~300MB parts
for a resumable pull (the CLI's chunked writer corrupted a 5GB download and
SDK read_file stalls without retry on a flaky link).

Usage (profile proxynanmaga):
    python -m modal run eval_lr_fold/modal_split_gguf.py
"""
import json
import time
from pathlib import Path

import modal

vol = modal.Volume.from_name("satquery-data", create_if_missing=False)
DATA = "/data"
GGUF = Path(DATA) / "runs" / "lr_fold" / "gguf"
PARTS = GGUF / "parts"
PART_BYTES = 300 * 1024 * 1024  # 300MB parts

app = modal.App("satquery-lrfold-split")
img = modal.Image.debian_slim(python_version="3.12")


@app.function(image=img, volumes={DATA: vol}, cpu=2.0, memory=8 * 1024,
              timeout=1800)
def split() -> str:
    t0 = time.time()
    vol.reload()
    rep = {"parts": {}, "failures": []}
    PARTS.mkdir(parents=True, exist_ok=True)
    for name in ("Qwen3VL-8B-LRFOLD-Q4_K_M.gguf",
                 "mmproj-Qwen3VL-8B-LRFOLD-F16.gguf"):
        src = GGUF / name
        if not src.is_file():
            rep["failures"].append(f"missing {name}")
            continue
        n = src.stat().st_size
        idx = 0
        with src.open("rb") as fi:
            while True:
                chunk = fi.read(PART_BYTES)
                if not chunk:
                    break
                (PARTS / f"{name}.part{idx:03d}").write_bytes(chunk)
                idx += 1
        rep["parts"][name] = {"n_parts": idx, "bytes": n}
    (PARTS / "parts_manifest.json").write_text(json.dumps(rep, indent=2))
    vol.commit()
    rep["wall_seconds"] = round(time.time() - t0, 1)
    rep["ok"] = not rep["failures"]
    print("SPLIT " + json.dumps(rep), flush=True)
    return json.dumps(rep)


@app.local_entrypoint()
def main():
    print(split.remote())
