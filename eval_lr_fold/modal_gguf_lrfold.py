"""SWAP-8091 — convert the LR-fold merged tree (on proxynanmaga's
satquery-data volume) to GGUF: F16 text + F16 mmproj + Q4_K_M.

Replicates ATTACH-8091's conversion exactly, but on Modal CPU so the 17GB
bf16 tree never touches the laptop:
  * converter = the SAME pinned llama.cpp b10621 source
    (uploaded to runs/lr_fold/llamacpp_src from canonical/work/)
  * mmproj comes FROM THE MERGED MODEL (--mmproj on merged_eval) — the
    projector was further-trained in the LR fold; borrowing the narrator's
    or old canonical's mmproj would silently drop those 20 tensors.
  * quantize = official b10621 ubuntu-x64 release binary (same tag as the
    local llama-quantize.exe used for the previous artifact).
  * verifies every merged_eval file sha256 against merged_manifest.json
    before spending compute.

Outputs -> /data/runs/lr_fold/gguf/ + gguf_manifest.json (shas, sizes,
tensor counts). Pull back Q4 (~5GB) + mmproj (~1.2GB) + manifest only.

Usage (profile proxynanmaga):
    python -m modal run eval_lr_fold/modal_gguf_lrfold.py::convert
"""

import json
import subprocess
import time
from pathlib import Path

import modal

VOLUME_NAME = "satquery-data"
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
DATA = "/data"
MERGED = Path(DATA) / "runs" / "lr_fold" / "merged_eval"
SRC = Path(DATA) / "runs" / "lr_fold" / "llamacpp_src"
OUT = Path(DATA) / "runs" / "lr_fold" / "gguf"
BIN_URL = ("https://github.com/ggml-org/llama.cpp/releases/download/"
           "b10621/llama-b10621-bin-ubuntu-x64.tar.gz")

app = modal.App("satquery-lrfold-gguf")

img = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "libgomp1")
    .pip_install(
        "torch==2.8.0", "transformers==4.57.6", "safetensors==0.8.0",
        "sentencepiece==0.2.2", "numpy<2.4",
    )
    .run_commands(
        "mkdir -p /opt/llamacpp",
        f"curl -fsSL {BIN_URL} | tar -xz -C /opt/llamacpp",
        # the release tar drops a build/bin tree; normalize
        "find /opt/llamacpp -name llama-quantize -type f "
        "-exec cp {} /opt/llamacpp/llama-quantize \\; && "
        "chmod +x /opt/llamacpp/llama-quantize",
        "find /opt/llamacpp -name 'libggml*.so*' -exec dirname {} \\; "
        "| head -1 > /opt/llamacpp/libdir.txt",
    )
    .env({"PYTHONUTF8": "1", "TOKENIZERS_PARALLELISM": "false"})
)

USD_CPU_CORE_S = 0.0000131
USD_MEM_GIB_S = 0.00000222


def _sha256(p: Path, bufsize: int = 1 << 22) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def _gguf_info(p: Path) -> dict:
    """tensor count + arch field via the pinned gguf-py."""
    import sys
    sys.path.insert(0, str(SRC / "gguf-py"))
    import gguf  # noqa: E402
    r = gguf.GGUFReader(str(p))
    n_tensors = len(r.tensors)
    arch = r.fields.get("general.architecture")
    name = r.fields.get("general.name")
    def _v(f):
        try:
            return f.parts[f.indexes[0]].tolist() if f and f.indexes else None
        except Exception:  # noqa: BLE001
            return None
    a = _v(arch)
    nm = _v(name)
    if isinstance(a, (bytes, bytearray)):
        a = a.decode()
    if isinstance(nm, (bytes, bytearray)):
        nm = nm.decode()
    return {"n_tensors": n_tensors, "arch": a, "name": nm,
            "bytes": p.stat().st_size, "sha256": _sha256(p)}


@app.function(image=img, volumes={DATA: vol}, cpu=8.0, memory=64 * 1024,
              timeout=3 * 3600)
def convert() -> str:
    t0 = time.time()
    vol.reload()
    rep = {"step": "gguf_lrfold", "checks": {}, "failures": []}

    # ---- verify merged tree against manifest ------------------------------
    man = json.loads((MERGED / "merged_manifest.json").read_text())
    want = man["merged_files_sha256"]
    bad = []
    for rel, w in sorted(want.items()):
        p = MERGED / rel
        got = _sha256(p) if p.is_file() else "MISSING"
        if got != w:
            bad.append(f"{rel}: {got[:16]} != {w[:16]}")
    rep["checks"]["merged_tree_files"] = f"{len(want) - len(bad)}/{len(want)} sha-match"
    if bad:
        rep["failures"] += bad
        rep["status"] = "FAIL_PRECHECK"
        print("GGUF " + json.dumps(rep), flush=True)
        return json.dumps(rep)

    # ---- conversions ------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    env = {"PYTHONPATH": str(SRC / "gguf-py"),
           "PYTHONUTF8": "1", "PATH": "/usr/bin:/bin:/usr/local/bin"}
    libdir = Path("/opt/llamacpp/libdir.txt").read_text().strip()
    env["LD_LIBRARY_PATH"] = libdir

    f16 = OUT / "Qwen3VL-8B-LRFOLD-F16.gguf"
    mmp = OUT / "mmproj-Qwen3VL-8B-LRFOLD-F16.gguf"
    q4 = OUT / "Qwen3VL-8B-LRFOLD-Q4_K_M.gguf"

    def run(cmd, tag):
        t = time.time()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        rep["checks"][f"{tag}_rc"] = proc.returncode
        (OUT / f"{tag}.log").write_text(
            (proc.stdout or "")[-8000:] + "\n---STDERR---\n"
            + (proc.stderr or "")[-8000:])
        rep["checks"][f"{tag}_s"] = round(time.time() - t, 1)
        if proc.returncode != 0:
            rep["failures"].append(f"{tag} rc={proc.returncode}")
            return False
        return True

    py = "/usr/local/bin/python"
    ok = run([py, str(SRC / "convert_hf_to_gguf.py"), str(MERGED),
              "--outtype", "f16", "--outfile", str(f16)], "convert_f16")
    ok &= run([py, str(SRC / "convert_hf_to_gguf.py"), str(MERGED),
               "--mmproj", "--outtype", "f16", "--outfile", str(mmp)],
              "convert_mmproj")
    ok &= run(["/opt/llamacpp/llama-quantize", str(f16), str(q4), "Q4_K_M"],
              "quantize_q4km")

    # ---- hash + inspect outputs -------------------------------------------
    outs = {}
    for p in (f16, mmp, q4):
        if p.is_file():
            try:
                outs[p.name] = _gguf_info(p)
            except Exception as e:  # noqa: BLE001
                outs[p.name] = {"sha256": _sha256(p),
                                "bytes": p.stat().st_size,
                                "gguf_read_error": str(e)}
    rep["outputs"] = outs
    rep["status"] = "OK" if ok and not rep["failures"] else "FAIL"
    rep["wall_seconds"] = round(time.time() - t0, 1)
    rep["est_usd"] = round(rep["wall_seconds"]
                         * (8 * USD_CPU_CORE_S + 64 * USD_MEM_GIB_S), 4)
    (OUT / "gguf_manifest.json").write_text(json.dumps(rep, indent=2) + "\n")
    vol.commit()
    print("GGUF " + json.dumps(rep, default=str)[:4000], flush=True)
    return json.dumps(rep, default=str)


@app.local_entrypoint()
def main():
    print(convert.remote())
