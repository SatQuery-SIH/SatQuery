"""CLOUD-SEAT — OpenAI-compatible vLLM endpoints for both SatQuery seats on Modal.

Two seats, one app, each behind Modal proxy auth + scale-to-zero:

  narrator   — Qwen/Qwen3-VL-8B-Instruct @ 0c351dd0… (frozen product voice,
               bf16, weights served from the HF cache already on the volume)
  canonical  — the LR-FOLD merged tree at runs/lr_fold/merged_eval on the
               proxynanmaga satquery-data volume, served as "canonical-lrfold"
               (merged_tree_sha256 3a4fecb0… per eval_lr_fold/final/
               merged_manifest.json — verified at container start)

Deploy (profile proxynanmaga):

    python -m modal deploy deploy/modal_serve.py

Auth: requires_proxy_auth=True — callers send `Modal-Key: wk-…` +
`Modal-Secret: ws-…` (or `Authorization: Bearer wk-….ws-…`). Token lives in
deploy/cloud.env (gitignored), never committed. See deploy/README.md.

Lazy-GPU economics: min_containers=0 + scaledown_window=600 — zero spend when
idle. Weights + vLLM compile artifacts are volume-cached (HF_HOME=/data/_hf,
VLLM_CACHE_ROOT=/data/_vllm_cache) so cold starts never re-download or
re-compile. keep_warm for a demo window only: SATQUERY_SEAT_MIN_CONTAINERS=1
python -m modal deploy deploy/modal_serve.py  — never the default.

Warmup contract: GET /v1/models on a cold endpoint triggers scale-up; the
request hangs while the container boots, so probe loops must poll, not
single-shot. Cold-start latency is reported in deploy/README.md.
"""

import json
import os
import subprocess
from pathlib import Path

import modal

# --------------------------------------------------------------------------
# Pins (CLOUD-SEAT spec)
# --------------------------------------------------------------------------

VOLUME_NAME = "satquery-data"          # proxynanmaga account — volumes don't cross
DATA = "/data"
vol = modal.Volume.from_name(VOLUME_NAME)

NARRATOR_HF_ID = "Qwen/Qwen3-VL-8B-Instruct"
NARRATOR_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

CANONICAL_DIR = Path(DATA) / "runs" / "lr_fold" / "merged_eval"
CANONICAL_SERVED_NAME = "canonical-lrfold"
CANONICAL_TREE_SHA256 = (
    "3a4fecb0521ddaf42ed297ec8ca391f9b5d94bd6c884bf51cbab0c4562c7f11c"
)

GPU = "A10G"        # 24 GB — bf16 8B (~16.5 GB) + KV headroom
PORT = 8000
SCALEDOWN_WINDOW_S = 600               # ~10 min idle -> scale to zero
WEB_STARTUP_TIMEOUT_S = 900            # port-open wait covers cold vLLM boot

# keep_warm for the evaluation window only — 0 is the hard default.
MIN_CONTAINERS = int(os.environ.get("SATQUERY_SEAT_MIN_CONTAINERS", "0"))

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.11.2", "hf_transfer")
    .env(
        {
            # weights + compiled graphs persist on the volume across cold starts
            "HF_HOME": f"{DATA}/_hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_CACHE_ROOT": f"{DATA}/_vllm_cache",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

app = modal.App("satquery-serve")

_VLLM_COMMON = [
    "--host", "0.0.0.0",
    "--port", str(PORT),
    "--dtype", "bfloat16",
    # 0.92 leaves <1 GiB for KV on 24 GB after the vision encoder cache —
    # 0.95 is the minimum that clears the max_model_len=8192 KV check (A10G).
    "--gpu-memory-utilization", "0.95",
    "--max-model-len", "8192",
    "--limit-mm-per-prompt", '{"image": 4, "video": 0}',
    "--max-num-seqs", "32",
]


def _spawn_vllm(args: list[str]) -> None:
    print("SPAWN " + " ".join(args), flush=True)
    subprocess.Popen(args)


@app.function(
    image=vllm_image,
    gpu=GPU,
    volumes={DATA: vol},
    min_containers=MIN_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW_S,
    timeout=24 * 3600,
)
@modal.web_server(
    port=PORT,
    startup_timeout=WEB_STARTUP_TIMEOUT_S,
    label="narrator",
    requires_proxy_auth=True,
)
def narrator() -> None:
    """Frozen product voice — pinned HF revision, bf16."""
    _spawn_vllm(
        [
            "vllm", "serve", NARRATOR_HF_ID,
            "--revision", NARRATOR_REVISION,
            "--served-model-name", NARRATOR_HF_ID,
        ]
        + _VLLM_COMMON
    )


@app.function(
    image=vllm_image,
    gpu=GPU,
    volumes={DATA: vol},
    min_containers=MIN_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW_S,
    timeout=24 * 3600,
)
@modal.web_server(
    port=PORT,
    startup_timeout=WEB_STARTUP_TIMEOUT_S,
    label="canonical",
    requires_proxy_auth=True,
)
def canonical() -> None:
    """Canonical answer seat — LR-FOLD merged tree, bf16, served as
    canonical-lrfold. Asserts the on-volume manifest pin before serving."""
    vol.reload()
    manifest_path = CANONICAL_DIR / "merged_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    got = manifest.get("merged_tree_sha256")
    if got != CANONICAL_TREE_SHA256:
        raise RuntimeError(
            f"canonical merged_tree_sha256 mismatch: {got} != "
            f"{CANONICAL_TREE_SHA256} — refusing to serve"
        )
    _spawn_vllm(
        [
            "vllm", "serve", str(CANONICAL_DIR),
            "--served-model-name", CANONICAL_SERVED_NAME,
        ]
        + _VLLM_COMMON
    )


@app.local_entrypoint()
def main() -> None:
    print(
        "Deploy with: python -m modal deploy deploy/modal_serve.py\n"
        "Endpoints appear under https://proxynanmaga--satquery-serve-*.modal.run"
    )
