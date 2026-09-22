#!/bin/bash
# LR-FOLD studio setup — pip deps + pinned HF base snapshot.
# Runs detached; progress in run/setup.log; ends with SETUP_DONE|SETUP_FAIL.
set -u
R=/teamspace/studios/this_studio/lr_fold
mkdir -p "$R/staging" "$R/ckpt_init/adapter" "$R/run" "$R/hf"
exec >> "$R/run/setup.log" 2>&1
echo "SETUP_START $(date -u +%FT%TZ)"

pip install --no-cache-dir \
    torch==2.8.0 torchvision==0.23.0 \
    "transformers>=4.57.0,<4.60.0" "peft>=0.15.0,<0.19.0" \
    "accelerate>=1.6.0,<2.0.0" pillow "numpy<2.4" safetensors hf_transfer \
    || { echo "SETUP_FAIL pip $(date -u +%FT%TZ)"; exit 1; }
echo "PIP_DONE $(date -u +%FT%TZ)"

python - <<'PY' || { echo "SETUP_FAIL hf $(date -u +%FT%TZ)"; exit 1; }
import os
os.environ["HF_HOME"] = "/teamspace/studios/this_studio/lr_fold/hf"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3-VL-8B-Instruct",
                      revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
print("BASE", p)
PY
echo "SETUP_DONE $(date -u +%FT%TZ)"
