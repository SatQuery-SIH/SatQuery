"""PROD-PACK-10: 50-row Unsloth two-native-image collator preflight. Does NOT train."""
from __future__ import annotations

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent.parent  # SatQuery/
BEFORE_LOCAL = HERE / "demo" / "data" / "scene2" / "before.png"
AFTER_LOCAL = HERE / "demo" / "data" / "scene2" / "after.png"
OUT_LOCAL = HERE / "gates" / "_cache" / "prod10" / "two_image_preflight.json"
BASE_MODEL = "unsloth/Qwen3-VL-8B-Instruct"
CACHE_VOLUME_NAME = "satquery-hf-cache"
N_ROWS = 50

# Same pins as prod08_modal.py train_image so Modal can reuse the cached image.
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        extra_index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install(
        "pillow",
        "bitsandbytes>=0.43",
        "huggingface_hub",
        "sentencepiece",
        "protobuf",
        "hf-transfer",
        "numpy",
        "safetensors",
    )
    .pip_install("unsloth_zoo")
    .pip_install("unsloth @ git+https://github.com/unslothai/unsloth.git")
    .pip_install("transformers==4.57.6", "trl==0.22.2")
    .pip_install("torchao==0.13.0")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/model_cache",
        "UNSLOTH_COMPILE_DISABLE": "1",
    })
    .add_local_file(str(BEFORE_LOCAL), remote_path="/incoming/before.png")
    .add_local_file(str(AFTER_LOCAL), remote_path="/incoming/after.png")
)

app = modal.App("satquery-prod10-preflight")
cache_vol = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)


def _tensor_info(v) -> dict:
    import torch

    if isinstance(v, torch.Tensor):
        info = {
            "kind": "tensor",
            "shape": list(v.shape),
            "dtype": str(v.dtype),
            "device": str(v.device),
        }
        if v.numel() > 0 and v.dtype.is_floating_point:
            info["mean"] = float(v.float().mean().detach().cpu())
            info["min"] = float(v.float().min().detach().cpu())
            info["max"] = float(v.float().max().detach().cpu())
        return info
    if isinstance(v, (list, tuple)):
        return {"kind": type(v).__name__, "n": len(v), "elem0": _tensor_info(v[0]) if v else None}
    return {"kind": type(v).__name__, "repr": repr(v)[:240]}


def _n_images_from_batch(batch: dict, batch_size: int) -> dict:
    """Infer how many native image tensors survived the collator."""
    keys = sorted(batch.keys())
    grid = batch.get("image_grid_thw")
    pixel = batch.get("pixel_values")
    n_grid = None
    n_pixel_rows = None
    if grid is not None and hasattr(grid, "shape"):
        n_grid = int(grid.shape[0])
    if pixel is not None and hasattr(pixel, "shape"):
        n_pixel_rows = int(pixel.shape[0])
    per_ex_grid = (n_grid / batch_size) if n_grid is not None else None
    collage_suspect = False
    if grid is not None and hasattr(grid, "shape") and grid.shape[-1] >= 3:
        # Qwen grid_thw is (n_images, 3) = (t, h, w) in patch units.
        # A width-concat collage of 384 and 448 would be one wide grid, not two rows.
        collage_suspect = per_ex_grid is not None and per_ex_grid < 2
    return {
        "batch_keys": keys,
        "image_grid_thw": _tensor_info(grid) if grid is not None else None,
        "pixel_values": _tensor_info(pixel) if pixel is not None else None,
        "n_grid_rows": n_grid,
        "n_pixel_rows": n_pixel_rows,
        "n_images_per_example_grid": per_ex_grid,
        "collage_suspect": collage_suspect,
    }


def _make_row(im0, im1, i: int) -> dict:
    text = (
        "[change] Image 1 is before, Image 2 is after. What changed, and where? "
        f"(preflight row {i})"
    )
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": im0},
                    {"type": "image", "image": im1},
                    {"type": "text", "text": text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "built-up expanded in the north-east."}],
            },
        ]
    }


@app.function(
    image=train_image,
    gpu=["L40S", "A100-40GB"],
    timeout=40 * 60,
    memory=65536,
    volumes={"/model_cache": cache_vol},
)
def preflight() -> dict:
    import unsloth  # noqa: F401
    import torch
    from PIL import Image
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"preflight gpu={gpu_name} cuda={torch.cuda.is_available()}", flush=True)

    im0 = Image.open("/incoming/before.png").convert("RGB").resize((384, 384))
    im1 = Image.open("/incoming/after.png").convert("RGB").resize((448, 320))
    rows = [_make_row(im0, im1, i) for i in range(N_ROWS)]
    n_img_in_msg = [
        sum(1 for p in r["messages"][0]["content"] if p.get("type") == "image") for r in rows
    ]
    if min(n_img_in_msg) != 2 or max(n_img_in_msg) != 2:
        return {
            "pass": False,
            "status": "FAIL",
            "reason": "messages did not contain two image parts",
            "n_img_in_msg": n_img_in_msg[:5],
            "gpu": gpu_name,
            "serve_two_image": "already Scene 2 (llama-server two image_url); no new serve path",
        }

    model, tokenizer = FastVisionModel.from_pretrained(
        BASE_MODEL,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    # No get_peft_model. No trainer.train().
    collator = UnslothVisionDataCollator(
        model,
        tokenizer,
        train_on_responses_only=True,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    dumped = []
    n_ok = 0
    n_fail = 0
    reasons = []
    for i in range(0, N_ROWS, 2):
        chunk = rows[i : i + 2]
        batch = collator(chunk)
        info = _n_images_from_batch(batch, batch_size=len(chunk))
        per = info.get("n_images_per_example_grid")
        ok = per is not None and per >= 2 and not info.get("collage_suspect")
        if not ok:
            n_fail += 1
            reasons.append({"i": i, "info": info})
        else:
            n_ok += 1
        if len(dumped) < 3:
            dumped.append({"i": i, "ok": ok, **info})
        print(f"chunk i={i} ok={ok} per={per} keys={info['batch_keys']}", flush=True)

    # Single-example dump for a decoded sample (means of two grid rows if present).
    one = collator([rows[0]])
    one_info = _n_images_from_batch(one, batch_size=1)
    decoded = {
        "input_sizes": {"image0": [384, 384], "image1": [448, 320]},
        "note": (
            "Images resized to different shapes on purpose. A width-concat collage "
            "would be one grid; two native tensors keep two image_grid_thw rows."
        ),
        "batch": one_info,
    }
    grid = one.get("image_grid_thw")
    if grid is not None and hasattr(grid, "detach"):
        decoded["image_grid_thw_rows"] = grid.detach().cpu().tolist()

    passed = n_fail == 0 and (one_info.get("n_images_per_example_grid") or 0) >= 2
    payload = {
        "pass": bool(passed),
        "status": "PASS" if passed else "FAIL",
        "n_rows": N_ROWS,
        "n_chunks_ok": n_ok,
        "n_chunks_fail": n_fail,
        "gpu": gpu_name,
        "base_model": BASE_MODEL,
        "finetune": False,
        "lora": False,
        "trained": False,
        "input_image_sizes": {"before": [384, 384], "after": [448, 320]},
        "decoded_sample": decoded,
        "chunk_dumps": dumped,
        "fail_samples": reasons[:5],
        "serve_two_image": "already Scene 2 (llama-server two image_url); no new serve path this spec",
        "format": "two {type:image} then {type:text}; no collage / stitch / 4-band cube",
    }
    if not passed:
        payload["reason"] = (
            "Unsloth collator did not keep two native image tensors "
            "(dropped image 2 or concatenated on width)."
        )
    print("PREFLIGHT_RESULT", json.dumps({k: payload[k] for k in ("pass", "status", "n_chunks_ok", "n_chunks_fail", "gpu")}))
    return payload


@app.local_entrypoint()
def main():
    if not BEFORE_LOCAL.is_file() or not AFTER_LOCAL.is_file():
        raise FileNotFoundError(f"need {BEFORE_LOCAL} and {AFTER_LOCAL}")
    result = preflight.remote()
    OUT_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOCAL.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"wrote {OUT_LOCAL} pass={result.get('pass')} status={result.get('status')}")
