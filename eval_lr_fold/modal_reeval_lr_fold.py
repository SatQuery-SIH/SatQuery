"""LR-FOLD-EVAL — re-score frozen eval columns on the LR-fold merged tree.

Constants-only copy of modal_reeval_merged.py (approved diff: CKPT_DIR,
MERGED_DIR, EVAL_TAG, CKPT_SHA256 -> post-train values, lane labels, run
dir for reports). Protocol/helpers/decode/shards byte-identical.

Original docstring follows verbatim:
RE-EVAL-MERGED — re-score frozen RSVQA eval columns on the FULLY-merged
checkpoint (LoRA + trained projector), account proxynanmaga.

Why this lane exists: modal_rsvqa_adapt.py::_attach_trained loads merger.pt
(24 keys, peft-prefixed 'base_model.model.model.visual.*') into the RAW base
model with strict=False -> every key lands in unexpected_keys and is silently
dropped. The published eval_full_qwen3vl/* numbers are therefore
LoRA + base-projector FLOORS. This file re-scores the same frozen columns on
the merged full-weights tree produced by the modal_vrsbench_eval.py::merge_ckpt
recipe (PeftModel.from_pretrained -> load_state_dict(merger, all-matched,
unexpected==0 asserted) -> merge_and_unload -> save_pretrained bf16).

Protocol is identical to the floor run by construction: the same helpers are
copied verbatim (_chat_prompt, _gen_eval, tok_exact, canon, vocab_project,
_load_mix_rows, gold/*.jsonl, eval_ids, COLUMNS shard counts, batch_size=40,
greedy + repetition_penalty=1.08 + max_new_tokens=16, sdpa/bf16, A100-40GB).
The ONLY change is model load: merged full-weights dir instead of
base + adapter.

Run (profile proxynanmaga):

    python -m modal run modal_reeval_merged.py::e_stage     # input gate
    python -m modal run modal_reeval_merged.py::e_merge     # CPU merge
    python -m modal run modal_reeval_merged.py::e_smoke     # 200-row probe
    ONLY_COL=rsvqa_hr_val python -m modal run modal_reeval_merged.py::e_eval
    python -m modal run modal_reeval_merged.py::e_score
"""

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import modal

# --------------------------------------------------------------------------
# Constants / contract  (mirrors modal_rsvqa_adapt.py)
# --------------------------------------------------------------------------

VOLUME_NAME = "satquery-data"
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

DATA = "/data"
RUN = Path(DATA) / "runs" / "rsvqa_adapt"
EVAL_IDS_DIR = Path(DATA) / "eval_ids"
HF_CACHE = "/data/_hf"          # persisted HF downloads on the volume

CANDIDATES = {
    "qwen3vl": "Qwen/Qwen3-VL-8B-Instruct",
}
# pinned base revision (same as modal_vrsbench_eval.py MODELS["qwen3vl8b"])
BASE_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
BASE_WEIGHTS_SHA = (
    "aa8d7f7ef1f6867301225a39a924e49dea4b9c36fc9ce953eec52dc692c27a91"
)

# Decode contract (identical to floor run)
REP_PENALTY = 1.08
MAX_NEW_TOKENS = 16

# Frozen eval columns: file in /data/eval_ids + shard count (identical to
# floor run so per-shard row sets and batch composition match 1:1)
COLUMNS = {
    "rsvqa_hr_val":        {"ids": "rsvqa_hr_val_eval_ids.json",        "shards": 4},
    "rsvqa_hr_test":       {"ids": "rsvqa_hr_test_eval_ids.json",       "shards": 8},
    "rsvqa_hr_test_phili": {"ids": "rsvqa_hr_test_phili_eval_ids.json", "shards": 4},
    "rsvqa_lr_val":        {"ids": "rsvqa_lr_val_eval_ids.json",        "shards": 1},
    "rsvqa_lr_test":       {"ids": "rsvqa_lr_test_eval_ids.json",       "shards": 1},
    "vrsbench_vqa":        {"ids": "vrsbench_vqa_eval_ids.json",        "shards": 2},
}
BAR_COL = "rsvqa_hr_val"
BAR = 0.5077

# ckpt_final on the volume (uploaded from local modal_volume_backup)
CKPT_DIR = Path(DATA) / "runs" / "lr_fold" / "ckpt_final"
MERGED_DIR = Path(DATA) / "runs" / "lr_fold" / "merged_eval"
EVAL_TAG = "lr_fold_merged"

# shas asserted before any use — LR-fold ckpt_final (opt_step 4910), from
# report.json on the volume + locally re-hashed after download.
CKPT_SHA256 = {
    "adapter_model.safetensors":
        "60f1be46174b568f311e8f8e2aebf28a875a102c917c6f23ced5de884a4eeb68",
    "merger.pt":
        "1331936df4275a7c550341de3e9414c4d40ee3b902ee02c64517aadecdb28cf2",
    "combined":
        "143bbfd550c00643f0ce6a5ff3b287fc1dba47476b6d5dae1316eb32dbd7972d",
    "adapter_config.json":
        "4ee89ba31cc7534008c725d668735a618170ba79c0e8742ef1eadec65f923dab",
    "mix.jsonl":
        "e60f369567e922e4ca56b509da523d351c31c09a371b5dd8d95180918e1b7332",
}

# full sha256 of local authoritative copies — asserted on the volume copies
EVAL_IDS_SHA256 = {
    "rsvqa_hr_val_eval_ids.json":
        "3cc2518b92fc9c031591819bac7d58da2266dddc538dcd74c465d96f3493ab0b",
    "rsvqa_hr_test_eval_ids.json":
        "196c2a2ff6f6e27e21cb76c0bfb7a84b3dea3d703361831c7bea8756cd985fbf",
    "rsvqa_hr_test_phili_eval_ids.json":
        "9a1c6f00d3915b947cb2bd8f002be26ef9ddb31ab4a443f880d27537261a055c",
    "rsvqa_lr_val_eval_ids.json":
        "81fdedabc5aba181bf154b5f02eb496ded7c6f41916659ed4d0d3d56e2cbb664",
    "rsvqa_lr_test_eval_ids.json":
        "7735643c116bcce5dd2b387deb1c7661fd448cd8c99de2ef707bc5f114f0255b",
    "vrsbench_vqa_eval_ids.json":
        "bd858cb81c6450e231befa3e25caa3c2aefb43c7dd61f6ae561ba4caa7d0eae8",
}
GOLD_SHA256 = {
    "rsvqa_hr_val.jsonl":
        "825ca775cfb3670be55b3d18bd2224c0999ba562d6214653397d99e4604930fe",
    "rsvqa_hr_test.jsonl":
        "ddbfe342f5ed4314bec77f03fc04c41305c02e407d7cd07f129162af0b31ce6e",
    "rsvqa_hr_test_phili.jsonl":
        "f19fbcbca7847d0675e46cfea0cf276fdaaf7c21c03d71f744e8ba460185616b",
    "rsvqa_lr_val.jsonl":
        "38e064f49fc6446e2059041c61398dc817956bf38f1a52b3f19f80841fe64a13",
    "rsvqa_lr_test.jsonl":
        "773af16fb177886fe59e395b88c284416478fb023b2f999512e1547edc52d991",
    "vrsbench_vqa.jsonl":
        "3bc512282fd79ed8a87d2b577d4a8f057878a0b68af662142e8223c8fd6ca4c7",
}
VOCAB_SHA256 = (
    "2559751ed0a47eff9bcb9860ebdcc4c5c72cdd870048b681525d62018dfb9ee4"
)

# dataset source files sha-pinned to the OLD volume's recorded values
# (mix_manifest.json::source_sha256, computed there during build_mix) —
# proves the re-fetched datasets are the same bytes the floor run saw.
DATASET_SHA256 = {
    "rsvqa/hr/USGS_split_train_questions.json":
        "08a29d70d9ea0a1e01bac3b19a2b9f45a9410e337efe48216d6e40023742a37b",
    "rsvqa/hr/USGS_split_train_answers.json":
        "8d1db1ff9639afbbe22e72032a24f2ce15a40e9b968145ef9386373d969de5ec",
    "rsvqa/hr/USGS_split_train_images.json":
        "cc8aae453e37700cda5016a709bc2f9f617d9447c6413481afc4a3cb866125b0",
    "vrsbench/VRSBench_EVAL_vqa.json":
        "4a797f5fc456331a55938b689350228bb7aa200b7e9a3972a6604b14dcfc175c",
}

# Modal list pricing (modal.com/pricing) — same as modal_rsvqa_adapt.py
USD_A100_40_S = 0.000944   # ~$3.40/h
USD_CPU_CORE_S = 0.0000131
USD_MEM_GIB_S = 0.00000222

app = modal.App("satquery-reeval-lr-fold")

# identical to modal_rsvqa_adapt.py::train_image (same env for generation)
train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "transformers>=4.57.0,<4.60.0",
        "peft>=0.15.0,<0.19.0",
        "accelerate>=1.6.0,<2.0.0",
        "pillow",
        "numpy<2.4",
        "safetensors",
        "hf_transfer",
    )
    .env({"HF_HOME": HF_CACHE, "TOKENIZERS_PARALLELISM": "false",
          "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

cpu_image = modal.Image.debian_slim(python_version="3.12").pip_install("pillow")


def _usd(seconds: float, gpu_s_rate: float = 0.0, cpu: float = 2.0, mem_gib: float = 4.0) -> float:
    rate = gpu_s_rate + cpu * USD_CPU_CORE_S + mem_gib * USD_MEM_GIB_S
    return round(seconds * rate, 4)


def _runstats(t0: float, gpu_s_rate: float = 0.0, cpu: float = 2.0, mem_gib: float = 4.0) -> dict:
    secs = round(time.time() - t0, 1)
    stats = {
        "wall_seconds": secs,
        "est_usd": _usd(secs, gpu_s_rate, cpu, mem_gib),
        "gpu_usd_rate_s": gpu_s_rate,
        "cpu_cores": cpu,
        "memory_gib": mem_gib,
        "pricing": "Modal list: A100-40GB $0.000944/s; CPU $0.0000131/core-s; mem $0.00000222/GiB-s",
    }
    print("RUNSTATS " + json.dumps(stats), flush=True)
    return stats


def _sha256(path: Path, bufsize: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def _js(obj) -> str:
    """Remote fns return JSON strings (local env lacks torch for unpickle)."""
    return json.dumps(obj, default=str)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")
    print(f"WROTE {path}", flush=True)


# --------------------------------------------------------------------------
# Shared model helpers — VERBATIM from modal_rsvqa_adapt.py (protocol lock)
# --------------------------------------------------------------------------

def _set_use_cache(model, flag: bool) -> None:
    cfg = model.config
    if hasattr(cfg, "use_cache"):
        cfg.use_cache = flag
    for sub in ("text_config",):
        if hasattr(cfg, sub) and hasattr(getattr(cfg, sub), "use_cache"):
            getattr(cfg, sub).use_cache = flag


def _chat_prompt(processor, question: str, pil=None) -> str:
    img_item = {"type": "image"}
    if pil is not None:
        img_item["image"] = pil
    msgs = [{
        "role": "user",
        "content": [img_item, {"type": "text", "text": question}],
    }]
    return processor.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


def _load_rgb(path: Path):
    from PIL import Image
    import numpy as np

    im = Image.open(path)
    try:
        return im.convert("RGB")
    except Exception:
        arr = np.asarray(im)
        if arr.ndim == 3:
            arr = arr[..., :3]
        elif arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            lo, hi = np.percentile(arr, (1, 99))
            arr = np.clip((arr - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, "RGB")


def _preload_images(paths) -> dict:
    """unique path -> PIL.RGB. Returns {path_str: img}."""
    from concurrent.futures import ThreadPoolExecutor

    uniq = sorted({str(p) for p in paths})
    out = {}

    def _one(p):
        return p, _load_rgb(Path(p))

    with ThreadPoolExecutor(max_workers=32) as ex:
        for p, im in ex.map(_one, uniq):
            out[p] = im
    return out


def _gen_eval(model, processor, tokenizer, rows, device, batch_size=40) -> list[str]:
    """Batched greedy generation. rows: [{_pil, question}]. Returns raw texts."""
    import torch

    tokenizer.padding_side = "left"
    order = sorted(range(len(rows)), key=lambda i: str(rows[i]["_img"]))
    preds = [""] * len(rows)
    for i0 in range(0, len(order), batch_size):
        idxs = order[i0 : i0 + batch_size]
        prompts = [_chat_prompt(processor, rows[i]["question"], rows[i]["_pil"])
                   for i in idxs]
        images = [rows[i]["_pil"] for i in idxs]
        enc = processor(text=prompts, images=images, padding=True,
                        return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=REP_PENALTY,
            )
        gen = out[:, enc.input_ids.shape[1]:]
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for j, i in enumerate(idxs):
            preds[i] = texts[j].strip()
    return preds


# --------------------------------------------------------------------------
# Normalizer — VERBATIM from modal_rsvqa_adapt.py
# --------------------------------------------------------------------------

_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12",
}


def tok_exact(s: str) -> str:
    """Token-exact reconstruction of the wiped 0.5077 scorer: strip + casefold
    + whitespace collapse + trailing-period strip. THIS is the bar metric."""
    return re.sub(r"\s+", " ", s.strip().lower()).rstrip(".")


def canon(s: str) -> str:
    """Full canonicalizer: case fold, unit canonicalization, digit-word fold.
    Applied to BOTH sides before nearest-vocab projection."""
    s = tok_exact(s)
    s = s.replace("²", "2").replace("㎡", "m2")
    s = re.sub(r"(\d),(\d)", r"\1\2", s)                      # 1,234 -> 1234
    for phrase, unit in (
        ("square kilometers", "km2"), ("square kilometer", "km2"),
        ("sq kilometers", "km2"), ("sq km", "km2"), ("km 2", "km2"),
        ("square meters", "m2"), ("square meter", "m2"),
        ("sq meters", "m2"), ("sq m", "m2"), ("m 2", "m2"),
        ("hectares", "ha"), ("hectare", "ha"),
    ):
        s = s.replace(phrase, unit)
    s = re.sub(r"(\d)\s*(m2|km2|ha)\b", r"\1\2", s)           # 12 m2 -> 12m2
    if s in _NUM_WORDS:
        s = _NUM_WORDS[s]
    return s


def vocab_project(s: str, vocab: list[str], cutoff: float = 0.85) -> str:
    """Nearest-vocab projection over the TRAIN answer vocabulary (no eval gold
    is ever used to build the vocab)."""
    import difflib

    if s in vocab:
        return s
    m = difflib.get_close_matches(s, vocab, n=1, cutoff=cutoff)
    return m[0] if m else s


def _load_mix_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def eval_ids_read(col: str) -> list:
    return json.loads((EVAL_IDS_DIR / COLUMNS[col]["ids"]).read_text())["ids"]


def _load_merged(device: str = "cuda"):
    """Fully-merged full-weights tree (LoRA + trained merger baked in)."""
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as AutoModel
    except ImportError:  # older name
        from transformers import AutoModelForVision2Seq as AutoModel

    processor = AutoProcessor.from_pretrained(str(MERGED_DIR))
    model = AutoModel.from_pretrained(
        str(MERGED_DIR),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    return model, processor


# --------------------------------------------------------------------------
# Step 0 — stage gate: verify every input on the fresh volume by SHA256 and
# confirm every gold image exists (coverage=1.0 is a hard bar)
# --------------------------------------------------------------------------

@app.function(
    image=cpu_image,
    volumes={DATA: vol},
    cpu=4.0,
    memory=32 * 1024,
    timeout=3600,
)
def stage_check() -> dict:
    """Assert: ckpt shas == pinned; eval_ids + gold + vocab shas == local
    authoritative copies; every image referenced by gold rows exists."""
    t0 = time.time()
    vol.reload()
    report = {"step": "stage_check", "checks": {}}
    failures = []

    # ---- ckpt_final ---------------------------------------------------------
    ck = {}
    p = CKPT_DIR / "adapter" / "adapter_model.safetensors"
    ck["adapter_model.safetensors"] = _sha256(p) if p.exists() else "MISSING"
    p = CKPT_DIR / "adapter" / "adapter_config.json"
    ck["adapter_config.json"] = _sha256(p) if p.exists() else "MISSING"
    p = CKPT_DIR / "merger.pt"
    ck["merger.pt"] = _sha256(p) if p.exists() else "MISSING"
    report["checks"]["ckpt"] = ck
    for k in ("adapter_model.safetensors", "merger.pt", "adapter_config.json"):
        if ck[k] != CKPT_SHA256[k]:
            failures.append(f"ckpt {k}: {ck[k]} != {CKPT_SHA256[k]}")
    h = hashlib.sha256()
    for v in sorted([ck["adapter_model.safetensors"], ck["merger.pt"]]):
        h.update(v.encode())
    ck["combined"] = h.hexdigest()
    if ck["combined"] != CKPT_SHA256["combined"]:
        failures.append(f"ckpt combined: {ck['combined']}")

    # ---- frozen eval ids -----------------------------------------------------
    ids = {}
    for fn, want in EVAL_IDS_SHA256.items():
        p = EVAL_IDS_DIR / fn
        got = _sha256(p) if p.exists() else "MISSING"
        ids[fn] = got
        if got != want:
            failures.append(f"eval_ids {fn}: {got[:16]} != {want[:16]}")
    report["checks"]["eval_ids"] = {k: v[:16] for k, v in ids.items()}

    # ---- gold + vocab ---------------------------------------------------------
    gold = {}
    for fn, want in GOLD_SHA256.items():
        p = RUN / "gold" / fn
        got = _sha256(p) if p.exists() else "MISSING"
        gold[fn] = got
        if got != want:
            failures.append(f"gold {fn}: {got[:16]} != {want[:16]}")
    report["checks"]["gold"] = {k: v[:16] for k, v in gold.items()}
    pv = RUN / "vocab.json"
    v_sha = _sha256(pv) if pv.exists() else "MISSING"
    report["checks"]["vocab.json"] = v_sha[:16]
    if v_sha != VOCAB_SHA256:
        failures.append(f"vocab.json: {v_sha[:16]} != {VOCAB_SHA256[:16]}")

    # ---- dataset source files == old volume's recorded shas -------------------
    ds = {}
    for rel, want in DATASET_SHA256.items():
        p = Path(DATA) / rel
        got = _sha256(p) if p.exists() else "MISSING"
        ds[rel] = got[:16]
        if got != want:
            failures.append(f"dataset {rel}: {got[:16]} != {want[:16]}")
    report["checks"]["dataset_source"] = ds

    # ---- image existence (per gold row; listdir once per dir — per-file
    # stat on a mounted volume is a network round trip) -----------------------
    dir_listings: dict[str, set] = {}
    img = {}
    for col in COLUMNS:
        gp = RUN / "gold" / f"{col}.jsonl"
        if not gp.exists():
            img[col] = {"status": "NO_GOLD"}
            continue
        missing = []
        n = 0
        uniq = set()
        with gp.open() as f:
            for line in f:
                r = json.loads(line)
                n += 1
                ip = r["image"]
                uniq.add(ip)
                d, fn = ip.rsplit("/", 1)
                if d not in dir_listings:
                    try:
                        dir_listings[d] = set(os.listdir(d))
                    except OSError:
                        dir_listings[d] = set()
                if fn not in dir_listings[d]:
                    missing.append(ip)
        img[col] = {"rows": n, "uniq_images": len(uniq),
                    "missing": len(missing), "missing_sample": missing[:5]}
        if missing:
            failures.append(f"{col}: {len(missing)} missing images "
                            f"e.g. {missing[:3]}")
    report["checks"]["images"] = img
    report["checks"]["image_dir_sizes"] = {
        d: len(s) for d, s in sorted(dir_listings.items())}

    # ---- ids vs gold alignment --------------------------------------------------
    align = {}
    for col, cfg in COLUMNS.items():
        ip = EVAL_IDS_DIR / cfg["ids"]
        gp = RUN / "gold" / f"{col}.jsonl"
        if not (ip.exists() and gp.exists()):
            continue
        id_list = json.loads(ip.read_text())["ids"]
        with gp.open() as f:
            g_ids = [json.loads(line)["id"] for line in f]
        align[col] = {
            "n_ids": len(id_list), "n_gold": len(g_ids),
            "order_identical": id_list == g_ids,
            "set_overlap": len(set(id_list) & set(g_ids)),
        }
        if id_list != g_ids:
            failures.append(f"{col}: gold order/content != frozen ids")
    report["checks"]["id_gold_alignment"] = align

    report["status"] = "FAIL" if failures else "OK"
    report["failures"] = failures
    report["run"] = _runstats(t0, cpu=4.0, mem_gib=32)
    _write_json(Path(DATA) / "runs" / "lr_fold" / "stage_check.json", report)
    vol.commit()
    return _js(report)


# --------------------------------------------------------------------------
# Step 1 — merge (recipe verbatim from modal_vrsbench_eval.py::merge_ckpt)
# --------------------------------------------------------------------------

@app.function(
    image=train_image,
    volumes={DATA: vol},
    cpu=8.0,
    memory=96 * 1024,
    timeout=2 * 3600,
)
def merge_ckpt() -> dict:
    """ckpt_final (LoRA adapter + merger.pt) -> merged full-bf16 weights dir.
    ORDER MATTERS: adapter attached first so merger.pt's peft-prefixed keys
    (base_model.model.model.visual.merger.*) resolve — loading them into the
    raw base model silently no-ops under strict=False."""
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as AutoModel
    except ImportError:  # older name
        from transformers import AutoModelForVision2Seq as AutoModel

    t0 = time.time()
    vol.reload()
    ckpt = CKPT_DIR
    out = MERGED_DIR
    report = {"step": "merge_ckpt", "ckpt_dir": str(ckpt)}
    mman_path = out / "merged_manifest.json"
    cpu_rate = 8 * USD_CPU_CORE_S + 96 * USD_MEM_GIB_S

    sha_adapter = _sha256(ckpt / "adapter" / "adapter_model.safetensors")
    sha_merger = _sha256(ckpt / "merger.pt")
    assert sha_adapter == CKPT_SHA256["adapter_model.safetensors"], \
        f"adapter sha {sha_adapter} != pinned"
    assert sha_merger == CKPT_SHA256["merger.pt"], \
        f"merger sha {sha_merger} != pinned"
    report["input_sha256"] = {"adapter_model.safetensors": sha_adapter,
                              "merger.pt": sha_merger}

    base_dir = snapshot_download(CANDIDATES["qwen3vl"], revision=BASE_REVISION)
    vol.commit()

    if not (out / "model.safetensors").exists() and \
            not list(out.glob("model-*.safetensors")):
        processor = AutoProcessor.from_pretrained(base_dir)
        model = AutoModel.from_pretrained(
            base_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        peft_model = PeftModel.from_pretrained(
            model, str(ckpt / "adapter"), is_trainable=False)
        mp = torch.load(ckpt / "merger.pt", map_location="cpu")
        res = peft_model.load_state_dict(mp, strict=False)
        n_unexpected = len(res.unexpected_keys)
        assert n_unexpected == 0, \
            f"merger.pt keys unmatched (strict load would drop them): " \
            f"{res.unexpected_keys[:5]}"
        report["merger_tensors_loaded"] = len(mp)
        merged = peft_model.merge_and_unload()
        out.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(out)
        processor.save_pretrained(out)
        n_merger_tensors = len(mp)
    else:
        prev = json.loads(mman_path.read_text()) if mman_path.exists() else {}
        n_merger_tensors = prev.get("merge", {}).get("merger_tensors")
        report["status"] = "SKIP_MERGE_EXISTS"

    # mm preprocessing configs: this transformers version's
    # processor.save_pretrained omits preprocessor_config.json — copy the
    # base snapshot's mm files so eval applies identical geometry.
    import shutil
    for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
        src = Path(base_dir) / name
        if src.exists():
            shutil.copyfile(src, out / name)

    file_shas = {}
    for p in sorted(out.iterdir()):
        if p.is_file():
            file_shas[p.name] = _sha256(p)
    tree_sha = hashlib.sha256("\n".join(
        f"{k}:{v}" for k, v in sorted(file_shas.items())).encode()).hexdigest()
    mman = {
        "step": "merge_ckpt",
        "lane": "LR-FOLD-EVAL",
        "merged_dir": str(out),
        "base": {"hf_id": CANDIDATES["qwen3vl"], "revision": BASE_REVISION,
                 "weights_sha256": BASE_WEIGHTS_SHA},
        "ckpt_sha256": CKPT_SHA256,
        "merge": {
            "recipe": "PeftModel.from_pretrained(base,adapter) -> "
                      "load_state_dict(merger.pt, strict=False, all-matched) "
                      "-> merge_and_unload() -> save_pretrained(bf16)",
            "merger_tensors": n_merger_tensors,
            "lora": {"r": 16, "alpha": 32, "targets": "q/k/v/o/gate/up/down",
                     "exclude": "visual"},
            "note": "fp32 adapter deltas merged into bf16 base; equivalent "
                    "to runtime adapter application within bf16 rounding",
        },
        "merged_files_sha256": file_shas,
        "merged_tree_sha256": tree_sha,
        "prior_tree_sha256_geetha_volume":
            "5b720abf… (per spec; cross-account volume unreachable — "
            "compare only if same file set)",
        "account": "proxynanmaga",
        "run": _runstats(t0, cpu=8.0, mem_gib=96),
    }
    mman_path.write_text(json.dumps(mman, indent=2) + "\n")
    vol.commit()
    report.setdefault("status", "OK")
    report.update({"merged_tree_sha256": tree_sha, "n_files": len(file_shas)})
    report["run"] = mman["run"]
    return _js(report)


# --------------------------------------------------------------------------
# Step 2 — tiny smoke on merged model (200 val rows) before spending on full
# --------------------------------------------------------------------------

@app.function(
    image=train_image,
    volumes={DATA: vol},
    gpu="A100-40GB",
    memory=96 * 1024,
    timeout=3600,
)
def smoke_merged(n: int = 200) -> dict:
    """First n rows of rsvqa_hr_val gold through the merged tree. tok_exact
    should land near the 0.8138 floor — a garbage result halts the lane."""
    import torch

    t0 = time.time()
    vol.reload()
    report = {"step": "smoke_merged", "n": n}
    try:
        gold = _load_mix_rows(RUN / "gold" / "rsvqa_hr_val.jsonl")[:n]
        img_cache = _preload_images([g["image"] for g in gold])
        model, processor = _load_merged("cuda")
        model = model.cuda()
        model.eval()
        _set_use_cache(model, True)
        for g in gold:
            g["_img"] = g["image"]
            g["_pil"] = img_cache[g["image"]]
        preds = _gen_eval(model, processor, processor.tokenizer, gold, "cuda")
        em = sum(1 for g, p in zip(gold, preds)
                 if tok_exact(p) == tok_exact(g["gold"])) / len(gold)
        report["n"] = len(gold)
        report["tok_exact_em"] = round(em, 4)
        report["samples"] = [
            {"id": g["id"], "gold": g["gold"], "pred": p}
            for g, p in list(zip(gold, preds))[:15]
        ]
        report["status"] = "OK"
    except Exception as e:  # noqa: BLE001
        report["status"] = "ERROR"
        report["error"] = f"{type(e).__name__}: {e}"
        print(f"SMOKE_MERGED ERROR {e}", flush=True)
    report["run"] = _runstats(t0, gpu_s_rate=USD_A100_40_S, mem_gib=96)
    _write_json(Path(DATA) / "runs" / "lr_fold" / "smoke_merged.json", report)
    vol.commit()
    return _js(report)


# --------------------------------------------------------------------------
# Step 3 — frozen eval on the merged tree (sharded fan-out) + scoring
# --------------------------------------------------------------------------

@app.function(
    image=train_image,
    volumes={DATA: vol},
    gpu="A100-40GB",
    memory=96 * 1024,
    timeout=4 * 3600,
)
def eval_shard(column: str, shard_idx: int, n_shards: int) -> dict:
    """Generate preds for gold[column] contiguous shard — identical chunking,
    batching, decode and output schema as the floor run's eval_shard."""
    import torch

    t0 = time.time()
    vol.reload()
    out_dir = RUN / "eval" / EVAL_TAG / column
    out_path = out_dir / f"preds_{shard_idx:03d}.jsonl"
    report = {"step": "eval_shard", "ckpt": EVAL_TAG, "column": column,
              "shard": shard_idx, "n_shards": n_shards,
              "model": "merged_eval (LoRA+trained merger baked)"}
    try:
        if out_path.exists() and os.environ.get("FORCE_EVAL") != "1":
            report["status"] = "SKIP_EXISTS"
            report["run"] = _runstats(t0, gpu_s_rate=USD_A100_40_S, mem_gib=96)
            return _js(report)
        gold = _load_mix_rows(RUN / "gold" / f"{column}.jsonl")
        chunk = math.ceil(len(gold) / n_shards)
        rows = gold[shard_idx * chunk : (shard_idx + 1) * chunk]
        report["n_rows"] = len(rows)
        if not rows:
            report["status"] = "EMPTY"
            return _js(report)

        t_pre = time.time()
        needed = sorted({r["image"] for r in rows})
        img_cache = _preload_images(needed)
        report["preload_s"] = round(time.time() - t_pre, 1)
        report["n_images"] = len(img_cache)
        for r in rows:
            r["_img"] = r["image"]
            r["_pil"] = img_cache[r["image"]]

        model, processor = _load_merged("cuda")
        model = model.cuda()
        model.eval()
        _set_use_cache(model, True)

        t_ev = time.time()
        preds = _gen_eval(model, processor, processor.tokenizer, rows, "cuda")
        report["gen_s"] = round(time.time() - t_ev, 1)
        report["per_s"] = round(len(rows) / report["gen_s"], 2)

        out_dir.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for r, p in zip(rows, preds):
                f.write(json.dumps({"id": r["id"], "question": r["question"],
                                    "gold": r["gold"], "pred": p,
                                    "type": r["type"]},
                                   ensure_ascii=False) + "\n")
        report["status"] = "OK"
    except Exception as e:  # noqa: BLE001
        report["status"] = "ERROR"
        report["error"] = f"{type(e).__name__}: {e}"
        print(f"EVAL {column}[{shard_idx}] ERROR {e}", flush=True)
    report["run"] = _runstats(t0, gpu_s_rate=USD_A100_40_S, mem_gib=96)
    vol.commit()
    return _js(report)


@app.function(
    image=cpu_image,
    volumes={DATA: vol},
    cpu=4.0,
    memory=16 * 1024,
    timeout=3600,
)
def score_eval() -> dict:
    """Merge shard preds -> per column OA + AA + per-family; RAW EM +
    token-exact EM + normalized EM. VERBATIM from the floor run's score_eval
    (ckpt_tag fixed to EVAL_TAG)."""
    t0 = time.time()
    vol.reload()
    ckpt_tag = EVAL_TAG
    vocab = json.loads((RUN / "vocab.json").read_text())["vocab"]
    vocab_set = set(vocab)
    proj_cache: dict[str, str] = {}

    def project(s: str) -> str:
        if s in vocab_set:
            return s
        if len(s) > 60:
            return s
        if s not in proj_cache:
            proj_cache[s] = vocab_project(s, vocab)
        return proj_cache[s]

    metrics = {"ckpt": ckpt_tag, "columns": {}, "contract": {
        "raw_em": "pred.strip() == gold.strip()",
        "token_exact_em": "tok_exact(): strip+casefold+ws-collapse+rstrip('.') — "
                          "reconstruction of the wiped 0.5077 scorer; bar metric",
        "normalized_em": "canon() (units, digit-words, commas) + nearest-vocab "
                         "projection onto TRAIN answer vocab (cutoff 0.85)",
        "vocab_source": "train answers only — no eval gold used",
        "decode": f"greedy, repetition_penalty={REP_PENALTY}, "
                  f"max_new_tokens={MAX_NEW_TOKENS}",
        "model": "merged_eval: qwen3vl8b base + LoRA merged + trained "
                 "merger (see merged_manifest.json)",
    }}

    for col, cfg in COLUMNS.items():
        cdir = RUN / "eval" / ckpt_tag / col
        if not cdir.exists():
            metrics["columns"][col] = {"status": "MISSING"}
            continue
        rows = []
        for p in sorted(cdir.glob("preds_*.jsonl")):
            with p.open() as f:
                for line in f:
                    rows.append(json.loads(line))
        # de-dup by id (reruns)
        seen = {}
        for r in rows:
            seen[r["id"]] = r
        rows = list(seen.values())
        gold_n = len(eval_ids_read(col))
        fams = {}
        n_raw = n_tok = n_norm = 0
        for r in rows:
            g, p = r["gold"], r["pred"]
            fams.setdefault(r["type"], [0, 0])
            ok_tok = tok_exact(p) == tok_exact(g)
            fams[r["type"]][1] += int(ok_tok)
            fams[r["type"]][0] += 1
            n_raw += int(p.strip() == g.strip())
            n_tok += int(ok_tok)
            n_norm += int(project(canon(p)) == project(canon(g)))
        n = len(rows)
        aa = sum(v[1] / v[0] for v in fams.values()) / len(fams) if fams else 0.0
        metrics["columns"][col] = {
            "n_preds": n, "n_gold": gold_n,
            "coverage": round(n / gold_n, 4) if gold_n else None,
            "oa_raw_em": round(n_raw / n, 4) if n else None,
            "oa_token_exact_em": round(n_tok / n, 4) if n else None,
            "oa_normalized_em": round(n_norm / n, 4) if n else None,
            "aa_token_exact": round(aa, 4),
            "per_family_token_exact": {
                k: {"n": v[0], "acc": round(v[1] / v[0], 4)} for k, v in sorted(fams.items())
            },
        }
    bar = metrics["columns"].get(BAR_COL, {})
    metrics["bar"] = {
        "column": BAR_COL, "target": BAR,
        "achieved_token_exact": bar.get("oa_token_exact_em"),
        "achieved_normalized": bar.get("oa_normalized_em"),
        "verdict": ("BEAT" if (bar.get("oa_token_exact_em") or 0) > BAR else "PARK"),
    }
    metrics["run"] = _runstats(t0, cpu=4.0, mem_gib=16)
    _write_json(RUN / "eval" / ckpt_tag / "metrics.json", metrics)
    vol.commit()
    return _js(metrics)


# --------------------------------------------------------------------------
# Local entrypoints
# --------------------------------------------------------------------------

@app.local_entrypoint()
def e_stage():
    print(stage_check.remote())


@app.local_entrypoint()
def e_merge():
    print(merge_ckpt.remote())


@app.local_entrypoint()
def e_smoke():
    n = int(os.environ.get("SMOKE_N", "200"))
    print(smoke_merged.remote(n))


@app.local_entrypoint()
def e_eval():
    only = os.environ.get("ONLY_COL")
    cols = [c.strip() for c in only.split(",")] if only else list(COLUMNS)
    for c in cols:
        assert c in COLUMNS, f"unknown column {c}"
    jobs = []
    for col in cols:
        cfg = COLUMNS[col]
        for i in range(cfg["shards"]):
            jobs.append((col, i, cfg["shards"]))
    print(f"fan-out {len(jobs)} shard jobs: {cols}")
    results = list(eval_shard.starmap(jobs))
    for r in results:
        print(r)


@app.local_entrypoint()
def e_score():
    print(score_eval.remote())
