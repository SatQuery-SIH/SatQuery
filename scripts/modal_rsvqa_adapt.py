"""RSVQA-ADAPT — joint LoRA canonical-answer producer (MASTER_PLAN_V2 §4.1/§5).

Bounded task. One Modal app, gated pipeline. Modal profile `harsha-610vmg`
(workspace harsha-610vmg) only — legacy spare/ripper2005 profiles are
forbidden. Every function logs RUNSTATS {wall_seconds, est_usd} and writes
JSON reports under /data/runs/rsvqa_adapt/ on volume `satquery-data`.

Data on volume:
    /data/rsvqa/hr/Data/{img_id}.png        RSVQA-HR PNG mirrors (tif also exist)
    /data/rsvqa/hr/USGS_split_*_{questions,answers,images}.json
    /data/rsvqa/lr/Images_LR/{img_id}.tif
    /data/rsvqa/lr/LR_split_*_{questions,answers,images}.json
    /data/vrsbench/Images_train/{image_id}.png, Images_val/, VRSBench_train.json,
        VRSBench_EVAL_{vqa,Cap,referring}.json
    /data/eval_ids/*.json                   frozen eval ids (uploaded, authoritative)

Pipeline (local entrypoints — run in order):

    python -m modal run modal_rsvqa_adapt.py::e_preflight    # Step 0 gate
    python -m modal run modal_rsvqa_adapt.py::e_mix          # Step 1 mix + gold + overlap gate
    CANDIDATE=qwen3vl python -m modal run modal_rsvqa_adapt.py::e_smoke    # Step 2 gate
    CANDIDATE=qwen3vl  python -m modal run modal_rsvqa_adapt.py::e_bakeoff # Step 3
    CANDIDATE=qwen25vl python -m modal run modal_rsvqa_adapt.py::e_bakeoff
    CANDIDATE=<winner> python -m modal run modal_rsvqa_adapt.py::e_train   # Step 4
    CKPT_TAG=full_<cand> python -m modal run modal_rsvqa_adapt.py::e_eval  # Step 5 fan-out
    CKPT_TAG=full_<cand> python -m modal run modal_rsvqa_adapt.py::e_score # Step 5 metrics

Hard rules: frozen eval ids before preds; zero train/eval overlap asserted +
written to manifest; masked loss (-100 on prompt+vision); inverse-frequency
class weights; count oversampled ~2.5x; 15-20% caption regularizer; RAW EM +
token-exact EM + normalized EM reported separately; bar = beat 0.5077
token-exact on frozen RSVQA-HR val.
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
# Constants / contract
# --------------------------------------------------------------------------

VOLUME_NAME = "satquery-data"
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

DATA = "/data"
RUN = Path(DATA) / "runs" / "rsvqa_adapt"
EVAL_IDS_DIR = Path(DATA) / "eval_ids"
HF_CACHE = "/data/_hf"          # persisted HF downloads on the volume

RSVQA_HR = Path(DATA) / "rsvqa" / "hr"
RSVQA_LR = Path(DATA) / "rsvqa" / "lr"
VRSB = Path(DATA) / "vrsbench"

CANDIDATES = {
    "qwen3vl": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen25vl": "Qwen/Qwen2.5-VL-7B-Instruct",
}

# Recipe (MASTER_PLAN_V2 §4.1 / DR1)
LORA_R = 16
LORA_ALPHA = 32
LORA_LR = 1e-4
PROJ_LR = 1e-5
EFF_BATCH = 48           # micro_b x grad_accum
MICRO_B = 8
EPOCHS = 2
WARMUP_FRAC = 0.03
MAX_PIXELS = 768 * 768   # vision-token bound; RSVQA 512px images unaffected
SEED = 42

# Mix (Step 1)
N_RSVQA = 50_000
N_VRSB_VQA = 50_000
COUNT_OVERSAMPLE = 2.5   # count is weakest family (0.415 zero-shot)

# Decode contract (§5)
REP_PENALTY = 1.08
MAX_NEW_TOKENS = 16

# Frozen eval columns: file in /data/eval_ids + shard count for Step 5
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

# Modal list pricing (modal.com/pricing)
USD_A100_40_S = 0.000944   # ~$3.40/h
USD_CPU_CORE_S = 0.0000131
USD_MEM_GIB_S = 0.00000222

app = modal.App("satquery-rsvqa-adapt")

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
    )
    .env({"HF_HOME": HF_CACHE, "TOKENIZERS_PARALLELISM": "false"})
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
# Shared model helpers (run inside containers only)
# --------------------------------------------------------------------------

def _load_base(model_id: str):
    """bf16 base on cuda. Tolerates transformers auto-class renames."""
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as AutoModel
    except ImportError:  # older name
        from transformers import AutoModelForVision2Seq as AutoModel

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    return model, processor


def _attach_lora(model):
    """LoRA r16/a32 on LM attn+MLP; visual tower excluded (ViT frozen anyway)."""
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        exclude_modules=[r".*visual.*", r".*vision_tower.*"],
    )
    return get_peft_model(model, cfg)


def _freeze_and_unfreeze_merger(model) -> list[str]:
    """Freeze everything peft didn't already; unfreeze the vision projector
    (merger). Returns list of unfrozen merger param names."""
    for n, p in model.named_parameters():
        if "lora_" in n:
            continue
        p.requires_grad = False
    merger = []
    for n, p in model.named_parameters():
        if "merger" in n:
            p.requires_grad = True
            merger.append(n)
    return merger


def _set_use_cache(model, flag: bool) -> None:
    cfg = model.config
    if hasattr(cfg, "use_cache"):
        cfg.use_cache = flag
    for sub in ("text_config",):
        if hasattr(cfg, sub) and hasattr(getattr(cfg, sub), "use_cache"):
            getattr(cfg, sub).use_cache = flag


def _trainable_summary(model) -> dict:
    lora_p = sum(p.numel() for n, p in model.named_parameters()
                 if p.requires_grad and "lora_" in n)
    proj_p = sum(p.numel() for n, p in model.named_parameters()
                 if p.requires_grad and "lora_" not in n)
    lora_in_visual = [n for n, p in model.named_parameters()
                      if "lora_" in n and ("visual" in n or "vision_tower" in n)]
    return {
        "lora_params": lora_p,
        "projector_params": proj_p,
        "lora_in_visual_count": len(lora_in_visual),
        "lora_in_visual_names": lora_in_visual[:10],
    }


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


def _make_train_batch(rows, processor, tokenizer, device):
    """Prompt+vision masked -100; answer + <|im_end|> supervised."""
    import torch

    prompts = [_chat_prompt(processor, r["question"], r["_pil"]) for r in rows]
    images = [r["_pil"] for r in rows]
    enc = processor(text=prompts, images=images, padding=True, return_tensors="pt")

    seqs, labs = [], []
    for i, r in enumerate(rows):
        m = enc.attention_mask[i].bool()
        p_ids = enc.input_ids[i][m].tolist()          # real (unpadded) prompt ids
        a_ids = tokenizer(
            r["answer"] + "<|im_end|>\n", add_special_tokens=False
        ).input_ids
        seqs.append(p_ids + a_ids)
        labs.append([-100] * len(p_ids) + a_ids)

    T = max(len(s) for s in seqs)
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids = torch.full((len(seqs), T), pad, dtype=torch.long)
    labels = torch.full((len(seqs), T), -100, dtype=torch.long)
    attn = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        input_ids[i, : len(s)] = torch.tensor(s)
        labels[i, : len(l)] = torch.tensor(l)
        attn[i, : len(s)] = 1

    batch = {
        "input_ids": input_ids.to(device),
        "attention_mask": attn.to(device),
        "labels": labels.to(device),
    }
    for k in ("pixel_values", "image_grid_thw"):
        if k in enc and enc[k] is not None:
            batch[k] = enc[k].to(device)
    return batch


def _weighted_loss(out, labels, weights):
    """Per-sample mean CE over supervised tokens, reweighted by class weight."""
    import torch
    import torch.nn.functional as F

    logits = out.logits[:, :-1, :].float()
    targets = labels[:, 1:]
    tok_ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view(targets.size())
    tok_mask = (targets != -100).float()
    per_sample = (tok_ce * tok_mask).sum(1) / tok_mask.sum(1).clamp(min=1.0)
    # token-level accuracy on supervised positions (diagnostic)
    pred = logits.argmax(-1)
    tok_acc = ((pred == targets) & (targets != -100)).float().sum() / tok_mask.sum().clamp(min=1.0)
    w = weights.to(per_sample.device)
    return (per_sample * w).sum() / w.sum().clamp(min=1e-6), tok_acc.item()


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
# Normalizer — DISCLOSED output-contract post-proc (never hidden)
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


# --------------------------------------------------------------------------
# Step 0 — framework pre-flight (gate)
# --------------------------------------------------------------------------

@app.function(
    image=train_image,
    volumes={DATA: vol},
    gpu="A100-40GB",
    memory=64 * 1024,
    timeout=3600,
)
def preflight() -> dict:
    """(a) load model, (b) LoRA attn/MLP + train projector, (c) one masked
    fwd/bwd. Candidate A = qwen3vl; on ANY failure also test qwen25vl so the
    forfeit decision is evidence-based."""
    import torch

    t0 = time.time()
    vol.reload()
    report = {"step": "preflight", "results": {}}

    import transformers, peft
    report["versions"] = {
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    print("VERSIONS " + json.dumps(report["versions"]), flush=True)

    # two real rows for the fwd/bwd probe
    qs = json.loads((RSVQA_HR / "USGS_split_train_questions.json").read_text())["questions"]
    act = [q for q in qs if q.get("active")][:2]
    rows = []
    for q in act:
        rows.append({
            "question": q["question"],
            "answer": "yes" if q["type"] == "presence" else "3",
            "_pil": _load_rgb(RSVQA_HR / "Data" / f"{q['img_id']}.png"),
        })

    for cand in ("qwen3vl", "qwen25vl"):
        if cand == "qwen25vl" and report["results"].get("qwen3vl", {}).get("pass"):
            continue  # only probe B if A failed
        res = {"model_id": CANDIDATES[cand]}
        try:
            model, processor = _load_base(CANDIDATES[cand])
            model = model.cuda()
            res["load"] = "ok"
        except Exception as e:  # noqa: BLE001
            res["load"] = f"FAIL: {type(e).__name__}: {e}"
            res["pass"] = False
            report["results"][cand] = res
            print(f"PREFLIGHT {cand} load FAIL {e}", flush=True)
            continue
        try:
            model = _attach_lora(model)
            merger = _freeze_and_unfreeze_merger(model)
            res["merger_params_unfrozen"] = len(merger)
            res["merger_sample"] = merger[:6]
            ts = _trainable_summary(model)
            res["trainable"] = ts
            assert ts["lora_params"] > 0, "no LoRA params attached"
            assert ts["lora_in_visual_count"] == 0, \
                f"LoRA leaked into ViT: {ts['lora_in_visual_names']}"
            assert ts["projector_params"] > 0, "projector (merger) not unfrozen"

            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            _set_use_cache(model, False)
            batch = _make_train_batch(rows, processor, processor.tokenizer, "cuda")
            n_sup = int((batch["labels"] != -100).sum())
            res["supervised_tokens"] = n_sup
            assert n_sup > 0
            img_ids = [t for t in batch["input_ids"].unique().tolist()
                       if processor.tokenizer.convert_ids_to_tokens(t) in
                       ("<|image_pad|>", "<|image_soft_token|>")]
            if img_ids:
                pos = (batch["input_ids"] == img_ids[0])
                assert int((batch["labels"][pos] != -100).sum()) == 0, \
                    "vision tokens not masked"
            res["vision_tokens_masked"] = True

            lora_ids = [id(p) for n, p in model.named_parameters() if "lora_" in n]
            out = model(**{k: v for k, v in batch.items() if k != "labels"},
                        labels=batch["labels"])
            loss = out.loss
            loss.backward()
            res["loss"] = float(loss)
            grads_lora = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for n, p in model.named_parameters() if "lora_" in n
            )
            grads_merger = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for n, p in model.named_parameters()
                if "merger" in n and p.requires_grad
            )
            grads_vit = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for n, p in model.named_parameters()
                if "visual" in n and "merger" not in n and "lora_" not in n
            )
            res["grads"] = {
                "lora": bool(grads_lora), "merger": bool(grads_merger),
                "vit_blocks": bool(grads_vit),
            }
            assert grads_lora and grads_merger and not grads_vit

            # generate sanity
            model.eval()
            _set_use_cache(model, True)
            enc = processor(text=[_chat_prompt(processor, rows[0]["question"],
                                               rows[0]["_pil"])],
                            images=[rows[0]["_pil"]], return_tensors="pt").to("cuda")
            with torch.inference_mode():
                g = model.generate(**enc, max_new_tokens=8, do_sample=False)
            res["gen_sample"] = processor.tokenizer.decode(
                g[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
            res["pass"] = True
        except Exception as e:  # noqa: BLE001
            res["pass"] = False
            res["error"] = f"{type(e).__name__}: {e}"
            print(f"PREFLIGHT {cand} FAIL {e}", flush=True)
        finally:
            del model
            torch.cuda.empty_cache()
        report["results"][cand] = res
        print(f"PREFLIGHT {cand} " + json.dumps(res, default=str)[:600], flush=True)

    report["verdict"] = {
        "candidate_a": "qwen3vl " + ("PASS" if report["results"].get("qwen3vl", {}).get("pass") else "FORFEIT"),
        "candidate_b": "qwen25vl " + ("available" if report["results"].get("qwen25vl", {}).get("pass") else "not probed" if "qwen25vl" not in report["results"] else "FAIL"),
    }
    report["run"] = _runstats(t0, gpu_s_rate=USD_A100_40_S, mem_gib=64)
    _write_json(RUN / "preflight_report.json", report)
    vol.commit()
    return _js(report)


# --------------------------------------------------------------------------
# Step 1 — joint mix build + frozen-gold prep + overlap gate (CPU)
# --------------------------------------------------------------------------

def _rsvqa_rows(root: Path, split: str, prefix: str, img_subdir: str, img_ext: str):
    """Join questions/answers/images for a split. Returns active rows:
    {uid, qid, img_id, question, answer, type, image_path}."""
    qs = json.loads((root / f"{prefix}_split_{split}_questions.json").read_text())["questions"]
    ans = json.loads((root / f"{prefix}_split_{split}_answers.json").read_text())["answers"]
    ims = json.loads((root / f"{prefix}_split_{split}_images.json").read_text())["images"]
    amap = {a["id"]: a for a in ans}
    imap = {i["id"]: i for i in ims}
    rows = []
    for q in qs:
        if not q.get("active"):
            continue
        a = amap.get(q["answers_ids"][0])
        if a is None or not a.get("active"):
            continue
        rows.append({
            "qid": q["id"],
            "img_id": q["img_id"],
            "question": q["question"],
            "answer": str(a["answer"]),
            "type": q["type"],
            "image_path": str(root / img_subdir / f"{q['img_id']}{img_ext}"),
        })
    return rows


def _vrsb_rows():
    rows = []
    data = json.loads((VRSB / "VRSBench_train.json").read_text())
    for i, r in enumerate(data):
        conv = r["conversations"]
        q_raw = conv[0]["value"]
        m = re.search(r"\[(\w+)\]", q_raw)
        tag = m.group(1) if m else "none"
        text = re.sub(r"<image>\s*", "", q_raw)
        text = re.sub(r"\[\w+\]\s*", "", text).strip()
        rows.append({
            "row_idx": i,
            "tag": tag,
            "image": r["image"],
            "question": text,
            "answer": conv[1]["value"].strip(),
            "image_path": str(VRSB / "Images_train" / r["image"]),
        })
    return rows


@app.function(
    image=cpu_image,
    volumes={DATA: vol},
    cpu=4.0,
    memory=32 * 1024,
    timeout=3600,
)
def build_mix() -> dict:
    """~50k RSVQA-HR (count x2.5) + ~50k VRSBench-VQA + all caption rows
    (~17%). Writes mix.jsonl, vocab.json, gold/{col}.jsonl. HARD GATE: any
    overlap with frozen eval ids -> status=STOP."""
    import random

    t0 = time.time()
    vol.reload()
    rng = random.Random(SEED)
    manifest = {"step": "build_mix", "seed": SEED}

    # ---- frozen eval id sets -------------------------------------------------
    eval_ids = {}
    eval_ids_ordered = {}
    eval_sha = {}
    for col, cfg in COLUMNS.items():
        d = json.loads((EVAL_IDS_DIR / cfg["ids"]).read_text())
        eval_ids[col] = set(d["ids"])
        eval_ids_ordered[col] = list(d["ids"])   # frozen file order — "first 5k"
        eval_sha[col] = d.get("ids_sha256")
    # extra vrsbench eval columns (caption/referring) — for the image-overlap
    # gate only; they are not scored in this task
    vrsb_extra_eval = set()
    for extra in ("vrsbench_caption_eval_ids.json", "vrsbench_referring_eval_ids.json"):
        p = EVAL_IDS_DIR / extra
        if p.exists():
            vrsb_extra_eval |= set(json.loads(p.read_text())["ids"])
    manifest["eval_ids_sha256"] = eval_sha
    manifest["eval_n"] = {c: len(s) for c, s in eval_ids.items()}

    # ---- RSVQA-HR train ------------------------------------------------------
    src_sha = {}
    for f in ("USGS_split_train_questions.json", "USGS_split_train_answers.json",
              "USGS_split_train_images.json"):
        src_sha[f"rsvqa/hr/{f}"] = _sha256(RSVQA_HR / f)
    hr_rows = _rsvqa_rows(RSVQA_HR, "train", "USGS", "Data", ".png")
    by_type = {}
    for r in hr_rows:
        by_type.setdefault(r["type"], []).append(r)
    type_counts = {t: len(v) for t, v in by_type.items()}
    weights = {t: (COUNT_OVERSAMPLE if t == "count" else 1.0) for t in by_type}
    wsum = sum(type_counts[t] * weights[t] for t in by_type)
    quota = {t: int(round(N_RSVQA * type_counts[t] * weights[t] / wsum))
             for t in by_type}
    picked = []
    for t, rows in sorted(by_type.items()):
        rng.shuffle(rows)
        picked += rows[: quota[t]]
    rsvqa_pick = [{
        "uid": f"rsvqa_hr_train:{r['img_id']}:{r['qid']}",
        "eval_id": f"{r['img_id']}:{r['qid']}",
        "source": "rsvqa_hr", "qtype": r["type"],
        "question": r["question"], "answer": r["answer"],
        "image": r["image_path"],
    } for r in picked]

    # ---- VRSBench train -------------------------------------------------------
    vrows = _vrsb_rows()
    vqa = [r for r in vrows if r["tag"] == "vqa"]
    cap = [r for r in vrows if r["tag"] == "caption"]
    rng.shuffle(vqa)
    vqa_pick = [{
        "uid": f"vrsb_vqa_train:{r['row_idx']}",
        "eval_id": f"{r['image']}#?",      # train rows carry no eval qid; image checked below
        "source": "vrsbench_vqa", "qtype": "vqa",
        "question": r["question"], "answer": r["answer"],
        "image": r["image_path"],
    } for r in vqa[:N_VRSB_VQA]]
    cap_pick = [{
        "uid": f"vrsb_cap_train:{r['row_idx']}",
        "eval_id": f"{r['image']}#?",
        "source": "vrsbench_cap", "qtype": "caption",
        "question": r["question"], "answer": r["answer"],
        "image": r["image_path"],
    } for r in cap]

    mix = rsvqa_pick + vqa_pick + cap_pick

    # ---- OVERLAP GATE (dataset-scoped: 'img:qid' namespaces collide across
    # datasets — HR train id '232:23215' != LR eval '232:23215'. Question ids
    # are GLOBAL within a dataset: same table in every split file, `active`
    # marks membership — so same-dataset id overlap IS a real leak.) ----------
    hr_eval = eval_ids["rsvqa_hr_val"] | eval_ids["rsvqa_hr_test"] \
        | eval_ids["rsvqa_hr_test_phili"]
    lr_eval = eval_ids["rsvqa_lr_val"] | eval_ids["rsvqa_lr_test"]
    rsvqa_overlap = sorted(set(r["eval_id"] for r in rsvqa_pick) & hr_eval)
    # image-level check for vrsbench: no eval image may appear in the train mix
    eval_images = {eid.split("#")[0]
                   for eid in eval_ids["vrsbench_vqa"] | vrsb_extra_eval
                   if "#" in eid}
    mix_images = {r["image"].split("/")[-1] for r in mix if r["source"] != "rsvqa_hr"}
    vrsb_img_overlap = sorted(eval_images & mix_images)
    # diagnostics: cross-namespace collisions (expected, not leaks) + shared
    # HR image ids (legal: splits partition questions, images repeat)
    cross_ns = sorted(set(r["eval_id"] for r in rsvqa_pick) & lr_eval)
    rsvqa_eval_imgs = {eid.split(":")[0] for eid in hr_eval}
    rsvqa_img_overlap = sorted(
        {r["image"].split("/")[-1].split(".")[0] for r in rsvqa_pick}
        & rsvqa_eval_imgs
    )
    manifest["overlap"] = {
        "rsvqa_hr_id_overlap_n": len(rsvqa_overlap),
        "rsvqa_hr_id_overlap_sample": rsvqa_overlap[:10],
        "vrsb_image_overlap_n": len(vrsb_img_overlap),
        "vrsb_image_overlap_sample": vrsb_img_overlap[:10],
        "diag_cross_namespace_collisions_vs_lr_eval_n": len(cross_ns),
        "diag_shared_hr_image_ids_n": len(rsvqa_img_overlap),
        "note": "gate is dataset-scoped id-level; question ids are global "
                "within rsvqa-hr (same table, active=membership), so "
                "same-dataset overlap = real leak. LR/HR 'img:qid' strings "
                "collide across different datasets — not a leak.",
    }
    if rsvqa_overlap or vrsb_img_overlap:
        manifest["status"] = "STOP"
        manifest["reason"] = "train/eval overlap detected"
        _write_json(RUN / "mix_manifest.json", manifest)
        vol.commit()
        return _js(manifest)

    # ---- class weights (inverse frequency, captions = 1.0) --------------------
    from collections import Counter

    freq = Counter(r["answer"] for r in mix if r["qtype"] != "caption")
    n_cls = len(freq)
    n_w = sum(freq.values())
    w_rows = []
    for r in mix:
        if r["qtype"] == "caption":
            r["weight"] = 1.0
        else:
            w = n_w / (n_cls * freq[r["answer"]])
            r["weight"] = min(max(w, 0.25), 4.0)
        w_rows.append(r["weight"])
    wsum2 = sum(r["weight"] for r in mix if r["qtype"] != "caption")
    n2 = sum(1 for r in mix if r["qtype"] != "caption")
    for r in mix:
        if r["qtype"] != "caption":
            r["weight"] = round(r["weight"] * n2 / wsum2, 6)
    manifest["class_weight_rule"] = (
        "w = clip(N/(K*freq), 0.25, 4.0) renormalized to mean 1.0 over "
        "non-caption rows; caption rows fixed 1.0 (regularizer, not weighted)"
    )

    rng.shuffle(mix)
    mix_path = RUN / "mix.jsonl"
    mix_path.parent.mkdir(parents=True, exist_ok=True)
    with mix_path.open("w") as f:
        for r in mix:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- vocab (train answers only — never eval gold) -------------------------
    vocab = sorted({canon(r["answer"]) for r in mix if r["qtype"] != "caption"})
    _write_json(RUN / "vocab.json", {"rule": "canon() over train answers only", "vocab": vocab})

    # ---- frozen gold files for Step 5 -----------------------------------------
    gold_dir = RUN / "gold"
    gold_stats = {}
    hr_cache = {}

    def hr_split(split):
        if split not in hr_cache:
            hr_cache[split] = _rsvqa_rows(RSVQA_HR, split, "USGS", "Data", ".png")
        return hr_cache[split]

    lr_cache = {}

    def lr_split(split):
        if split not in lr_cache:
            lr_cache[split] = _rsvqa_rows(RSVQA_LR, split, "LR", "Images_LR", ".tif")
        return lr_cache[split]

    def write_gold(col, rows):
        path = gold_dir / f"{col}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        gold_stats[col] = len(rows)

    for col in ("rsvqa_hr_val", "rsvqa_hr_test", "rsvqa_hr_test_phili"):
        split = col.replace("rsvqa_hr_", "")
        rows = hr_split(split)
        by_id = {f"{r['img_id']}:{r['qid']}": r for r in rows}
        out, missing = [], 0
        for eid in eval_ids_ordered[col]:
            r = by_id.get(eid)
            if r is None:
                missing += 1
                continue
            out.append({"id": eid, "question": r["question"], "gold": r["answer"],
                        "type": r["type"], "image": r["image_path"]})
        write_gold(col, out)
        gold_stats[col + "_missing"] = missing

    for col in ("rsvqa_lr_val", "rsvqa_lr_test"):
        split = col.replace("rsvqa_lr_", "")
        rows = lr_split(split)
        by_id = {f"{r['img_id']}:{r['qid']}": r for r in rows}
        out, missing = [], 0
        for eid in eval_ids_ordered[col]:
            r = by_id.get(eid)
            if r is None:
                missing += 1
                continue
            out.append({"id": eid, "question": r["question"], "gold": r["answer"],
                        "type": r["type"], "image": r["image_path"]})
        write_gold(col, out)
        gold_stats[col + "_missing"] = missing

    # vrsbench eval images: resolve against both image dirs
    vrsb_img_dirs = {}
    for d in ("Images_val", "Images_train"):
        for fn in os.listdir(VRSB / d):
            vrsb_img_dirs.setdefault(fn, str(VRSB / d / fn))
    ev = json.loads((VRSB / "VRSBench_EVAL_vqa.json").read_text())
    src_sha["vrsbench/VRSBench_EVAL_vqa.json"] = _sha256(VRSB / "VRSBench_EVAL_vqa.json")
    ev_by_id = {f"{r['image_id']}#{r['question_id']}": r for r in ev}
    out, missing, missing_img = [], 0, 0
    for eid in eval_ids_ordered["vrsbench_vqa"]:
        r = ev_by_id.get(eid)
        if r is None:
            missing += 1
            continue
        ip = vrsb_img_dirs.get(r["image_id"])
        if ip is None:
            missing_img += 1
            continue
        out.append({"id": eid, "question": r["question"],
                    "gold": str(r["ground_truth"]), "type": r.get("type", "vqa"),
                    "image": ip})
    write_gold("vrsbench_vqa", out)
    gold_stats["vrsbench_vqa_missing"] = missing
    gold_stats["vrsbench_vqa_missing_img"] = missing_img

    manifest["source_sha256"] = src_sha
    manifest["mix"] = {
        "total": len(mix),
        "rsvqa_hr": len(rsvqa_pick),
        "rsvqa_hr_by_type": {t: sum(1 for r in rsvqa_pick if r["qtype"] == t)
                             for t in sorted(by_type)},
        "vrsbench_vqa": len(vqa_pick),
        "vrsbench_cap": len(cap_pick),
        "caption_frac": round(len(cap_pick) / len(mix), 4),
        "count_oversample": COUNT_OVERSAMPLE,
        "train_pool_type_counts": type_counts,
    }
    manifest["gold"] = gold_stats
    manifest["mix_sha256"] = _sha256(mix_path)
    manifest["status"] = "OK"
    manifest["run"] = _runstats(t0, cpu=4.0, mem_gib=32)
    _write_json(RUN / "mix_manifest.json", manifest)
    vol.commit()
    return _js(manifest)


# --------------------------------------------------------------------------
# Step 2 — overfit smoke (gate)
# --------------------------------------------------------------------------

def _load_mix_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _stratified(rows, n, seed):
    import random
    rng = random.Random(seed)
    by = {}
    for r in rows:
        by.setdefault((r["source"], r["qtype"]), []).append(r)
    out = []
    rest = []
    keys = sorted(by)
    per = max(1, n // len(keys))
    for k in keys:
        rs = by[k]
        rng.shuffle(rs)
        out += rs[:per]
        rest += rs[per:]
    rng.shuffle(rest)
    out += rest[: max(0, n - len(out))]
    rng.shuffle(out)
    return out[:n]


def _train(model, processor, rows, image_cache, *, steps_or_epochs, lr_lora=LORA_LR,
           lr_proj=PROJ_LR, micro_b=MICRO_B, accum=None, log_every=25,
           warmup=WARMUP_FRAC, constant_lr=False, out_dir: Path | None = None,
           ckpt_every=0, resume=False, time_cap_s=None):
    """Masked-loss training loop. Returns log list + final step."""
    import torch
    from transformers import get_cosine_schedule_with_warmup, get_constant_schedule

    device = "cuda"
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    _set_use_cache(model, False)

    accum = accum or max(1, EFF_BATCH // micro_b)
    params_lora = [p for n, p in model.named_parameters()
                   if p.requires_grad and "lora_" in n]
    params_proj = [p for n, p in model.named_parameters()
                   if p.requires_grad and "lora_" not in n]
    opt = torch.optim.AdamW(
        [{"params": params_lora, "lr": lr_lora},
         {"params": params_proj, "lr": lr_proj}],
        weight_decay=0.0,
    )
    import random
    rng = random.Random(SEED)

    if steps_or_epochs[0] == "steps":
        total_opt = steps_or_epochs[1]
        order = list(range(len(rows)))
        rng.shuffle(order)
        # repeat indefinitely; we slice per step
        seq = []
        while len(seq) < (total_opt * accum * micro_b) + micro_b:
            seq += order
            rng.shuffle(order)
    else:
        epochs = steps_or_epochs[1]
        total_opt = (len(rows) * epochs) // (micro_b * accum)
        seq = []
        for _ in range(epochs):
            order = list(range(len(rows)))
            rng.shuffle(order)
            seq += order

    sched = (get_constant_schedule(opt) if constant_lr
             else get_cosine_schedule_with_warmup(
                 opt, int(total_opt * warmup), total_opt))

    start_step = 0
    if resume and out_dir and (out_dir / "ckpt_last" / "step.json").exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        sd = json.loads((out_dir / "ckpt_last" / "step.json").read_text())
        start_step = sd["opt_step"]
        set_peft_model_state_dict(
            model, load_file(str(out_dir / "ckpt_last" / "adapter"
                                 / "adapter_model.safetensors")))
        mp = torch.load(out_dir / "ckpt_last" / "merger.pt", map_location="cpu")
        model.load_state_dict(mp, strict=False)
        opt.load_state_dict(torch.load(out_dir / "ckpt_last" / "opt.pt",
                                       map_location="cpu"))
        for st in opt.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(device)
        sched.step(start_step)
        print(f"RESUME from opt_step {start_step}", flush=True)

    log = []
    model.train()
    t_start = time.time()
    i = start_step * accum * micro_b
    micro = 0
    running = {"loss": 0.0, "acc": 0.0, "n": 0}
    for step in range(start_step, total_opt):
        for _ in range(accum):
            idxs = seq[i : i + micro_b]
            i += micro_b
            batch_rows = []
            for j in idxs:
                r = dict(rows[j])
                r["_pil"] = image_cache[r["image"]]
                batch_rows.append(r)
            batch = _make_train_batch(batch_rows, processor,
                                      processor.tokenizer, device)
            w = torch.tensor([r["weight"] for r in batch_rows],
                             dtype=torch.float32)
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch.get("pixel_values"),
                        image_grid_thw=batch.get("image_grid_thw"))
            loss, tok_acc = _weighted_loss(out, batch["labels"], w)
            (loss / accum).backward()
            running["loss"] += float(loss)
            running["acc"] += tok_acc
            running["n"] += 1
            micro += 1
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % log_every == 0 or step == total_opt - 1:
            rec = {
                "opt_step": step + 1,
                "micro_steps": micro,
                "loss": round(running["loss"] / max(running["n"], 1), 5),
                "tok_acc": round(running["acc"] / max(running["n"], 1), 4),
                "lr_lora": sched.get_last_lr()[0],
                "lr_proj": sched.get_last_lr()[1],
                "elapsed_s": round(time.time() - t_start, 1),
            }
            running = {"loss": 0.0, "acc": 0.0, "n": 0}
            log.append(rec)
            print("TRAINLOG " + json.dumps(rec), flush=True)
        if ckpt_every and out_dir and (step + 1) % ckpt_every == 0:
            _save_ckpt(model, opt, out_dir / "ckpt_last", step + 1)
            vol.commit()
        if time_cap_s and (time.time() - t_start) > time_cap_s:
            print(f"TIME CAP hit at opt_step {step + 1}", flush=True)
            _save_ckpt(model, opt, out_dir / "ckpt_last", step + 1) if out_dir else None
            break
    return log, step + 1


def _save_ckpt(model, opt, path: Path, step: int) -> None:
    import torch
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path / "adapter")
    mp = {n: p.detach().cpu() for n, p in model.named_parameters()
          if "merger" in n and p.requires_grad}
    torch.save(mp, path / "merger.pt")
    torch.save(opt.state_dict(), path / "opt.pt")
    (path / "step.json").write_text(json.dumps({"opt_step": step}))


def _save_adapter_only(model, path: Path) -> None:
    import torch
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path / "adapter")
    mp = {n: p.detach().cpu() for n, p in model.named_parameters()
          if "merger" in n and p.requires_grad}
    torch.save(mp, path / "merger.pt")


def _attach_trained(model, ckpt_dir: Path):
    """Load merger.pt + LoRA adapter for eval."""
    import torch
    from peft import PeftModel

    mp = torch.load(ckpt_dir / "merger.pt", map_location="cpu")
    peft_model = PeftModel.from_pretrained(model, ckpt_dir / "adapter", is_trainable=False)
    res = peft_model.load_state_dict(mp, strict=False)
    assert len(res.unexpected_keys) == 0, \
        f"merger.pt keys unmatched on PEFT model ({len(res.unexpected_keys)} " \
        f"unexpected — merger keys are peft-prefixed; adapter must be attached " \
        f"before the merger load, see merge_ckpt): {res.unexpected_keys[:5]}"
    return peft_model


@app.function(
    image=train_image,
    volumes={DATA: vol},
    gpu="A100-40GB",
    memory=64 * 1024,
    timeout=3600,
)
def smoke(candidate: str = "qwen3vl") -> dict:
    """100-row overfit: masked loss + class weights, ~200 steps, must reach
    >=95% train EM (generated). Gate for the full recipe."""
    import torch

    t0 = time.time()
    vol.reload()
    report = {"step": "smoke", "candidate": candidate}
    try:
        mix = _load_mix_rows(RUN / "mix.jsonl")
        rows = _stratified(mix, 100, seed=7)
        img_cache = _preload_images([r["image"] for r in rows])
        model, processor = _load_base(CANDIDATES[candidate])
        model = model.cuda()
        model = _attach_lora(model)
        merger = _freeze_and_unfreeze_merger(model)
        report["trainable"] = _trainable_summary(model)
        report["n_merger_tensors"] = len(merger)

        log, last = _train(model, processor, rows, img_cache,
                           steps_or_epochs=("steps", 200),
                           micro_b=10, accum=1, log_every=10,
                           constant_lr=True, warmup=0.0)
        report["loss_curve"] = log
        report["final_loss"] = log[-1]["loss"] if log else None
        report["final_tok_acc"] = log[-1]["tok_acc"] if log else None

        model.eval()
        _set_use_cache(model, True)
        for r in rows:
            r["_img"] = r["image"]
            r["_pil"] = img_cache[r["image"]]
        preds = _gen_eval(model, processor, processor.tokenizer, rows, "cuda",
                          batch_size=20)
        # EM gate measured on VQA-answer rows only: caption gold is a paragraph
        # and cannot EM-match by construction — captions are the regularizer,
        # not the gate. Reported per-source for transparency.
        per_src = {}
        fails = []
        for r, p in zip(rows, preds):
            ok = tok_exact(p) == tok_exact(r["answer"])
            s = per_src.setdefault(r["source"], [0, 0])
            s[0] += 1
            s[1] += int(ok)
            if not ok and r["qtype"] != "caption" and len(fails) < 15:
                fails.append({"q": r["question"][:60], "gold": r["answer"],
                              "pred": p, "type": r["qtype"], "src": r["source"]})
        vqa_rows = [(r, p) for r, p in zip(rows, preds) if r["qtype"] != "caption"]
        em = (sum(1 for r, p in vqa_rows
                  if tok_exact(p) == tok_exact(r["answer"])) / len(vqa_rows))
        report["train_em_generated_vqa_only"] = round(em, 4)
        report["per_source_em"] = {
            k: {"n": v[0], "em": round(v[1] / v[0], 4)}
            for k, v in sorted(per_src.items())
        }
        report["vqa_failures"] = fails
        report["pred_samples"] = [
            {"q": r["question"][:60], "gold": r["answer"], "pred": p}
            for r, p in list(zip(rows, preds))[:10]
        ]
        report["gate"] = "PASS" if em >= 0.95 else "FAIL"
        report["status"] = "OK"
    except Exception as e:  # noqa: BLE001
        report["status"] = "ERROR"
        report["error"] = f"{type(e).__name__}: {e}"
        report["gate"] = "FAIL"
        print(f"SMOKE ERROR {e}", flush=True)
    report["run"] = _runstats(t0, gpu_s_rate=USD_A100_40_S, mem_gib=64)
    _write_json(RUN / "smoke" / f"{candidate}_report.json", report)
    vol.commit()
    return _js(report)


# --------------------------------------------------------------------------
# Step 3 — bake-off (10k subsample, eval first 5k frozen val ids)
# --------------------------------------------------------------------------

@app.function(
    image=train_image,
    volumes={DATA: vol},
    gpu="A100-40GB",
    memory=96 * 1024,
    timeout=4 * 3600,
)
def bakeoff(candidate: str = "qwen3vl") -> dict:
    """Same recipe, same seeded 10k subsample for every surviving candidate;
    eval on first 5k frozen RSVQA-HR val ids. Winner = higher tok-exact EM."""
    import torch

    t0 = time.time()
    vol.reload()
    out_dir = RUN / "bakeoff" / candidate
    report = {"step": "bakeoff", "candidate": candidate,
              "model_id": CANDIDATES[candidate]}
    try:
        mix = _load_mix_rows(RUN / "mix.jsonl")
        rows = _stratified(mix, 10_000, seed=11)   # identical across candidates
        gold = _load_mix_rows(RUN / "gold" / "rsvqa_hr_val.jsonl")[:5000]

        t_pre = time.time()
        needed = {r["image"] for r in rows} | {g["image"] for g in gold}
        img_cache = _preload_images(needed)
        report["preload_s"] = round(time.time() - t_pre, 1)
        report["n_images"] = len(img_cache)

        model, processor = _load_base(CANDIDATES[candidate])
        model = model.cuda()
        model = _attach_lora(model)
        _freeze_and_unfreeze_merger(model)
        report["trainable"] = _trainable_summary(model)

        log, last = _train(model, processor, rows, img_cache,
                           steps_or_epochs=("epochs", 1),
                           micro_b=MICRO_B, log_every=20)
        report["loss_tail"] = log[-5:]
        report["opt_steps"] = last

        _save_adapter_only(model, out_dir / "ckpt")

        model.eval()
        _set_use_cache(model, True)
        for g in gold:
            g["_img"] = g["image"]
            g["_pil"] = img_cache[g["image"]]
        t_ev = time.time()
        preds = _gen_eval(model, processor, processor.tokenizer, gold, "cuda")
        report["eval_s"] = round(time.time() - t_ev, 1)
        report["eval_per_s"] = round(len(gold) / report["eval_s"], 2)

        em = sum(1 for g, p in zip(gold, preds)
                 if tok_exact(p) == tok_exact(g["gold"])) / len(gold)
        report["em_token_exact"] = round(em, 4)
        report["n_eval"] = len(gold)
        with (out_dir / "preds_5k.jsonl").open("w") as f:
            for g, p in zip(gold, preds):
                f.write(json.dumps({"id": g["id"], "gold": g["gold"],
                                    "pred": p, "type": g["type"]},
                                   ensure_ascii=False) + "\n")
        report["status"] = "OK"
    except Exception as e:  # noqa: BLE001
        report["status"] = "ERROR"
        report["error"] = f"{type(e).__name__}: {e}"
        print(f"BAKEOFF {candidate} ERROR {e}", flush=True)
    report["run"] = _runstats(t0, gpu_s_rate=USD_A100_40_S, mem_gib=96)
    _write_json(out_dir / "report.json", report)
    vol.commit()
    return _js(report)


# --------------------------------------------------------------------------
# Step 4 — full training run (winner only)
# --------------------------------------------------------------------------

@app.function(
    image=train_image,
    volumes={DATA: vol},
    gpu="A100-40GB",
    memory=128 * 1024,
    timeout=20 * 3600,
)
def train_full(candidate: str = "qwen3vl") -> dict:
    """Full joint mix, recipe per §4.1. Saves adapter+merger, SHA256'd."""
    import torch

    t0 = time.time()
    vol.reload()
    out_dir = RUN / f"full_{candidate}"
    report = {"step": "train_full", "candidate": candidate,
              "model_id": CANDIDATES[candidate],
              "recipe": {
                  "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
                  "lora_lr": LORA_LR, "proj_lr": PROJ_LR,
                  "eff_batch": EFF_BATCH, "micro_b": MICRO_B,
                  "epochs": EPOCHS, "warmup_frac": WARMUP_FRAC,
                  "max_pixels": MAX_PIXELS, "seed": SEED,
                  "loss": "masked (-100 prompt+vision), per-sample mean CE x "
                          "inverse-freq class weight (clip 0.25-4, mean 1)",
                  "decode": f"greedy + rep_penalty {REP_PENALTY}, "
                            f"max_new {MAX_NEW_TOKENS}",
              }}
    try:
        mix = _load_mix_rows(RUN / "mix.jsonl")
        report["n_rows"] = len(mix)
        t_pre = time.time()
        img_cache = _preload_images([r["image"] for r in mix])
        report["preload_s"] = round(time.time() - t_pre, 1)
        report["n_images"] = len(img_cache)

        model, processor = _load_base(CANDIDATES[candidate])
        model = model.cuda()
        model = _attach_lora(model)
        _freeze_and_unfreeze_merger(model)
        report["trainable"] = _trainable_summary(model)

        resume = (out_dir / "ckpt_last" / "step.json").exists()
        log, last = _train(model, processor, mix, img_cache,
                           steps_or_epochs=("epochs", EPOCHS),
                           micro_b=MICRO_B, log_every=50,
                           out_dir=out_dir, ckpt_every=500, resume=resume,
                           time_cap_s=15 * 3600)
        report["opt_steps"] = last
        report["loss_tail"] = log[-10:]
        (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))

        _save_adapter_only(model, out_dir / "ckpt_final")
        report["adapter_sha256"] = {
            "adapter_model.safetensors": _sha256(
                out_dir / "ckpt_final" / "adapter" / "adapter_model.safetensors"),
            "merger.pt": _sha256(out_dir / "ckpt_final" / "merger.pt"),
        }
        h = hashlib.sha256()
        for v in sorted(report["adapter_sha256"].values()):
            h.update(v.encode())
        report["ckpt_combined_sha256"] = h.hexdigest()
        report["status"] = "OK"
    except Exception as e:  # noqa: BLE001
        report["status"] = "ERROR"
        report["error"] = f"{type(e).__name__}: {e}"
        print(f"TRAIN_FULL {candidate} ERROR {e}", flush=True)
    report["run"] = _runstats(t0, gpu_s_rate=USD_A100_40_S, mem_gib=128)
    _write_json(out_dir / "report.json", report)
    vol.commit()
    return _js(report)


# --------------------------------------------------------------------------
# Step 5 — frozen eval (sharded fan-out) + scoring
# --------------------------------------------------------------------------

@app.function(
    image=train_image,
    volumes={DATA: vol},
    gpu="A100-40GB",
    memory=96 * 1024,
    timeout=4 * 3600,
)
def eval_shard(ckpt_tag: str, column: str, shard_idx: int, n_shards: int) -> dict:
    """Generate preds for gold[column] contiguous shard."""
    import torch

    t0 = time.time()
    vol.reload()
    out_dir = RUN / "eval" / ckpt_tag / column
    out_path = out_dir / f"preds_{shard_idx:03d}.jsonl"
    report = {"step": "eval_shard", "ckpt": ckpt_tag, "column": column,
              "shard": shard_idx, "n_shards": n_shards}
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

        # candidate from ckpt_tag suffix: bakeoff_qwen3vl / full_qwen25vl
        cand = ckpt_tag.split("_")[-1]
        ckpt_dir = (RUN / "bakeoff" / cand / "ckpt") if ckpt_tag.startswith("bakeoff") \
            else (RUN / f"full_{cand}" / "ckpt_final")

        t_pre = time.time()
        needed = sorted({r["image"] for r in rows})
        img_cache = _preload_images(needed)
        report["preload_s"] = round(time.time() - t_pre, 1)
        report["n_images"] = len(img_cache)
        for r in rows:
            r["_img"] = r["image"]
            r["_pil"] = img_cache[r["image"]]

        model, processor = _load_base(CANDIDATES[cand])
        model = model.cuda()
        model = _attach_trained(model, ckpt_dir)
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
def score_eval(ckpt_tag: str) -> dict:
    """Merge shard preds -> per column OA + AA + per-family; RAW EM +
    token-exact EM + normalized EM (canon + train-vocab projection)."""
    t0 = time.time()
    vol.reload()
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


def eval_ids_read(col: str) -> list:
    return json.loads((EVAL_IDS_DIR / COLUMNS[col]["ids"]).read_text())["ids"]


# --------------------------------------------------------------------------
# Local entrypoints
# --------------------------------------------------------------------------

@app.local_entrypoint()
def e_preflight():
    print(preflight.remote())


@app.local_entrypoint()
def e_mix():
    print(build_mix.remote())


@app.local_entrypoint()
def e_smoke():
    cand = os.environ.get("CANDIDATE", "qwen3vl")
    print(smoke.remote(cand))


@app.local_entrypoint()
def e_bakeoff():
    cand = os.environ.get("CANDIDATE", "qwen3vl")
    print(bakeoff.remote(cand))


@app.local_entrypoint()
def e_train():
    cand = os.environ.get("CANDIDATE", "qwen3vl")
    print(train_full.remote(cand))


@app.local_entrypoint()
def e_eval():
    ckpt = os.environ.get("CKPT_TAG", "full_qwen3vl")
    only = os.environ.get("ONLY_COL")
    jobs = []
    for col, cfg in COLUMNS.items():
        if only and col != only:
            continue
        for i in range(cfg["shards"]):
            jobs.append((ckpt, col, i, cfg["shards"]))
    print(f"fan-out {len(jobs)} shard jobs")
    results = list(eval_shard.starmap(jobs))
    for r in results:
        print(r)


@app.local_entrypoint()
def e_score():
    ckpt = os.environ.get("CKPT_TAG", "full_qwen3vl")
    print(score_eval.remote(ckpt))
