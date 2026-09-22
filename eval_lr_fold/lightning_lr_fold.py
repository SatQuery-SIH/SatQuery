"""LR-FOLD trainer — Lightning-ported driver of modal_rsvqa_adapt.py.

PORTED, not rewritten: every model/math helper below is copied VERBATIM from
modal_rsvqa_adapt.py (masked loss, class weights, LoRA r16/a32 attn+MLP,
merger FT, cosine+warmup, greedy eval decode). Modal decorators stripped;
paths come from env vars so the SAME file runs on a Lightning Studio or in a
Modal container fallback.

Env contract:
    DATA_ROOT   root the mix's relative image paths resolve under
                (rsvqa_lr/Images_LR, rsvqa_hr/Data, vrsbench/Images_train)
    MIX         path to mix_lr_fold.jsonl
    RUN_DIR     output dir (ckpt_last/, ckpt_final/, STATUS.json,
                train_log.jsonl, report.json)
    INIT_CKPT   ckpt_final dir to continue from (adapter/ + merger.pt)
                empty  -> fresh LoRA attach (Arm B / debug)
    MODE        smoke | train
    TIME_CAP_S  hard wall cap inside the trainer (default 14h train, 1h smoke)
    CKPT_EVERY  optimizer-step checkpoint cadence (default 500)
    LOG_EVERY   TRAINLOG cadence (default 50 train / 10 smoke)
    MODEL_ID    HF id (default Qwen/Qwen3-VL-8B-Instruct)
    BASE_REVISION  pinned snapshot revision (default 0c351dd0…)

Continuation semantics (Arm A): weights init from INIT_CKPT
(adapter-first + merger.pt strict-load with unexpected==0 assert — the
_attach_trained fix kept), FRESH AdamW + fresh cosine — stale momentum and
schedule are NOT resumed. Crash resume is separate: RUN_DIR/ckpt_last/
step.json present -> full resume of weights+opt+scheduler step (the same
machinery that crossed accounts bit-identically at step 4,000).

Every ckpt save writes STATUS.json + appends train_log.jsonl; an external
syncer pulls RUN_DIR back to the laptop. Nothing here imports lightning_sdk.
"""

import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path

# --------------------------------------------------------------------------
# Recipe constants — VERBATIM from modal_rsvqa_adapt.py
# --------------------------------------------------------------------------

LORA_R = 16
LORA_ALPHA = 32
LORA_LR = 1e-4
PROJ_LR = 1e-5
EFF_BATCH = 48
MICRO_B = 8
EPOCHS = 2
WARMUP_FRAC = 0.03
MAX_PIXELS = 768 * 768
SEED = 42

REP_PENALTY = 1.08
MAX_NEW_TOKENS = 16

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


def _sha256(path: Path, bufsize: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")
    print(f"WROTE {path}", flush=True)


# --------------------------------------------------------------------------
# Model helpers — VERBATIM from modal_rsvqa_adapt.py
# --------------------------------------------------------------------------

def _load_base(model_id: str, revision: str | None = None):
    """bf16 base on cuda. Tolerates transformers auto-class renames."""
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as AutoModel
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoModel

    kw = {"revision": revision} if revision else {}
    processor = AutoProcessor.from_pretrained(model_id, **kw)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        **kw,
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


def _init_continuation(model, ckpt_dir: Path):
    """Init trainable model from a prior ckpt_final (adapter + merger.pt).

    Order matters — identical to _attach_trained: the adapter is attached
    FIRST so merger.pt's peft-prefixed keys resolve; strict=False is safe
    only because unexpected==0 is asserted.
    """
    import torch
    from peft import PeftModel

    peft_model = PeftModel.from_pretrained(
        model, str(ckpt_dir / "adapter"), is_trainable=True)
    for n, p in peft_model.named_parameters():
        if "lora_" in n:
            continue
        p.requires_grad = False
    mp = torch.load(ckpt_dir / "merger.pt", map_location="cpu")
    res = peft_model.load_state_dict(mp, strict=False)
    assert len(res.unexpected_keys) == 0, (
        f"merger.pt keys unmatched on PEFT model "
        f"({len(res.unexpected_keys)} unexpected): {res.unexpected_keys[:5]}")
    merger = []
    for n, p in peft_model.named_parameters():
        if "merger" in n:
            p.requires_grad = True
            merger.append(n)
    return peft_model, merger


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
        p_ids = enc.input_ids[i][m].tolist()
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


_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12",
}


def tok_exact(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).rstrip(".")


def canon(s: str) -> str:
    s = tok_exact(s)
    s = s.replace("²", "2").replace("㎡", "m2")
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    for phrase, unit in (
        ("square kilometers", "km2"), ("square kilometer", "km2"),
        ("sq kilometers", "km2"), ("sq km", "km2"), ("km 2", "km2"),
        ("square meters", "m2"), ("square meter", "m2"),
        ("sq meters", "m2"), ("sq m", "m2"), ("m 2", "m2"),
        ("hectares", "ha"), ("hectare", "ha"),
    ):
        s = s.replace(phrase, unit)
    s = re.sub(r"(\d)\s*(m2|km2|ha)\b", r"\1\2", s)
    if s in _NUM_WORDS:
        s = _NUM_WORDS[s]
    return s


def vocab_project(s: str, vocab: list[str], cutoff: float = 0.85) -> str:
    import difflib

    if s in vocab:
        return s
    m = difflib.get_close_matches(s, vocab, n=1, cutoff=cutoff)
    return m[0] if m else s


# --------------------------------------------------------------------------
# Mix / train loop — VERBATIM _stratified/_train/_save_ckpt/_attach_trained
# from modal_rsvqa_adapt.py, with two additions:
#   * every TRAINLOG record is appended to RUN_DIR/train_log.jsonl (tee)
#   * every _save_ckpt also writes STATUS.json for the laptop syncer
# --------------------------------------------------------------------------

def _load_mix_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _stratified(rows, n, seed):
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


def _write_status(run_dir: Path, **kw) -> None:
    st = {"ts": time.time(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                  time.gmtime())}
    st.update(kw)
    _write_json(run_dir / "STATUS.json", st)


def _train(model, processor, rows, image_cache, *, steps_or_epochs, lr_lora=LORA_LR,
           lr_proj=PROJ_LR, micro_b=MICRO_B, accum=None, log_every=25,
           warmup=WARMUP_FRAC, constant_lr=False, out_dir: Path | None = None,
           ckpt_every=0, resume=False, time_cap_s=None,
           status_extra: dict | None = None):
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
    rng = random.Random(SEED)

    if steps_or_epochs[0] == "steps":
        total_opt = steps_or_epochs[1]
        order = list(range(len(rows)))
        rng.shuffle(order)
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
    log_path = (out_dir / "train_log.jsonl") if out_dir else None
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
            if log_path:
                with log_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
        if ckpt_every and out_dir and (step + 1) % ckpt_every == 0:
            _save_ckpt(model, opt, out_dir / "ckpt_last", step + 1)
            _write_status(out_dir, event="ckpt_saved", opt_step=step + 1,
                          total_opt=total_opt, elapsed_s=round(time.time() - t_start, 1),
                          last_log=log[-1] if log else None,
                          **(status_extra or {}))
            print(f"CKPT_SAVED opt_step={step + 1}", flush=True)
        if time_cap_s and (time.time() - t_start) > time_cap_s:
            print(f"TIME CAP hit at opt_step {step + 1}", flush=True)
            _save_ckpt(model, opt, out_dir / "ckpt_last", step + 1) if out_dir else None
            if out_dir:
                _write_status(out_dir, event="time_cap", opt_step=step + 1,
                              total_opt=total_opt,
                              elapsed_s=round(time.time() - t_start, 1),
                              **(status_extra or {}))
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
    """Load merger.pt + LoRA adapter for eval (adapter-first + assert kept)."""
    import torch
    from peft import PeftModel

    mp = torch.load(ckpt_dir / "merger.pt", map_location="cpu")
    peft_model = PeftModel.from_pretrained(model, str(ckpt_dir / "adapter"),
                                           is_trainable=False)
    res = peft_model.load_state_dict(mp, strict=False)
    assert len(res.unexpected_keys) == 0, \
        f"merger.pt keys unmatched on PEFT model ({len(res.unexpected_keys)} " \
        f"unexpected): {res.unexpected_keys[:5]}"
    return peft_model


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _resolve(row: dict, data_root: Path) -> None:
    row["image"] = str(data_root / row["image"])


def run_smoke(data_root: Path, mix_path: Path, run_dir: Path,
              init_ckpt: Path | None, model_id: str, revision: str | None,
              time_cap_s: float) -> dict:
    """100-row overfit: masked loss + class weights, ~200 steps, gate >=95%
    train EM (VQA rows only — captions are the regularizer, not the gate)."""
    import torch

    t0 = time.time()
    report = {"step": "smoke", "model_id": model_id, "revision": revision,
              "init_ckpt": str(init_ckpt) if init_ckpt else None}
    try:
        mix = _load_mix_rows(mix_path)
        for r in mix:
            _resolve(r, data_root)
        rows = _stratified(mix, 100, seed=7)
        img_cache = _preload_images([r["image"] for r in rows])
        report["n_images"] = len(img_cache)

        model, processor = _load_base(model_id, revision)
        model = model.cuda()
        if init_ckpt:
            model, merger = _init_continuation(model, init_ckpt)
            report["init"] = "continuation_from_ckpt_final"
        else:
            model = _attach_lora(model)
            merger = _freeze_and_unfreeze_merger(model)
            report["init"] = "fresh_lora"
        report["n_merger_tensors"] = len(merger)
        report["trainable"] = _trainable_summary(model)

        log, last = _train(model, processor, rows, img_cache,
                           steps_or_epochs=("steps", 200),
                           micro_b=10, accum=1, log_every=10,
                           constant_lr=True, warmup=0.0,
                           out_dir=run_dir, time_cap_s=time_cap_s,
                           status_extra={"phase": "smoke"})
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
    report["wall_seconds"] = round(time.time() - t0, 1)
    _write_json(run_dir / "smoke_report.json", report)
    _write_status(run_dir, event="smoke_done", gate=report.get("gate"),
                  em=report.get("train_em_generated_vqa_only"),
                  wall_seconds=report["wall_seconds"])
    return report


def run_train(data_root: Path, mix_path: Path, run_dir: Path,
              init_ckpt: Path | None, model_id: str, revision: str | None,
              time_cap_s: float, ckpt_every: int, log_every: int) -> dict:
    """Full LR-fold mix, 2 epochs, continuation init. Saves adapter+merger,
    SHA256'd. Resume auto-detects ckpt_last/step.json."""
    import torch

    t0 = time.time()
    report = {"step": "train_lr_fold", "model_id": model_id,
              "revision": revision,
              "init_ckpt": str(init_ckpt) if init_ckpt else None,
              "recipe": {
                  "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
                  "lora_lr": LORA_LR, "proj_lr": PROJ_LR,
                  "eff_batch": EFF_BATCH, "micro_b": MICRO_B,
                  "epochs": EPOCHS, "warmup_frac": WARMUP_FRAC,
                  "max_pixels": MAX_PIXELS, "seed": SEED,
                  "init": "ckpt_final adapter+merger; FRESH AdamW + fresh "
                          "cosine (new fold — no stale momentum/schedule)",
                  "loss": "masked (-100 prompt+vision), per-sample mean CE x "
                          "inverse-freq class weight (clip 0.25-4, mean 1)",
                  "decode": f"greedy + rep_penalty {REP_PENALTY}, "
                            f"max_new {MAX_NEW_TOKENS}",
              }}
    try:
        mix = _load_mix_rows(mix_path)
        for r in mix:
            _resolve(r, data_root)
        report["n_rows"] = len(mix)
        report["mix_sha256"] = _sha256(mix_path)

        resume = (run_dir / "ckpt_last" / "step.json").exists()
        t_pre = time.time()
        img_cache = _preload_images([r["image"] for r in mix])
        report["preload_s"] = round(time.time() - t_pre, 1)
        report["n_images"] = len(img_cache)

        model, processor = _load_base(model_id, revision)
        model = model.cuda()
        if resume:
            # peft structure must exist before _train's resume path loads
            # adapter+merger+opt from ckpt_last; weight values here don't
            # matter — they get overwritten
            if init_ckpt:
                model, merger = _init_continuation(model, init_ckpt)
            else:
                model = _attach_lora(model)
                merger = _freeze_and_unfreeze_merger(model)
            report["init"] = "resume_from_ckpt_last"
        elif init_ckpt:
            model, merger = _init_continuation(model, init_ckpt)
            report["init"] = "continuation_from_ckpt_final"
        else:
            model = _attach_lora(model)
            merger = _freeze_and_unfreeze_merger(model)
            report["init"] = "fresh_lora"
        report["n_merger_tensors"] = len(merger)
        report["trainable"] = _trainable_summary(model)
        _write_status(run_dir, event="train_start", n_rows=len(mix),
                      n_images=len(img_cache), resume=resume,
                      preload_s=report["preload_s"])

        log, last = _train(model, processor, mix, img_cache,
                           steps_or_epochs=("epochs", EPOCHS),
                           micro_b=MICRO_B, log_every=log_every,
                           out_dir=run_dir, ckpt_every=ckpt_every,
                           resume=resume, time_cap_s=time_cap_s,
                           status_extra={"phase": "train"})
        report["opt_steps"] = last
        report["loss_tail"] = log[-10:]
        (run_dir / "train_log.json").write_text(json.dumps(log, indent=2))

        _save_adapter_only(model, run_dir / "ckpt_final")
        report["adapter_sha256"] = {
            "adapter_model.safetensors": _sha256(
                run_dir / "ckpt_final" / "adapter" / "adapter_model.safetensors"),
            "merger.pt": _sha256(run_dir / "ckpt_final" / "merger.pt"),
        }
        h = hashlib.sha256()
        for v in sorted(report["adapter_sha256"].values()):
            h.update(v.encode())
        report["ckpt_combined_sha256"] = h.hexdigest()
        report["status"] = "OK"
    except Exception as e:  # noqa: BLE001
        report["status"] = "ERROR"
        report["error"] = f"{type(e).__name__}: {e}"
        print(f"TRAIN_LR_FOLD ERROR {e}", flush=True)
    report["wall_seconds"] = round(time.time() - t0, 1)
    _write_json(run_dir / "report.json", report)
    _write_status(run_dir, event="train_done", status=report.get("status"),
                  opt_steps=report.get("opt_steps"),
                  wall_seconds=report["wall_seconds"],
                  adapter_sha256=report.get("adapter_sha256"),
                  ckpt_combined_sha256=report.get("ckpt_combined_sha256"))
    return report


def main() -> int:
    data_root = Path(os.environ["DATA_ROOT"])
    mix_path = Path(os.environ["MIX"])
    run_dir = Path(os.environ["RUN_DIR"])
    init_ckpt = Path(os.environ["INIT_CKPT"]) if os.environ.get("INIT_CKPT") else None
    mode = os.environ.get("MODE", "train")
    model_id = os.environ.get("MODEL_ID", DEFAULT_MODEL)
    revision = os.environ.get("BASE_REVISION") or None
    ckpt_every = int(os.environ.get("CKPT_EVERY", "500"))
    log_every = int(os.environ.get("LOG_EVERY", "50"))
    run_dir.mkdir(parents=True, exist_ok=True)

    if mode == "smoke":
        cap = float(os.environ.get("TIME_CAP_S", "3600"))
        rep = run_smoke(data_root, mix_path, run_dir, init_ckpt, model_id,
                        revision, cap)
    elif mode == "train":
        cap = float(os.environ.get("TIME_CAP_S", str(14 * 3600)))
        rep = run_train(data_root, mix_path, run_dir, init_ckpt, model_id,
                        revision, cap, ckpt_every, log_every)
    else:
        raise SystemExit(f"unknown MODE {mode}")
    print("FINAL " + json.dumps({"status": rep.get("status"),
                                 "gate": rep.get("gate")}, default=str), flush=True)
    return 0 if rep.get("status") == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
