"""VRSBENCH-EVAL — zero-shot baselines on frozen ids for all three VRSBench
columns (VQA / caption / grounding), per MASTER_PLAN_V2 §5.

Producer: Qwen3-VL-8B-Instruct (pinned revision). Optional second row:
Qwen2.5-VL-7B-Instruct on the grounding column only (published anchor
41.95 P@0.5 sits on it — direct harness comparison).

Data:    Modal volume `satquery-data` -> /data/vrsbench/{Images_val, *.json}
Weights: Modal volume `satquery-models` -> /models/hf (persistent HF cache)
Frozen ids (local, uploaded at image-build time, never regenerated):
    gates/_cache/eval_ids/vrsbench_{vqa,caption,referring}_eval_ids.json

DECLARED PROTOCOL (fixed before first scored prediction; written verbatim
into manifest.json):

  VQA      prompt = question + "\\nAnswer the question using a single word or
           phrase." Greedy (temperature 0), max_tokens 32.
           Metrics: EM_raw = exact string equality after strip().
           EM_norm = EM after canonical normalizer (case fold, punctuation
           strip, leading-article drop, number-word->digit, unit
           canonicalization). EM_norm_proj = EM_norm plus nearest-vocab
           projection over the frozen-id answer vocabulary (difflib
           cutoff 0.88). All three reported side by side; the raw column
           is never replaced.

  Caption  prompt = annotation question verbatim ("Describe the image in
           detail"). Greedy, max_tokens 256. Single deterministic caption
           per image. Metrics: pycocoevalcap BLEU-1..4 / METEOR /
           ROUGE-L / CIDEr on the frozen caption ids (1 ref per id).

  Grounding prompt = "Please provide the bounding box coordinate of the
           region this sentence describes: {question}" — the canonical
           Qwen-VL REC prompt, so the model emits its native
           <|box_start|>(x1,y1),(x2,y2)<|box_end|> in the *resized-image
           pixel frame*. Greedy, max_tokens 128.
           Parse (deterministic, per-row tag recorded in preds.jsonl):
             * <|box_start|> span -> resized-frame pixels -> / resized dims
               (model's own smart_resize rule: factor = patch*merge,
               min/max pixels from pinned preprocessor_config.json)
             * "bbox_2d": [a,b,c,d] -> per-model grid: qwen3vl8b = 0-1000
               normalized (sanity-verified), qwen25vl7b = resized-px frame
             * else first 4-number tuple / first 4 numbers -> grid
               heuristic: max<=1.5 -> 0-1; <=100.5 -> /100;
               <=1000.5 -> /1000; else -> original pixels.
             * <4 numbers -> parse_fail, IoU scored 0.
           GT grid is 0-100 (official VRSBench note, verified on data:
           max coord = 100.0) -> /100 -> [0,1].
           Metrics (official eval_fianl/compute_metrics convention):
             acc@0.5, acc@0.7, meanIoU, cumIoU computed on the integer 0-100
             grid with the +1 inclusive-area convention (pred rescaled and
             rounded to the 0-100 int grid first). Float IoU variant
             reported as a secondary column. Splits: is_unique true/false.
           PRE-FLIGHT: sanity() runs 8 grounding + 3 VQA + 2 caption samples
           and writes a hand-check table (raw output, detected grid,
           rescaled box, GT box, IoU) to sanity_report.json BEFORE the
           scored run. If the emitted grid is not what the parser assumes,
           the parser is corrected and re-declared in the manifest before
           any scored prediction is written.

Outputs -> /data/vrsbench/evals/ :
    manifest.json                      model rev + weights sha, prompt hash,
                                       ids_sha256, decoding params, seconds, USD
    sanity_report.json                 grounding hand-check table
    <model>/<column>_preds.jsonl       one line per frozen id (raw output kept)
    <model>/<column>_metrics.json

Adapted-caption column (CAPTION-EVAL, same lane):
    model=qwen3vl8b_rsvqa serves ckpt_final merged on-volume by merge_ckpt()
    (LoRA adapter + trained visual merger -> full bf16 weights dir).
    Protocol identical to the zero-shot caption row (verbatim annotation
    prompt, greedy, max_tokens 256, pycocoevalcap) and is written into the
    manifest BEFORE any adapted prediction is generated. Overlap gate:
    mix.jsonl rows vs frozen caption ids — id- and image-level; nonzero
    halts the run.

Run:
    python -m modal run modal_vrsbench_eval.py::sanity
    python -m modal run modal_vrsbench_eval.py::run --column all
    python -m modal run modal_vrsbench_eval.py::run --column referring --model qwen25vl7b
    python -m modal run modal_vrsbench_eval.py::merge_ckpt
    python -m modal run modal_vrsbench_eval.py::run_caption_adapted

Modal rules: profile `harsha-610vmg` only by bootstrap; the adapted-caption
run executes under `ripper2005` on explicit user override (geethadmohan1980
out of credits) — recorded in the manifest. GPU = A100-40GB, list
$0.000583/s (modal.com/pricing). Halt + ask if projected spend > $5.
"""

import hashlib
import json
import math
import os
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

# ---------------------------------------------------------------- config

DATA_ROOT = "/data"
MODELS_ROOT = "/models"
IDS_ROOT = "/ids"
EVAL_ROOT = f"{DATA_ROOT}/vrsbench/evals"
IMG_DIR = f"{DATA_ROOT}/vrsbench/Images_val"

data_vol = modal.Volume.from_name("satquery-data")
model_vol = modal.Volume.from_name("satquery-models", create_if_missing=True)

app = modal.App("satquery-vrsbench-eval")

LOCAL = Path(__file__).resolve().parent
ID_FILES = {
    "vqa": "vrsbench_vqa_eval_ids.json",
    "caption": "vrsbench_caption_eval_ids.json",
    "referring": "vrsbench_referring_eval_ids.json",
}
GT_FILES = {
    "vqa": "VRSBench_EVAL_vqa.json",
    "caption": "VRSBench_EVAL_Cap.json",
    "referring": "VRSBench_EVAL_referring.json",
}
# sha256 of the laptop copies of the GT files (computed 2026-09-13);
# the job asserts the volume copies are identical before scoring.
GT_SHA256 = {
    "vqa": "4a797f5fc456331a55938b689350228bb7aa200b7e9a3972a6604b14dcfc175c",
    "caption": "d45fea7288bd7b243c968add4f9d1d2fae473826a4dccb32df0663e209826be2",
    "referring": "fd63f7c6b77a158f4cc933a1ead88fa63aa23ebcf30f2ec9be111f3567ff1b44",
}

ADAPTED_CKPT_DIR = f"{DATA_ROOT}/runs/rsvqa_adapt/full_qwen3vl/ckpt_final"
ADAPTED_MERGED_DIR = f"{ADAPTED_CKPT_DIR}/merged_vllm"
MIX_PATH = f"{DATA_ROOT}/runs/rsvqa_adapt/mix.jsonl"
# shas of the adapted artifact (RSVQA-ADAPT eval manifest) — asserted before merge.
CKPT_SHA256 = {
    "adapter_model.safetensors":
        "88d82e090e557f1e70c3f9a32650e325965f4d4c293e0ba8ebde5f9eecca5bd3",
    "merger.pt":
        "7c77c99a3abaff393cfa06637ff7a5e3696713cb3da85b222f17c9166ab240dc",
    "combined":
        "8fccee8dd09970c29e742d4053a60cb7433d2dbf463a219e5f057f0a0e5b5c99",
    "mix.jsonl":
        "e60f369567e922e4ca56b509da523d351c31c09a371b5dd8d95180918e1b7332",
}

MODELS = {
    "qwen3vl8b": {
        "hf_id": "Qwen/Qwen3-VL-8B-Instruct",
        "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    },
    "qwen25vl7b": {
        "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
    },
    # adapted ckpt: qwen3vl8b + RSVQA-ADAPT LoRA + trained visual merger,
    # merged into a full weights dir on the volume by merge_ckpt().
    "qwen3vl8b_rsvqa": {
        "merged_dir": ADAPTED_MERGED_DIR,
        "ckpt_dir": ADAPTED_CKPT_DIR,
        "ckpt_sha256": CKPT_SHA256,
    },
}

# Declared prompts (verbatim into manifest; sha256'd as prompt_template_hash)
VQA_PROMPT = "{q}\nAnswer the question using a single word or phrase."
CAP_PROMPT = "{q}"
REF_PROMPT = (
    "Please provide the bounding box coordinate of the region this "
    "sentence describes: {q}"
)

DECODING = {
    "vqa": {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": 32},
    "caption": {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": 256},
    "referring": {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": 128},
}

GPU_NAME = "A100"            # A100-40GB
USD_PER_S_GPU = 0.000583     # Modal list, modal.com/pricing
MEM_GIB = 48
CPU_CORES = 8
USD_PER_S_CPU = CPU_CORES * 0.0000131 + MEM_GIB * 0.00000222
USD_PER_S = USD_PER_S_GPU + USD_PER_S_CPU

BUDGET_HALT_USD = 5.0

eval_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.29.0",
        "qwen-vl-utils",
        "pycocoevalcap",
        "nltk",
        "pillow",
        "hf_transfer",
        # merge_ckpt only: attach adapter + load merger.pt before save.
        "peft==0.18.1",
    )
    # pycocoevalcap METEOR shells out to a bundled meteor-1.5.jar -> needs JRE
    .apt_install("default-jre-headless")
    .env(
        {
            "HF_HOME": f"{MODELS_ROOT}/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_LOGGING_LEVEL": "WARNING",
            # flashinfer JIT-compiles CUDA kernels -> needs nvcc; use the
            # precompiled torch sampler + FlashAttention backend instead.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        }
    )
)
for col, fname in ID_FILES.items():
    eval_image = eval_image.add_local_file(
        str(LOCAL / "gates" / "_cache" / "eval_ids" / fname),
        f"{IDS_ROOT}/{col}.json",
    )

# ---------------------------------------------------------------- helpers


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _runstats(t0: float, extra: dict | None = None,
              usd_per_s: float = USD_PER_S, gpu: str = GPU_NAME) -> dict:
    secs = round(time.time() - t0, 1)
    stats = {
        "wall_seconds": secs,
        "est_usd": round(secs * usd_per_s, 4),
        "gpu": gpu,
        "rate": f"${usd_per_s}/s",
        "pricing": "Modal on-demand list (modal.com/pricing)",
    }
    if extra:
        stats.update(extra)
    print("RUNSTATS " + json.dumps(stats), flush=True)
    return stats


def _smart_resize(h: int, w: int, factor: int, min_px: int, max_px: int):
    """Qwen-VL image resize rule (matches qwen_vl_utils.smart_resize)."""
    h_bar = max(factor, round(h / factor) * factor)
    w_bar = max(factor, round(w / factor) * factor)
    if h_bar * w_bar > max_px:
        beta = math.sqrt(h * w / max_px)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_px:
        beta = math.sqrt(min_px / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return int(h_bar), int(w_bar)


def _load_mm_geometry(model_dir: str):
    """(factor, min_pixels, max_pixels) from the model's own config."""
    cfg = json.loads(Path(model_dir, "preprocessor_config.json").read_text())
    patch = cfg.get("patch_size", 16)
    merge = cfg.get("merge_size", 2)
    factor = patch * merge
    size = cfg.get("size", {}) or {}
    min_px = cfg.get("min_pixels") or size.get("shortest_edge") or 4 * factor * factor
    max_px = cfg.get("max_pixels") or size.get("longest_edge") or 16_384 * factor * factor
    return factor, int(min_px), int(max_px)


def _resized_dims(img_w: int, img_h: int, factor: int, min_px: int, max_px: int):
    return _smart_resize(img_h, img_w, factor, min_px, max_px)[::-1]  # (w, h)


def _load_frozen(col: str) -> dict:
    return json.loads(Path(IDS_ROOT, f"{col}.json").read_text())


def _load_gt(col: str) -> dict:
    """GT lookup keyed by 'image_id#question_id'; asserts volume file == laptop file."""
    path = Path(DATA_ROOT, "vrsbench", GT_FILES[col])
    sha = _sha256_file(path)
    assert sha == GT_SHA256[col], f"GT sha mismatch {col}: {sha} != {GT_SHA256[col]}"
    rows = json.loads(path.read_text())
    lut = {}
    for i, e in enumerate(rows):
        qid = e.get("question_id", i)
        lut[f"{e['image_id']}#{qid}"] = e
    return lut


def _load_llm(model_key: str):
    """Download pinned weights to the model volume, return (llm, geometry, meta).
    Adapted model keys carry `merged_dir` — a full-weights dir on the data
    volume produced by merge_ckpt(); served directly, no HF download."""
    from huggingface_hub import HfApi, snapshot_download
    from vllm import LLM

    cfg = MODELS[model_key]
    if "merged_dir" in cfg:
        model_dir = cfg["merged_dir"]
        mman = json.loads(
            Path(model_dir, "merged_manifest.json").read_text())
        geometry = _load_mm_geometry(model_dir)
        t0 = time.time()
        llm = LLM(
            model=model_dir,
            dtype="bfloat16",
            max_model_len=8192,
            limit_mm_per_prompt={"image": 1},
            enable_prefix_caching=True,
            gpu_memory_utilization=0.92,
            max_num_seqs=256,
            seed=0,
        )
        print(f"[model] merged {model_dir} load={time.time()-t0:.0f}s",
              flush=True)
        meta = {
            "hf_id": mman["base"]["hf_id"],
            "revision": mman["base"]["revision"],
            "weights_sha256": mman["merged_tree_sha256"],
            "snapshot_dir": model_dir,
            "mm_resize": {"factor": geometry[0], "min_pixels": geometry[1],
                          "max_pixels": geometry[2]},
            "adapted": {
                "ckpt_dir": cfg["ckpt_dir"],
                "ckpt_sha256": cfg["ckpt_sha256"],
                "merge": mman["merge"],
            },
        }
        return llm, geometry, meta
    info = HfApi().model_info(cfg["hf_id"], revision=cfg["revision"], files_metadata=True)
    fp = sorted(
        f"{s.rfilename}:{(s.lfs or {}).get('sha256') or (s.lfs or {}).get('oid') or 'blob'}"
        for s in info.siblings
    )
    weights_sha = _sha256_str("\n".join(fp))
    t0 = time.time()
    model_dir = snapshot_download(cfg["hf_id"], revision=cfg["revision"])
    model_vol.commit()
    print(f"[model] {cfg['hf_id']}@{cfg['revision'][:12]} "
          f"weights_sha={weights_sha[:16]} dl={time.time()-t0:.0f}s", flush=True)

    geometry = _load_mm_geometry(model_dir)
    llm = LLM(
        model=model_dir,
        dtype="bfloat16",
        max_model_len=8192,
        limit_mm_per_prompt={"image": 1},
        enable_prefix_caching=True,
        gpu_memory_utilization=0.92,
        max_num_seqs=256,
        seed=0,
    )
    meta = {
        "hf_id": cfg["hf_id"],
        "revision": cfg["revision"],
        "weights_sha256": weights_sha,
        "snapshot_dir": str(model_dir),
        "mm_resize": {"factor": geometry[0], "min_pixels": geometry[1],
                      "max_pixels": geometry[2]},
    }
    return llm, geometry, meta


def _msgs(image_path: str, text: str):
    from PIL import Image

    im = Image.open(image_path).convert("RGB")
    return (
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_pil", "image_pil": im},
                    {"type": "text", "text": text},
                ],
            }
        ],
        im.size,  # (w, h)
    )


def _gen(llm, items, col: str):
    """items: list of dicts {id, image, prompt}. Yields (item, raw_text)."""
    from vllm import SamplingParams

    sp = SamplingParams(**DECODING[col])
    convs, sizes = [], []
    for it in items:
        conv, wh = _msgs(os.path.join(IMG_DIR, it["image"]), it["prompt"])
        convs.append(conv)
        sizes.append(wh)
    outs = llm.chat(convs, sp)
    for it, out, wh in zip(items, outs, sizes):
        yield it, out.outputs[0].text.strip(), wh


# ------------------------------------------------- parsing + normalizers

_NUM_RE = r"-?\d+(?:\.\d+)?"


_BBOX_KEY_RE = re.compile(
    r'"(?:bbox_2d|bbox|bounding_box|box)"\s*:\s*\[([^\]]+)\]', re.I)
_BOX_TOK_RE = re.compile(r"<\|box_start\|>(.*?)<\|box_end\|>", re.S)
_ARR4_RE = re.compile(
    rf"[\[\(<]?\s*({_NUM_RE})\s*,\s*({_NUM_RE})\s*,\s*({_NUM_RE})\s*,\s*"
    rf"({_NUM_RE})\s*[\]\)>]?")


def _clip01(box):
    x1, y1, x2, y2 = box
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return [min(max(x1, 0.0), 1.0), min(max(y1, 0.0), 1.0),
            min(max(x2, 0.0), 1.0), min(max(y2, 0.0), 1.0)]


def parse_pred_box(text: str, img_w: int, img_h: int, res_w: int, res_h: int,
                   model_key: str = "qwen3vl8b"):
    """-> (box01 [x1,y1,x2,y2] or None, scale_tag). Declared rules, see header.

    Extraction priority (first match wins; the 'bbox_2d' key text itself
    contains a digit and must never leak into coordinates):
      1. <|box_start|>...<|box_end|> span   -> resized-pixel frame
      2. "bbox_2d"/"bbox"/"box": [a,b,c,d]  -> model-declared grid:
           qwen3vl8b -> 0-1000 normalized (verified in sanity);
           qwen25vl7b -> resized-pixel frame
      3. any 4-number tuple [a,b,c,d]/(a,b),(c,d) in text
      4. first 4 numbers anywhere ('first4')
      5. <4 numbers -> parse_fail
    Grid for paths 3-4: max<=1.5 -> 0-1; <=100.5 -> /100; <=1000.5 -> /1000;
    else -> original pixels.
    """
    v = None
    tag = None
    m = _BOX_TOK_RE.search(text)
    if m:
        nums = re.findall(_NUM_RE, m.group(1))
        if len(nums) >= 4:
            v = [float(x) for x in nums[:4]]
            tag = "qwen_native_px"
            box = [v[0] / res_w, v[1] / res_h, v[2] / res_w, v[3] / res_h]
            return _clip01(box), tag
    if v is None:
        m = _BBOX_KEY_RE.search(text)
        if m:
            nums = re.findall(_NUM_RE, m.group(1))
            if len(nums) >= 4:
                v = [float(x) for x in nums[:4]]
                if model_key == "qwen25vl7b":
                    tag = "bbox2d_resized_px"
                    return _clip01(
                        [v[0] / res_w, v[1] / res_h,
                         v[2] / res_w, v[3] / res_h]), tag
                tag = "bbox2d_0_1000"
                return _clip01([x / 1000 for x in v]), tag
    if v is None:
        m = _ARR4_RE.search(text)
        if m:
            v = [float(m.group(i)) for i in range(1, 5)]
            tag = "tuple4"
    if v is None:
        nums = re.findall(_NUM_RE, text)
        if len(nums) >= 4:
            v = [float(x) for x in nums[:4]]
            tag = "first4"
    if v is None:
        return None, "parse_fail"
    mx = max(abs(x) for x in v)
    if mx <= 1.5:
        gtag, box = "01", v
    elif mx <= 100.5:
        gtag, box = "100", [x / 100 for x in v]
    elif mx <= 1000.5:
        gtag, box = "1000", [x / 1000 for x in v]
    else:
        gtag = "px"
        box = [v[0] / img_w, v[1] / img_h, v[2] / img_w, v[3] / img_h]
    return _clip01(box), f"{tag}_{gtag}"


def parse_gt_box(gt: str):
    nums = re.findall(_NUM_RE, gt)
    if len(nums) < 4:
        return None
    return [float(x) / 100.0 for x in nums[:4]]  # official grid 0-100


def iou_official(pred01, gt01) -> float:
    """Official eval_fianl convention: integer 0-100 grid, +1 inclusive area."""
    p = [round(v * 100) for v in pred01]
    g = [round(v * 100) for v in gt01]
    ix1, iy1 = max(p[0], g[0]), max(p[1], g[1])
    ix2, iy2 = min(p[2], g[2]), min(p[3], g[3])
    inter = max(0, ix2 - ix1 + 1) * max(0, iy2 - iy1 + 1)
    a = (p[2] - p[0] + 1) * (p[3] - p[1] + 1)
    b = (g[2] - g[0] + 1) * (g[3] - g[1] + 1)
    return inter / max(a + b - inter, 1e-9)


def iou_float(pred01, gt01) -> float:
    ix1, iy1 = max(pred01[0], gt01[0]), max(pred01[1], gt01[1])
    ix2, iy2 = min(pred01[2], gt01[2]), min(pred01[3], gt01[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a = (pred01[2] - pred01[0]) * (pred01[3] - pred01[1])
    b = (gt01[2] - gt01[0]) * (gt01[3] - gt01[1])
    return inter / max(a + b - inter, 1e-9)


_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}
_UNIT_MAP = {
    "kilometers": "km", "kilometres": "km", "kms": "km", "km": "km",
    "meters": "m", "metres": "m", "meter": "m", "metre": "m",
    "sq": "square", "hectares": "ha", "hectare": "ha",
    "feet": "ft", "foot": "ft", "miles": "mile", "mile": "mile",
    "square": "square", "km²": "km2", "m²": "m2",
}


def normalize_answer(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("km²", "km2").replace("m²", "m2")
    s = re.sub(r"[^\w\s]", " ", s)
    toks = [_UNIT_MAP.get(t, t) for t in s.split()]
    toks = [_NUM_WORDS.get(t, t) for t in toks]
    while toks and toks[0] in ("the", "a", "an"):
        toks.pop(0)
    return " ".join(toks)


def project_to_vocab(norm_pred: str, vocab: list[str], cutoff: float = 0.88) -> str:
    if not norm_pred or norm_pred in vocab:
        return norm_pred
    import difflib

    best = difflib.get_close_matches(norm_pred, vocab, n=1, cutoff=cutoff)
    return best[0] if best else norm_pred


# ------------------------------------------------------------- metrics


def metrics_vqa(rows, lut):
    vocab = sorted({normalize_answer(e["ground_truth"]) for e in lut.values()})
    n = em_raw = em_norm = em_proj = 0
    per_type = {}
    for r in rows:
        e = lut.get(r["id"])
        if e is None:
            continue
        n += 1
        gt_raw = e["ground_truth"].strip()
        pr_raw = r["raw_output"].strip()
        gt_n, pr_n = normalize_answer(gt_raw), normalize_answer(pr_raw)
        pr_p = project_to_vocab(pr_n, vocab)
        ok_raw = pr_raw == gt_raw
        ok_n = pr_n == gt_n
        ok_p = pr_p == gt_n
        em_raw += ok_raw
        em_norm += ok_n
        em_proj += ok_p
        t = e.get("type", "?")
        d = per_type.setdefault(t, [0, 0, 0, 0])
        d[0] += 1
        d[1] += ok_raw
        d[2] += ok_n
        d[3] += ok_p
    out = {
        "n": n,
        "em_raw": round(em_raw / max(n, 1), 4),
        "em_norm": round(em_norm / max(n, 1), 4),
        "em_norm_proj": round(em_proj / max(n, 1), 4),
        "per_type": {
            t: {"n": d[0], "em_raw": round(d[1] / d[0], 4),
                "em_norm": round(d[2] / d[0], 4),
                "em_norm_proj": round(d[3] / d[0], 4)}
            for t, d in sorted(per_type.items())
        },
        "normalizer": "strip, lower, punct-strip, article-drop, "
                      "number-word->digit, unit map; projection: difflib "
                      "cutoff 0.88 over frozen-id answer vocab",
    }
    return out


def metrics_caption(rows, lut):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge

    gts, res = {}, {}
    for r in rows:
        e = lut.get(r["id"])
        if e is None:
            continue
        cap = " ".join(r["raw_output"].split())
        gts[r["id"]] = [e["ground_truth"].strip().replace("\n", " ")]
        res[r["id"]] = [cap]
    n = len(res)
    b, _ = Bleu().compute_score(gts, res)
    m, _ = Meteor().compute_score(gts, res)
    rg, _ = Rouge().compute_score(gts, res)
    c, _ = Cider().compute_score(gts, res)
    return {
        "n": n,
        "bleu1": round(b[0], 4), "bleu2": round(b[1], 4),
        "bleu3": round(b[2], 4), "bleu4": round(b[3], 4),
        "meteor": round(m, 4), "rouge_l": round(rg, 4),
        "cider": round(c, 4),
        "scorer": "pycocoevalcap",
    }


def metrics_referring(rows, lut):
    per = {"n": 0, "acc05": 0, "acc07": 0, "miou": 0.0, "I": 0.0, "U": 0.0,
           "f_acc05": 0, "f_miou": 0.0, "parse_fail": 0}
    splits = {}
    scale_hist = {}
    for r in rows:
        e = lut.get(r["id"])
        if e is None:
            continue
        per["n"] += 1
        tag = r.get("scale_tag", "?")
        scale_hist[tag] = scale_hist.get(tag, 0) + 1
        key = "unique" if e.get("unique") else "non_unique"
        sp = splits.setdefault(key, [0, 0, 0.0])
        sp[0] += 1
        if r.get("pred01") is None:
            per["parse_fail"] += 1
            continue
        gt01 = parse_gt_box(e["ground_truth"])
        iou = iou_official(r["pred01"], gt01)
        fiou = iou_float(r["pred01"], gt01)
        per["acc05"] += iou >= 0.5
        per["acc07"] += iou >= 0.7
        per["miou"] += iou
        per["f_acc05"] += fiou >= 0.5
        per["f_miou"] += fiou
        sp[1] += iou >= 0.5
        sp[2] += iou
    n = max(per["n"], 1)
    return {
        "n": per["n"],
        "acc_iou_0.5": round(per["acc05"] / n, 4),
        "acc_iou_0.7": round(per["acc07"] / n, 4),
        "mean_iou": round(per["miou"] / n, 4),
        "mean_iou_float": round(per["f_miou"] / n, 4),
        "acc_iou_0.5_float": round(per["f_acc05"] / n, 4),
        "parse_fail": per["parse_fail"],
        "by_unique": {
            k: {"n": v[0], "acc_iou_0.5": round(v[1] / max(v[0], 1), 4),
                "mean_iou": round(v[2] / max(v[0], 1), 4)}
            for k, v in splits.items()
        },
        "scale_tags": scale_hist,
        "convention": "official eval_fianl: int 0-100 grid, +1 inclusive IoU; "
                      "_float = plain IoU on 0-1 boxes",
    }


# ------------------------------------------------------------- core loop


def _run_column(llm, geometry, col: str, out_dir: Path,
                limit: int | None = None, model_key: str = "qwen3vl8b"):
    """Generate preds for one column (resume-safe), then score."""
    t0 = time.time()
    frozen = _load_frozen(col)
    lut = _load_gt(col)
    prompts = {"vqa": VQA_PROMPT, "caption": CAP_PROMPT, "referring": REF_PROMPT}

    preds_path = out_dir / f"{col}_preds.jsonl"
    done = set()
    if preds_path.exists():
        for line in preds_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])

    samples = []
    missing_img = 0
    for fid in frozen["ids"]:
        if fid in done:
            continue
        e = lut.get(fid)
        if e is None:
            continue
        img = e["image_id"]
        if not Path(IMG_DIR, img).exists():
            missing_img += 1
            continue
        samples.append({"id": fid, "image": img,
                        "prompt": prompts[col].format(q=e["question"])})
    if limit:
        samples = samples[:limit]
    print(f"[{col}] frozen={len(frozen['ids'])} done={len(done)} "
          f"todo={len(samples)} missing_img={missing_img}", flush=True)

    f = open(preds_path, "a", buffering=1)
    factor, min_px, max_px = geometry
    CHUNK = 1024
    done_n = 0
    for c0 in range(0, len(samples), CHUNK):
        chunk = samples[c0:c0 + CHUNK]
        for it, raw, (w, h) in _gen(llm, chunk, col):
            row = {"id": it["id"], "image": it["image"],
                   "prompt": it["prompt"], "raw_output": raw}
            if col == "referring":
                rw, rh = _resized_dims(w, h, factor, min_px, max_px)
                box01, tag = parse_pred_box(raw, w, h, rw, rh, model_key)
                row.update({"pred01": box01, "scale_tag": tag,
                            "img_wh": [w, h], "resized_wh": [rw, rh]})
            f.write(json.dumps(row) + "\n")
            done_n += 1
        el = time.time() - t0
        rate = done_n / max(el, 1e-9)
        eta = (len(samples) - c0 - len(chunk)) / max(rate, 1e-9)
        print(f"[{col}] {c0 + len(chunk)}/{len(samples)} "
              f"{rate:.2f}/s eta={eta/60:.0f}min "
              f"proj_usd={el*USD_PER_S:.2f}", flush=True)
        if (c0 // CHUNK) % 4 == 3:
            data_vol.commit()
    f.close()
    data_vol.commit()

    # ---- score from the full preds file (all rows, frozen order)
    rows = [json.loads(l) for l in preds_path.read_text().splitlines() if l.strip()]
    frozen_set = set(frozen["ids"])
    scored = [r for r in rows if r["id"] in frozen_set]
    if col == "vqa":
        metrics = metrics_vqa(scored, lut)
    elif col == "caption":
        metrics = metrics_caption(scored, lut)
    else:
        metrics = metrics_referring(scored, lut)
    metrics["ids_sha256"] = frozen["ids_sha256"]
    metrics["n_frozen"] = frozen["n"]
    metrics["n_preds"] = len(scored)
    metrics["seconds"] = round(time.time() - t0, 1)
    (out_dir / f"{col}_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n")
    print(f"[{col}] METRICS {json.dumps({k: v for k, v in metrics.items() if not isinstance(v, dict)})}",
          flush=True)
    data_vol.commit()
    return metrics


def _update_manifest(model_key: str, model_meta: dict, col_results: dict,
                     run_stats: dict):
    mpath = Path(EVAL_ROOT) / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {}
    man.setdefault("task", "VRSBENCH-EVAL")
    man["updated_utc"] = datetime.now(timezone.utc).isoformat()
    man["protocol"] = {
        "vqa": {"prompt": VQA_PROMPT, "prompt_sha256": _sha256_str(VQA_PROMPT),
                "decoding": DECODING["vqa"],
                "metric": "em_raw | em_norm | em_norm_proj (see metrics.json)"},
        "caption": {"prompt": CAP_PROMPT + " (annotation question verbatim)",
                    "prompt_sha256": _sha256_str(CAP_PROMPT),
                    "decoding": DECODING["caption"],
                    "metric": "pycocoevalcap bleu1-4/meteor/rouge_l/cider"},
        "referring": {"prompt": REF_PROMPT,
                      "prompt_sha256": _sha256_str(REF_PROMPT),
                      "decoding": DECODING["referring"],
                      "metric": "acc@0.5 acc@0.7 meanIoU cumIoU, official "
                                "int-0-100 +1 convention; float variant "
                                "secondary",
                      "rescale": "box_start span -> resized-px frame -> /resized; "
                                 "bbox_2d key -> qwen3: /1000, qwen2.5: /resized; "
                                 "bare tuple/first4 -> grid heuristic "
                                 "(<=1.5: 0-1, <=100.5: /100, <=1000.5: /1000, "
                                 "else /img dims); per-row scale_tag in preds",
                      "gt_grid": "0-100 -> /100"},
    }
    man["ids_sha256"] = {
        c: _load_frozen(c)["ids_sha256"] for c in ID_FILES
    }
    man.setdefault("models", {})[model_key] = model_meta
    man.setdefault("columns", {}).update(col_results)
    man.setdefault("runs", []).append(run_stats)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(man, indent=2) + "\n")
    data_vol.commit()


# ---------------------------------------------------------------- entry


@app.function(
    image=eval_image,
    gpu=GPU_NAME,
    volumes={DATA_ROOT: data_vol, MODELS_ROOT: model_vol},
    cpu=CPU_CORES,
    memory=int(MEM_GIB * 1024),
    timeout=3600,
)
def sanity(model: str = "qwen3vl8b") -> dict:
    """Pre-flight hand-check: 8 grounding + 3 VQA + 2 caption samples."""
    t0 = time.time()
    llm, geometry, meta = _load_llm(model)
    report = {"model": meta, "checks": {}}

    lut = _load_gt("referring")
    frozen = _load_frozen("referring")
    factor, min_px, max_px = geometry
    items = []
    for fid in frozen["ids"][:8]:
        e = lut[fid]
        items.append({"id": fid, "image": e["image_id"],
                      "prompt": REF_PROMPT.format(q=e["question"])})
    rows = []
    for it, raw, (w, h) in _gen(llm, items, "referring"):
        e = lut[it["id"]]
        rw, rh = _resized_dims(w, h, factor, min_px, max_px)
        box01, tag = parse_pred_box(raw, w, h, rw, rh, model)
        gt01 = parse_gt_box(e["ground_truth"])
        iou = iou_official(box01, gt01) if box01 else 0.0
        rows.append({"id": it["id"], "q": e["question"], "raw": raw,
                     "tag": tag, "pred01": box01, "gt01": gt01,
                     "img_wh": [w, h], "resized_wh": [rw, rh],
                     "iou_official": round(iou, 4)})
        print(f"SANITY-REF {it['id']} tag={tag} iou={iou:.3f}\n"
              f"  raw={raw!r}\n  pred01={box01}\n  gt01={gt01}", flush=True)
    report["checks"]["referring"] = rows

    lut_v = _load_gt("vqa")
    fz_v = _load_frozen("vqa")
    items = [{"id": fid, "image": lut_v[fid]["image_id"],
              "prompt": VQA_PROMPT.format(q=lut_v[fid]["question"])}
             for fid in fz_v["ids"][:3]]
    rows = []
    for it, raw, _ in _gen(llm, items, "vqa"):
        e = lut_v[it["id"]]
        rows.append({"id": it["id"], "q": e["question"], "gt": e["ground_truth"],
                     "raw": raw})
        print(f"SANITY-VQA {it['id']} gt={e['ground_truth']!r} raw={raw!r}",
              flush=True)
    report["checks"]["vqa"] = rows

    lut_c = _load_gt("caption")
    fz_c = _load_frozen("caption")
    items = [{"id": fid, "image": lut_c[fid]["image_id"],
              "prompt": CAP_PROMPT.format(q=lut_c[fid]["question"])}
             for fid in fz_c["ids"][:2]]
    rows = []
    for it, raw, _ in _gen(llm, items, "caption"):
        e = lut_c[it["id"]]
        rows.append({"id": it["id"], "gt": e["ground_truth"][:160], "raw": raw[:300]})
        print(f"SANITY-CAP {it['id']}\n  gt={e['ground_truth'][:160]!r}\n"
              f"  raw={raw[:300]!r}", flush=True)
    report["checks"]["caption"] = rows

    report["run"] = _runstats(t0)
    Path(EVAL_ROOT).mkdir(parents=True, exist_ok=True)
    (Path(EVAL_ROOT) / f"sanity_report_{model}.json").write_text(
        json.dumps(report, indent=2) + "\n")
    data_vol.commit()
    return report


@app.function(
    image=eval_image,
    gpu=GPU_NAME,
    volumes={DATA_ROOT: data_vol, MODELS_ROOT: model_vol},
    cpu=CPU_CORES,
    memory=int(MEM_GIB * 1024),
    timeout=6 * 3600,
    retries=1,
)
def run(column: str = "all", model: str = "qwen3vl8b",
        limit: int | None = None) -> dict:
    """Scored run. column in {vqa, caption, referring, all}."""
    t0 = time.time()
    cols = ["vqa", "caption", "referring"] if column == "all" else [column]
    llm, geometry, meta = _load_llm(model)
    out_dir = Path(EVAL_ROOT) / model
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for col in cols:
        results[col] = _run_column(llm, geometry, col, out_dir,
                                   limit=limit, model_key=model)
        proj = (time.time() - t0) * USD_PER_S
        if proj > BUDGET_HALT_USD:
            print(f"[budget] projected ${proj:.2f} > ${BUDGET_HALT_USD} — "
                  f"stopping after {col}", flush=True)
            break

    stats = _runstats(t0, {"columns_done": list(results)})
    _update_manifest(model, meta,
                     {f"{model}/{c}": {"metrics": f"{model}/{c}_metrics.json",
                                       "preds": f"{model}/{c}_preds.jsonl",
                                       "n": m["n"]}
                      for c, m in results.items()},
                     stats)
    return {"stats": stats,
            "metrics": {c: {k: v for k, v in m.items()
                            if not isinstance(v, dict)}
                        for c, m in results.items()}}


# --------------------------------------------------------------------------
# CAPTION-EVAL — adapted ckpt (rsvqa_adapt full_qwen3vl) on frozen caption ids
# --------------------------------------------------------------------------


def _caption_len_stats(preds_path: Path):
    """Word-count stats over raw_output of a caption preds.jsonl."""
    ws = []
    if not preds_path.exists():
        return None
    for line in preds_path.read_text().splitlines():
        if line.strip():
            ws.append(len(json.loads(line)["raw_output"].split()))
    if not ws:
        return None
    s = sorted(ws)
    return {
        "n": len(s),
        "mean": round(statistics.mean(s), 2),
        "median": statistics.median(s),
        "p10": s[int(0.10 * (len(s) - 1))],
        "p90": s[int(0.90 * (len(s) - 1))],
        "min": s[0],
        "max": s[-1],
    }


def _assert_caption_overlap() -> dict:
    """Overlap gate: training mix vs frozen caption ids. The caption fold of
    the mix is a regularizer built from VRSBench_train rows — its eval_id
    field is 'image#?' (no question id), so the load-bearing check is at
    image level; id-string overlap is reported too. Any overlap -> raise."""
    frozen = _load_frozen("caption")
    fids = set(frozen["ids"])
    fimgs = {i.split("#")[0] for i in fids}
    n_rows = 0
    mix_ids, mix_imgs = set(), set()
    with open(MIX_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n_rows += 1
            if r.get("eval_id"):
                mix_ids.add(r["eval_id"].split("#")[0])
            if r.get("image"):
                mix_imgs.add(r["image"].rsplit("/", 1)[-1])
    id_hit = sorted(fimgs & mix_ids)
    img_hit = sorted(fimgs & mix_imgs)
    res = {
        "mix_path": MIX_PATH,
        "mix_sha256": CKPT_SHA256["mix.jsonl"],
        "mix_rows": n_rows,
        "frozen_n": len(fids),
        "frozen_images_n": len(fimgs),
        "id_overlap_n": len(id_hit),
        "image_overlap_n": len(img_hit),
        "sample_hits": img_hit[:5],
    }
    if id_hit or img_hit:
        raise RuntimeError(f"OVERLAP GATE FAILED {json.dumps(res)}")
    print(f"[overlap] OK {json.dumps(res)}", flush=True)
    return res


def _declare_run(model_key: str, model_meta: dict, extra: dict) -> None:
    """Write protocol + pending run entry into manifest BEFORE any scored
    prediction (spec: declare before preds)."""
    mpath = Path(EVAL_ROOT) / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {}
    man.setdefault("task", "VRSBENCH-EVAL")
    man["updated_utc"] = datetime.now(timezone.utc).isoformat()
    man.setdefault("protocol", {})["caption"] = {
        "prompt": CAP_PROMPT + " (annotation question verbatim)",
        "prompt_sha256": _sha256_str(CAP_PROMPT),
        "decoding": DECODING["caption"],
        "metric": "pycocoevalcap bleu1-4/meteor/rouge_l/cider",
    }
    man.setdefault("ids_sha256", {})["caption"] = \
        _load_frozen("caption")["ids_sha256"]
    man.setdefault("models", {})[model_key] = model_meta
    man.setdefault("declared_runs", []).append(extra)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(man, indent=2) + "\n")
    data_vol.commit()


@app.function(
    image=eval_image,
    volumes={DATA_ROOT: data_vol, MODELS_ROOT: model_vol},
    cpu=8.0,
    memory=96 * 1024,
    timeout=3600,
)
def merge_ckpt() -> dict:
    """ckpt_final (LoRA adapter + merger.pt) -> merged full-bf16 weights dir
    for vLLM. ORDER MATTERS: the adapter is attached first so merger.pt's
    peft-prefixed keys (base_model.model.model.visual.merger.*) resolve —
    loading them into the raw base model silently no-ops under
    strict=False. merge_and_unload() then bakes the LoRA deltas in."""
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as AutoModel
    except ImportError:  # older name
        from transformers import AutoModelForVision2Seq as AutoModel

    t0 = time.time()
    data_vol.reload()
    ckpt = Path(ADAPTED_CKPT_DIR)
    out = Path(ADAPTED_MERGED_DIR)
    report = {"step": "merge_ckpt", "ckpt_dir": str(ckpt)}
    mman_path = out / "merged_manifest.json"
    cpu_rate = 8 * 0.0000131 + 96 * 0.00000222

    sha_adapter = _sha256_file(ckpt / "adapter" / "adapter_model.safetensors")
    sha_merger = _sha256_file(ckpt / "merger.pt")
    assert sha_adapter == CKPT_SHA256["adapter_model.safetensors"], \
        f"adapter sha {sha_adapter} != pinned"
    assert sha_merger == CKPT_SHA256["merger.pt"], \
        f"merger sha {sha_merger} != pinned"
    report["input_sha256"] = {"adapter_model.safetensors": sha_adapter,
                              "merger.pt": sha_merger}

    base = MODELS["qwen3vl8b"]
    base_dir = snapshot_download(base["hf_id"], revision=base["revision"])
    model_vol.commit()

    if not (out / "model.safetensors").exists():
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
    # base snapshot's mm files so vLLM applies identical geometry.
    import shutil
    for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
        src = Path(base_dir) / name
        if src.exists():
            shutil.copyfile(src, out / name)

    file_shas = {}
    for p in sorted(out.iterdir()):
        if p.is_file():
            file_shas[p.name] = _sha256_file(p)
    tree_sha = _sha256_str("\n".join(
        f"{k}:{v}" for k, v in sorted(file_shas.items())))
    mman = {
        "step": "merge_ckpt",
        "merged_dir": str(out),
        "base": {"hf_id": base["hf_id"], "revision": base["revision"],
                 # HF-file sha set computed in the zero-shot lane manifest
                 "weights_sha256":
                 "aa8d7f7ef1f6867301225a39a924e49dea4b9c36fc9ce953eec52dc692c27a91"},
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
        "run": _runstats(t0, usd_per_s=cpu_rate, gpu="none (CPU merge)"),
    }
    mman_path.write_text(json.dumps(mman, indent=2) + "\n")
    data_vol.commit()
    report.setdefault("status", "OK")
    report.update({"merged_tree_sha256": tree_sha, "n_files": len(file_shas)})
    report["run"] = mman["run"]
    return report


@app.function(
    image=eval_image,
    gpu=GPU_NAME,
    volumes={DATA_ROOT: data_vol, MODELS_ROOT: model_vol},
    cpu=CPU_CORES,
    memory=int(MEM_GIB * 1024),
    timeout=6 * 3600,
    retries=1,
)
def run_caption_adapted(limit: int | None = None) -> dict:
    """Adapted-ckpt caption column on the frozen ids. Overlap gate + manifest
    declaration happen BEFORE the first prediction."""
    t0 = time.time()
    data_vol.reload()
    model = "qwen3vl8b_rsvqa"
    overlap = _assert_caption_overlap()          # raises -> halt
    llm, geometry, meta = _load_llm(model)
    _declare_run(model, meta, {
        "column": f"{model}/caption",
        "status": "declared_before_preds",
        "declared_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": "ripper2005 (user override; bootstrap profile is "
                     "harsha-610vmg — see run notes)",
        "prompt_sha256": _sha256_str(CAP_PROMPT),
        "decoding": DECODING["caption"],
        "ids_sha256": _load_frozen("caption")["ids_sha256"],
        "overlap_gate": overlap,
        "model": meta,
    })
    out_dir = Path(EVAL_ROOT) / model
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = _run_column(llm, geometry, "caption", out_dir,
                          limit=limit, model_key="qwen3vl8b")

    # length distribution — the zero-shot CIDEr collapse was a length-penalty
    # artifact, so lengths are part of the finding.
    adapted_len = _caption_len_stats(out_dir / "caption_preds.jsonl")
    base_len = _caption_len_stats(
        Path(EVAL_ROOT) / "qwen3vl8b" / "caption_preds.jsonl")
    metrics["pred_len_words"] = adapted_len
    metrics["baseline_pred_len_words"] = base_len
    metrics["overlap_gate"] = overlap
    metrics["model"] = meta
    (out_dir / "caption_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n")

    stats = _runstats(t0, {"columns_done": [f"{model}/caption"],
                           "workspace": "ripper2005"})
    _update_manifest(model, meta,
                     {f"{model}/caption": {
                         "metrics": f"{model}/caption_metrics.json",
                         "preds": f"{model}/caption_preds.jsonl",
                         "n": metrics["n"]}},
                     stats)
    return {"stats": stats,
            "metrics": {k: v for k, v in metrics.items()
                        if not isinstance(v, dict)},
            "pred_len_words": adapted_len,
            "baseline_pred_len_words": base_len}
