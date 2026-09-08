"""PROD-PACK-10 CPU mix: quarantined indices + samples.jsonl. Does NOT train. Does NOT download Images_train.zip."""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # SatQuery/
GATES = ROOT / "gates"
CACHE = GATES / "_cache"
OUT = CACHE / "prod10"
EVAL_IDS_PATH = GATES / "baseline_eval_ids.json"
CDVQA_EVAL_PATH = GATES / "cdvqa_eval_ids.json"
BEN_SAMPLES = CACHE / "prod" / "samples.jsonl"
LEVIR_SCORE = ROOT / "demo" / "data" / "scene2" / "levir_score.json"
PREFLIGHT = OUT / "two_image_preflight.json"
QUARANTINE_PATH = OUT / "quarantine_report.json"
INDICES_PATH = OUT / "prod10_indices.json"
SAMPLES_PATH = OUT / "samples.jsonl"
PACK_LOG_PATH = OUT / "pack_log.json"
VERBATIM_PATH = OUT / "samples_verbatim.json"
CDVQA_RAW = OUT / "cdvqa_raw"

HF_REPO = "xiang709/VRSBench"
HF_TRAIN_JSON = "VRSBench_train.json"
SEED = 42
N_VQA = 4_800
N_CAPTION = 8_400
N_BRIEF = 3_600
N_CROSS = 4_800
N_CHANGE = 2_400
N_DIR_HARD = 300
N_TOTAL_WITH_CHANGE = 24_000
N_TOTAL_NO_CHANGE = 21_600

PREFIX = {
    "vqa": "[vqa] Answer with a single word or phrase.",
    "caption": "[caption] Describe the scene in detail.",
    "brief": "[brief] Write an evidence-grounded briefing from the tools.",
    "cross": "[cross] Use optical and SAR together. Image 1 is optical, Image 2 is SAR.",
    "change": "[change] Image 1 is before, Image 2 is after. What changed, and where?",
    "cross_sar": "[cross] This is a SAR VV gray-RGB preview (single image; optical not packed).",
}
DIR_SUFFIX = "Answer one direction pair only; do not mention the other axis."
DIR_RE = re.compile(
    r"(east-?west|west-?east|north-?south|south-?north|north-?east|north-?west|"
    r"south-?east|south-?west|which direction|what direction|oriented|"
    r"running east|running west|running north|running south|"
    r"point(?:s|ing)? (?:to|toward)|heading)",
    re.I,
)
TAG_RE = re.compile(r"^(?:<image>\s*)?\[([^\]]+)\]\s*", re.I)
SHORT_TAIL = re.compile(r"\s*\.?\s*A short answer to the question is\s*$", re.I)
BAN_GOLD_RE = re.compile(r"(270611|81882|12689|test_45)", re.I)
CDVQA_BASE = "https://raw.githubusercontent.com/YZHJessica/CDVQA/main/"
GSD_M = 0.5
PIXEL_M2 = GSD_M * GSD_M
LEVIR_HW = 1024 * 1024


def _hf(filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=HF_REPO,
            filename=filename,
            repo_type="dataset",
            cache_dir=str(CACHE / "hf"),
        )
    )


def load_eval_ban() -> dict:
    meta = json.loads(EVAL_IDS_PATH.read_text(encoding="utf-8"))
    example_ids = {str(x) for x in meta["example_ids"]}
    if len(example_ids) != 300:
        raise RuntimeError(f"STOP: baseline_eval_ids.json n={len(example_ids)} expected 300")
    image_ids = set()
    stems = set()
    questions = set()
    for eid in example_ids:
        img = eid.split("::", 1)[0]
        image_ids.add(img)
        stems.add(Path(img).stem)
    subset = CACHE / "vrsbench" / "subset.json"
    if subset.exists():
        for row in json.loads(subset.read_text(encoding="utf-8")):
            questions.add((str(row["image_id"]), str(row["question"]).strip().lower()))
    return {
        "n_eval_ids": len(example_ids),
        "example_ids": example_ids,
        "image_ids": image_ids,
        "stems": stems,
        "questions": questions,
        "meta": meta,
    }


def load_cdvqa_ban() -> dict:
    if not CDVQA_EVAL_PATH.exists():
        raise FileNotFoundError(f"STOP: {CDVQA_EVAL_PATH} missing; run python train10/prod10_cdvqa_ids.py first")
    meta = json.loads(CDVQA_EVAL_PATH.read_text(encoding="utf-8"))
    pairs = {str(x) for x in meta.get("pair_ids_test_union") or []}
    stems = {Path(x).stem for x in pairs}
    holdout = {str(x) for x in meta.get("qa_ids_holdout_n100") or []}
    return {
        "pair_ids": pairs,
        "stems": stems,
        "holdout_qa": holdout,
        "meta": meta,
        "n_pairs": len(pairs),
    }


def strip_instruction(human: str, tag: str) -> str:
    t = (human or "").replace("<image>", " ").strip()
    t = TAG_RE.sub("", t).strip()
    if tag == "vqa":
        t = SHORT_TAIL.sub("", t).strip()
    return t.strip()


def parse_train_row(raw: dict, idx: int) -> dict | None:
    conv = raw.get("conversations") or []
    human = next((c.get("value", "") for c in conv if c.get("from") == "human"), "")
    gpt = next((c.get("value", "") for c in conv if c.get("from") == "gpt"), "")
    m = TAG_RE.match((human or "").strip())
    tag = m.group(1).lower() if m else "none"
    image = str(raw.get("image") or "").strip()
    if not image or not gpt or not str(gpt).strip():
        return None
    instruction = strip_instruction(human, tag)
    if not instruction:
        return None
    return {
        "idx": idx,
        "image_id": image,
        "tag": tag,
        "instruction": instruction,
        "answer": str(gpt).strip(),
        "example_id": f"{image}::train::{idx}",
        "direction": bool(DIR_RE.search(instruction)),
    }


def vrs_banned(r: dict, ban: dict) -> bool:
    hit_eid = r["example_id"] in ban["example_ids"]
    hit_img = r["image_id"] in ban["image_ids"] or Path(r["image_id"]).stem in ban["stems"]
    hit_q = (r["image_id"], r["instruction"].lower()) in ban["questions"]
    return hit_eid or hit_img or hit_q


def gold_banned(text: str) -> bool:
    return bool(BAN_GOLD_RE.search(text or ""))


def sample_choice(pool: list, n: int, seed: int) -> list:
    rng = np.random.RandomState(seed)
    if n > len(pool):
        raise RuntimeError(f"STOP: want {n} from pool {len(pool)}")
    order = rng.choice(len(pool), size=n, replace=False)
    return [pool[int(i)] for i in order]


def single_messages(prefix: str, instruction: str, answer: str, extra: str = "") -> list:
    human = f"{prefix} {instruction}".strip()
    if extra:
        human = f"{human} {extra}".strip()
    return [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": human}]},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]


def two_messages(prefix: str, instruction: str, answer: str) -> list:
    human = f"{prefix} {instruction}".strip()
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": human},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]


def read_preflight() -> dict:
    if not PREFLIGHT.exists():
        return {"pass": False, "status": "MISSING", "reason": f"{PREFLIGHT} not written"}
    return json.loads(PREFLIGHT.read_text(encoding="utf-8"))


def _download_cdvqa(name: str) -> Path:
    CDVQA_RAW.mkdir(parents=True, exist_ok=True)
    dest = CDVQA_RAW / name
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    url = CDVQA_BASE + name
    print(f"GET {url}", flush=True)
    with urllib.request.urlopen(url, timeout=180) as r:
        dest.write_bytes(r.read())
    return dest


def load_cdvqa_train() -> tuple[list[dict], dict]:
    images = json.loads(_download_cdvqa("Train_images.json").read_text(encoding="utf-8"))["images"]
    questions = json.loads(_download_cdvqa("Train_questions.json").read_text(encoding="utf-8"))["questions"]
    answers = json.loads(_download_cdvqa("Train_answers.json").read_text(encoding="utf-8"))["answers"]
    img_by_id = {int(r["id"]): r for r in images}
    ans_by_id = {int(r["id"]): r for r in answers}
    rows = []
    n_miss_img = n_miss_ans = 0
    for q in questions:
        img = img_by_id.get(int(q["img_id"]))
        if img is None:
            n_miss_img += 1
            continue
        aids = q.get("answers_ids") or []
        if not aids:
            n_miss_ans += 1
            continue
        ans = ans_by_id.get(int(aids[0]))
        if ans is None:
            n_miss_ans += 1
            continue
        gold = str(ans.get("answer") or ans.get("ans") or "").strip()
        if not gold:
            n_miss_ans += 1
            continue
        fn = str(img.get("file_name") or "")
        rows.append(
            {
                "qa_id": int(q["id"]),
                "pair_id": fn,
                "question": str(q.get("question") or "").strip(),
                "qtype": str(q.get("type") or ""),
                "answer": gold,
            }
        )
    stats = {
        "n_train_images_json": len(images),
        "n_train_questions_json": len(questions),
        "n_train_answers_json": len(answers),
        "n_joined": len(rows),
        "n_miss_img": n_miss_img,
        "n_miss_ans": n_miss_ans,
        "unique_pairs": len({r["pair_id"] for r in rows}),
    }
    return rows, stats


BRIEF_OPENERS = [
    "Tool-grounded briefing for this before/after pair.",
    "Evidence briefing from the frozen change-detection scores.",
    "The following figures come only from the scored LEVIR tool JSON.",
    "Change briefing (numbers frozen; the VLM did not invent km²).",
    "Built-up change summary using the official 0.5 m GSD formula.",
    "Paired-image briefing: Image 1 is before, Image 2 is after.",
    "Scored-mask briefing for this LEVIR-CD test pair.",
    "Area briefing derived from predicted-change pixel count.",
    "This note restates the frozen IoU, F1, and area_calc formula.",
    "Evidence-only briefing; prose must not add extra measurements.",
]
BRIEF_MIDDLES = [
    "Predicted changed pixels are {pred}. Ground-truth changed pixels are {gt}.",
    "The scored mask reports pred_changed={pred} and gt_changed={gt}.",
    "Pixel counts from the frozen JSON: predicted {pred}, reference {gt}.",
    "ChangeFormer-scored counts: {pred} predicted changed pixels versus {gt} GT pixels.",
    "Frozen row counts: pred_changed {pred}; gt_changed {gt}.",
    "The tool JSON lists pred_changed={pred} and gt_changed={gt} on this pair.",
]
BRIEF_METRICS = [
    "IoU is {iou:.6f} and F1 is {f1:.6f}.",
    "Against GT the scored mask has IoU {iou:.6f} and F1 {f1:.6f}.",
    "Mask agreement: IoU={iou:.6f}, F1={f1:.6f} (from the frozen JSON, not the VLM).",
    "Reported overlap metrics are IoU {iou:.6f} and F1 {f1:.6f}.",
    "Keep IoU {iou:.6f} and F1 {f1:.6f} exactly as scored.",
    "Do not round away the frozen IoU {iou:.6f} or F1 {f1:.6f}.",
]
BRIEF_AREAS = [
    "Using area_m2 = count * 0.5^2, predicted area is {area_m2:.4f} m² ({area_km2:.8f} km²), {pct:.6f}% of the 1024×1024 image.",
    "GSD is 0.5 m/px so each pixel is 0.25 m². Predicted area = {pred} × 0.25 = {area_m2:.4f} m² ({area_km2:.8f} km²); {pct:.6f}% of the image.",
    "Formula count * 0.5^2 on pred_changed yields {area_m2:.4f} m², equal to {area_km2:.8f} km², covering {pct:.6f}% of the array.",
    "Predicted built-up change area is {area_m2:.4f} square metres ({area_km2:.8f} km²) at 0.5 m GSD ({pct:.6f}% of 1024²).",
    "area_calc on the predicted mask: {area_m2:.4f} m² / {area_km2:.8f} km² / {pct:.6f}% (GSD 0.5 m).",
    "Recomputed from pred_changed only: {area_m2:.4f} m² ({area_km2:.8f} km²), {pct:.6f}% of the image.",
]
BRIEF_CLOSERS = [
    "Cite these tool numbers; do not invent quadrant counts that are not in the JSON.",
    "No additional km² may be added beyond the formula above.",
    "This briefing is evidence-grounded and must stay within the frozen JSON.",
    "Image 1 is before and Image 2 is after; the numbers describe the scored change mask.",
    "Quadrant splits are omitted because they are not present in levir_score.json.",
]


def brief_gold(row: dict, opener: str, mid: str, met: str, area: str, closer: str) -> str:
    pred = int(row["pred_changed"])
    gt = int(row["gt_changed"])
    iou = float(row["iou"])
    f1 = float(row["f1"])
    area_m2 = pred * PIXEL_M2
    area_km2 = area_m2 / 1e6
    pct = 100.0 * pred / LEVIR_HW
    ctx = dict(pred=pred, gt=gt, iou=iou, f1=f1, area_m2=area_m2, area_km2=area_km2, pct=pct)
    parts = [
        opener,
        mid.format(**ctx),
        met.format(**ctx),
        area.format(**ctx),
        closer,
    ]
    return " ".join(parts)


def build_brief_rows() -> tuple[list[dict], dict]:
    score = json.loads(LEVIR_SCORE.read_text(encoding="utf-8"))
    names = [str(x) for x in score["names"]]
    if len(names) != 20:
        raise RuntimeError(f"STOP: levir_score names n={len(names)} expected 20")
    if any(n.lower() == "test_45.png" or "test_45" in n.lower() for n in names):
        raise RuntimeError("STOP: test_45 appeared in levir_score names")
    by_name = {r["name"]: r for r in score["rows"]}
    templates = []
    for o in BRIEF_OPENERS:
        for m in BRIEF_MIDDLES:
            for a in BRIEF_AREAS[:3]:
                templates.append((o, m, BRIEF_METRICS[len(templates) % len(BRIEF_METRICS)], a, BRIEF_CLOSERS[len(templates) % len(BRIEF_CLOSERS)]))
    # 10 * 6 * 3 = 180 templates
    if len(templates) != 180:
        raise RuntimeError(f"STOP: brief templates n={len(templates)} expected 180")
    rows = []
    for ti, tmpl in enumerate(templates):
        for name in names:
            src = by_name[name]
            gold = brief_gold(src, *tmpl)
            if gold_banned(gold) or "test_45" in gold.lower():
                raise RuntimeError(f"STOP: banned token in brief gold for {name}")
            n_sent = gold.count(".") + gold.count("!")
            rows.append(
                {
                    "slice": "brief",
                    "id": f"brief::{name}::t{ti}",
                    "pair_name": name,
                    "template_id": ti,
                    "instruction": "Image 1 is before, Image 2 is after. Write the evidence-grounded briefing from the tools.",
                    "answer": gold,
                    "n_sentences_approx": n_sent,
                    "pack_images": [f"png/levir/A/{name}", f"png/levir/B/{name}"],
                    "pred_changed": src["pred_changed"],
                    "gt_changed": src["gt_changed"],
                    "iou": src["iou"],
                    "f1": src["f1"],
                    "n_images": 2,
                }
            )
    if len(rows) != N_BRIEF:
        raise RuntimeError(f"STOP: brief n={len(rows)} expected {N_BRIEF}")
    stats = {
        "n_unique_pairs": len(names),
        "n_templates": len(templates),
        "n": len(rows),
        "banned_test_45": False,
        "gsd_m": GSD_M,
        "formula": "count * 0.5^2",
        "pairs": names,
    }
    return rows, stats


def build_vqa_caption(ban: dict) -> tuple[list[dict], list[dict], dict]:
    train_path = _hf(HF_TRAIN_JSON)
    print(f"train_json={train_path} bytes={train_path.stat().st_size}", flush=True)
    raw = json.loads(train_path.read_text(encoding="utf-8"))
    parsed = []
    tag_counts = Counter()
    for i, row in enumerate(raw):
        rec = parse_train_row(row, i)
        if rec is None:
            continue
        tag_counts[rec["tag"]] += 1
        parsed.append(rec)
    vqa_all = [r for r in parsed if r["tag"] == "vqa"]
    cap_all = [r for r in parsed if r["tag"] == "caption"]
    vqa_ok = [r for r in vqa_all if not vrs_banned(r, ban) and not gold_banned(r["answer"])]
    cap_ok = [r for r in cap_all if not vrs_banned(r, ban) and not gold_banned(r["answer"])]
    dir_ok = [r for r in vqa_ok if r["direction"]]
    non_dir = [r for r in vqa_ok if not r["direction"]]
    n_dir = min(N_DIR_HARD, len(dir_ok))
    dir_chosen = sample_choice(dir_ok, n_dir, SEED + 1) if n_dir else []
    need_rest = N_VQA - n_dir
    rest = sample_choice(non_dir, need_rest, SEED + 2)
    vqa_rows = []
    for r in dir_chosen:
        vqa_rows.append(
            {
                "slice": "vqa",
                "id": r["example_id"],
                "image_id": r["image_id"],
                "instruction": r["instruction"],
                "answer": r["answer"],
                "direction_hard_neg": True,
                "pack_images": [f"png/vrs/{r['image_id']}"],
                "n_images": 1,
                "src_idx": r["idx"],
            }
        )
    for r in rest:
        vqa_rows.append(
            {
                "slice": "vqa",
                "id": r["example_id"],
                "image_id": r["image_id"],
                "instruction": r["instruction"],
                "answer": r["answer"],
                "direction_hard_neg": False,
                "pack_images": [f"png/vrs/{r['image_id']}"],
                "n_images": 1,
                "src_idx": r["idx"],
            }
        )
    if len(vqa_rows) != N_VQA:
        raise RuntimeError(f"STOP: vqa n={len(vqa_rows)}")
    stats = {
        "tag_counts": dict(tag_counts),
        "n_vqa_all": len(vqa_all),
        "n_vqa_eligible": len(vqa_ok),
        "n_caption_all": len(cap_all),
        "n_caption_eligible": len(cap_ok),
        "n_direction_eligible": len(dir_ok),
        "n_direction_sampled": n_dir,
        "train_json_bytes": train_path.stat().st_size,
    }
    return vqa_rows, cap_ok, stats


def fill_captions(cap_ok: list[dict], n_need: int, ban: dict) -> tuple[list[dict], dict]:
    vrs_cap = sample_choice(cap_ok, min(n_need, len(cap_ok)), SEED + 3)
    rows = []
    for r in vrs_cap:
        rows.append(
            {
                "slice": "caption",
                "id": r["example_id"],
                "image_id": r["image_id"],
                "instruction": r["instruction"] or "Describe the scene in detail.",
                "answer": r["answer"],
                "pack_images": [f"png/vrs/{r['image_id']}"],
                "n_images": 1,
                "source": "vrsbench_caption",
            }
        )
    n_short = n_need - len(rows)
    ben_used = 0
    if n_short > 0:
        ben_caps = []
        with BEN_SAMPLES.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if str(rec.get("type", "")).lower() != "captioning":
                    continue
                pid = str(rec.get("patch_id", ""))
                if pid in ban["image_ids"] or pid in ban["stems"] or Path(pid).name in ban["image_ids"]:
                    continue
                ans = str(rec.get("answer") or "").strip()
                if not ans or gold_banned(ans):
                    continue
                ben_caps.append(rec)
        extra = sample_choice(ben_caps, n_short, SEED + 4)
        for rec in extra:
            rows.append(
                {
                    "slice": "caption",
                    "id": f"ben_cap::{rec.get('id')}",
                    "image_id": str(rec.get("patch_id")),
                    "instruction": str(rec.get("instruction") or "Describe the scene in detail."),
                    "answer": str(rec["answer"]).strip(),
                    "pack_images": [f"png/ben_s2/{int(rec['i']):05d}.png"],
                    "n_images": 1,
                    "source": "ben_captioning",
                    "country": rec.get("country"),
                    "season": rec.get("season"),
                    "ben_i": int(rec["i"]),
                }
            )
        ben_used = len(extra)
    if len(rows) != n_need:
        raise RuntimeError(f"STOP: caption n={len(rows)} need={n_need}")
    return rows, {"n_vrs_caption": len(vrs_cap), "n_ben_caption_fill": ben_used, "n": len(rows)}


def load_ben_pool(ban: dict) -> list[dict]:
    pool = []
    with BEN_SAMPLES.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pid = str(rec.get("patch_id", ""))
            if pid in ban["image_ids"] or pid in ban["stems"] or Path(pid).name in ban["image_ids"]:
                continue
            ans = str(rec.get("answer") or "").strip()
            if not ans or gold_banned(ans):
                continue
            if str(rec.get("id", "")) in ban["example_ids"]:
                continue
            pool.append(rec)
    return pool


CROSS_SAR_NOTE = (
    "Image 2 is SAR VV rendered as gray-RGB. Bright returns are typically rough or man-made "
    "surfaces; dark returns are typically smooth water or flat ground. Combine both images; "
    "do not treat either alone as complete. Do not output CLC class IDs."
)


def build_cross(pool: list[dict], two_image: bool) -> tuple[list[dict], dict]:
    chosen = sample_choice(pool, N_CROSS, SEED + 5)
    rows = []
    for rec in chosen:
        pid = str(rec.get("patch_id"))
        typ = str(rec.get("type", "")).lower()
        instr = str(rec.get("instruction") or "").strip()
        ans = str(rec.get("answer") or "").strip()
        if two_image:
            if typ == "captioning":
                gold = ans + " " + CROSS_SAR_NOTE
                q = "Describe what optical and SAR jointly show. Use land-cover vocabulary, not CLC IDs."
            else:
                gold = ans
                q = instr or "Answer using optical (image 1) and SAR VV (image 2) together."
            rows.append(
                {
                    "slice": "cross",
                    "id": f"cross::{rec.get('id')}",
                    "image_id": pid,
                    "instruction": q,
                    "answer": gold,
                    "pack_images": [
                        f"png/ben_s2/{int(rec['i']):05d}.png",
                        f"png/ben_s1/{int(rec['i']):05d}.png",
                    ],
                    "n_images": 2,
                    "ben_type": typ,
                    "country": rec.get("country"),
                    "season": rec.get("season"),
                    "ben_i": int(rec["i"]),
                }
            )
        else:
            gold = (
                f"SAR VV gray-RGB preview of a {rec.get('country')} {rec.get('season')} patch. "
                "Bright returns typically mark rough or built surfaces; dark returns typically "
                "mark smooth water or flat ground. This is not an optical RGB caption and not a CLC ID."
            )
            rows.append(
                {
                    "slice": "cross",
                    "id": f"cross_sar::{rec.get('id')}",
                    "image_id": pid,
                    "instruction": "Describe this SAR VV gray-RGB preview.",
                    "answer": gold,
                    "pack_images": [f"png/ben_s1/{int(rec['i']):05d}.png"],
                    "n_images": 1,
                    "ben_type": typ,
                    "country": rec.get("country"),
                    "season": rec.get("season"),
                    "ben_i": int(rec["i"]),
                    "shrunk_single_image_sar": True,
                }
            )
    stats = {"n": len(rows), "two_image": two_image, "ben_types": dict(Counter(r["ben_type"] for r in rows))}
    return rows, stats


def build_change(cdvqa_ban: dict) -> tuple[list[dict], dict]:
    joined, stats = load_cdvqa_train()
    eligible = []
    n_excl_pair = 0
    for r in joined:
        fn = r["pair_id"]
        if fn in cdvqa_ban["pair_ids"] or Path(fn).stem in cdvqa_ban["stems"]:
            n_excl_pair += 1
            continue
        if gold_banned(r["answer"]) or gold_banned(r["question"]):
            continue
        eligible.append(r)
    stats["n_excluded_test_pair"] = n_excl_pair
    stats["n_eligible"] = len(eligible)
    chosen = sample_choice(eligible, N_CHANGE, SEED + 6)
    overlap = [r for r in chosen if r["pair_id"] in cdvqa_ban["pair_ids"]]
    if overlap:
        raise RuntimeError(f"STOP: CDVQA test pair leaked into change sample n={len(overlap)}")
    rows = []
    for r in chosen:
        fn = r["pair_id"]
        rows.append(
            {
                "slice": "change",
                "id": f"cdvqa_train::{r['qa_id']}::{fn}",
                "pair_id": fn,
                "qa_id": r["qa_id"],
                "qtype": r["qtype"],
                "instruction": r["question"],
                "answer": r["answer"],
                "pack_images": [f"png/second/A/{fn}", f"png/second/B/{fn}"],
                "n_images": 2,
                "rasters_local": False,
            }
        )
    stats["n_sampled"] = len(rows)
    stats["n_unique_pairs_sampled"] = len({r["pair_id"] for r in rows})
    stats["rasters_note"] = (
        "QA/answers from official Train_*.json. Rasters are SECOND (not in GitHub). "
        "CPU pack stores A/B filenames for a later Modal fetch; QA is not invented from masks."
    )
    return rows, stats


def attach_messages(rec: dict) -> dict:
    sl = rec["slice"]
    extra = DIR_SUFFIX if rec.get("direction_hard_neg") else ""
    if rec.get("n_images") == 2:
        rec["messages"] = two_messages(PREFIX["cross" if sl == "cross" else sl], rec["instruction"], rec["answer"])
        if sl == "brief":
            rec["messages"] = two_messages(PREFIX["brief"], rec["instruction"], rec["answer"])
        if sl == "change":
            rec["messages"] = two_messages(PREFIX["change"], rec["instruction"], rec["answer"])
    else:
        key = "cross_sar" if rec.get("shrunk_single_image_sar") else sl
        rec["messages"] = single_messages(PREFIX[key], rec["instruction"], rec["answer"], extra=extra)
    rec["image"] = rec["pack_images"][0]
    rec["n_human_images"] = sum(1 for p in rec["messages"][0]["content"] if p.get("type") == "image")
    rec["human_text"] = next(p["text"] for p in rec["messages"][0]["content"] if p.get("type") == "text")
    return rec


def stratified_split(rows: list[dict]) -> list[dict]:
    rng = np.random.RandomState(SEED)
    by = {}
    for r in rows:
        by.setdefault(r["slice"], []).append(r)
    out = []
    split_counts = Counter()
    for sl, group in sorted(by.items()):
        order = rng.permutation(len(group))
        shuffled = [group[int(i)] for i in order]
        n_val = int(round(len(shuffled) * 0.05))
        val = shuffled[:n_val]
        train = shuffled[n_val:]
        for r in train:
            r["split"] = "train"
        for r in val:
            r["split"] = "val"
        out.extend(train)
        out.extend(val)
        split_counts[f"{sl}:train"] = len(train)
        split_counts[f"{sl}:val"] = len(val)
    for i, r in enumerate(out):
        r["i"] = i
    return out


def quarantine_check(records: list[dict], ban: dict, cdvqa_ban: dict, change_n: int) -> dict:
    n_vrs_ex = n_vrs_img = n_vrs_stem = 0
    n_cdvqa = 0
    n_gold_ban = 0
    n_test45 = 0
    examples = []
    for r in records:
        rid = str(r.get("id"))
        imgs = [str(x) for x in r.get("pack_images") or []]
        img_ids = [str(r.get("image_id") or ""), str(r.get("pair_id") or "")] + imgs
        stems = {Path(x).stem for x in img_ids if x}
        names = {Path(x).name for x in img_ids if x}
        if rid in ban["example_ids"]:
            n_vrs_ex += 1
            examples.append(("eval_example_id", rid))
        if any(x in ban["image_ids"] for x in img_ids + list(names)):
            n_vrs_img += 1
            examples.append(("eval_image", rid))
        if stems & ban["stems"]:
            n_vrs_stem += 1
            examples.append(("eval_stem", rid))
        if r.get("slice") == "change":
            if r.get("pair_id") in cdvqa_ban["pair_ids"] or Path(str(r.get("pair_id"))).stem in cdvqa_ban["stems"]:
                n_cdvqa += 1
                examples.append(("cdvqa_test_pair", rid))
        blob = " ".join([str(r.get("answer")), str(r.get("instruction")), " ".join(imgs)])
        if gold_banned(blob):
            n_gold_ban += 1
            examples.append(("banned_gold", rid))
        if "test_45" in blob.lower():
            n_test45 += 1
            examples.append(("test_45", rid))
    leakage = any(c > 0 for c in (n_vrs_ex, n_vrs_img, n_vrs_stem, n_cdvqa, n_gold_ban, n_test45))
    report = {
        "method": [
            "Ban gates/baseline_eval_ids.json (300 example ids + image stems + eval questions).",
            "Ban gates/cdvqa_eval_ids.json test1∪test2 pair file_names from [change].",
            "Ban gold integers 270611 and 81882, water_pixels fingerprint 12689, and test_45.",
            "Do not copy demo/traces/scene2.json or scene3.json into gold.",
            "VRSBench from xiang709/VRSBench VRSBench_train.json; BEN from 06 samples.jsonl.",
            "[brief] from demo/data/scene2/levir_score.json names n=20, test_45 absent.",
        ],
        "n_records": len(records),
        "n_eval_ids": ban["n_eval_ids"],
        "n_cdvqa_test_pairs": cdvqa_ban["n_pairs"],
        "overlaps": {
            "n_vrsbench_example_id": n_vrs_ex,
            "n_vrsbench_image_id": n_vrs_img,
            "n_vrsbench_stem": n_vrs_stem,
            "n_cdvqa_test_pair": n_cdvqa,
            "n_banned_gold_integers": n_gold_ban,
            "n_test_45": n_test45,
            "examples": examples[:20],
        },
        "n_overlap_total": n_vrs_ex + n_vrs_img + n_vrs_stem + n_cdvqa + n_gold_ban + n_test45,
        "leakage": leakage,
        "change_n": change_n,
    }
    return report


def verbatim_rows(records: list[dict]) -> list[dict]:
    by = {}
    for r in records:
        by.setdefault(r["slice"], []).append(r)
    out = []
    for sl in ("vqa", "caption", "brief", "cross", "change"):
        group = by.get(sl) or []
        for r in group[:2]:
            out.append(
                {
                    "slice": sl,
                    "id": r["id"],
                    "split": r["split"],
                    "n_images": r["n_images"],
                    "human_text": r["human_text"],
                    "answer": r["answer"][:500],
                    "pack_images": r["pack_images"],
                    "messages": r["messages"],
                }
            )
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ban = load_eval_ban()
    cdvqa_ban = load_cdvqa_ban()
    pre = read_preflight()
    if str(pre.get("status", "")).upper() == "MISSING":
        raise RuntimeError(f"STOP: two-image preflight not written ({pre.get('reason')})")
    two_ok = bool(pre.get("pass")) and str(pre.get("status", "")).upper() == "PASS"
    change_n = N_CHANGE if two_ok else 0
    # If change drops: the 2,400 go to captions (20/45/15/20/0 of 24000).
    # Do not invent VRSBench change pairs. n=21600 is the other reading of the spec;
    # this pack keeps 24000 and moves the quota onto real captions.
    caption_n = N_CAPTION + (0 if change_n else N_CHANGE)
    print(json.dumps({"preflight_pass": two_ok, "change_n": change_n, "caption_n": caption_n, "pre_status": pre.get("status")}), flush=True)

    vqa_rows, cap_ok, vrs_stats = build_vqa_caption(ban)
    cap_rows, cap_stats = fill_captions(cap_ok, caption_n, ban)
    brief_rows, brief_stats = build_brief_rows()
    ben_pool = load_ben_pool(ban)
    cross_rows, cross_stats = build_cross(ben_pool, two_image=two_ok)
    change_rows: list[dict] = []
    change_stats: dict = {"n": 0, "dropped": True, "reason": None}
    if change_n == N_CHANGE:
        change_rows, change_stats = build_change(cdvqa_ban)
        change_stats["dropped"] = False
    else:
        change_stats["reason"] = (
            "two-image preflight not PASS"
            if not two_ok
            else "change dropped"
        )
        if pre.get("status") == "MISSING":
            change_stats["reason"] = f"preflight missing: {pre.get('reason')}"
        # Spec: drop [change] if collator fails OR splits missing. Splits exist;
        # this branch is collator/preflight.
        change_stats["cdvqa_eval_ids_written"] = True
        change_stats["github_json_authentic"] = True
        change_stats["rasters_in_github"] = False

    all_rows = []
    for rec in vqa_rows + cap_rows + brief_rows + cross_rows + change_rows:
        all_rows.append(attach_messages(rec))
    records = stratified_split(all_rows)
    n = len(records)
    expected = N_VQA + caption_n + N_BRIEF + N_CROSS + change_n
    if n != expected:
        raise RuntimeError(f"STOP: n={n} expected={expected}")
    slice_counts = dict(Counter(r["slice"] for r in records))
    split_counts = dict(Counter(r["split"] for r in records))
    split_slice = dict(Counter(f"{r['split']}:{r['slice']}" for r in records))

    qrep = quarantine_check(records, ban, cdvqa_ban, change_n)
    QUARANTINE_PATH.write_text(json.dumps(qrep, indent=2), encoding="utf-8")
    if qrep["leakage"]:
        print("STOP: leakage=true", qrep["overlaps"], file=sys.stderr)
        return 2

    for r in records:
        if r["n_human_images"] != r["n_images"]:
            raise RuntimeError(f"STOP: messages image count {r['n_human_images']} != {r['n_images']} id={r['id']}")
        sl = r["slice"]
        ht = r["human_text"]
        if sl == "vqa" and not ht.startswith("[vqa]"):
            raise RuntimeError("STOP: vqa prefix missing")
        if sl == "caption" and not ht.startswith("[caption]"):
            raise RuntimeError("STOP: caption prefix missing")
        if sl == "brief" and not ht.startswith("[brief]"):
            raise RuntimeError("STOP: brief prefix missing")
        if sl == "cross" and not ht.startswith("[cross]"):
            raise RuntimeError("STOP: cross prefix missing")
        if sl == "change" and not ht.startswith("[change]"):
            raise RuntimeError("STOP: change prefix missing")

    payload = {
        "n": n,
        "n_train": split_counts.get("train", 0),
        "n_val": split_counts.get("val", 0),
        "seed": SEED,
        "slice_counts": slice_counts,
        "split_counts": split_counts,
        "split_slice_counts": split_slice,
        "shares_note": (
            "20/35/15/20/10 of 24000 with change, or 20/45/15/20/0 with 2400 moved to captions"
        ),
        "change_n": change_n,
        "two_image_preflight_pass": two_ok,
        "leakage": False,
        "vrs": vrs_stats,
        "caption": cap_stats,
        "brief": brief_stats,
        "cross": cross_stats,
        "change": change_stats,
        "hf_repo": HF_REPO,
        "hf_train_json": HF_TRAIN_JSON,
        "cdvqa_eval_ids": str(CDVQA_EVAL_PATH),
        "preflight": str(PREFLIGHT),
    }
    INDICES_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    SAMPLES_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    verb = verbatim_rows(records)
    VERBATIM_PATH.write_text(json.dumps(verb, indent=2, ensure_ascii=False), encoding="utf-8")
    pack_log = {
        **payload,
        "samples_path": str(SAMPLES_PATH),
        "samples_bytes": SAMPLES_PATH.stat().st_size,
        "n_verbatim": len(verb),
        "serve_ps1_untouched": True,
        "trained": False,
        "new_adapter": False,
        "new_gguf": False,
        "serve_two_image": "already Scene 2",
    }
    PACK_LOG_PATH.write_text(json.dumps(pack_log, indent=2), encoding="utf-8")
    print(json.dumps({
        "n": n,
        "slice_counts": slice_counts,
        "split_counts": split_counts,
        "leakage": False,
        "two_image": two_ok,
        "samples": str(SAMPLES_PATH),
        "samples_bytes": SAMPLES_PATH.stat().st_size,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"STOP: {type(e).__name__}: {e}", file=sys.stderr)
        raise
