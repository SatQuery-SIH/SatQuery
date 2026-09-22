"""LR-FOLD mix builder — deterministic, local-only.

Mix (user-clarified 2026-09-15):
  - LR share  : ALL 57,223 active LR_split_train rows; count-type rows at
                x2.5 multiplicity (literal repeats, seeded) — same treatment
                HR count got in the original recipe.
                NOTE: LR_split_train_questions.json has 77,232 entries but
                20,009 are bare stubs {id, active:false} whose qids are exactly
                the lr_val+lr_test eval qids. Only the 57,223 active joined
                rows are trainable.
  - HR replay : N_HR_REPLAY rows sampled VERBATIM from the old mix.jsonl
                (source=='rsvqa_hr') — count multiplicity already baked in;
                no re-oversample.  >=15k floor per spec.
  - Captions  : N_CAP rows sampled from old mix.jsonl (source=='vrsbench_cap').

Image field convention: RELATIVE paths under a data root the trainer owns
  rsvqa_lr/Images_LR/{img}.tif | rsvqa_hr/Data/{img}.png |
  vrsbench/Images_train/{fn}
so the same mix.jsonl works on Lightning and on a Modal fallback.

Overlap gate (dataset-scoped, same as original build_mix):
  - id level : rsvqa_lr rows vs (lr_val ∪ lr_test) frozen ids -> 0
               rsvqa_hr rows vs (hr_val ∪ hr_test ∪ phili) frozen ids -> 0
  - image lvl: LR train img_ids vs LR eval img_ids -> 0 (LR splits partition
               images AND questions — verified: train imgs 0-231/332-471/
               572-771, val 472-571, test 232-331)
               HR replay img_ids vs HR eval img_ids -> logged (HR splits
               partition questions only; images repeat — legal per original)
               vrsbench caption imgs vs (vqa ∪ caption ∪ referring) eval imgs -> 0
"""

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # SatQuery/
OUT = Path(__file__).resolve().parent                  # eval_lr_fold/
GATES = ROOT / "gates" / "_cache"
LR_DIR = GATES / "data" / "rsvqa" / "lr"
EVAL_IDS = GATES / "eval_ids"
OLD_MIX = ROOT / "modal_volume_backup" / "rsvqa_adapt" / "mix.jsonl"

SEED = 42
N_HR_REPLAY = 20_000
N_CAP = 15_000
COUNT_OVERSAMPLE = 2.5      # multiplicity inside the LR share
W_CLIP = (0.25, 4.0)

COLUMNS = {
    "rsvqa_hr_val": "rsvqa_hr_val_eval_ids.json",
    "rsvqa_hr_test": "rsvqa_hr_test_eval_ids.json",
    "rsvqa_hr_test_phili": "rsvqa_hr_test_phili_eval_ids.json",
    "rsvqa_lr_val": "rsvqa_lr_val_eval_ids.json",
    "rsvqa_lr_test": "rsvqa_lr_test_eval_ids.json",
    "vrsbench_vqa": "vrsbench_vqa_eval_ids.json",
}


def _sha256(path: Path, bufsize: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def canon(s: str) -> str:
    """VERBATIM from modal_rsvqa_adapt.py — vocab build only."""
    import re
    s = re.sub(r"\s+", " ", s.strip().lower()).rstrip(".")
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
    nw = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
          "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
          "ten": "10", "eleven": "11", "twelve": "12"}
    if s in nw:
        s = nw[s]
    return s


def main() -> int:
    rng = random.Random(SEED)
    manifest = {"step": "build_mix_lr_fold", "seed": SEED,
                    "n_hr_replay_target": N_HR_REPLAY, "n_cap_target": N_CAP,
                    "count_oversample_lr": COUNT_OVERSAMPLE}

    # ---- frozen eval id sets -------------------------------------------------
    eval_ids = {}
    for col, fn in COLUMNS.items():
        d = json.loads((EVAL_IDS / fn).read_text())
        eval_ids[col] = set(d["ids"])
    manifest["eval_n"] = {c: len(s) for c, s in eval_ids.items()}
    manifest["eval_ids_file_sha256"] = {
        c: _sha256(EVAL_IDS / fn) for c, fn in COLUMNS.items()}

    # ---- LR share ------------------------------------------------------------
    qs = json.loads((LR_DIR / "LR_split_train_questions.json").read_text())["questions"]
    ans = json.loads((LR_DIR / "LR_split_train_answers.json").read_text())["answers"]
    ims = json.loads((LR_DIR / "LR_split_train_images.json").read_text())["images"]
    src_sha = {
        "rsvqa/lr/LR_split_train_questions.json": _sha256(LR_DIR / "LR_split_train_questions.json"),
        "rsvqa/lr/LR_split_train_answers.json": _sha256(LR_DIR / "LR_split_train_answers.json"),
        "rsvqa/lr/LR_split_train_images.json": _sha256(LR_DIR / "LR_split_train_images.json"),
    }
    amap = {a["id"]: a for a in ans}
    n_stub = n_noans = 0
    lr_rows = []
    for q in qs:
        if not q.get("active"):
            n_stub += 1
            continue
        a = amap.get(q["answers_ids"][0])
        if a is None or not a.get("active"):
            n_noans += 1
            continue
        lr_rows.append({
            "uid": f"rsvqa_lr_train:{q['img_id']}:{q['id']}",
            "eval_id": f"{q['img_id']}:{q['id']}",
            "source": "rsvqa_lr", "qtype": q["type"],
            "question": q["question"], "answer": str(a["answer"]),
            "image": f"rsvqa_lr/Images_LR/{q['img_id']}.tif",
        })
    manifest["lr"] = {
        "train_file_entries": len(qs), "inactive_stubs": n_stub,
        "active_joined": len(lr_rows), "active_no_answer": n_noans,
        "uniq_images": len({r["image"] for r in lr_rows}),
        "type_counts": dict(Counter(r["qtype"] for r in lr_rows)),
        "note": "inactive stubs' qids == lr_val+lr_test eval qids "
                "(verified disjoint from active) — they carry no "
                "image/question and are never trainable.",
    }

    # count x2.5 multiplicity inside the LR share (literal repeats, seeded)
    counts = [r for r in lr_rows if r["qtype"] == "count"]
    others = [r for r in lr_rows if r["qtype"] != "count"]
    target_count_rows = int(len(counts) * COUNT_OVERSAMPLE + 0.5)
    extra3 = target_count_rows - 2 * len(counts)   # entries beyond 2 copies
    assert 0 <= extra3 <= len(counts), f"oversample math off: {extra3}"
    extra_idx = list(range(len(counts)))
    rng.shuffle(extra_idx)
    extra_set = set(extra_idx[:extra3])
    lr_share = list(others)
    for i, r in enumerate(counts):
        lr_share.append(r)                                  # base copy
        lr_share.append({**r, "uid": r["uid"] + ":rep1"})   # x2
        if i in extra_set:
            lr_share.append({**r, "uid": r["uid"] + ":rep2"})  # x2.5 tail
    manifest["lr"]["effective_rows_after_oversample"] = len(lr_share)
    manifest["lr"]["count_effective"] = target_count_rows
    manifest["lr"]["count_uniq"] = len(counts)

    # ---- HR replay + captions, verbatim rows from old mix ---------------------
    old = [json.loads(l) for l in OLD_MIX.open()]
    manifest["old_mix_sha256"] = _sha256(OLD_MIX)
    hr_pool = [r for r in old if r["source"] == "rsvqa_hr"]
    cap_pool = [r for r in old if r["source"] == "vrsbench_cap"]
    assert len(hr_pool) >= N_HR_REPLAY and len(cap_pool) >= N_CAP
    rng.shuffle(hr_pool)
    rng.shuffle(cap_pool)
    hr_pick = []
    for r in hr_pool[:N_HR_REPLAY]:
        r = dict(r)
        assert r["image"].startswith("/data/rsvqa/hr/Data/")
        r["image"] = "rsvqa_hr/Data/" + r["image"].rsplit("/", 1)[-1]
        r.pop("weight", None)
        hr_pick.append(r)
    cap_pick = []
    for r in cap_pool[:N_CAP]:
        r = dict(r)
        assert r["image"].startswith("/data/vrsbench/Images_train/")
        r["image"] = "vrsbench/Images_train/" + r["image"].rsplit("/", 1)[-1]
        r.pop("weight", None)
        cap_pick.append(r)

    mix = lr_share + hr_pick + cap_pick

    # ---- OVERLAP GATE ---------------------------------------------------------
    lr_eval = eval_ids["rsvqa_lr_val"] | eval_ids["rsvqa_lr_test"]
    hr_eval = (eval_ids["rsvqa_hr_val"] | eval_ids["rsvqa_hr_test"]
               | eval_ids["rsvqa_hr_test_phili"])
    lr_id_overlap = sorted({r["eval_id"] for r in lr_share} & lr_eval)
    hr_id_overlap = sorted({r["eval_id"] for r in hr_pick} & hr_eval)

    lr_eval_imgs = {e.split(":")[0] for e in lr_eval}
    hr_eval_imgs = {e.split(":")[0] for e in hr_eval}
    lr_img_overlap = sorted({r["image"].rsplit("/", 1)[-1].split(".")[0]
                             for r in lr_share} & lr_eval_imgs)
    hr_img_overlap = sorted({r["image"].rsplit("/", 1)[-1].split(".")[0]
                             for r in hr_pick} & hr_eval_imgs)

    vrsb_extra = set()
    for extra_fn in ("vrsbench_caption_eval_ids.json",
                     "vrsbench_referring_eval_ids.json"):
        p = EVAL_IDS / extra_fn
        if p.exists():
            vrsb_extra |= set(json.loads(p.read_text())["ids"])
    vrsb_eval_imgs = {e.split("#")[0]
                      for e in (eval_ids["vrsbench_vqa"] | vrsb_extra)
                      if "#" in e}
    vrsb_img_overlap = sorted({r["image"].rsplit("/", 1)[-1]
                               for r in cap_pick} & vrsb_eval_imgs)

    manifest["overlap"] = {
        "rsvqa_lr_id_overlap_n": len(lr_id_overlap),
        "rsvqa_lr_id_overlap_sample": lr_id_overlap[:10],
        "rsvqa_lr_image_overlap_n": len(lr_img_overlap),
        "rsvqa_hr_id_overlap_n": len(hr_id_overlap),
        "rsvqa_hr_id_overlap_sample": hr_id_overlap[:10],
        "vrsb_image_overlap_n": len(vrsb_img_overlap),
        "vrsb_image_overlap_sample": vrsb_img_overlap[:10],
        "diag_hr_shared_image_ids_n": len(hr_img_overlap),
        "note": "id gate is dataset-scoped (LR rows vs LR eval, HR rows vs "
                "HR eval). LR image-level gate is literal — LR splits "
                "partition images as well as questions. HR image overlap is "
                "diagnostic only (splits partition questions; images repeat) "
                "— same semantics as the original build_mix.",
    }
    hard_fail = lr_id_overlap or lr_img_overlap or hr_id_overlap or vrsb_img_overlap
    if hard_fail:
        manifest["status"] = "STOP"
        manifest["reason"] = "train/eval overlap detected"
        (OUT / "mix_manifest_lr_fold.json").write_text(
            json.dumps(manifest, indent=2) + "\n")
        print("STOP — overlap:", manifest["overlap"])
        return 2

    # ---- class weights (same rule as original) --------------------------------
    freq = Counter(r["answer"] for r in mix if r["qtype"] != "caption")
    n_cls, n_w = len(freq), sum(freq.values())
    for r in mix:
        if r["qtype"] == "caption":
            r["weight"] = 1.0
        else:
            r["weight"] = min(max(n_w / (n_cls * freq[r["answer"]]), *W_CLIP[:1]), W_CLIP[1])
    wsum2 = sum(r["weight"] for r in mix if r["qtype"] != "caption")
    n2 = sum(1 for r in mix if r["qtype"] != "caption")
    for r in mix:
        if r["qtype"] != "caption":
            r["weight"] = round(r["weight"] * n2 / wsum2, 6)
    manifest["class_weight_rule"] = (
        "w = clip(N/(K*freq), 0.25, 4.0) renormalized to mean 1.0 over "
        "non-caption rows; caption rows fixed 1.0 — freq computed on the "
        "final mix incl. LR count multiplicity (same as original).")

    rng.shuffle(mix)
    mix_path = OUT / "mix_lr_fold.jsonl"
    with mix_path.open("w") as f:
        for r in mix:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    vocab = sorted({canon(r["answer"]) for r in mix if r["qtype"] != "caption"})
    (OUT / "vocab_lr_fold.json").write_text(json.dumps(
        {"rule": "canon() over lr_fold train answers only", "vocab": vocab},
        indent=2) + "\n")

    manifest["source_sha256"] = src_sha
    manifest["mix"] = {
        "total": len(mix),
        "rsvqa_lr_effective": len(lr_share),
        "rsvqa_lr_unique": len(lr_rows),
        "rsvqa_hr_replay": len(hr_pick),
        "vrsbench_cap": len(cap_pick),
        "hr_replay_by_type": dict(Counter(r["qtype"] for r in hr_pick)),
        "lr_effective_by_type": dict(Counter(r["qtype"] for r in lr_share)),
        "caption_frac": round(len(cap_pick) / len(mix), 4),
        "uniq_images": {
            "rsvqa_lr": len({r["image"] for r in lr_share}),
            "rsvqa_hr": len({r["image"] for r in hr_pick}),
            "vrsbench_cap": len({r["image"] for r in cap_pick}),
        },
    }
    manifest["mix_sha256"] = _sha256(mix_path)
    manifest["status"] = "OK"
    (OUT / "mix_manifest_lr_fold.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print("OK total:", len(mix), "| lr_eff:", len(lr_share),
          "| hr:", len(hr_pick), "| cap:", len(cap_pick))
    print("uniq images:", manifest["mix"]["uniq_images"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
