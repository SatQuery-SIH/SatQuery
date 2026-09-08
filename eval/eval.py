"""Reusable VQA eval harness (BASELINE-SPEC-02 + BASELINE-RUN-03 amendments).

Loads the cached VRSBench subset from fetch_data.py, calls a locally served
vision-language model, writes RAW predictions JSONL (no score-side rewriting).

CLI (MUST): --model --limit --split --out --seed --backend --url

Backends:
  ollama  POST {url}/api/chat          (Gate 1 path; default url http://localhost:11434)
  openai  POST {url}/v1/chat/completions  (llama.cpp llama-server; default http://127.0.0.1:8080)

Arm A:  python eval/eval.py --backend ollama --url http://localhost:11434 --model qwen2.5vl:7b --limit 300 --out gates/_cache/baseline/preds_qwen25vl.jsonl
Arm B:  python eval/eval.py --backend openai --url http://127.0.0.1:8080 --model qwen3vl --limit 300 --out gates/_cache/baseline/preds_qwen3vl.jsonl

Optional local-judge (GPT-protocol proxy, local model only):
  python eval/eval.py --judge-existing gates/_cache/baseline/preds_qwen25vl.jsonl --backend ollama --model qwen2.5vl:7b --url http://localhost:11434
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # SatQuery/
SUBSET_PATH = ROOT / "gates" / "_cache" / "vrsbench" / "subset.json"
DEFAULT_SEED = 42

JUDGE_PROMPT = (
    "Question: {question}\n"
    "Ground Truth Answer: {gold}\n"
    "Predicted Answer: {pred}\n"
    "Does the predicted answer match the ground truth? Answer 1 for match and 0 "
    "for not match. Use semantic meaning not exact match. Synonyms are also "
    "treated as a match, e.g., football and soccer, playground and ground track "
    "field, building and rooftop, pond and swimming pool. Do not explain the reason.\n"
)


def _mime(path: Path) -> str:
    guess, _ = mimetypes.guess_type(str(path))
    if guess:
        return guess
    ext = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")


def encode_image(path: Path) -> tuple[str, str]:
    return base64.b64encode(path.read_bytes()).decode("ascii"), _mime(path)


def ask_ollama(
    url: str,
    model: str,
    prompt: str,
    img_b64: str | None,
    mime: str,
    timeout: float,
    max_tokens: int,
) -> str:
    content = prompt
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0},
    }
    if img_b64 is not None:
        payload["messages"][0]["images"] = [img_b64]
    req = urllib.request.Request(
        url.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return (body.get("message") or {}).get("content") or ""


def ask_openai(
    url: str,
    model: str,
    prompt: str,
    img_b64: str | None,
    mime: str,
    timeout: float,
    max_tokens: int,
) -> str:
    parts: list[dict] = [{"type": "text", "text": prompt}]
    if img_b64 is not None:
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{img_b64}"},
            }
        )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": parts}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"].get("content") or ""


def load_subset(split: str, seed: int) -> list[dict]:
    if split != "test":
        raise SystemExit(
            f"Only split='test' is supported (VRSBench_EVAL_vqa.json). Got {split!r}. "
            "Do not evaluate on train."
        )
    if not SUBSET_PATH.is_file():
        raise SystemExit(f"Subset not found: {SUBSET_PATH}. Run python eval/fetch_data.py first.")
    rows = json.loads(SUBSET_PATH.read_text(encoding="utf-8"))
    meta_path = SUBSET_PATH.parent / "subset_ids.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(meta.get("seed", seed)) != seed:
            print(
                f"WARNING: subset was drawn with seed={meta.get('seed')}, "
                f"eval --seed={seed}. Using the logged subset as-is "
                f"(ids stay quarantined).",
                file=sys.stderr,
            )
    return rows


def load_done_ids(out_path: Path) -> dict[str, dict]:
    done = {}
    if not out_path.is_file():
        return done
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            eid = rec.get("example_id") or f"{rec.get('image_id')}::{rec.get('question_id')}"
            if rec.get("error"):
                continue
            pred = rec.get("prediction")
            if pred is None:
                continue
            if isinstance(pred, str) and pred.startswith("ERROR"):
                continue
            done[eid] = rec
    return done


def append_jsonl(out_path: Path, rec: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def call_with_retry(ask, **kwargs) -> tuple[str, float, str | None]:
    last_err = None
    t0 = time.perf_counter()
    for attempt in range(2):
        try:
            text = ask(**kwargs)
            elapsed = time.perf_counter() - t0
            return (text if text is not None else ""), elapsed, None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, OSError) as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(1.0)
    elapsed = time.perf_counter() - t0
    return "", elapsed, last_err


def run_eval(args: argparse.Namespace) -> int:
    rows = load_subset(args.split, args.seed)
    n = min(args.limit, len(rows)) if args.limit else len(rows)
    rows = rows[:n]
    out_path = Path(args.out)
    done = load_done_ids(out_path)
    print(
        f"backend={args.backend} model={args.model} url={args.url} "
        f"n={len(rows)} already_done={len(done)} out={out_path}"
    )

    ask = ask_ollama if args.backend == "ollama" else ask_openai
    first_img = Path(rows[0]["image_path"])
    print("Warm-up (excluded from per-example latency)...")
    b64, mime = encode_image(first_img)
    t0 = time.perf_counter()
    _, _, err = call_with_retry(
        ask,
        url=args.url,
        model=args.model,
        prompt="Reply with the single word: ready.",
        img_b64=b64,
        mime=mime,
        timeout=args.timeout,
        max_tokens=16,
    )
    warmup = time.perf_counter() - t0
    if err:
        print(f"WARM-UP FAILED: {err}", file=sys.stderr)
        return 2
    print(f"warm-up {warmup:.1f}s")

    latencies = []
    for i, rec in enumerate(rows, 1):
        eid = rec["example_id"]
        if eid in done:
            latencies.append(float(done[eid].get("latency_s") or 0.0))
            print(f"[{i}/{n}] SKIP {eid}")
            continue
        img_path = Path(rec["image_path"])
        if not img_path.is_file():
            out_rec = {
                **{k: rec[k] for k in (
                    "example_id", "question_id", "image_id", "question", "gold",
                    "category", "type_raw",
                )},
                "prediction": "",
                "latency_s": 0.0,
                "error": f"missing image: {img_path}",
                "model": args.model,
                "backend": args.backend,
            }
            append_jsonl(out_path, out_rec)
            print(f"[{i}/{n}] MISSING IMAGE {img_path}")
            continue
        b64, mime = encode_image(img_path)
        text, elapsed, err = call_with_retry(
            ask,
            url=args.url,
            model=args.model,
            prompt=rec["question"],
            img_b64=b64,
            mime=mime,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
        )
        latencies.append(elapsed)
        out_rec = {
            "example_id": rec["example_id"],
            "question_id": rec["question_id"],
            "image_id": rec["image_id"],
            "question": rec["question"],
            "gold": rec["gold"],
            "prediction": text if not err else f"ERROR: {err}",
            "category": rec["category"],
            "type_raw": rec["type_raw"],
            "latency_s": round(elapsed, 4),
            "error": err,
            "model": args.model,
            "backend": args.backend,
        }
        append_jsonl(out_path, out_rec)
        preview = (text if not err else f"ERROR {err}")[:80].replace("\n", " ")
        print(f"[{i}/{n}] {elapsed:6.2f}s  {rec['category']:10s}  {preview}", flush=True)

    if latencies:
        sl = sorted(latencies)
        mean = sum(sl) / len(sl)
        p95 = sl[min(len(sl) - 1, int(round(0.95 * (len(sl) - 1))))]
        print(f"latency n={len(sl)} mean={mean:.2f}s p95={p95:.2f}s warmup={warmup:.1f}s")
    print(f"raw predictions → {out_path}")
    return 0


def parse_judge_bit(text: str) -> int | None:
    s = (text or "").strip()
    if not s:
        return None
    # Official notebook stores '1' / '0'; take the first isolated bit.
    for ch in s:
        if ch == "1":
            return 1
        if ch == "0":
            return 0
    low = s.lower()
    if "match" in low and "not match" not in low and "no match" not in low:
        return 1
    return None


def official_prefilter(gold: str, pred: str) -> int | None:
    """Reproduce eval_vqa_gpt.ipynb's pre-GPT shortcuts (local, no API)."""
    g = (gold or "").strip().lower()
    p = (pred or "").strip().lower()
    if not g:
        return 0
    if g in p:
        return 1
    closed = {"yes", "no"} | {str(i) for i in range(100)}
    if g in closed:
        return 1 if g == p else 0
    return None


def run_judge(args: argparse.Namespace) -> int:
    src = Path(args.judge_existing)
    rows = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    dest = Path(args.out) if args.out else src.with_name(src.stem + ".judged.jsonl")
    if dest.exists():
        dest.unlink()
    ask = ask_ollama if args.backend == "ollama" else ask_openai
    print(f"local-judge n={len(rows)} backend={args.backend} model={args.model} → {dest}")
    n_skip = n_model = n_fail = 0
    for i, rec in enumerate(rows, 1):
        gold = rec.get("gold", "")
        pred = rec.get("prediction", "")
        bit = official_prefilter(gold, pred)
        source = "prefilter"
        raw_judge = None
        elapsed = 0.0
        if bit is None:
            prompt = JUDGE_PROMPT.format(
                question=rec.get("question", ""), gold=gold, pred=pred
            )
            text, elapsed, err = call_with_retry(
                ask,
                url=args.url,
                model=args.model,
                prompt=prompt,
                img_b64=None,
                mime="image/jpeg",
                timeout=args.timeout,
                max_tokens=8,
            )
            raw_judge = text if not err else f"ERROR: {err}"
            bit = parse_judge_bit(text if not err else "")
            source = "local_model"
            n_model += 1
            if bit is None:
                n_fail += 1
        else:
            n_skip += 1
        rec = dict(rec)
        rec["local_judge"] = bit
        rec["local_judge_source"] = source
        rec["local_judge_raw"] = raw_judge
        rec["local_judge_latency_s"] = round(elapsed, 4)
        rec["local_judge_model"] = args.model
        append_jsonl(dest, rec)
        if i % 25 == 0 or i == len(rows):
            print(f"judge [{i}/{len(rows)}] prefilter={n_skip} model={n_model} unparsed={n_fail}", flush=True)
    print(f"wrote {dest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SatQuery VQA eval harness")
    ap.add_argument("--model", default="qwen2.5vl:7b")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=str(ROOT / "gates" / "_cache" / "baseline" / "preds.jsonl"))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    ap.add_argument("--url", default=None)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument(
        "--judge-existing",
        default=None,
        help="score an existing preds JSONL with the local-judge proxy (text-only)",
    )
    args = ap.parse_args(argv)
    if args.url is None:
        args.url = (
            "http://localhost:11434" if args.backend == "ollama" else "http://127.0.0.1:8080"
        )
    if args.judge_existing:
        return run_judge(args)
    return run_eval(args)


if __name__ == "__main__":
    sys.exit(main())
