"""The only program that runs on dropped eval data.

Uploads shape (read-only from demo/ingest.py bind_inputs):
  single      -> input_mode="single",      uploads={"image": path}
  change      -> input_mode="bi-temporal", uploads={"before": path, "after": path}
  sar         -> input_mode="optical+sar", uploads={"optical": path, "sar": path}
               bind_inputs also accepts sar_vv / sar_npz aliases for the SAR file.

SAR input_mode string (read-only from demo/planner.py INPUT_MODES): "optical+sar".
Every row calls demo.pipeline.run_query — the same entry point as the GUI.
There is no second inference path.

Frozen decode (recorded in MANIFEST; client is the demo stack):
  temperature 0.0, top_p 1.0, short-answer wait timeout 180s per row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

KIT_DIR = Path(__file__).resolve().parent
ROOT = KIT_DIR.parent
DEMO = ROOT / "demo"
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from judge_kit.schema import KIT_MODE_TO_INPUT_MODE, validate_row  # noqa: E402

FROZEN_GGUF_SHA = "67d1659bfe71b89d50b45a4ad1a9e5b997e5bb16ce5da66a6a6167abd569e9e2"
FROZEN_MMPROJ_SHA = "ca524100ebf825c9a870db1c580d03879e0da0ab2541697e2458e64891cf9d38"
QWEN_DIR = ROOT / "gates" / "qwen3vl"
GGUF_PATH = QWEN_DIR / "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
MMPROJ_PATH = QWEN_DIR / "mmproj-Qwen3VL-8B-Instruct-F16.gguf"
PROTOCOL_PATH = ROOT / "gates" / "_cache" / "scoreboard" / "protocol.json"
ROW_TIMEOUT_S = 180.0
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 16
_ANSWER_PREFIX = re.compile(
    r"^(?:answer\s*(?:is|=|:)\s*|the answer is\s+)",
    flags=re.I,
)
_PUNCT_TRAIL = re.compile(r"[\s\.,;:!\?\"'`]+$")
_WS = re.compile(r"\s+")


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen_weights(gguf: Path = GGUF_PATH, mmproj: Path = MMPROJ_PATH) -> dict[str, str]:
    """SHA-check frozen GGUF + mmproj BEFORE any launch. Mismatch raises."""
    if not gguf.is_file():
        raise FileNotFoundError(f"missing GGUF: {gguf}")
    if not mmproj.is_file():
        raise FileNotFoundError(f"missing mmproj: {mmproj}")
    g = sha256_file(gguf)
    m = sha256_file(mmproj)
    if g != FROZEN_GGUF_SHA:
        raise RuntimeError(f"STOP: GGUF sha {g} != {FROZEN_GGUF_SHA}")
    if m != FROZEN_MMPROJ_SHA:
        raise RuntimeError(f"STOP: mmproj sha {m} != {FROZEN_MMPROJ_SHA}")
    return {"gguf_sha256": g, "mmproj_sha256": m, "gguf": str(gguf), "mmproj": str(mmproj)}


def port_health(url: str, timeout: float = 3.0) -> bool:
    health = url.rstrip("/") + "/health"
    try:
        req = urllib.request.Request(health, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= int(resp.status) < 300
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


def port_open(port: int) -> bool:
    s = socket.socket()
    s.settimeout(0.3)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def load_synonym_table(path: Path = PROTOCOL_PATH) -> dict[str, str]:
    """Reuse the frozen scoreboard synonym table verbatim. Never retune."""
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    table = rec.get("synonym_table")
    if not isinstance(table, dict) or not table:
        raise RuntimeError(f"STOP: synonym_table missing in {path}")
    return {str(k): str(v) for k, v in table.items()}


def normalize_answer(text: str | None, table: dict[str, str]) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    s = s.splitlines()[0].strip()
    s = _ANSWER_PREFIX.sub("", s).strip()
    s = s.lower()
    s = _WS.sub(" ", s)
    s = _PUNCT_TRAIL.sub("", s)
    s = _WS.sub(" ", s).strip()
    return table.get(s, s)


def exact_match(pred: str | None, gold: str | None, table: dict[str, str]) -> bool:
    g = normalize_answer(gold, table)
    p = normalize_answer(pred, table)
    return bool(g) and p == g


def resolve_path(raw: str, images_dir: Path | None) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    if images_dir is not None:
        return images_dir / raw
    return p


def _uploads_for(row: dict[str, Any], images_dir: Path | None) -> tuple[dict[str, Path] | None, str | None]:
    mode = row["mode"]
    if mode == "single":
        p = resolve_path(row["image"], images_dir)
        if not p.is_file():
            return None, f"missing image: {p}"
        return {"image": p}, None
    a = resolve_path(row["image_a"], images_dir)
    b = resolve_path(row["image_b"], images_dir)
    missing = []
    if not a.is_file():
        missing.append(f"image_a={a}")
    if not b.is_file():
        missing.append(f"image_b={b}")
    if missing:
        return None, "missing image: " + ", ".join(missing)
    if mode == "change":
        return {"before": a, "after": b}, None
    return {"optical": a, "sar": b}, None


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        skip = {"mask", "water_mask", "cls_a", "cls_b", "direction"}
        return {k: _jsonable(v) for k, v in obj.items() if k not in skip}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _pick_answer(trace: dict[str, Any]) -> tuple[str, str]:
    """canonical from packet when present; keep prediction_raw beside it."""
    raw = trace.get("answer")
    if raw is None:
        raw = trace.get("visible_answer") or ""
    raw_s = str(raw) if raw is not None else ""
    pkt = trace.get("evidence_packet") or {}
    canon = pkt.get("canonical_answer") if isinstance(pkt, dict) else None
    if canon is not None and str(canon).strip() != "":
        return str(canon), raw_s
    return raw_s, raw_s


def load_questions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            rows.append({"_raw_error": f"schema: line {i} is not JSON ({e})", "_raw_obj": None})
            continue
        rows.append(obj)
    return rows


def load_done_ids(preds_path: Path) -> set[str]:
    done: set[str] = set()
    if not preds_path.is_file():
        return done
    for line in preds_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = obj.get("id")
        if rid:
            done.add(str(rid))
    return done


def _default_run_query(query: str, input_mode: str, uploads: dict, vlm_url: str, live: bool) -> dict[str, Any]:
    from pipeline import run_query

    return run_query(
        query,
        input_mode,
        scene=None,
        live=live,
        vlm_url=vlm_url,
        device_cd="cpu",
        uploads=uploads,
    )


def _blank_rec(rid: str, mode: str, error: str, t0: float) -> dict[str, Any]:
    return {
        "id": rid,
        "mode": mode,
        "answer": "",
        "prediction_raw": "",
        "latency_s": round(time.perf_counter() - t0, 3),
        "error": error,
    }


def eval_one(
    obj: Any,
    *,
    images_dir: Path | None,
    vlm_url: str,
    live: bool,
    run_query_fn: Callable[..., dict[str, Any]],
    seen_ids: set[str],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    t0 = time.perf_counter()
    if isinstance(obj, dict) and obj.get("_raw_error"):
        rid = str(obj.get("id") or f"invalid_{len(seen_ids)}")
        return _blank_rec(rid, "", str(obj["_raw_error"]), t0), None
    row, err = validate_row(obj, seen_ids=seen_ids)
    rid = ""
    mode = ""
    if isinstance(obj, dict):
        rid = str(obj.get("id") or "")
        mode = str(obj.get("mode") or "")
    if err or row is None:
        return _blank_rec(rid or f"invalid_{len(seen_ids)}", mode, err or "schema: invalid row", t0), None
    uploads, uerr = _uploads_for(row, images_dir)
    if uerr or uploads is None:
        return _blank_rec(row["id"], row["mode"], uerr or "missing image", t0), None
    input_mode = KIT_MODE_TO_INPUT_MODE[row["mode"]]
    try:
        trace = run_query_fn(
            row["question"],
            input_mode,
            uploads,
            vlm_url,
            live,
        )
    except Exception as e:
        return _blank_rec(row["id"], row["mode"], f"{type(e).__name__}: {e}", t0), None
    if not isinstance(trace, dict):
        return _blank_rec(row["id"], row["mode"], "run_query returned non-dict", t0), None
    plan = trace.get("plan") or {}
    tout = trace.get("tool_outputs") or {}
    answer, raw = _pick_answer(trace)
    error = None
    vlm = trace.get("vlm") or {}
    # Bind failure: supported plan, no tools, no VLM call. Live VQA stores text on
    # trace["vlm"] / trace["answer"] with empty tool_outputs — that is success.
    if plan.get("supported") and not tout and not vlm:
        msg = str(trace.get("answer") or "").strip()
        if msg:
            error = msg
            answer, raw = "", ""
    elif not plan.get("supported"):
        if not answer:
            answer = str(plan.get("refusal") or trace.get("answer") or "")
            raw = answer
    rec = {
        "id": row["id"],
        "mode": row["mode"],
        "answer": answer,
        "prediction_raw": raw,
        "latency_s": round(time.perf_counter() - t0, 3),
        "error": error,
    }
    return rec, trace


def write_manifest(
    path: Path,
    *,
    weights: dict[str, str],
    server: str,
    started_here: bool,
    n: int,
    n_fail: int,
    n_skip: int,
    wall_s: float,
    mode_counts: dict[str, int],
    vlm_url: str,
    questions_path: Path,
    preds_path: Path,
) -> None:
    lines = [
        f"utc={utcnow()}",
        f"gguf_sha256={weights.get('gguf_sha256', FROZEN_GGUF_SHA)}",
        f"mmproj_sha256={weights.get('mmproj_sha256', FROZEN_MMPROJ_SHA)}",
        f"server={server}",
        f"started_here={str(started_here).lower()}",
        f"vlm_url={vlm_url}",
        f"n={n}",
        f"n_fail={n_fail}",
        f"n_resume_skip={n_skip}",
        f"wall_s={wall_s:.3f}",
        f"mode_single={mode_counts.get('single', 0)}",
        f"mode_change={mode_counts.get('change', 0)}",
        f"mode_sar={mode_counts.get('sar', 0)}",
        f"temperature={TEMPERATURE}",
        f"top_p={TOP_P}",
        f"max_tokens={MAX_TOKENS}",
        f"timeout_s={ROW_TIMEOUT_S}",
        f"questions={questions_path}",
        f"preds={preds_path}",
        "usd=0.00",
        "entry=demo.pipeline.run_query",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def score_if_gold(preds_path: Path, gold_path: Path, out_path: Path) -> dict[str, Any]:
    table = load_synonym_table()
    gold_by: dict[str, str] = {}
    for line in gold_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        rid = str(obj.get("id") or "")
        g = obj.get("gold", obj.get("answer", obj.get("ground_truth")))
        if rid:
            gold_by[rid] = "" if g is None else str(g)
    n = 0
    n_exact = 0
    n_fail = 0
    for line in preds_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        pred = json.loads(line)
        rid = str(pred.get("id") or "")
        if rid not in gold_by:
            continue
        n += 1
        if pred.get("error"):
            n_fail += 1
            continue
        if exact_match(pred.get("answer"), gold_by[rid], table):
            n_exact += 1
    rec = {
        "n": n,
        "n_exact": n_exact,
        "n_fail_counted_wrong": n_fail,
        "accuracy": (n_exact / n) if n else None,
        "synonym_table_sha256": hashlib.sha256(
            json.dumps(table, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "metric": "exact-match",
        "gold": str(gold_path),
        "preds": str(preds_path),
        "utc": utcnow(),
    }
    out_path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    return rec


def evaluate(
    questions_path: Path,
    out_dir: Path,
    *,
    images_dir: Path | None = None,
    vlm_url: str = "http://127.0.0.1:8080",
    live: bool = True,
    resume: bool = True,
    with_traces: bool = False,
    server: str = "unknown",
    started_here: bool = False,
    score_gold_path: Path | None = None,
    run_query_fn: Callable[..., dict[str, Any]] | None = None,
    verify_weights_on_disk: bool = False,
) -> dict[str, Any]:
    questions_path = Path(questions_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "preds.jsonl"
    manifest_path = out_dir / "MANIFEST.txt"
    traces_dir = out_dir / "traces"
    if images_dir is None:
        guess = questions_path.parent / "images"
        images_dir = guess if guess.is_dir() else questions_path.parent
    else:
        images_dir = Path(images_dir)

    runner = run_query_fn or _default_run_query
    weights = {
        "gguf_sha256": FROZEN_GGUF_SHA,
        "mmproj_sha256": FROZEN_MMPROJ_SHA,
        "gguf": str(GGUF_PATH),
        "mmproj": str(MMPROJ_PATH),
    }
    if verify_weights_on_disk:
        weights = verify_frozen_weights()

    raw_rows = load_questions(questions_path)
    done = load_done_ids(preds_path) if resume else set()
    n_skip = 0
    n_fail = 0
    mode_counts: Counter[str] = Counter()
    seen_ids: set[str] = set(done)
    wall0 = time.perf_counter()
    preds_existed = preds_path.is_file()
    wrote_any = False

    fh = None
    try:
        for obj in raw_rows:
            rid_hint = ""
            if isinstance(obj, dict) and obj.get("id"):
                rid_hint = str(obj["id"])
            if rid_hint and rid_hint in done:
                n_skip += 1
                mode_counts[str(obj.get("mode") or "")] += 1
                continue
            rec, trace = eval_one(
                obj,
                images_dir=images_dir,
                vlm_url=vlm_url,
                live=live,
                run_query_fn=runner,
                seen_ids=seen_ids,
            )
            mode_counts[str(rec.get("mode") or "")] += 1
            if rec.get("error"):
                n_fail += 1
            if fh is None:
                fh = preds_path.open("a", encoding="utf-8")
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            fh.flush()
            wrote_any = True
            done.add(str(rec.get("id") or ""))
            if with_traces and trace is not None:
                traces_dir.mkdir(parents=True, exist_ok=True)
                safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(rec.get("id") or "row"))
                (traces_dir / f"{safe}.json").write_text(
                    json.dumps(_jsonable(trace), indent=2, default=str) + "\n",
                    encoding="utf-8",
                )
    finally:
        if fh is not None:
            fh.close()

    n = len(raw_rows)
    wall_s = time.perf_counter() - wall0
    write_manifest(
        manifest_path,
        weights=weights,
        server=server,
        started_here=started_here,
        n=n,
        n_fail=n_fail,
        n_skip=n_skip,
        wall_s=wall_s,
        mode_counts=dict(mode_counts),
        vlm_url=vlm_url,
        questions_path=questions_path,
        preds_path=preds_path,
    )
    scores = None
    if score_gold_path is not None:
        scores = score_if_gold(preds_path, Path(score_gold_path), out_dir / "scores.json")
    return {
        "n": n,
        "n_fail": n_fail,
        "n_resume_skip": n_skip,
        "preds": str(preds_path),
        "manifest": str(manifest_path),
        "wrote_any": wrote_any,
        "preds_existed": preds_existed,
        "scores": scores,
        "wall_s": wall_s,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run frozen-schema questions through demo.pipeline.run_query")
    p.add_argument("--questions", required=True, help="questions.jsonl")
    p.add_argument("--out", required=True, help="output directory (preds.jsonl + MANIFEST.txt)")
    p.add_argument("--images", default="", help="images directory (default: <questions>/images)")
    p.add_argument("--vlm-url", default="http://127.0.0.1:8080")
    p.add_argument("--server", default="unknown")
    p.add_argument("--started-here", action="store_true")
    p.add_argument("--live", action="store_true", default=True)
    p.add_argument("--no-live", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--with-traces", action="store_true")
    p.add_argument("--score-if-gold", default="", help="optional gold jsonl; exact-match only")
    p.add_argument("--verify-weights", action="store_true")
    args = p.parse_args(argv)
    live = not args.no_live
    rec = evaluate(
        Path(args.questions),
        Path(args.out),
        images_dir=Path(args.images) if args.images else None,
        vlm_url=args.vlm_url,
        live=live,
        resume=not args.no_resume,
        with_traces=args.with_traces,
        server=args.server,
        started_here=args.started_here,
        score_gold_path=Path(args.score_if_gold) if args.score_if_gold else None,
        verify_weights_on_disk=args.verify_weights,
    )
    print(json.dumps({k: rec[k] for k in ("n", "n_fail", "n_resume_skip", "preds", "wall_s")}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
