"""LOOKING-08-12: n=20 original/shuffle/blank identity probe on Run 08. No train.

Serves adapted08 on :8091 only. Never binds :8080. Never edits demo/serve.ps1.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent  # SatQuery/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval import ask_openai, call_with_retry, encode_image  # noqa: E402
from score import normalize  # noqa: E402

SEED = 42
CATS = ("Presence", "Quantity", "Direction", "Color")
N_PER = 5
PORT = 8091
URL = f"http://127.0.0.1:{PORT}"
MODEL = "qwen3vl-adapted08"
MAX_TOKENS = 128
TIMEOUT = 180.0

QWEN = ROOT / "gates" / "qwen3vl"
SERVER = QWEN / "llama_cpp" / "llama-server.exe"
Q4 = QWEN / "preflight" / "Qwen3VL-8B-Instruct-adapted08-Q4_K_M.gguf"
MMPROJ = QWEN / "mmproj-Qwen3VL-8B-Instruct-adapted08-F16.gguf"
EVAL_IDS = ROOT / "gates" / "baseline_eval_ids.json"
SUBSET = ROOT / "gates" / "_cache" / "vrsbench" / "subset.json"
ORIG_300 = ROOT / "gates" / "_cache" / "baseline" / "preds_adapted08.jsonl"
OUT = ROOT / "gates" / "_cache" / "looking08"
SLICE_PATH = OUT / "slice.json"
BLANK_PATH = OUT / "blank_gray.png"
IDENT_PATH = OUT / "identity.json"
REPORT_PATH = OUT / "looking08_report.md"
PRED_ORIG = OUT / "preds_original.jsonl"
PRED_SHUF = OUT / "preds_shuffle.jsonl"
PRED_BLANK = OUT / "preds_blank.jsonl"


def _port_open(port: int) -> bool:
    s = socket.socket()
    s.settimeout(0.3)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def gpu_used_mib() -> int | None:
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if p.returncode != 0:
        return None
    line = (p.stdout or "").strip().splitlines()
    if not line:
        return None
    try:
        return int(float(line[0].strip()))
    except ValueError:
        return None


def derangement(n: int, seed: int) -> list[int]:
    rng = np.random.RandomState(seed)
    while True:
        perm = rng.permutation(n)
        if not any(int(perm[i]) == i for i in range(n)):
            return [int(x) for x in perm]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_frozen_slice() -> list[dict]:
    """LOOKING-08-12B: never resample. The 20 ids in slice.json are the contract."""
    if not SLICE_PATH.is_file():
        raise FileNotFoundError(f"STOP: frozen slice missing {SLICE_PATH}")
    rows = json.loads(SLICE_PATH.read_text(encoding="utf-8"))
    if len(rows) != 20:
        raise RuntimeError(f"STOP: frozen slice n={len(rows)} expected 20")
    cats = Counter(str(r.get("category")) for r in rows)
    if any(cats.get(c, 0) != N_PER for c in CATS):
        raise RuntimeError(f"STOP: frozen slice cats={dict(cats)} expected 5 each of {CATS}")
    for r in rows:
        img = Path(r["image_path"])
        if not img.is_file():
            raise FileNotFoundError(f"STOP: missing image {img}")
    return rows


def load_frozen_perm(n: int) -> list[int]:
    if not IDENT_PATH.is_file():
        raise FileNotFoundError(f"STOP: identity.json missing (need frozen derangement) {IDENT_PATH}")
    ident = json.loads(IDENT_PATH.read_text(encoding="utf-8"))
    perm = ident.get("derangement")
    if not isinstance(perm, list) or len(perm) != n:
        raise RuntimeError(f"STOP: frozen derangement missing/len={perm!r}")
    perm_i = [int(x) for x in perm]
    if any(perm_i[i] == i for i in range(n)):
        raise RuntimeError("STOP: frozen derangement has a fixed point")
    if sorted(perm_i) != list(range(n)):
        raise RuntimeError("STOP: frozen derangement is not a permutation of 0..n-1")
    return perm_i


def load_frozen_original(slice_rows: list[dict]) -> tuple[list[dict], list[str]]:
    by_eid = {}
    for rec in load_jsonl(PRED_ORIG):
        eid = rec.get("example_id")
        if eid:
            by_eid[str(eid)] = rec
    out = []
    missing = []
    for r in slice_rows:
        hit = by_eid.get(r["example_id"])
        pred = hit.get("prediction") if hit else None
        if hit is None or not pred or str(pred).startswith("ERROR"):
            missing.append(r["example_id"])
            continue
        out.append(
            {
                **r,
                "prediction": pred,
                "latency_s": hit.get("latency_s"),
                "error": None,
                "condition": "original",
                "source": hit.get("source") or str(PRED_ORIG),
                "model": hit.get("model"),
            }
        )
    return out, missing


def build_slice() -> list[dict]:
    meta = json.loads(EVAL_IDS.read_text(encoding="utf-8"))
    ids = set(str(x) for x in meta["example_ids"])
    if len(ids) != 300:
        raise RuntimeError(f"STOP: baseline_eval_ids n={len(ids)} expected 300")
    subset = json.loads(SUBSET.read_text(encoding="utf-8"))
    by_cat: dict[str, list[dict]] = {c: [] for c in CATS}
    for row in subset:
        if str(row["example_id"]) not in ids:
            continue
        cat = str(row.get("category") or "")
        if cat in by_cat:
            by_cat[cat].append(row)
    rng = np.random.RandomState(SEED)
    picked: list[dict] = []
    for cat in CATS:
        pool = by_cat[cat]
        if len(pool) < N_PER:
            raise RuntimeError(f"STOP: {cat} pool={len(pool)} < {N_PER}")
        order = rng.choice(len(pool), size=N_PER, replace=False)
        for i in order:
            r = pool[int(i)]
            img = Path(r["image_path"])
            if not img.is_file():
                raise FileNotFoundError(f"STOP: missing image {img}")
            picked.append(
                {
                    "example_id": r["example_id"],
                    "image_id": r["image_id"],
                    "category": r["category"],
                    "question": r["question"],
                    "gold": r["gold"],
                    "image_path": str(img),
                    "question_id": r.get("question_id"),
                    "type_raw": r.get("type_raw"),
                }
            )
    return picked


def load_orig_300() -> dict[str, dict]:
    out = {}
    if not ORIG_300.is_file():
        return out
    with ORIG_300.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            eid = rec.get("example_id")
            if not eid:
                continue
            if rec.get("error"):
                continue
            pred = rec.get("prediction")
            if pred is None or (isinstance(pred, str) and pred.startswith("ERROR")):
                continue
            out[str(eid)] = rec
    return out


def write_blank() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (512, 512), (128, 128, 128)).save(BLANK_PATH)


def infer_one(question: str, image_path: Path) -> dict:
    b64, mime = encode_image(image_path)
    text, elapsed, err = call_with_retry(
        ask_openai,
        url=URL,
        model=MODEL,
        prompt=question,
        img_b64=b64,
        mime=mime,
        timeout=TIMEOUT,
        max_tokens=MAX_TOKENS,
    )
    return {
        "prediction": text if not err else f"ERROR: {err}",
        "latency_s": round(elapsed, 4),
        "error": err,
    }


def start_8091() -> subprocess.Popen:
    if _port_open(8080):
        used = gpu_used_mib()
        if used is not None and used >= 2000:
            raise RuntimeError(
                f"STOP: port 8080 is listening and GPU memory.used={used} MiB. "
                "LOOKING-08-12 must not bind or kill :8080. Cannot load adapted08 on :8091 "
                "on this 8 GB card while the demo llama-server holds VRAM."
            )
    if _port_open(PORT):
        raise RuntimeError(f"STOP: port {PORT} already in use (refusing to reuse an unknown brain)")
    if not Q4.is_file():
        raise FileNotFoundError(f"STOP: 08 GGUF missing {Q4}")
    if not MMPROJ.is_file():
        raise FileNotFoundError(f"STOP: 08 mmproj missing {MMPROJ}")
    if not SERVER.is_file():
        raise FileNotFoundError(f"STOP: llama-server missing {SERVER}")
    err = OUT / "looking08_server_err.log"
    out = OUT / "looking08_server.log"
    args = [
        str(SERVER),
        "-m", str(Q4),
        "--mmproj", str(MMPROJ),
        "-ngl", "99",
        "-c", "4096",
        "--port", str(PORT),
        "--host", "127.0.0.1",
    ]
    print(">", " ".join(args), flush=True)
    with err.open("w", encoding="utf-8") as fe, out.open("w", encoding="utf-8") as fo:
        proc = subprocess.Popen(args, stdout=fo, stderr=fe, cwd=str(QWEN / "llama_cpp"))
    try:
        healthy = False
        for _ in range(80):
            time.sleep(3)
            if proc.poll() is not None:
                tail = err.read_text(encoding="utf-8", errors="replace")[-3000:]
                raise RuntimeError(f"STOP: llama-server :8091 exited {proc.returncode}\n{tail}")
            try:
                urllib.request.urlopen(f"{URL}/health", timeout=3)
                healthy = True
                break
            except (urllib.error.URLError, TimeoutError, OSError):
                continue
        if not healthy:
            proc.terminate()
            raise RuntimeError("STOP: llama-server :8091 never healthy")
    except Exception:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        raise
    return proc


def stop_8091(proc: subprocess.Popen | None) -> dict:
    if proc is None:
        return {"stopped": False, "reason": "never_started", "port_open": _port_open(PORT)}
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    # Confirm nothing we started is still bound.
    leftover = _port_open(PORT)
    if leftover:
        # Only kill listeners on 8091, never 8080.
        try:
            p = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            pids = set()
            for line in (p.stdout or "").splitlines():
                if ":8091" in line and "LISTENING" in line:
                    pids.add(line.split()[-1])
            for pid in pids:
                subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True, timeout=15)
        except Exception as e:
            return {"stopped": False, "error": f"{type(e).__name__}: {e}", "port_open": _port_open(PORT)}
    return {"stopped": True, "pid": proc.pid, "returncode": proc.returncode, "port_open": _port_open(PORT)}


def presence_yes_rate(rows: list[dict]) -> dict:
    pres = [r for r in rows if r["category"] == "Presence"]
    n_yes = 0
    preds = []
    for r in pres:
        p = r.get("prediction") or ""
        preds.append(p)
        if normalize(p) == "yes":
            n_yes += 1
    n = len(pres)
    return {
        "n": n,
        "n_yes": n_yes,
        "yes_rate": (n_yes / n) if n else None,
        "preds": preds,
        "full300_context": "Presence on frozen n=300 adapted08: 30/34 Yes, chance identity 0.7924",
    }


def identity_table(orig: list[dict], shuf: list[dict], blank: list[dict]) -> dict:
    by_eid_s = {r["example_id"]: r for r in shuf}
    by_eid_b = {r["example_id"]: r for r in blank}
    items = []
    for o in orig:
        eid = o["example_id"]
        s = by_eid_s[eid]
        b = by_eid_b[eid]
        no = normalize(o.get("prediction"))
        ns = normalize(s.get("prediction"))
        nb = normalize(b.get("prediction"))
        items.append(
            {
                "example_id": eid,
                "category": o["category"],
                "question": o["question"],
                "gold": o["gold"],
                "pred_original": o.get("prediction"),
                "pred_shuffle": s.get("prediction"),
                "pred_blank": b.get("prediction"),
                "identity_shuffle": no == ns,
                "identity_blank": no == nb,
            }
        )
    def mean(mask) -> float:
        xs = [1.0 if it[mask] else 0.0 for it in items]
        return float(sum(xs) / len(xs)) if xs else float("nan")

    def mean_cat(cats: set[str], key: str) -> float:
        xs = [1.0 if it[key] else 0.0 for it in items if it["category"] in cats]
        return float(sum(xs) / len(xs)) if xs else float("nan")

    by_cat = {}
    for cat in CATS:
        by_cat[cat] = {
            "n": sum(1 for it in items if it["category"] == cat),
            "shuffle_identity": mean_cat({cat}, "identity_shuffle"),
            "blank_identity": mean_cat({cat}, "identity_blank"),
        }
    return {
        "n": len(items),
        "shuffle_identity": mean("identity_shuffle"),
        "blank_identity": mean("identity_blank"),
        "quantity_color_shuffle_identity": mean_cat({"Quantity", "Color"}, "identity_shuffle"),
        "direction_shuffle_identity": mean_cat({"Direction"}, "identity_shuffle"),
        "presence_shuffle_identity": mean_cat({"Presence"}, "identity_shuffle"),
        "by_category": by_cat,
        "items": items,
    }


def write_report(ident: dict, yes: dict, serve: dict, extra: str = "") -> None:
    lines = [
        "# LOOKING-08-12 report",
        "",
        "Identity = `score.normalize(pred) == score.normalize(pred_original)`. No PASS/FAIL.",
        "",
        f"- n={ident.get('n')} seed={SEED} cats={list(CATS)}",
        f"- prompt = raw `question` (no `[vqa]`, no short-answer suffix)",
        f"- original source: {ident.get('original_source')}",
        f"- 8091 stop: {json.dumps(serve.get('stop'))}",
        extra,
        "",
        "| Rate | Value |",
        "|---|---|",
        f"| Shuffle identity | {ident.get('shuffle_identity')} |",
        f"| Blank identity | {ident.get('blank_identity')} |",
        f"| Quantity+Color shuffle identity | {ident.get('quantity_color_shuffle_identity')} |",
        f"| Direction shuffle identity | {ident.get('direction_shuffle_identity')} |",
        f"| Presence shuffle identity (supporting) | {ident.get('presence_shuffle_identity')} |",
        f"| Presence Yes-rate on this slice (original) | {yes.get('n_yes')}/{yes.get('n')} = {yes.get('yes_rate')} |",
        "",
        f"Full-300 context: {yes.get('full300_context')}",
        "",
        "Bars (orchestrator): blank ≥ 0.70 A100 kill; 0.40–0.70 gray; ≤ 0.40 unblocked on this axis. "
        "Quantity+Color shuffle ≤ 0.40 is a later attach requirement. Direction shuffle ≥ 0.60 red flag.",
        "",
        "## Per item",
        "",
        "| example_id | cat | orig | shuffle | blank | id_shuf | id_blank |",
        "|---|---|---|---|---|---|---|",
    ]
    for it in ident.get("items") or []:
        lines.append(
            f"| `{it['example_id']}` | {it['category']} | {json.dumps(it['pred_original'])} | "
            f"{json.dumps(it['pred_shuffle'])} | {json.dumps(it['pred_blank'])} | "
            f"{it['identity_shuffle']} | {it['identity_blank']} |"
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not Q4.is_file():
        print(f"STOP: 08 GGUF missing {Q4}", file=sys.stderr)
        return 2
    if not MMPROJ.is_file():
        print(f"STOP: 08 mmproj missing {MMPROJ}", file=sys.stderr)
        return 2

    slice_rows = load_frozen_slice()
    frozen_ids = [r["example_id"] for r in slice_rows]
    print(f"FROZEN_SLICE n={len(slice_rows)} ids={frozen_ids}", flush=True)
    if not BLANK_PATH.is_file():
        raise FileNotFoundError(f"STOP: frozen blank missing {BLANK_PATH}")
    im = Image.open(BLANK_PATH)
    if im.size != (512, 512):
        raise RuntimeError(f"STOP: blank size {im.size} expected (512, 512)")
    perm = load_frozen_perm(len(slice_rows))
    orig_rows, missing = load_frozen_original(slice_rows)
    original_source = str(PRED_ORIG)
    print(f"slice n={len(slice_rows)} cats={dict(Counter(r['category'] for r in slice_rows))}", flush=True)
    print(f"frozen_derangement={perm} fixed_points=0", flush=True)
    print(f"blank={BLANK_PATH} size={BLANK_PATH.stat().st_size}", flush=True)
    print(f"original frozen {len(orig_rows)}/{len(slice_rows)} missing={len(missing)}", flush=True)

    proc = None
    stop_info = {"stopped": False, "reason": "not_started"}
    shuf_rows: list[dict] | None = None
    blank_rows: list[dict] | None = None
    blocked_err: str | None = None
    try:
        proc = start_8091()
        print("LOOKING08_SERVER_OK", URL, flush=True)
        if missing:
            original_source = "preds_adapted08.jsonl + re-infer missing on 8091"
            have = {r["example_id"] for r in orig_rows}
            for r in slice_rows:
                if r["example_id"] in have:
                    continue
                inf = infer_one(r["question"], Path(r["image_path"]))
                orig_rows.append({**r, **inf, "condition": "original", "source": "8091", "model": MODEL})
            order = {r["example_id"]: i for i, r in enumerate(slice_rows)}
            orig_rows.sort(key=lambda x: order[x["example_id"]])

        shuf_rows = []
        for i, r in enumerate(slice_rows):
            img = Path(slice_rows[perm[i]]["image_path"])
            inf = infer_one(r["question"], img)
            shuf_rows.append(
                {
                    **r,
                    **inf,
                    "condition": "shuffle",
                    "shuffle_image_id": slice_rows[perm[i]]["image_id"],
                    "shuffle_image_path": str(img),
                    "perm_index": perm[i],
                    "model": MODEL,
                }
            )
            print(f"shuffle {i+1}/20 {r['category']} {inf.get('prediction')!r}", flush=True)

        blank_rows = []
        for i, r in enumerate(slice_rows):
            inf = infer_one(r["question"], BLANK_PATH)
            blank_rows.append({**r, **inf, "condition": "blank", "image_path": str(BLANK_PATH), "model": MODEL})
            print(f"blank {i+1}/20 {r['category']} {inf.get('prediction')!r}", flush=True)
    except (RuntimeError, FileNotFoundError, OSError) as e:
        blocked_err = f"{type(e).__name__}: {e}"
        print(f"STOP: {blocked_err}", file=sys.stderr, flush=True)
    finally:
        stop_info = stop_8091(proc)
        print("STOP_8091", json.dumps(stop_info), flush=True)

    if orig_rows and missing:
        dump_jsonl(PRED_ORIG, orig_rows)
    if blocked_err is not None or shuf_rows is None or blank_rows is None:
        yes = presence_yes_rate(orig_rows) if orig_rows else {}
        payload = {
            "blocked": True,
            "reason": blocked_err or "shuffle/blank not inferred",
            "stop_8091": stop_info,
            "n_slice": len(slice_rows),
            "original_cached": len(orig_rows),
            "original_missing": missing,
            "presence_yes_rate_slice": yes,
            "derangement": perm,
            "q4": str(Q4),
            "q4_bytes": Q4.stat().st_size,
            "mmproj": str(MMPROJ),
            "mmproj_bytes": MMPROJ.stat().st_size,
            "port_8080_open": _port_open(8080),
            "port_8091_open": _port_open(PORT),
            "gpu_used_mib": gpu_used_mib(),
            "serve_ps1_untouched": True,
            "did_not_fallback_to_base_q4": True,
        }
        IDENT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        write_report(
            {
                "n": len(slice_rows),
                "original_source": original_source,
                "items": [],
                "shuffle_identity": None,
                "blank_identity": None,
                "quantity_color_shuffle_identity": None,
                "direction_shuffle_identity": None,
                "presence_shuffle_identity": None,
            },
            yes,
            {"stop": stop_info},
            extra=f"- **BLOCKED:** {blocked_err}",
        )
        print("LOOKING08_BLOCKED", json.dumps({k: payload[k] for k in payload if k != "original_missing"}, indent=2))
        return 2

    ident = identity_table(orig_rows, shuf_rows, blank_rows)
    ident["blocked"] = False
    ident["frozen_example_ids"] = [r["example_id"] for r in slice_rows]
    ident["original_source"] = original_source
    ident["derangement"] = perm
    ident["seed"] = SEED
    ident["port"] = PORT
    ident["model"] = MODEL
    ident["gguf"] = str(Q4)
    ident["mmproj"] = str(MMPROJ)
    ident["q4_bytes"] = Q4.stat().st_size
    ident["mmproj_bytes"] = MMPROJ.stat().st_size
    yes = presence_yes_rate(orig_rows)
    ident["presence_yes_rate_slice"] = yes
    ident["stop_8091"] = stop_info
    ident["port_8080_open"] = _port_open(8080)
    ident["port_8091_open"] = _port_open(PORT)
    ident["did_not_fallback_to_base_q4"] = True
    dump_jsonl(PRED_SHUF, shuf_rows)
    dump_jsonl(PRED_BLANK, blank_rows)
    IDENT_PATH.write_text(json.dumps(ident, indent=2), encoding="utf-8")
    write_report(ident, yes, {"stop": stop_info})
    print(json.dumps({
        "shuffle_identity": ident["shuffle_identity"],
        "blank_identity": ident["blank_identity"],
        "quantity_color_shuffle_identity": ident["quantity_color_shuffle_identity"],
        "direction_shuffle_identity": ident["direction_shuffle_identity"],
        "presence_shuffle_identity": ident["presence_shuffle_identity"],
        "presence_yes": yes,
        "stop_8091": stop_info,
        "port_8091_open": ident["port_8091_open"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"STOP: {type(e).__name__}: {e}", file=sys.stderr)
        # Ensure 8091 is not left up if we spawned it; main()'s finally handles the usual path.
        if _port_open(PORT):
            print("WARNING: :8091 still listening after exception", file=sys.stderr)
        raise
