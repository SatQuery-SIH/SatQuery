"""CLOUD-SEAT smoke — exercise the deployed Modal endpoints end to end.

Local client (real external path, proxy-auth headers included):

  1. GET {cloud}/v1/models on both seats -> assert served model ids
  2. First N frozen rsvqa_hr_val ids through the CLOUD canonical endpoint
     (eval decode contract: greedy, repetition_penalty=1.08, max_tokens=16)
     -> tok_exact EM vs gold. Eval ids used ONLY for this smoke — never
     training, per gates/_cache/eval_ids policy.
  3. One image + prompt through the narrator endpoint -> prose check.
  4. --cold-start flag: poll a zero-scaled endpoint until first 200 and
     record honest cold-start seconds.

Env (from deploy/cloud.env or the environment):
  SATQUERY_MODAL_KEY        wk-… proxy token id
  SATQUERY_MODAL_SECRET     ws-… proxy token secret
  SATQUERY_CLOUD_NARRATOR_URL    e.g. https://proxynanmaga--satquery-serve-narrator.modal.run
  SATQUERY_CLOUD_CANONICAL_URL   e.g. https://proxynanmaga--satquery-serve-canonical.modal.run

Run from the repo root:
  python deploy/smoke_cloud.py                 # full smoke (endpoints warm)
  python deploy/smoke_cloud.py --cold-start    # measure scale-from-zero first
  python deploy/smoke_cloud.py --n 20
"""

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAT = HERE.parent
EVAL_IDS = SAT / "gates" / "_cache" / "eval_ids" / "rsvqa_hr_val_eval_ids.json"
GOLD = SAT / "modal_volume_backup" / "rsvqa_adapt" / "gold" / "rsvqa_hr_val.jsonl"
GOLD_SHA256 = "825ca775cfb3670be55b3d18bd2224c0999ba562d6214653397d99e4604930fe"
IMG_CACHE = SAT / "tmp" / "smoke_images"          # tmp/ is gitignored
REPORT = HERE / "smoke_report.json"
VOLUME = "satquery-data"

# eval decode contract (identical to demo/tools.py CANONICAL_DECODE)
DECODE = {"temperature": 0.0, "repetition_penalty": 1.08, "max_tokens": 16}


def _load_env() -> dict:
    """deploy/cloud.env (gitignored) -> env vars; real env wins."""
    env = dict(os.environ)
    dotenv = HERE / "cloud.env"
    if dotenv.is_file():
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip())
    return env


def _headers(env: dict) -> dict:
    return {
        "Modal-Key": env["SATQUERY_MODAL_KEY"],
        "Modal-Secret": env["SATQUERY_MODAL_SECRET"],
    }


def _req(url: str, headers: dict, payload: dict | None, timeout: float):
    req = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode())


def tok_exact(s: str) -> str:
    """Frozen scorer: strip + casefold + ws-collapse + rstrip('.')."""
    return re.sub(r"\s+", " ", s.strip().lower()).rstrip(".")


def get_models(url: str, headers: dict, timeout: float = 30.0):
    return _req(url.rstrip("/") + "/v1/models", headers, None, timeout)


def wait_warm(url: str, headers: dict, deadline_s: float, label: str) -> float:
    """Poll /v1/models until 200; returns seconds elapsed."""
    t0 = time.perf_counter()
    last = ""
    while True:
        try:
            status, body = get_models(url, headers, timeout=120.0)
            if status == 200:
                return time.perf_counter() - t0
            last = f"http {status}"
        except Exception as exc:  # cold start: proxy holds/drops while booting
            last = f"{type(exc).__name__}: {exc}"
        if time.perf_counter() - t0 > deadline_s:
            raise TimeoutError(f"{label}: no 200 after {deadline_s}s ({last})")
        print(f"  [{label}] waiting… {time.perf_counter() - t0:7.1f}s  ({last})",
              flush=True)
        time.sleep(5)


def fetch_image(vol_path: str, dest_dir: Path) -> Path:
    """vol_path like /data/rsvqa/hr/Data/0.png -> local file via modal CLI."""
    rel = vol_path.split("/data/", 1)[-1].lstrip("/")
    dest = dest_dir / rel.replace("/", "__")
    if dest.is_file():
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "modal", "volume", "get", VOLUME, rel, str(dest)],
        check=True, capture_output=True,
    )
    return dest


def chat_image(url: str, headers: dict, model: str, question: str,
               image: Path, max_tokens: int, timeout: float = 300.0):
    b64 = base64.b64encode(image.read_bytes()).decode()
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": question},
            ],
        }],
        **{k: v for k, v in DECODE.items() if k != "max_tokens"},
        "max_tokens": max_tokens,
        "stream": False,
    }
    t0 = time.perf_counter()
    status, body = _req(url.rstrip("/") + "/v1/chat/completions", headers,
                        payload, timeout)
    lat = time.perf_counter() - t0
    text = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    return status, text.strip(), round(lat, 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--cold-start", action="store_true",
                    help="measure zero-scaled -> first 200 on both seats first")
    ap.add_argument("--warm-deadline", type=float, default=1200.0)
    args = ap.parse_args()

    env = _load_env()
    for k in ("SATQUERY_MODAL_KEY", "SATQUERY_MODAL_SECRET",
              "SATQUERY_CLOUD_NARRATOR_URL", "SATQUERY_CLOUD_CANONICAL_URL"):
        if not env.get(k):
            print(f"missing env {k} — see deploy/README.md", file=sys.stderr)
            return 2
    headers = _headers(env)
    nurl, curl_ = (env["SATQUERY_CLOUD_NARRATOR_URL"].rstrip("/"),
                   env["SATQUERY_CLOUD_CANONICAL_URL"].rstrip("/"))

    report: dict = {"t_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "endpoints": {"narrator": nurl, "canonical": curl_}}

    # ---- cold start --------------------------------------------------------
    if args.cold_start:
        print("COLD-START probe (endpoints must be zero-scaled)", flush=True)
        report["cold_start_s"] = {
            "narrator": round(wait_warm(nurl, headers, args.warm_deadline,
                                        "narrator"), 1),
            "canonical": round(wait_warm(curl_, headers, args.warm_deadline,
                                         "canonical"), 1),
        }
        print(json.dumps(report["cold_start_s"], indent=2), flush=True)

    # ---- /v1/models ---------------------------------------------------------
    _, nm = get_models(nurl, headers)
    _, cm = get_models(curl_, headers)
    report["models"] = {"narrator": (nm.get("data") or [{}])[0].get("id"),
                        "canonical": (cm.get("data") or [{}])[0].get("id")}
    print("models:", json.dumps(report["models"]), flush=True)
    assert report["models"]["narrator"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert report["models"]["canonical"] == "canonical-lrfold"

    # ---- canonical 20-id EM --------------------------------------------------
    import hashlib
    got = hashlib.sha256(GOLD.read_bytes()).hexdigest()
    assert got == GOLD_SHA256, f"gold sha {got} != pinned {GOLD_SHA256}"
    frozen = json.loads(EVAL_IDS.read_text())["ids"][: args.n]
    gold_by_id = {}
    with GOLD.open() as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in frozen:
                gold_by_id[r["id"]] = r
    assert len(gold_by_id) == len(frozen), "frozen ids missing from gold"

    rows, correct, lat = [], 0, []
    for i, rid in enumerate(frozen):
        g = gold_by_id[rid]
        img = fetch_image(g["image"], IMG_CACHE)
        status, text, s = chat_image(
            curl_, headers, report["models"]["canonical"], g["question"], img,
            max_tokens=DECODE["max_tokens"])
        ok = tok_exact(text) == tok_exact(g["gold"])
        correct += ok
        lat.append(s)
        rows.append({"id": rid, "type": g.get("type"), "gold": g["gold"],
                     "pred": text, "em": ok, "latency_s": s})
        print(f"  [{i + 1:2d}/{len(frozen)}] {rid:>8} {g.get('type') or '?':<9} "
              f"gold={g['gold']!r:<22} pred={text!r:<22} {'OK' if ok else 'MISS'} "
              f"{s:.1f}s", flush=True)
    report["canonical_smoke"] = {
        "n": len(frozen), "em_tok_exact": round(correct / len(frozen), 4),
        "decode": DECODE, "frozen_ids_head": frozen[:3],
        "latency_s": {"mean": round(sum(lat) / len(lat), 2), "max": max(lat)},
        "rows": rows,
    }
    print(f"EM {correct}/{len(frozen)} = {correct / len(frozen):.3f}", flush=True)

    # ---- narrator prose -------------------------------------------------------
    img = fetch_image(gold_by_id[frozen[0]]["image"], IMG_CACHE)
    status, text, s = chat_image(
        nurl, headers, report["models"]["narrator"],
        "Describe this satellite image in 3-4 sentences: land cover, visible "
        "structures, and anything notable about the terrain.",
        img, max_tokens=220)
    report["narrator_probe"] = {"status": status, "latency_s": s, "text": text}
    print(f"narrator ({s:.1f}s): {text[:400]}", flush=True)

    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"WROTE {REPORT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
