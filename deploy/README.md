# CLOUD-SEAT — Modal deployment of both VLM seats

Deployable OpenAI-compatible endpoints for the two product seats, on Modal
(account `proxynanmaga`, app `satquery-serve`). vLLM 0.11.2, bf16, one A10G
(24 GB) per seat, scale-to-zero.

| seat | endpoint | served model id (`GET /v1/models`) | artifact |
|---|---|---|---|
| narrator | `https://proxynanmaga--narrator.modal.run` | `Qwen/Qwen3-VL-8B-Instruct` | HF snapshot @ `0c351dd0…` (frozen voice) |
| canonical | `https://proxynanmaga--canonical.modal.run` | `canonical-lrfold` | `runs/lr_fold/merged_eval` on volume `satquery-data`, merged_tree_sha256 `3a4fecb0…` (asserted at container start) |

## Auth — `modal-proxy-auth`

Both endpoints set `requires_proxy_auth=True`. Every request needs the
workspace proxy token pair as headers:

```
Modal-Key: wk-…
Modal-Secret: ws-…
# or: Authorization: Bearer wk-….ws-…
```

Token + URLs live in `deploy/cloud.env` (gitignored via `*.env`). Rotate with
`python -m modal workspace proxy-tokens create`. **Never commit the secret.**

## Env vars

| var | used by | meaning |
|---|---|---|
| `SATQUERY_MODAL_KEY` | cloud callers | proxy token id (`wk-…`) |
| `SATQUERY_MODAL_SECRET` | cloud callers | proxy token secret (`ws-…`) |
| `SATQUERY_CLOUD_NARRATOR_URL` | smoke / wiring | narrator endpoint |
| `SATQUERY_CLOUD_CANONICAL_URL` | smoke / wiring | canonical endpoint |
| `SATQUERY_SEAT_MIN_CONTAINERS` | deploy-time | `=1` keeps a warm container for a demo window — **never the default** |

Backend wiring (API-0080 lane): `config/seats.json` is the seat model —
`profile` flips `local`↔`cloud`; cloud callers must inject the two proxy-auth
headers (e.g. `tools.canonical_vqa(url=cloud_url)` needs a header-adding
transport — provenance fields themselves are unchanged).

## Commands

```bash
# deploy / redeploy (profile: proxynanmaga)
python -m modal deploy deploy/modal_serve.py

# eval window only — one warm container per seat:
SATQUERY_SEAT_MIN_CONTAINERS=1 python -m modal deploy deploy/modal_serve.py

# smoke: /v1/models ids, 20 frozen rsvqa_hr_val ids EM, narrator prose
python deploy/smoke_cloud.py --n 20

# measure cold start (wait ~11 min idle first so containers hit zero)
python deploy/smoke_cloud.py --cold-start
```

## Warmup contract

`GET {cloud_url}/v1/models` on a zero-scaled seat triggers scale-up. The
request hangs while the container boots — **probe loops must poll**, a single
request can hit the client timeout before the GPU is ready. Fire this probe
on `/health`/page-load so the GPU warms while the evaluator reads the UI.

## Measured (2026-09-22, proxynanmaga, A10G)

- Cold start → first 200 on `/v1/models` (zero-scaled, volume-cached weights):
  - narrator: **83.2 s / 84.3 s** (two boots)
  - canonical: **191.7 s / 263.5 s** (two boots — consistently slower; the
    spread is Modal scheduling + volume/compile-cache variance, not a code
    path difference)
  - **Frontend "models warming" budget: ~4.5 min worst-case** (seats boot in
    parallel — wall time ≈ canonical's). Probe poll interval ≥10 s; a single
    request can outlive a 120 s client timeout while the GPU warms.
- Warm latency: canonical `chat/completions` ~**2.3 s**/req (16-tok greedy,
  512 px image, mean 2.53 s incl. first-hit); narrator 220-tok prose ~**11 s**.
- Canonical smoke: **EM 17/20 = 0.850** tok_exact on frozen rsvqa_hr_val head
  (bf16 merged — tracks the 0.8073 hr_val class number; misses are count/area
  near-misses: `390m2` vs `521m2`, `0` vs `1`, `107m2` vs `14m2`).

## Scale-to-zero economics

`min_containers=0`, `scaledown_window=600` — zero GPU spend when idle. Weights
come from the volume (`HF_HOME=/data/_hf`, narrator snapshot already cached;
canonical served straight from `runs/lr_fold/merged_eval`) and vLLM compile
artifacts persist in `VLLM_CACHE_ROOT=/data/_vllm_cache` — nothing is
re-downloaded or re-compiled per cold start.

First-boot note: `gpu_memory_utilization=0.92` fails on 24 GB (0.77 GiB KV
< 1.12 GiB needed for `max_model_len=8192`); `0.95` clears it.
