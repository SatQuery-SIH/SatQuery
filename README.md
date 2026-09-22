# SatQuery AI — SIH26167 (ISRO / Smart India Hackathon 2026)

SatQuery AI is a **satellite-imagery assistant**: you ask a question about satellite imagery in plain English, and it answers with evidence. Two open-weight Qwen3-VL-8B seats work in tandem — a **remote-sensing-adapted canonical** produces typed answers and claims, while a **frozen narrator** turns tool evidence into prose; deterministic specialist tools do every measurement — area, change masks, SAR statistics. Nothing on screen comes from the VLM's imagination: every number traces to a tool, and when evidence is insufficient the system says so rather than guessing.

Built for **Smart India Hackathon 2026**, problem statement SIH26167 (ISRO / Space Technology).

| Key dates | |
|---|---|
| Internal hackathon | 16–17 Sep 2026 |
| SIH idea-submission deadline | 20 Sep 2026 |

---

## How it works

```
                        ┌─────────────────────────────────────────────┐
 USER QUERY + inputs ──►│  PLANNER  (scores visible JSON plan — not    │
 (single / bi-temporal /│  hidden chain-of-thought; refuses cleanly    │
  optical+SAR)          │  when the request is unsupported)            │
                        └──────────────┬──────────────────────────────┘
                                       │ plan: {task, tools[], mode}
                ┌──────────────────────┼───────────────────────────┐
                ▼                      ▼                           ▼
        ┌───────────────┐    ┌──────────────────┐        ┌──────────────────┐
        │   INGEST      │    │ SPECIALIST TOOLS │        │   VLM SEATS      │
        │ rasterio bind │    │ (deterministic)  │        │ (Qwen3-VL-8B ×2) │
        │ GSD + CRS +   │    │ cdvqa_map /      │        │                  │
        │ dtype + SAR   │    │ changeformer     │        │ canonical-lrfold │
        │ calibration   │    │ area_calc        │        │  → typed answers │
        │ detection     │    │ water/SDWI masks │        │   + claims       │
        └───────────────┘    │ sar_read/stats   │        │                  │
                             │ sar_agreement    │        │ narrator (frozen)│
                             │ coreg_check      │        │  → prose from    │
                             │ geo_export       │        │   evidence only  │
                             └────────┬─────────┘        └────────┬─────────┘
                                      │ tool_outputs (numbers,    │
                                      │ masks, typed verdicts)    │ claims +
                                      ▼                           ▼ prose
                        ┌─────────────────────────────────────────────┐
                        │  EVIDENCE PACKET — every claim typed:        │
                        │  {predicate, value, confidence, source_tool, │
                        │   model_id, artifact_sha256, seat}           │
                        │  withheld when evidence is insufficient      │
                        └──────────────┬──────────────────────────────┘
                                       ▼
                        ┌─────────────────────────────────────────────┐
                        │  REPORT — findings, measurement, confidence, │
                        │  limitations, narration_check (blocks        │
                        │  invented numbers incl. bad % derivations),  │
                        │  artifacts + GeoTIFF exports                 │
                        └──────────────┬──────────────────────────────┘
                                       ▼
            React SPA (web/) ◄── FastAPI (api/) ──► SQLite run store
            SSE stage events stream the pipeline live — plan → tools →
            packet → narration — so the audit trail is visible in real time.
```

**The design rule that makes this defensible to judges:** the VLM never computes. Ask "how much did built-up area grow?" and the change-detection tool produces the mask, the area tool converts pixels × ground-sample-distance into km², and only then does the VLM turn that evidence into a sentence — citing the numbers the tools produced.

**Three input modes (the PS's mandatory trio):** single-image VQA/captioning · bi-temporal change description · co-registered optical+SAR analysis.

**Honest withholding, by design:** when the evidence isn't there, the system says so — `withheld` claims (e.g. SAR agreement withheld when measured misregistration >5px), `inconclusive(weak_peak)` coregistration verdicts, planner refusals for unsupported requests, and `skipped_no_transform` geo exports on un-georeferenced uploads. Withholding is a recorded event in the packet, never silent.

---

## Model seats — local & cloud, labeled per claim

```
                 seats.json profile: "local" | "cloud"
        ┌──────────────────────┴──────────────────────┐
        ▼ LOCAL (default, offline-capable)            ▼ CLOUD (deployable)
  ┌───────────────────────────────┐         ┌───────────────────────────────┐
  │ narrator  :8080  llama.cpp    │         │ narrator   Modal vLLM bf16    │
  │  Q4_K_M, frozen base          │         │  scale-to-zero, proxy-auth    │
  │ canonical :8091  llama.cpp    │         │ canonical  Modal vLLM bf16    │
  │  Q4_K_M canonical-lrfold      │         │  canonical-lrfold merged tree │
  │  --image-min-tokens 384       │         │  (3a4fecb0…)                  │
  └───────────────────────────────┘         └───────────────────────────────┘
        Both profiles: per-claim provenance {model_id, sha256, seat}.
        A down seat withholds — it never silently substitutes another model.
```

The seat model lives in `config/seats.json`; the API exposes it at `GET/POST /seats`; the frontend toggle switches profiles explicitly. Cloud seats scale to zero when idle — a `GET /v1/models` probe warms them (cold start measured ~1.5–4.5 min).

---

## Training recipe — how the canonical model was adapted

The served canonical (`canonical-lrfold`) is a continuation-trained artifact built in two auditable stages:

**Stage 1 — RSVQA-HR adaptation (Modal, A100):**

| Ingredient | Value |
|---|---|
| Base | `Qwen/Qwen3-VL-8B-Instruct` @ `0c351dd` |
| Method | LoRA r16/α32 on attn+MLP **+ trained visual projector** (`merger.pt`) |
| Data | ~120k rows: RSVQA-HR train + VRSBench captions (count-type ×2.5 oversampled) |
| Loss | masked CE on supervised tokens, class-weighted; vision tokens masked |
| Schedule | 2 epochs, cosine, lr 1e-4 adapter / 1e-5 merger, batch 16 |
| Merge | adapter-first `PeftModel.from_pretrained` → `merger.pt` load with `unexpected_keys == 0` assert (the fix that caught a silently-dropped projector) |

**Stage 2 — LR-fold continuation (Lightning A100-40GB → Modal eval):**

| Ingredient | Value |
|---|---|
| Init | stage-1 `ckpt_final` (adapter + merger + optimizer state) |
| LR rows | 57,223 **active** only — 20,009 inactive stubs excluded (their qids ARE the LR eval sets — training them = direct leak) |
| Replay | ~20k HR rows verbatim from the original mix (anti-forgetting arm) + ~15k caption rows |
| Mix | 117,857 rows, **zero** id-overlap AND zero image-overlap vs all eval sets |
| Resume | rolling `ckpt_last` (every 500 steps) synced to local — survived a cross-platform migration |

**Artifact chain:** merged bf16 tree (`3a4fecb0…`) → llama.cpp b10621 convert → F16 → **Q4_K_M** (`24df79c4…`) + **mmproj from the merged model** (`3197a0db…` — the projector was further-trained; borrowing the base mmproj would drop it). Every step SHA-recorded in `gates/qwen3vl/canonical/SHA256SUMS.txt`.

**Serving flag that matters:** `--image-min-tokens 384` — llama.cpp under-tokenizes 256px imagery; without it the LR smoke dropped to 12/20.

---

## Results so far (locked — please don't re-litigate in a PR)

All scores are **token-exact EM on frozen eval ids**, official dataset files, coverage 1.0, greedy decode. Manifests + metrics + REPORTs under `eval_*/`; raw prediction shards stay local.

**VQA columns (merged bf16 eval):**

| Column | Zero-shot baseline | Stage-1 (HR adapt) | LR-fold (served) |
|---|---:|---:|---:|
| RSVQA-HR val (n=102,843) | 0.5077 | **0.8136** | 0.8073 ✓ regression bar ≥0.80 |
| RSVQA-HR test | — | **0.8164** | deferred* |
| RSVQA-HR test_phili | — | **0.7813** | deferred* |
| RSVQA-LR val (n=10,005) | 0.5427 | 0.5401 | **0.7343 (+0.194)** |
| RSVQA-LR test (n=10,004) | 0.5597 | 0.5566 | **0.7294 (+0.173)** |
| VRSBench VQA | 0.6597 | **0.6603** | deferred* |

\* three LR-fold columns deferred on budget — `SKIP_EXISTS` resumes the shards cleanly; nothing partial is published.

**Caption (VRSBench, pycocoevalcap, 9,350 frozen ids):** CIDEr **0.2803**, BLEU-4 0.1207, METEOR 0.2170, ROUGE-L 0.3348 — vs **0.0** CIDEr zero-shot (the adapted model answers in caption register, mean length 43 vs 199 words).

**Other measured:**
- VRSBench grounding: **0.6114** · CDVQA (deterministic mapper path): **0.62 / 0.62 / 0.51** — honestly labeled as a mapper, not a trained change-VQA
- Served-artifact sanity (Q4, n=20 smokes, ±10pt noise): HR 16/20 · LR 18/20 · cloud bf16 17/20
- LR weakest family: `count` ~0.25 even after ×2.5 oversample — the honest residual

**Adapter discipline:** seats swap only on measured evidence — `canonical-lrfold` shipped after its HR-val regression bar passed (0.8073 ≥ 0.80) and its provenance re-pinned to the served bytes. Every model-produced claim carries `{model_id, artifact_sha256, seat}` — local Q4 and cloud bf16 are labeled, never conflated.

---

## Run the demo

Weights and scene PNGs are **not** in git (size and licensing). You need:

1. `llama-server` (llama.cpp), plus `Qwen3VL-8B-Instruct-Q4_K_M.gguf` and `mmproj-Qwen3VL-8B-Instruct-F16.gguf` under `gates/qwen3vl/`
2. Scene PNGs already live under `demo/data/` on this laptop. Rebuild from source only if those folders are missing.

```text
python -m pip install -r demo/requirements.txt
.\demo\serve.ps1 live                 # narrator :8080 + canonical :8091 + Gradio :7860
uvicorn api.main:app --port 8000      # FastAPI backend (from api/)
cd web && npm install && npm run dev  # React SPA → :5173
```

- Narrator → `:8080` · canonical → `:8091` · Gradio fallback → `:7860` · API → `:8000` · web → `:5173`
- No GPU? Cached rehearsal replays all scenes: `.\demo\rehearse_cached.ps1`
- Serving note: llama.cpp is required because Ollama cannot import Qwen3-VL's separate mmproj vision file. Canonical serves with `--image-min-tokens 384` (llama.cpp under-tokenizes 256px LR imagery without it).

---

## Repository layout

| Path | What it is |
|---|---|
| `CONTRIBUTING.md` | How to send a PR |
| `demo/` | The Gradio app — offline fallback UI |
| `web/` | React SPA — the product frontend |
| `api/` | FastAPI backend over the pipeline (contract: `api/README.md`) |
| `deploy/` | Modal serving — cloud seats + smoke harness |
| `config/` | `seats.json` — local/cloud seat model + provenance |
| `scripts/` | Modal training/eval harnesses (RSVQA adapt, re-eval, artifact sync) |
| `docs/` | Problem-statement decode + architecture (read these second) |
| `strategy/` | Internal north star — gitignored |
| `ops/` | Laptop board (`LOCKS.md`, `NEXT_TASKS.md`). Gitignored. |
| `gates/` | Local GGUF + live ChangeFormer / semantic checkpoints |
| `cf_ft/` | Live ChangeFormer + SECOND semantic package (imported by the demo) |
| `eval_*` | Frozen-id eval records — manifests/metrics/REPORTs (raw preds gitignored) |
| `judge_kit/` | Plug-and-play eval runner + rehearsal + Sundarbans case study |

---

## Ground rules

- **No commercial vision APIs.** Open-weight models only — local seats offline-capable, cloud seats self-hosted on Modal.
- **Seat swaps are measured, never silent.** A seat changes only with a regression-barred eval + re-pinned provenance (see Results).
- **No duplicate 24k training runs** on other Modal accounts.
- **Never commit** weights, `.env`, `ops/`, or `gates/_cache/`.

---

## Documentation map

| File | Owns |
|---|---|
| `docs/SIH26167_Team_Brief.md` | What the PS asks, FAQ, dates, roles |
| `docs/SIH26167_Final_Plan.md` | Architecture, PS compliance map, evaluation + demo design |
| `strategy/MASTER_ARCHITECTURE_AND_STRATEGY.md` | Internal architecture + adaptation gates (laptop) |
| `CONTRIBUTING.md` | PR rules |

These docs are **product facts** — they do not track live GPU jobs. For background on how we work: every experiment is pre-registered, every verdict is verified against artifacts on disk, and a negative result is reported as honestly as a positive one.

