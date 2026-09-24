# SatQuery AI — SIH26167 (ISRO / Smart India Hackathon 2026)

SatQuery AI is a **satellite-imagery assistant**: you ask a question about satellite imagery in plain English, and it answers with evidence. Two open-weight Qwen3-VL-8B models work in tandem — a **domain-adapted answer model** produces the typed answers and claims, while a **frozen narration model** (unmodified base weights) turns tool evidence into prose; deterministic specialist tools do every measurement — area, change masks, SAR statistics. Nothing on screen comes from the VLM's imagination: every number traces to a tool, and when evidence is insufficient the system says so rather than guessing.

Built for **Smart India Hackathon 2026**, problem statement SIH26167 (ISRO / Space Technology).

| Key dates | |
|---|---|
| Internal hackathon | 16–17 Sep 2026 — **done, team nominated** |
| SIH idea-submission deadline | 20 Sep 2026 — **done** |

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
        │   INGEST      │    │ SPECIALIST TOOLS │        │ LANGUAGE MODELS  │
        │ load + check  │    │ (deterministic)  │        │ (Qwen3-VL-8B ×2) │
        │ the rasters:  │    │ cdvqa_map /      │        │                  │
        │ pixel size,   │    │ changeformer     │        │ answer model     │
        │ CRS, dtype,   │    │ area_calc        │        │ (adapted)        │
        │ SAR calib.    │    │ water/SDWI masks │        │  → typed answers │
        └───────────────┘    │ sar_read/stats   │        │   + claims       │
                             │ sar_agreement    │        │                  │
                             │ coreg_check      │        │ narration model  │
                             │ geo_export       │        │ (frozen) → prose │
                             └────────┬─────────┘        │   from evidence  │
                                      │ tool_outputs (numbers,    └────────┬─────────┘
                                      │ masks, typed verdicts)    │ claims +
                                      ▼                           ▼ prose
                        ┌─────────────────────────────────────────────┐
                        │  EVIDENCE PACKET — every claim is typed:     │
                        │  {predicate, value, confidence, source_tool, │
                        │   model_id, artifact_sha256, which endpoint} │
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

## Model serving — local & cloud, labeled per answer

```
              serving profile: "local" | "cloud"   (config/seats.json)
        ┌──────────────────────┴──────────────────────┐
        ▼ LOCAL (default, runs offline)               ▼ CLOUD (deployable)
  ┌───────────────────────────────┐         ┌───────────────────────────────┐
  │ narration endpoint :8080      │         │ narration endpoint (Modal     │
  │  llama.cpp, 4-bit frozen base │         │  vLLM, full-precision bf16)   │
  │ answer endpoint    :8091      │         │ answer endpoint (Modal vLLM,  │
  │  llama.cpp, 4-bit adapted     │         │  full-precision merged        │
  │  (id: canonical-lrfold)       │         │  weights 3a4fecb0…)           │
  │  --image-min-tokens 384       │         │  scale-to-zero + proxy-auth   │
  └───────────────────────────────┘         └───────────────────────────────┘
        Either way, every answer records which model produced it
        {model id, weights SHA, which endpoint}. A down endpoint
        withholds — it never silently substitutes another model.
```

The serving profile lives in `config/seats.json`; the API exposes it at `GET/POST /seats`; the frontend toggle switches profiles explicitly. Cloud endpoints scale to zero when idle — a `GET /v1/models` probe warms them (cold start measured ~1.5–4.5 min).

---

## Training recipe — how the answer model was adapted

The adapted answer model (served id `canonical-lrfold`) is a continuation-trained artifact built in two auditable stages:

**Stage 1 — high-resolution adaptation (Modal, A100):**

| Ingredient | Value |
|---|---|
| Base | `Qwen/Qwen3-VL-8B-Instruct` @ `0c351dd` |
| Method | LoRA adapters (rank 16, α32) on attention+MLP layers **+ trained vision projector** |
| Data | ~120k rows: RSVQA-HR train + VRSBench captions (count-type questions oversampled ×2.5) |
| Loss | cross-entropy on supervised tokens only, class-weighted; vision tokens masked out |
| Schedule | 2 epochs, cosine decay, lr 1e-4 (adapters) / 1e-5 (projector), batch 16 |
| Merge | adapters loaded first, then projector weights, with a strict check that nothing silently failed to load (this is how a dropped-projector bug was caught) |

**Stage 2 — low-resolution continuation (Lightning A100-40GB → Modal eval):**

| Ingredient | Value |
|---|---|
| Init | stage-1 checkpoint (adapters + projector + optimizer state) |
| LR rows | 57,223 **active** only — 20,009 inactive stubs excluded: their question ids ARE the LR eval sets, so training them would be a direct leak |
| Replay | ~20k HR rows verbatim from the original mix (anti-forgetting) + ~15k caption rows |
| Mix | 117,857 rows, **zero** question-id overlap AND zero image overlap vs every eval set |
| Resume | rolling checkpoint (every 500 steps) synced to local — survived a cross-platform migration |

**Artifact chain:** full-precision merged weights (`3a4fecb0…`) → llama.cpp b10621 convert → F16 → **4-bit quantized GGUF** (`24df79c4…`) + **vision projector file regenerated from the merged model** (`3197a0db…` — the projector was further-trained; reusing the base one would silently drop that training). Every step's hash is recorded in `gates/qwen3vl/canonical/SHA256SUMS.txt`.

**Serving flag that matters:** `--image-min-tokens 384` — llama.cpp gives small (256px) images too few visual tokens without it; the low-res sanity check dropped to 12/20 before this flag.

---

## Results so far (locked — please don't re-litigate in a PR)

All scores are **exact-match accuracy on frozen evaluation question ids** — official dataset files, every question answered (coverage 1.0), deterministic decoding. Manifests + metrics + reports under `eval_*/`; raw prediction files stay local.

**Question-answering benchmarks (full-precision model):**

| Benchmark | Zero-shot baseline | HR-adapted | Current adapted (served) |
|---|---:|---:|---:|
| RSVQA-HR val (n=102,843) | 0.5077 | **0.8136** | 0.8073 ✓ above the 0.80 regression threshold |
| RSVQA-HR test | — | **0.8164** | deferred* |
| RSVQA-HR test_phili | — | **0.7813** | deferred* |
| RSVQA-LR val (n=10,005) | 0.5427 | 0.5401 | **0.7343 (+0.194)** |
| RSVQA-LR test (n=10,004) | 0.5597 | 0.5566 | **0.7294 (+0.173)** |
| VRSBench VQA | 0.6597 | **0.6603** | deferred* |

\* three columns on the newest model were deferred on budget — the eval resumes shard-by-shard where it stopped; nothing partial is published.

**Caption (VRSBench, pycocoevalcap, 9,350 frozen ids):** CIDEr **0.2803**, BLEU-4 0.1207, METEOR 0.2170, ROUGE-L 0.3348 — vs **0.0** CIDEr zero-shot (the adapted model answers in caption register, mean length 43 vs 199 words).

**Other measured:**
- VRSBench grounding acc@0.5: **0.6114** — zero-shot base Qwen3-VL-8B (bf16, Modal), n=16,159 frozen ids, 2026-09-14; raw preds not retained (per policy, the wiped column is not re-run "to confirm"). The live product's `ground` tool is in progress; the adapted seat has not yet been measured on this column.
- CDVQA change-detection QA: **0.62 / 0.62 / 0.51** — produced by the deterministic tool+mapping layer (honestly labeled: not a trained change-VQA model)
- Sanity checks on the served 4-bit model (20-question samples, ±10pt noise): HR 16/20 · LR 18/20 · cloud full-precision 17/20
- LR weakest family: counting questions ~0.25 even after ×2.5 oversampling — the honest residual

**Model discipline:** endpoints change only on measured evidence — the current adapted model shipped after passing a regression threshold on held-out data (0.8073 ≥ 0.80), with provenance re-pinned to the exact bytes served. Every model-produced claim carries `{model id, weights SHA, which endpoint}` — the local quantized model and the cloud full-precision model are labeled separately, never conflated.

---

## Run the demo

Weights and scene PNGs are **not** in git (size and licensing). You need:

1. `llama-server` (llama.cpp), plus `Qwen3VL-8B-Instruct-Q4_K_M.gguf` and `mmproj-Qwen3VL-8B-Instruct-F16.gguf` under `gates/qwen3vl/`
2. Scene PNGs already live under `demo/data/` on this laptop. Rebuild from source only if those folders are missing.

```text
python -m pip install -r demo/requirements.txt
.\demo\serve.ps1 live                 # narration model :8080 + answer model :8091 + Gradio :7860
uvicorn api.main:app --port 8000      # FastAPI backend (from api/)
cd web && npm install && npm run dev  # React SPA → :5173
```

- Narration model → `:8080` · answer model → `:8091` · Gradio fallback → `:7860` · API → `:8000` · web → `:5173`
- No GPU? Cached rehearsal replays all scenes: `.\demo\rehearse_cached.ps1`
- Serving note: llama.cpp is required because Ollama cannot import Qwen3-VL's separate vision-projector file. The answer model serves with `--image-min-tokens 384` (llama.cpp under-tokenizes small imagery without it).

---

## Repository layout

| Path | What it is |
|---|---|
| `CONTRIBUTING.md` | How to send a PR |
| `demo/` | The Gradio app — offline fallback UI |
| `web/` | React SPA — the product frontend |
| `api/` | FastAPI backend over the pipeline (contract: `api/README.md`) |
| `deploy/` | Modal serving — cloud endpoints + smoke harness |
| `config/` | `seats.json` — local/cloud serving profiles + provenance |
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

- **No commercial vision APIs.** Open-weight models only — local endpoints run offline, cloud endpoints are self-hosted on Modal.
- **Model swaps are measured, never silent.** An endpoint's model changes only with a regression-barred eval + re-pinned provenance (see Results).
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

