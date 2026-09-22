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
USER QUERY + input config (single / bi-temporal / optical+SAR)
        │
        ▼
Planner  ──► visible JSON tool trace (scored; not hidden chain-of-thought)
        ▼
Specialist tools (deterministic)          Two VLM seats (Qwen3-VL-8B)
  ChangeFormer + area_calc                  canonical-lrfold: adapted answers/claims
  SAR stats / coreg check / geo export      narrator: frozen, narrates evidence only
        ▼
React SPA (web/) over FastAPI (api/) — Gradio (demo/) remains the
zero-dependency offline fallback. Local llama.cpp seats or Modal cloud
seats — labeled per claim, never silently swapped.
```

**The design rule that makes this defensible to judges:** the VLM never computes. Ask "how much did built-up area grow?" and the change-detection tool produces the mask, the area tool converts pixels × ground-sample-distance into km², and only then does the VLM turn that evidence into a sentence — citing the numbers the tools produced.

**Three input modes (the PS's mandatory trio):** single-image VQA/captioning · bi-temporal change description · co-registered optical+SAR analysis. All three run in one offline Gradio UI.

**Constraints we designed for:** the demo laptop is an RTX 5060 8 GB card running exactly one VLM (no GeoChat or second 7B). BigEarthNet training data is Europe-only; the hidden ISRO evaluation set (Cartosat/RISAT or otherwise) is unknown — so the hedge is the geography-agnostic tool layer, not a retrained vision encoder.

---

## Results so far (locked — please don't re-litigate in a PR)

| What | Result |
|---|---|
| Canonical seat (served) | `canonical-lrfold` — Qwen3-VL-8B + RSVQA-adapted LoRA + trained projector, Q4_K_M, `:8091` |
| Narrator seat (served) | Frozen **Qwen3-VL-8B-Instruct** Q4_K_M, `:8080` — narrates evidence only |
| RSVQA-HR val / test / phili (frozen ids, token-exact) | **0.8136 / 0.8164 / 0.7813** (merged-tree eval; baseline zero-shot 0.5077) |
| RSVQA-LR val / test (frozen ids) | **0.7343 / 0.7294** — LR-fold continuation lifted the weakest column ~19 pts |
| VRSBench VQA / caption CIDEr | **0.6603 / 0.2803** (vs 0.0 CIDEr zero-shot) |
| Cloud seats | Same models on Modal vLLM (scale-to-zero) — labeled `cloud` profile in `config/seats.json` |
| Product | FastAPI (`api/`) + React SPA (`web/`); Gradio stays the offline fallback; audit trace + evidence packets + georeferenced GeoTIFF exports |

**Adapter discipline:** seats swap only on measured evidence — the canonical seat is the LR-fold merged artifact (`3a4fecb0…` on cloud / `24df79c4…` Q4 local), regression-barred on HR val (0.8073 ≥ 0.80) before it shipped. Provenance is per-claim: every model-produced answer carries its model id + artifact SHA + seat.

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

