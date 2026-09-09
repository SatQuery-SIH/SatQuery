# SatQuery AI — SIH26167 (ISRO / Smart India Hackathon 2026)

SatQuery AI is an **offline satellite-imagery assistant**: you ask a question about satellite imagery in plain English, and it answers with evidence. A local vision-language model (Qwen3-VL-8B) **plans, interprets, and narrates**; deterministic specialist tools do every measurement — area, change masks, SAR statistics. Nothing on screen comes from the VLM's imagination: every number traces to a tool.

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
Specialist tools (deterministic)          VLM (Qwen3-VL-8B, currently zero-shot)
  ChangeFormer + area_calc                narrates tool evidence; VQA/caption
  SAR stats / coreg check                 never invents a measured number
        ▼
Gradio demo (demo/) — 3 scenes, fully offline
```

**The design rule that makes this defensible to judges:** the VLM never computes. Ask "how much did built-up area grow?" and the change-detection tool produces the mask, the area tool converts pixels × ground-sample-distance into km², and only then does the VLM turn that evidence into a sentence — citing the numbers the tools produced.

**Three input modes (the PS's mandatory trio):** single-image VQA/captioning · bi-temporal change description · co-registered optical+SAR analysis. All three run in one offline Gradio UI.

**Constraints we designed for:** the demo laptop is an RTX 5060 8 GB card running exactly one VLM (no GeoChat or second 7B). BigEarthNet training data is Europe-only; the hidden ISRO evaluation set (Cartosat/RISAT or otherwise) is unknown — so the hedge is the geography-agnostic tool layer, not a retrained vision encoder.

---

## Results so far (locked — please don't re-litigate in a PR)

| What | Result |
|---|---|
| Base narrator | **Qwen3-VL-8B-Instruct**, zero-shot, via llama.cpp |
| VRSBench VQA n=300 (frozen local judge) | **0.7833** — continuity column, not an attach gate |
| LoRA fine-tuning pipeline (Gate 2) | **PASS** — proven end-to-end on Modal |
| Vision+language LoRA on BigEarthNet | **Failed as a general adapter** (0.6433). Parked as a domain specialist. |
| Language-only LoRA on short VQA | Collapsed to ~1-token answers. Do not attach. |
| Internal demo | Three scenes, airplane-mode-proven offline path, **running zero-shot (no adapter attached)** |

**Why zero-shot?** Our pre-registered experiment discipline: an adapter only replaces the base model if it clears attach bars **on disk** (caption style ≥40 tokens + cross-tag; untagged exact-match ≥ 0.55; tripwire; identity checks). None has yet. Until one does, the demo runs the stronger zero-shot model — and we say so plainly.

---

## Run the demo

Weights and scene PNGs are **not** in git (size and licensing). You need:

1. `llama-server` (llama.cpp), plus `Qwen3VL-8B-Instruct-Q4_K_M.gguf` and `mmproj-Qwen3VL-8B-Instruct-F16.gguf` under `gates/qwen3vl/`
2. Rebuild scenes: `python demo/prepare_scenes.py`

```text
python -m pip install -r demo/requirements.txt
.\demo\serve.ps1 live
```

- llama-server → `:8080` · Gradio UI → `:7860`
- No GPU? Cached rehearsal replays all three scenes: `.\demo\rehearse_cached.ps1`
- Serving note: llama.cpp is required because Ollama cannot import Qwen3-VL's separate mmproj vision file.

---

## Repository layout

| Path | What it is |
|---|---|
| `CONTRIBUTING.md` | How to send a PR |
| `demo/` | The Gradio app — default place to work |
| `eval/` | Frozen n=300 VRSBench evaluation harness |
| `docs/` | Problem-statement decode + architecture (read these second) |
| `gates/` | Frozen eval ids + data prep |
| `hunt/`, `train10/` | Internal GPU tooling. Do not `modal run` from a clone. |
| `archive/` | Closed runs and historic paperwork |

---

## Ground rules

- **No commercial vision APIs.** Open-weight models only; everything runs locally.
- **No adapter attaches to `demo/serve.ps1`** until attach bars pass on disk (see Results).
- **No duplicate 24k training runs** on other Modal accounts.
- **Never commit** weights, `.env`, `archive/`, or `gates/_cache/`.

---

## Documentation map

| File | Owns |
|---|---|
| `docs/SIH26167_Team_Brief.md` | What the PS asks, FAQ, dates, roles |
| `docs/SIH26167_Final_Plan.md` | Architecture, PS compliance map, evaluation + demo design |
| `docs/demo_spec.md` / `docs/demo_report.md` | Demo design + measured demo evidence |
| `CONTRIBUTING.md` | PR rules |

These docs are **product facts** — they do not track live GPU jobs. For background on how we work: every experiment is pre-registered, every verdict is verified against artifacts on disk, and a negative result is reported as honestly as a positive one.

