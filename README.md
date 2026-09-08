# SatQuery AI — SIH26167 (ISRO)

Offline satellite-imagery assistant. A local VLM **narrates**; deterministic tools compute area, change masks, and SAR stats. No number on screen may come only from the VLM.

**Internal show:** 16–17 Sep 2026. **Idea deadline:** 20 Sep 2026.

PS decode: `docs/SIH26167_Team_Brief.md`. Architecture: `docs/SIH26167_Final_Plan.md`. How to send a PR: `CONTRIBUTING.md`.

These files are product facts. They do not track a live GPU job.

## Architecture (locked)

```
USER QUERY + input config (single / bi-temporal / optical+SAR)
        │
        ▼
Planner  ──► visible JSON tool trace (scored; not hidden chain-of-thought)
        ▼
Specialist tools (deterministic)     VLM (Qwen3-VL-8B, currently zero-shot)
  ChangeFormer + area_calc           narrates tool evidence; VQA/caption
  SAR stats / coreg check            never invents a measured number
        ▼
Gradio demo (`demo/`) — 3 scenes, offline
```

**PS checkboxes:** open-weight only (no commercial vision APIs); at least one adapted VL component for the *finale*; three input modes; change description; optical+SAR; agentic trace; local GUI.

**Serving:** llama.cpp `llama-server` (Q4_K_M + mmproj). Ollama cannot import this mmproj.

**Hardware (demo laptop):** RTX 5060 8 GB. One VLM. No GeoChat / second 7B on that machine.

**Geography:** BigEarthNet is Europe-only. Hidden ISRO/Cartosat+RISAT set unknown. Hedge = tools, not a RISAT encoder.

## Locked results (do not re-litigate in a PR)

| What | Result |
| --- | --- |
| Base narrator | **Qwen3-VL-8B-Instruct**, zero-shot, via llama.cpp |
| VRSBench VQA n=300 local-judge | **0.7833** — continuity column, not an attach bar |
| LoRA pipe (Gate 2) | **PASS** (smoke on Modal) |
| Vision+language LoRA on BEN | Failed (0.6433). Parked. |
| Language-only short VQA | Collapsed to mean **~1 token**. Do not attach. |
| Internal demo | Three scenes, airplane-mode path, **adapter off** |

Do not attach a parked or later adapter to `demo/serve.ps1` unless attach bars pass **on disk**: caption style ≥40 + cross-tag; untagged exact ≥ 0.55; tripwire; looking blank ≤ 0.40 **and** Qty+Color shuffle ≤ 0.40. Fail any → stay zero-shot through 16–17 Sep.

## Run the demo (after you have local weights)

Weights and scene PNGs are **not** in git. You need llama-server plus `Qwen3VL-8B-Instruct-Q4_K_M.gguf` and `mmproj-Qwen3VL-8B-Instruct-F16.gguf` under `gates/qwen3vl/`. Rebuild scenes with `python demo/prepare_scenes.py`.

```text
python -m pip install -r demo/requirements.txt
.\demo\serve.ps1 live
```

llama-server on `:8080`. Gradio on `:7860`. Cached rehearsal (no GPU): `.\demo\rehearse_cached.ps1`.

## Layout

| Path | What |
| --- | --- |
| `CONTRIBUTING.md` | PR rules |
| `demo/` | Gradio app. Default place to work |
| `eval/` | Frozen n=300 VRSBench harness |
| `docs/` | Team brief + architecture |
| `gates/` | Frozen eval ids + `data_prep.py` |
| `hunt/` `train10/` | Owner GPU / mix packers. Do not `modal run` from a clone |

## Do not

- Use commercial vision APIs.
- Attach parked or later adapters to `serve.ps1` until attach bars pass on disk.
- Duplicate a 24k training job on another Modal account.
- Commit weights, `.env`, `archive/`, or `gates/_cache/`.
