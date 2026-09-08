# SIH26167 — SatQuery AI: architecture and PS compliance

Stable product architecture. Not a live GPU log. Team FAQ: `SIH26167_Team_Brief.md`. Repo map: `../README.md`.

**Internal show:** 16–17 Sep 2026. **Idea deadline:** 20 Sep 2026.

**Base narrator:** Qwen3-VL-8B-Instruct (llama.cpp Q4_K_M + mmproj), currently **zero-shot**.  
**Judge / fallback:** Qwen2.5-VL-7B 4-bit (Ollama). Local-judge VRSBench VQA n=300 = **0.7833** (continuity column, not an attach bar).

---

## 1. PS compliance map

| # | Official requirement | What we ship | Proof |
| --- | --- | --- | --- |
| 1 | Open-source vision only; **no commercial vision APIs** | Qwen3-VL + ChangeFormer + raster/SAR tools, all local | Airplane-mode demo |
| 2 | **At least one VL component adapted** for remote sensing (finale) | LoRA pipe proven (Gate 2 smoke). Two production mixes did **not** attach (vision+language BEN 0.6433; language-only ~1-token collapse). Finale adapter only if attach bars pass | Closed-run numbers in §2.3 |
| 3 | Three input configs: single, bi-temporal, optical+SAR | Three Gradio tabs | `demo/` |
| 4 | Single-image VQA + captioning or grounding | Scene 1 + VLM VQA/caption | `demo/` + `eval/` |
| 5 | Change description / change-VQA from a pair | ChangeFormer mask + tool area + VLM narration | Scene 2 |
| 6 | Co-registered optical+SAR analysis | SAR stats + optical + VLM late fusion | Scene 3 |
| 7 | Agentic orchestration; **only the observable trace is scored** | Deterministic planner → JSON trace panel | `demo/planner.py` |
| 8 | Validation, confidence, summaries, downloadable reports | Ingest preview, metric badge, report export | `demo/report.py` |
| 9 | Interactive GUI | Gradio | `demo/app.py` |

Missing one mandatory item is how complete-looking teams lose.

---

## 2. Architecture

```
USER QUERY + INPUT CONFIG (single / bi-temporal / optical+SAR)
        │
        ▼
PLANNER  ──► visible JSON plan (task, tools, parameters)
        ▼
SPECIALIST TOOLS (deterministic)
  ├─ vqa / caption     → Qwen3-VL-8B (zero-shot until attach)
  ├─ change_detect     → ChangeFormerV6 (LEVIR-CD pretrained)
  ├─ area_calc         → pixel count × GSD²
  └─ sar_read          → backscatter / water threshold (no learned fusion)
        ▼
VLM NARRATION — quotes tool numbers; does not invent a second set
        ▼
UI — three tabs, mask overlay, agent-trace, report
```

Rules:

- **The VLM never computes.** Areas, IoU, counts come from tools.
- **Planner is boring.** Constrained tool grammar, not a free-form agent.
- **Every specialist is independently testable.**
- **One VLM on 8 GB.** No GeoChat, no second 7B, on the demo laptop.

### 2.1 Serving

llama.cpp `llama-server` on `127.0.0.1:8080` (OpenAI-compatible). Gradio on `7860`. Ollama cannot import the Qwen3-VL mmproj.

### 2.2 SAR scope

Input pairs are **pre-co-registered**. No registration solver. No pixel-level optical→SAR network. Fusion is **late**: structured SAR reading + optical reading + VLM reasoning. BigEarthNet is Europe Sentinel-1/2; the hidden set is Cartosat-2S + RISAT. Hedge = tools.

### 2.3 Adaptation doctrine (finale, not internal)

- Internal show: **zero-shot** narrator. Gate 2 already proved a LoRA pipe exists.
- Vision+language BEN LoRA **failed** the local-judge bar (0.6433 vs 0.7833).
- Language-only short VQA **collapsed** to ~1-token answers. Do not attach.
- A later adapter may attach only if these bars pass **on disk**: caption style ≥40 + cross-tag; untagged exact ≥ 0.55; tripwire; looking blank ≤ 0.40 **and** Qty+Color shuffle ≤ 0.40. Fail any → stay zero-shot through 16–17 Sep and keep the finale checkbox honest.

---

## 3. Evaluation harness

Build order was harness-first. The exam is the official splits, not a self-made quiz.

| Piece | Where |
| --- | --- |
| Frozen VRSBench n=300 ids | `gates/baseline_eval_ids.json` — never train on these |
| Frozen CDVQA test pair ids | `gates/cdvqa_eval_ids.json` |
| Harness | `eval/eval.py` + `eval/score.py` |

The hidden ISRO set is never disclosed. Theatrical tuning to the visible 300 loses. Adaptation data, if used later, must be broader than that split.

The catalogue evaluation-criteria table is still a placeholder. Optimize official benchmark scores; expect normalization.

---

## 4. Demo design

Airplane mode shown on. Three scenes + free-ask through the same planner.

- **Scene 1 — single image:** land-cover / objects. Caption + VQA.
- **Scene 2 — bi-temporal (money scene):** change mask, area from tools, VLM quotes those figures.
- **Scene 3 — optical+SAR:** structured SAR stats + optical; late fusion. Not a claim of learned pixel fusion.

Latency budget: < 20 s first token, < 45 s complete. Cached trapdoor (`demo/rehearse_cached.ps1`) if the GPU is busy. Recorded fallback exists.

How to run (weights are local, not in git): `../demo/README.md`.

---

## 5. Dates

| Date | Meaning |
| --- | --- |
| **16–17 Sep 2026** | Internal: working 3-scene demo, measured baseline, honest adaptation story |
| **20 Sep 2026** | Idea deadline |
| **After internal** | Finale adapter only if attach bars pass; otherwise zero-shot + proven pipe |

Hackathon days = polish and rehearsal, not a first train.

---

## 6. Pitch (two audiences, one artifact)

- **Internal (faculty):** "Ask a satellite a question in plain English; it answers with evidence." Open on Scene 2. The trace is "the system showing its work."
- **National (ISRO):** compliance matrix (§1), official-harness numbers, adapter ablation if we attach, routing accuracy. Every measured number on screen comes from a tool or the harness.
