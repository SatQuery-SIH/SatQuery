# SIH26167 "SatQuery AI" — Team Brief

**Read this before the internal show (16–17 Sep 2026).** Idea-submission deadline: **20 Sep 2026**. Architecture and PS-compliance detail: `SIH26167_Final_Plan.md`. Product map: `../README.md`.

This brief is the single reference for what we are building, what the problem statement demands, and what has already been proven or ruled out. Where something is unverified, it says so. It is a product brief — it does not track live GPU jobs.

---

## 1. The 60-second version

ISRO/SAC wants a system where a **non-expert user types a plain-English question about satellite imagery and gets an evidence-backed answer** — not a land-cover map with a legend only a GIS analyst can read.

Three kinds of input: one image, two images of the same place at different times (what changed?), and an optical + SAR image pair of the same place (see through clouds). The system must **decide for itself which specialist tools to run**, show its work in a visible, scored execution trace, and prove that at least one of its AI components was actually **adapted to remote sensing** — a generic off-the-shelf chatbot explicitly fails the requirement.

We build it from open-source models that run **on our own machine, fully offline**, with the VLM narrating and deterministic tools doing every measurement.

---

## 2. What the problem statement actually asks

### 2.1 The mandatory checklist

| # | Requirement (official) | What it means |
| --- | --- | --- |
| 1 | Open-source vision components only; **no commercial/proprietary vision APIs** | All weights local. Prove it with an airplane-mode demo. |
| 2 | **At least one vision/VL component fine-tuned or adapted** for remote sensing | Mandatory for the *finale*. Internal can ship zero-shot + a proven LoRA pipe. |
| 3 | Three input modes: **single image**, **bi-temporal pair**, **co-registered optical+SAR pair** | One UI, three tabs. GeoTIFF/TIFF for real geospatial; PNG/JPEG for prescribed benchmarks. |
| 4 | **Single-image VQA is mandatory** + captioning *or* text-guided grounding | Answer questions about one image, plus describe it or highlight regions. |
| 5 | **Change description or change-VQA from a bi-temporal pair is mandatory** | "What changed between these two dates and where?" Change map is optional; we generate it anyway. |
| 6 | **Cross-modal pair analysis** from a co-registered optical+SAR pair | Use both images together (e.g. water under cloud). |
| 7 | **Agentic orchestration**: automatically select, sequence, and execute tools | Planner routes each query. Visible trace is scored. |
| 8 | Auditable execution summary + confidence + visual evidence + **downloadable reports** | Trace panel + report export. |
| 9 | Interactive **GUI or web application** | Not a notebook. Not a CLI. |

### 2.2 What the judges actually evaluate

- **"Only the observable execution trace ... will be evaluated. Internal reasoning text is neither required nor evaluated."** Judges score *what the system did* (task, tools, parameters, outputs). The agent-trace panel is a **scored artifact**, not decoration.
- Evaluation uses **prescribed public splits** — **VRSBench** and **RSVQA** (single-image) and **CDVQA** (change-VQA) — plus a **hidden ISRO/SAC set**: pre-georeferenced, co-registered **Cartosat-2S optical + RISAT SAR** pairs whose **annotations are never disclosed**. Scores are normalised across metrics.
- You cannot overfit the demo. The system has to generalise.

### 2.3 Two catalogue traps (same for every team)

The SIH site evaluation-criteria table is still an unfilled placeholder. Metric formulas are unpublished. Optimize each **named benchmark's official scoring**, expect normalization.

The dataset link cell on the site is truncated (~300 chars). Locating the real artifacts is part of the filter; we already did that (Section 3.3).

---

## 3. FAQ

### 3.1 What are we building?

A municipal planner drags in two satellite images a year apart and types: *"Has the built-up area increased, and where?"* The system checks compatibility, runs change detection, and answers with a number, a highlighted mask, confidence, and a downloadable report.

A **satellite analyst you can talk to**, with the analyst's work shown. Three input configurations, same idea.

### 3.2 Is local inference feasible?

Yes — measured, not hoped, on the demo laptop (RTX 5060 8 GB):

- **Qwen3-VL-8B-Instruct** via llama.cpp `llama-server` (Q4_K_M + mmproj), fully offline.
- Demo scenes complete in a few seconds (budget was < 20 s first token, < 45 s complete).
- Ollama cannot import this mmproj. The serving path is llama.cpp, not Ollama.
- Qwen2.5-VL-7B 4-bit remains the **local-judge** model and fallback, not the narrator.
- One VLM on 8 GB. No GeoChat / second 7B on the demo machine.

### 3.3 Do the datasets exist?

| Dataset | What it is |
| --- | --- |
| **VRSBench** | PS-named single-image eval. Frozen n=300 ids in `gates/baseline_eval_ids.json`. |
| **CDVQA** | Change-VQA exam (TGRS 2022). Frozen pair ids in `gates/cdvqa_eval_ids.json`. |
| **LEVIR-CD** | Change-detection specialist; pretrained ChangeFormer weights exist. |
| **BigEarthNet / BEN text** | Named adaptation source (Europe Sentinel-1/2). Not Indian Cartosat+RISAT. |

RSVQA was located but not used as a second harness. Hidden ISRO/Cartosat+RISAT annotations stay unknown.

### 3.4 Can we fine-tune a VLM? Do we have to for the internal show?

The **pipe is proven** (Gate 2 LoRA smoke on Modal). The **production mixes we already ran did not earn a demo attach**:

- Vision+language LoRA on BEN: local-judge **0.6433** (worse than zero-shot **0.7833**). Parked as a domain specialist.
- Language-only short VQA: exact-match rose, but mean answer length collapsed to **~1 token**. Unusable as a narrator.

**Internal 16–17 Sep ships zero-shot Qwen3-VL-8B.** Adaptation remains a **finale** checkbox. Do not plug an adapter into `demo/serve.ps1` unless these bars pass **on disk**: caption style ≥40 tokens + cross-tag; untagged exact ≥ 0.55; tripwire; looking blank ≤ 0.40 **and** Qty+Color shuffle ≤ 0.40. Fail any → stay zero-shot through the internal show.

### 3.5 Isn't "agentic AI" a buzzword here?

No. The PS scores the **observable execution trace**, and ours is real: a constrained grammar of a few tools, deterministic routing, every step logged. Routing accuracy on a 20-query suite is a metric we can show on demand.

### 3.6 How does SAR fit in?

- Pairs are **pre-co-registered**. We do not solve registration.
- We do **not** train pixel-level optical↔SAR fusion. The SAR tool produces structured statistics; the optical tool produces its reading; **fusion happens at the VLM reasoning layer**, quoting both.
- BigEarthNet is Europe Sentinel-1/2. The hidden set is Indian Cartosat+RISAT. Our hedge is the geography-agnostic tool layer, not a retrained RISAT encoder.

### 3.7 Can we demo this without it dying on stage?

Three rehearsed scenes (single / bi-temporal change / optical+SAR), a latency budget, a cached trapdoor that replays everything without a GPU, and a recorded fallback. Scene 2 is the money scene: a messy pair goes in, a change mask comes out, and a number lands on screen with the tool math visible beside it.

### 3.8 Too crowded / too hard?

Space Technology has produced SIH winners in every edition 2017–2025. Given-benchmark + hidden-eval PSs punish theatrical teams — which is exactly why we built harness-first. Feasibility and competition are our honest weaknesses; the deterministic demo and the tool-grounded numbers are the answers to both.

---

## 4. Architecture (what we build)

```
USER QUERY + INPUT (single / bi-temporal / optical+SAR)
        │
        ▼
INPUT VALIDATOR ── count, modality, format, compatibility
        │            (fail here → friendly refusal, not a hallucinated answer)
        ▼
AGENTIC PLANNER ── task, tool sequence, parameters
        │            (deterministic grammar; every step logged)
        ▼
SPECIALIST TOOLS ── small, independent, individually testable
   ├─ VLM (Qwen3-VL-8B, currently zero-shot) → VQA / caption
   ├─ ChangeFormer (pretrained LEVIR-CD)     → change mask
   ├─ Raster math                            → area = mask pixels × GSD²
   └─ SAR reader                             → backscatter / water stats
        │
        ▼
EVIDENCE FUSION ── every tool output lands in a structured record
        │            (VLM quotes these figures; never invents a second set)
        ▼
VLM NARRATION ── answer + cited regions + confidence
        │
        ▼
UI ── three tabs · mask overlays · agent-trace panel · report export
        └─ cached trapdoor + recorded fallback for demo day
```

**The VLM never computes.** Areas, percentages, counts come from tools. When a judge asks where a number came from, we point at the tool output and the mask — both on screen.

---

## 5. Dates (stable)

| Date | What the team shows |
| --- | --- |
| **16–17 Sep 2026** | Internal: 3-scene offline demo, zero-shot narrator, measured baseline, honest adaptation story. |
| **20 Sep 2026** | Idea deadline. |
| **After internal → finale** | Production adapter only if attach bars pass on disk. Otherwise stay zero-shot and still claim the Gate-2 pipe + closed-run evidence. |

Hackathon days are integration and rehearsal, not a first training run.

---

## 6. Who owns what

| Role | Owns |
| --- | --- |
| **Demo / frontend** | Three-tab UI, trace panel, scene choreography (`demo/`) |
| **Planner / tools** | Routing, ChangeFormer, area, SAR (`demo/planner.py`, `demo/tools.py`) |
| **Eval** | Frozen n=300 harness (`eval/`) |
| **Adaptation** | Finale LoRA, attach bars — not a PR to flip `serve.ps1` |
| **Pitch** | Internal: "ask a satellite a question." National: compliance matrix + measured numbers |

---

## 7. Risks

1. Live VLM latency/VRAM — **largely retired** (measured). Keep cache trapdoor + recorded fallback.
2. "Just GeoChat with a UI" — we do not ship GeoChat. Differentiators are the trace, tool-grounded numbers, adapter on/off if we attach later.
3. Specialist judge on SAR fusion — late-fusion framing; never claim learned pixel fusion.
4. Hidden Cartosat+RISAT set — tools + split discipline, not a Europe-only encoder.
5. Scope creep — three scenes, then freeze.

---

## 8. Why this PS

Named data, named benchmarks, and a hidden eval set filter out polish-without-substance. Easier PSs are easier for everyone. This one is hard in ways a deterministic, measured demo specifically answers.