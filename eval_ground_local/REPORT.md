# GROUND-MEASURE (local, decision-grade) — D-014 seat verdict

Date: 2026-09-26. Cost: $0 (two local llama.cpp seats + 294 staged images).

## Question

Did the RSVQA LoRA damage the adapted seat's box output, and which served
seat should own the `ground` tool? Decision-grade only — these are **Q4_K_M
llama.cpp** numbers on a **300-id seeded subset**; they are NOT comparable
to the reportable bf16 vLLM column (0.6114, n=16,159). The reportable
number is a separate approved run.

## Protocol

- Frozen ids: `gates/_cache/eval_ids/vrsbench_referring_eval_ids.json`
  (n=16,159; frozen before any preds).
- Sample: `random.Random(20260925).sample(ids, 300)` — same 300 ids for
  both seats, paired.
- GT: `VRSBench_EVAL_referring.json`, sha256 pinned to
  `fd63f7c6…ff1b44` (verified in harness).
- Images: 294 unique files staged from `proxynanmaga:satquery-data
  /vrsbench/Images_val` to `gates/_cache/data/vrsbench/images_val`
  (129 MB, retries on transient storage errors).
- Prompt/decoding: `REF_PROMPT` + `DECODING["referring"]` verbatim from
  `scripts/modal_vrsbench_eval.py` (temp 0.0, top_p 1.0, max_tokens 128).
- Parser/scorer: verbatim copy of `parse_pred_box` / `parse_gt_box` /
  `iou_official` / `iou_float` / `_smart_resize` / `_resized_dims` —
  same protocol, only transport differs (llama.cpp HTTP vs vLLM batch).
- Harness: `scripts/ground_measure_local.py`; raw outputs + parse tags
  preserved per item in `ground_local_*.json`.

## Seats

| seat | artifact | llama.cpp flags |
|---|---|---|
| canonical-lrfold-q4 (:8091) | `gates/qwen3vl/canonical/lrfold/Qwen3VL-8B-RSVQA-LRFOLD-Q4_K_M.gguf` + LRFOLD mmproj | `-ngl 99 -c 4096 --image-min-tokens 384` |
| narrator-base-q4 (:8080) | `gates/qwen3vl/Qwen3VL-8B-Instruct-Q4_K_M.gguf` + F16 mmproj | `-ngl 99 -c 4096` (defaults, first pass) |
| narrator-base-q4-mintok384 (:8080) | same artifact | `-ngl 99 -c 4096 --image-min-tokens 384` (matched-geometry rerun) |

## Results (n=300, paired ids)

| metric | canonical :8091 (384) | narrator :8080 (default) | narrator :8080 (384) |
|---|---|---|---|
| **acc@0.5 (official int-grid IoU)** | 0.5667 | 0.5767 | **0.6067** |
| acc@0.7 | 0.3433 | 0.3633 | **0.3967** |
| mean IoU (official) | 0.4887 | 0.4939 | **0.5174** |
| acc@0.5 (float IoU) | 0.5233 | 0.5400 | 0.5767 |
| parse failures | **0** | 9 | 6 |
| parse tags | tuple4_1000 ×296, tuple4_px ×3, tuple4_100 ×1 | bbox2d_0_1000 ×291, parse_fail ×9 | bbox2d_0_1000 ×294, parse_fail ×6 |
| wall | 1746 s | 1516 s | 1996 s |

### Paired analysis — first pass (as-served, narrator at defaults)

- both pass: 137, both fail: 94, canonical-only: 33, narrator-only: 36
- **McNemar p = 0.810 → no detectable difference** on acc@0.5.
- On the 291 ids the narrator parsed: canonical 0.5773 vs narrator 0.5945
  (+1.7pp, still noise at this n).
- Narrator's 9 parse fails are **refusals**, not format breakage:
  8× "There are none." + 1× "The provided sentence is incorrect…".
  Canonical predicted on all 9; it scored IoU ≥0.5 on 2 of them
  (0.80, 0.63) — objects the base claimed absent actually existed.

### Paired analysis — matched-geometry rerun (both at `--image-min-tokens 384`)

- both pass: 140, neither: 88, canonical-only: 30, narrator-only: 42
- **McNemar p = 0.195** — still not significant at α=0.05, but the
  narrator's point-estimate lead widened from +1.0pp to **+4.0pp**
  (95% CI on the delta ≈ −1.5..+9.5pp). Extra image tokens helped the
  narrator (+3.0pp) while canonical was already at 384.
- Direction is now consistent across both runs: narrator ≥ canonical
  on every metric at matched geometry, and its remaining 6 parse fails
  are still refusals (desired product behavior), not format breakage.
- **This fires the D-014 revisit condition** ("matched-resolution rerun
  changes the result"): the tie-breaker rationale must now overcome a
  4pp adverse point estimate rather than a 1pp one.

### Paired analysis

- both pass: 137, both fail: 94, canonical-only: 33, narrator-only: 36
- **McNemar p = 0.810 → statistical tie** on acc@0.5.
- On the 291 ids the narrator parsed: canonical 0.5773 vs narrator 0.5945
  (+1.7pp, still noise at this n).
- Narrator's 9 parse fails are **refusals**, not format breakage:
  8× "There are none." + 1× "The provided sentence is incorrect…".
  Canonical predicted on all 9; it scored IoU ≥0.5 on 2 of them
  (0.80, 0.63) — objects the base claimed absent actually existed.

### Anomaly review (immaterial)

4 canonical outputs fell outside the 0-1000 frame (3 `tuple4_px`,
1 `tuple4_100`). Each alternative interpretation moves that item's IoU
by <0.15 and changes acc@0.5 by ≤0.003. No action.

## Verdict — D-014 revisited after matched-geometry rerun

**Original pick (as-served): canonical LR-fold seat owns `ground`** —
recorded when the first pass showed no detectable difference
(McNemar p=0.81) and canonical's zero-refusal/zero-parse-failure profile
won the tie-breakers. That verdict is **superseded** by the
matched-geometry rerun + absent-target probe:

- Matched geometry: narrator **+4.0pp** acc@0.5 (p=0.195 — not
  significant, but the point estimate moved against canonical on every
  metric and the direction is consistent across runs).
- Absent-target probe: narrator withholds 48% unaided, canonical 2%.
  "Zero refusals" — the original tie-breaker — is a defect under
  absent-target pressure, not a feature.
- Canonical's output-format drift worsens under that pressure
  (14/150 non-0-1000 frames vs 4/300 matched), including degenerate
  full-frame boxes.

**Revised recommendation for `ground` v1:** presence-gated, narrator
generates the box — `canonical_vqa` presence oracle ("Is there a
{target}?" → no/uncertain → withhold, no artifacts) → narrator
`bbox_2d` box → strict per-response frame detection → degenerate
full-frame box ⇒ withhold. This keeps the adapted seat meaningfully
integrated (the presence oracle is exactly the binary-VQA family the
LoRA trained on — the reviewer's cited ~0.92 RSVQA family score), adds
the narrator's native refusals as a second withhold signal, and picks
the seat with the better matched-geometry point estimate. Canonical's
cleaner tuple format does not offset a −4pp estimate + 98% invention
rate; the fenced-JSON form parses fine (its "failures" are refusals).

### Caveats added on 2026-09-26 external review

- **As-served, not matched-geometry:** the first pass ran canonical at
  `--image-min-tokens 384` vs narrator at llama.cpp defaults — the
  confound *favored canonical* (more image tokens = higher input
  resolution). The matched-geometry rerun
  (`ground_local_narrator-base-q4-mintok384_300.json`) removed it and
  the narrator's lead grew to +4.0pp — i.e. the first pass likely
  understated the narrator, not canonical.
- **This benchmark rewards never-refusing — measured.** Absent-target
  probe (`absent_probe_*.json`, n=150 mismatched image↔question pairs,
  invention rate = upper bound since coincidental matches exist):
  **canonical invents a box on 98%** of absent-target queries (147/150;
  the 3 non-box outputs were "No"/degenerate, not structured refusals),
  **narrator invents on 52%** (78/150; 72 refusals). Canonical's absent-
  target fallback is often a degenerate full-frame box
  (`[0,0,1024,1024]`, `[0,0,1000,1000]`, `[0,0,100,100]` — 9 `tuple4_px`
  + 5 `tuple4_100` of 150 vs 4/300 on matched data) — a detectable
  signature, but not reliable withholding. **Consequence for `ground`
  v1: a presence check is mandatory regardless of seat** — `canonical_vqa`
  "Is there a {target}?" before grounding; "no" → withheld, no box
  rendered. The 98% number is exactly why grounding can't ship as a
  bare box call.
- **Grid drift:** 4/300 canonical outputs weren't 0-1000 on matched
  data (14/150 under absent-target pressure) — `ground` v1 detects the
  frame per response, records non-1000 tags in the trace, never
  rescales silently.
- **llama.cpp grounding warning:** the server logs "Qwen-VL models
  require at minimum 1024 image tokens to function correctly on
  grounding tasks" — both seats run 384 (canonical's as-served flag).
  If a judge-facing ground demo underperforms, retrying at
  `--image-min-tokens 1024` is a cheap lever — flagged, not measured.

Format note for `ground` v1: canonical emits bare `[x1,y1,x2,y2]`
0-1000-grid tuples; narrator emits ```json `bbox_2d` arrays. The
existing parser handles both. Whichever seat generates, the tool treats
empty/absent-box responses and degenerate full-frame boxes as withhold
signals — never as silent full-image boxes — and records the detected
frame tag (`tuple4_1000`/`bbox2d_0_1000`/`tuple4_px`/`tuple4_100`/
`parse_fail`) per response in the trace.

Caveat kept explicit: this comparison picks the *seat*. It does not
establish the reportable public number — that needs the bf16/full-n run
(approved on the designated Modal profile; volume staging required since
Images_val and the merged tree currently live on a different account's
`satquery-data`).

## Product-level eval — shipped gate on frozen ids (2026-09-26)

The seat comparison measured the bare box model. This run measures the
**shipped `tools.ground` contract end-to-end** — presence oracle (:8091
`canonical_vqa`) -> narrator box (:8080) -> frame/full-frame gate —
replayed offline per presence variant on the same frozen items:
n=300 present (same seeded ids) + n=150 absent (same mismatched pairs
as the seat probes). Presence was asked TWO ways per item (full
expression vs head-noun class question); one temp-0 box call shared.

| variant | present acc@0.5 | present withhold | absent false-box | absent withhold |
|---|---|---|---|---|
| full phrase | 0.4467 | 17.3% | 0.173 | 82.7% |
| **head noun** | **0.4800** | **10.7%** | 0.253 | 74.7% |

**Pick rule (pre-registered in scripts/ground_product_local.py):** best
present acc@0.5 provided absent false-box <= 0.30 -> **head noun wins
both halves** and ships as `ground(presence_mode="head")` default.
Head recovers 25 presence false-negatives (36->11 `target_absent`
withholds on real targets — the adapted seat answers "Is there a
vehicle?" far more reliably than a 20-word referring expression) at the
cost of ~12 more absent inventions; both variants keep absent
invention far below narrator-alone (~52%).

Withhold anatomy (present set): presence says no (36 full / 11 head),
box unparsed i.e. narrator free-text refusal (14/18 — a second withhold
layer), degenerate full-frame (2/3).

Product-level acc@0.5 ~0.48 vs bare-narrator 0.6067: the presence gate
trades ~13pp of present accuracy to cut absent-target invention from
~52% to 17-25% — the honest-product trade, now measured at n=450.

Artifacts: `ground_product_300p_150a.json` (per-item presence answers,
box text, per-variant decisions, IoU). Harness:
`scripts/ground_product_local.py` (incremental partial-resume).
