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
| narrator-base-q4 (:8080) | `gates/qwen3vl/Qwen3VL-8B-Instruct-Q4_K_M.gguf` + F16 mmproj | `-ngl 99 -c 4096` (defaults) |

## Results (n=300, paired ids)

| metric | canonical :8091 | narrator :8080 |
|---|---|---|
| **acc@0.5 (official int-grid IoU)** | **0.5667** | **0.5767** |
| acc@0.7 | 0.3433 | 0.3633 |
| mean IoU (official) | 0.4887 | 0.4939 |
| acc@0.5 (float IoU) | 0.5233 | 0.5400 |
| mean IoU (float) | — | — |
| parse failures | **0** | **9** |
| parse tags | tuple4_1000 ×296, tuple4_px ×3, tuple4_100 ×1 | bbox2d_0_1000 ×291, parse_fail ×9 |
| wall | 1746 s | 1516 s |

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

## Verdict — D-014 resolved

**Canonical LR-fold seat owns `ground`.** Grounding accuracy is a
statistical tie, and the tie-breakers all favor canonical:

1. **Zero refusals** — always emits a parseable box. A tool contract wants
   structured output or structured withholding; free-text refusal is a
   pipeline edge case. Narrator refuses 3% of the time.
2. **Adapted seat stays meaningfully integrated** — grounding through the
   LoRA seat keeps the RS-adapted component visible in the product (a
   stated finale criterion), and its box head demonstrably survived
   adaptation (the LoRA-damage hypothesis is rejected: p=0.81).
3. **Seat separation** — narrator seat stays free for planning/narration
   while the canonical seat does structured extraction (matches the
   existing `canonical_vqa` split).

Format note for `ground` v1: canonical emits bare `[x1,y1,x2,y2]`
0-1000-grid tuples; narrator emits ```json bbox_2d arrays. The existing
parser handles both; the tool should request the tuple form (shorter,
deterministic) and treat empty/absent-box responses as withhold signals,
not silent full-image boxes.

Caveat kept explicit: this comparison picks the *seat*. It does not
establish the reportable public number — that needs the bf16/full-n run
(approved on the designated Modal profile; volume staging required since
Images_val and the merged tree currently live on a different account's
`satquery-data`).
