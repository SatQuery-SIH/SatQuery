# LR-FOLD Lane — Final Report

**Result: LR fold trained to 4,910/4,910 steps, merged, evaluated on 3 of 6
frozen columns. LR → parity achieved, HR regression bar PASSED.**

## Headline numbers (token-exact EM — the bar metric)

| Column | Published baseline | lr_fold_merged | Δ | Status |
|---|---|---|---|---|
| rsvqa_lr_val | 0.5401 | **0.7343** | **+0.1942** | measured, coverage 1.0 |
| rsvqa_lr_test | 0.5566 | **0.7294** | **+0.1728** | measured, coverage 1.0 |
| rsvqa_hr_val | 0.8136 | **0.8073** | −0.0063 | measured — **bar ≥0.80 PASS** |
| rsvqa_hr_test | 0.8164 | — | — | NOT RUN (budget) — bar ≥0.79 unverified |
| rsvqa_hr_test_phili | 0.7813 | — | — | NOT RUN (budget) |
| vrsbench_vqa | 0.6603 | — | — | NOT RUN (budget) |

Per-family (token-exact): lr_val — comp 0.9545, presence 0.9212,
rural_urban 0.93, **count 0.2454** (weakest family; ×2.5 oversample applied
but count remains hardest). hr_val — comp 0.9156, presence 0.9277,
count 0.689, area 0.5671.

Metrics: `eval_lr_fold/final/metrics.json` (volume:
`runs/rsvqa_adapt/eval/lr_fold_merged/metrics.json`).

## Training

- 2-epoch continue-train of the published `ckpt_final` on
  `mix_lr_fold.jsonl` (117,857 rows: 82,857 LR-effective incl. count ×2.5,
  20,000 HR replay verbatim, 15,000 VRSB captions; id+image overlap = 0).
- Lightning L40S ran to ~step 3,400 before credit exhaustion; last synced
  ckpt_last = step 3,000.
- **Modal (proxynanmaga) resumed bit-continuously from @3000** — same
  trainer, deps, mix, image trees; ckpt_last 6-file SHA-verified pre-launch;
  post-resume losses at steps 3250–3600 matched the Lightning log
  digit-for-digit. Survived one infra retry via auto-resume from ckpt@3500.
- Finished: `report.json` status OK, opt_steps 4910, wall 12,623s (Modal
  segment), final loss 0.489, tok_acc 0.7854, lr → 0.

## Hashes (ckpt_final @4910)

- adapter_model.safetensors `60f1be46174b568f311e8f8e2aebf28a875a102c917c6f23ced5de884a4eeb68`
- merger.pt `1331936df4275a7c550341de3e9414c4d40ee3b902ee02c64517aadecdb28cf2`
- adapter_config.json `4ee89ba31cc7534008c725d668735a618170ba79c0e8742ef1eadec65f923dab`
- combined `143bbfd550c00643f0ce6a5ff3b287fc1dba47476b6d5dae1316eb32dbd7972d`
- mix_lr_fold `ce25242f6a12e061cd3e67cb96190cd001219f5ef253126829030c61683cf2c7`
- merged_eval tree sha256 `3a4fecb0521ddaf42ed297ec8ca391f9b5d94bd6c884bf51cbab0c4562c7f11c` (16 files)
- Local verified copy: `eval_lr_fold/ckpt_final/ckpt_final/`

## Gates

| Gate | Result |
|---|---|
| Smoke overfit (100 rows, Lightning) | PASS EM=1.0 |
| Full train 4910 steps | DONE (Lightning→Modal resume) |
| stage_check | ALL GREEN — ckpt/ids/gold/vocab/dataset SHAs match, 0 missing images, id↔gold order-identical ×6 |
| merge | OK — adapter-first, 24 merger tensors, strict load, bf16 |
| smoke_merged (200 hr_val) | 0.81 tok_exact ≈ floor |
| Eval | lr_val, lr_test, hr_val complete; 3 columns pending budget |

## Cost

- Modal proxynanmaga eval lane: stage $0.002 + merge $0.024 + smoke $0.049
  + lr_val $0.281 + lr_test $0.268 + hr_val $7.455 (4 shards) + score $0.002
  ≈ **$8.08**
- Modal resume train (L40S, 12,623s) ≈ ~$6.8 + tar/staging ≈ $0.25.
- Lightning: L40S studio ~9h train + setup (credit-metered, exhausted).

## Remaining (needs burner rotation, est. ~$26)

- rsvqa_hr_test (222,684 rows, 8 shards) ≈ $16 — the ≥0.79 regression bar
- rsvqa_hr_test_phili (105,647, 4 shards) ≈ $7.5
- vrsbench_vqa (37,409, 2 shards) ≈ $2.6
- rerun `e_eval` per column (SKIP_EXISTS resumes shards), then `e_score`.

## Platform handoff

- All artifacts on proxynanmaga `satquery-data`: `runs/lr_fold/{ckpt_final,
  merged_eval, report.json, train_log.json, STATUS.json, stage_check.json,
  smoke_merged.json}` + preds at `runs/rsvqa_adapt/eval/lr_fold_merged/`.
- ripperscience `satquery-data` holds a replicated eval-minimal copy
  (eval trees via verified tars + smalls + ckpt_final) — `eval_lr_fold/
  modal_unpack.py` untars+verifies it if eval continues there; otherwise
  safe to delete. Stale volumes (`satquery-rsvqa-hr-full`, `satquery-vrsbench`,
  `satquery-hf-cache`) untouched pending that decision.
- Arm B (fresh LR-only lane): SKIPPED per spec (main run landed).
