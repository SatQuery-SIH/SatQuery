# gates/ — frozen evaluation ids and data prep

This folder holds the **quarantine**: the id lists that define our exams. They never change, and they are never trained on — that is what makes every score comparable.

| Path | What |
| --- | --- |
| `baseline_eval_ids.json` | The frozen n=300 VRSBench VQA evaluation ids (seed=42). The continuity baseline **0.7833** was measured on exactly these. **Never train on them** — leakage invalidates every delta. |
| `cdvqa_eval_ids.json` | The frozen CDVQA test pair ids for the change-VQA exam. Same quarantine rules. |
| `data_prep.py` | Helper that turns BEN text rows into training-ready chat format. |

**Sub-folders (gitignored, not in the repo):**

- `_cache/` — cached datasets, predictions, judge outputs, hunt/train artifacts (`_cache/prod10/` is the live hunt/train workspace). Regenerable; never commit.
- `qwen3vl/` — model weights (base + adapted GGUFs, mmprojs) and the llama.cpp runtime. Multi-GB; never commit.

Related docs: `eval/README.md` (how the exam is run) · `../README.md` (repo map).
