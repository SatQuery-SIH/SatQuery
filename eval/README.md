# eval/ — the frozen VRSBench exam

This folder **is** our measurement instrument: a frozen n=300 slice of the official VRSBench VQA split, run identically every time so results are comparable across experiments. Run from `SatQuery/` (the parent of this folder).

**The rules that make scores comparable:** the 300 ids in `gates/baseline_eval_ids.json` never change, are never trained on, and the scoring protocol never moves. The headline **0.7833** (zero-shot Qwen3-VL local-judge) is a *continuity column* — it lets every future adapter be compared against the same baseline. It is **not** an attach bar by itself.

| File | What it does |
| --- | --- |
| `eval.py` | Runs the n=300 VQA calls against any OpenAI-compatible server (Ollama or llama.cpp). Raw predictions go to JSONL — never edited by hand. |
| `score.py` | CPU-only scoring: exact-match after normalization + synonyms, plus per-category aggregation. `python eval/score.py` runs its unit tests — do that before any GPU spend. |
| `fetch_data.py` | Caches the VRSBench eval split into `gates/_cache/vrsbench/` (gitignored). Handles the dataset's mixed-type schema quirk by loading raw JSON, not the HF viewer. |
| `looking08_probe.py` | The n=20 identity probe: original vs shuffled vs blanked golds. Diagnoses *whether a model actually reads the image* (a shuffled-gold score near 1.00 means it doesn't). Do not resample the frozen slice. |

**Reading a result honestly:** exact-match is a strict floor (verbose answers can contain the gold phrase; terse answers rarely do), the local-judge is the frozen primary, and the looking-08 probe catches image-blindness. Report all three when the verdict matters.
