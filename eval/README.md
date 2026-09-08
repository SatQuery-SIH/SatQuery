# eval/ — frozen VRSBench harness

Run from `SatQuery/` (parent of this folder). This is the exam script, not a reason to attach an adapter.

| File | What |
| --- | --- |
| `eval.py` | n=300 VQA calls (Ollama or llama.cpp OpenAI API) |
| `score.py` | CPU exact-match + synonyms. `python eval/score.py` runs unit tests |
| `fetch_data.py` | Cache VRSBench eval split into `gates/_cache/vrsbench/` (gitignored) |
| `looking08_probe.py` | n=20 original / shuffle / blank identity probe. Do not resample the slice |

Ids: `gates/baseline_eval_ids.json`. Never train on them. Headline **0.7833** is a continuity column, not an attach bar.
