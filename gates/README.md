# gates/ — frozen eval ids

`_cache/` and `qwen3vl/` (weights, llama.cpp) are gitignored.

| Path | What |
| --- | --- |
| `baseline_eval_ids.json` | Frozen n=300 VRSBench eval ids. Never train on these |
| `cdvqa_eval_ids.json` | Frozen CDVQA test pair ids |
| `data_prep.py` | BEN text → chat helper |
