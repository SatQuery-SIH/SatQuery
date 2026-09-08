# train10/ — 24k mix packers (owner GPU)

Pack helpers for a later language-only mix. Mix JSON stays on the owner laptop. Do not `modal run` these from a clone. Do not train on `gates/baseline_eval_ids.json`.

| File | What |
| --- | --- |
| `prod10_data.py` | CPU packer |
| `prod10_manifest.py` | Unique image paths |
| `prod10_preflight.py` | Two-image collator check |
| `prod10_cdvqa_ids.py` | Freeze CDVQA test pair ids |
