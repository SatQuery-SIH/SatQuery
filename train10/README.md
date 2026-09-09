# train10/ — the 24k production-mix packers (owner GPU)

Tools that build the **24,000-row language-only training mix** (20% VRS / 35% brief / 15% LEVIR / 20% BEN / 10% CDVQA) for the production adapter attempt. **Owner-only:** the mix JSON lives on the owner's laptop; do not `modal run` these from a clone, and never train on the frozen eval ids in `gates/`.

**Status:** spec written, dispatch gated on hunt verification (`hunt/`). Not dispatched.

| File | What it does |
| --- | --- |
| `prod10_data.py` | CPU packer — builds the stratified 24k mix with quarantine checks |
| `prod10_manifest.py` | Emits unique image paths + counts (the loader contract) |
| `prod10_preflight.py` | **Modal** collator check — catches the two-image (`pack_images[]`) footgun before any GPU spend |
| `prod10_cdvqa_ids.py` | Freezes the CDVQA test pair ids used by the change-VQA exam |

Specs: `prod_train10_spec.md` (training + attach bars) · `prod10_pack_spec.md` (pack contract). Related: `hunt/` (raster materialization) · `eval/` (the exam).
