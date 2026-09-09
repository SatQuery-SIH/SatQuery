# hunt/ — raster materialization (owner GPU)

Before any training can start, the training images must physically exist somewhere a loader can read them. This folder finds and materializes those rasters onto Modal volumes. **Owner-only:** not a teammate task; do not `modal run` from a clone.

**Status:** RASTER-HUNT-13 found the rasters (evidence in `gates/_cache/prod10/`); **VERIFY-HUNT-14** (independent verification + path map) is the live gate before TRAIN-10 dispatch.

| File | What it does |
| --- | --- |
| `raster_hunt.py` | Laptop-side probes: local LEVIR PNGs, zip keys, basename checks |
| `raster_hunt_modal.py` | Modal stages: VRS zip verify, SECOND download+extract, BEN S2/S1 rematerialization |

Outputs land under `gates/_cache/prod10/` (gitignored). The known landmine: hunt materialized BEN/SECOND under `/png/png/…` while the training mix expects `png/…` — verify resolves this with an explicit path map before TRAIN-10.
