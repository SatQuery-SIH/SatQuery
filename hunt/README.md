# hunt/ — raster materialize (owner GPU)

These scripts pull training rasters onto a Modal volume. They are **not** a teammate task. Do not `modal run` them from a clone.

| File | What |
| --- | --- |
| `raster_hunt.py` | Laptop-side probes |
| `raster_hunt_modal.py` | Modal VRS / SECOND / BEN stages |

Outputs go under `gates/_cache/` (gitignored).
