# demo/ — the 3-scene offline Gradio app

The product. Three scenes, one planner, fully offline — this is what runs on stage.

**Narrator status:** zero-shot Qwen3-VL-8B. Adapters exist (06, 08, X1) but **none has cleared its attach bars**, so none is served. Do not point `serve.ps1` at a parked adapter.

## Setup (weights and scene PNGs are not in git)

1. Put `llama-server` (llama.cpp) plus `Qwen3VL-8B-Instruct-Q4_K_M.gguf` and `mmproj-Qwen3VL-8B-Instruct-F16.gguf` under `gates/qwen3vl/`.
2. Rebuild the three scenes from their sources: `python demo/prepare_scenes.py`
3. Install the UI requirements: `python -m pip install -r demo/requirements.txt`

## Run

```text
.\demo\serve.ps1 live        # llama-server :8080 + Gradio :7860
.\demo\rehearse_cached.ps1   # cached trapdoor: all 3 scenes, zero GPU
```

## Demo-day insurance (already built)

- **Cached trapdoor:** if the GPU dies or llama-server won't start, the cached rehearsal replays every scene with real outputs in under a second — visibly labeled CACHED in the UI, so the audience is never misled.
- **Recorded fallback:** `fallback_recording.mp4` captures the live scene outputs and tool numbers.

## The three scenes

| Scene | Input | What the audience sees |
|---|---|---|
| 1 — single image | one satellite PNG | VQA + caption |
| 2 — bi-temporal *(the money scene)* | before/after pair | ChangeFormer mask → area in km² from the raster tool → VLM narration quoting those exact figures |
| 3 — optical+SAR | Sentinel-2 RGB + Sentinel-1 VV/VH | classical SAR water threshold + optical reading, fused at the VLM layer |

Every number on screen traces to a tool output in the agent-trace panel. Architecture: `docs/SIH26167_Final_Plan.md`. Internal show: **16–17 Sep**.
