# demo/ — 3-scene Gradio app

Narrator is **zero-shot** Qwen3-VL-8B until attach bars pass on disk. Do not point `serve.ps1` at old adapters.

Weights and scene PNGs are **not** in git. Place llama-server, `Qwen3VL-8B-Instruct-Q4_K_M.gguf`, and `mmproj-Qwen3VL-8B-Instruct-F16.gguf` under `gates/qwen3vl/`. Rebuild scenes with `python demo/prepare_scenes.py`.

```text
python -m pip install -r demo/requirements.txt
.\demo\serve.ps1 live
```

llama-server on `:8080`. Gradio on `:7860`. Cached rehearsal (no GPU): `.\demo\rehearse_cached.ps1`.

Architecture: `docs/SIH26167_Final_Plan.md`. Dates: **16–17 Sep**.
