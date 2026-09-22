# gates/ — local weights and live checkpoints

Not in git (size). Needed to run the demo:

| Path | What |
| --- | --- |
| `qwen3vl/llama_cpp/llama-server.exe` | Local VLM server |
| `qwen3vl/Qwen3VL-8B-Instruct-Q4_K_M.gguf` | Frozen narrator |
| `qwen3vl/mmproj-Qwen3VL-8B-Instruct-F16.gguf` | Vision projector |
| `_cache/changeformer/` | Imported LEVIR ChangeFormer checkpoint |
| `_cache/cf_ft/train/best_by_val.pt` | Housing-change specialist |
| `_cache/cf_ft/semantic/best_semantic.pt` | Land-cover describer |
| `_cache/cf_ft/ckpt_registry.json` | Which ckpt is attached |
| `data_prep.py` | Used only if you rebuild scenes with `demo/prepare_scenes.py` |
