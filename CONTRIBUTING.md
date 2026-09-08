# Contributing

`main` is protected by review. Open a branch, push, and open a PR. Do not commit straight to `main`.

Default lane: **`demo/`** (UI, planner, tools) and the public docs under `docs/`.

## Standing rules

- No commercial vision APIs.
- Do not attach adapters to `demo/serve.ps1` unless attach bars already passed on disk.
- Do not `modal run` `hunt/` or `train10/` from a clone.
- Do not invent a second large training job.

## Do not commit

Weights, caches, venvs, zips, adapters, `.env`, logs, demo scenes/vendor/traces, `archive/`. If `git status` shows `gates/_cache/`, `*.gguf`, `archive/`, or `teammate_pack/`, stop.

Operator IPC, closed archive, executor specs, and `teammate_pack/` stay on the owner laptop (see `.gitignore`). Public docs under `docs/` are product facts, not a live GPU board.

## PR checklist

- [ ] One concern per PR (demo vs eval vs docs).
- [ ] `python demo/test_harden.py` and/or `python eval/score.py` if you touched those paths.
- [ ] No secrets, no `C:\Users\...` paths, no local Wi-Fi names.
