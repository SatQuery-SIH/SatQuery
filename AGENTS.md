# AGENTS.md — SatQuery AI (SIH26167)

Agentic vision-language assistant for remote-sensing imagery, built for
Smart India Hackathon 2026 (ISRO/SAC). If you are an agent working in
this repo, this file is your contract — read it before doing anything.

## Layout

- `demo/` — deterministic tools, planner, pipeline, evidence packet,
  report, prepared scenes, tests (the product core)
- `api/` — FastAPI service (`:8000`) wrapping the pipeline
- `web/` — React/Vite frontend (`:5173`)
- `ops/` — private orchestrator board (gitignored): bootstraps, queue,
  decision log, locks
- `strategy/` — master plan + research
- `gates/_cache/` — frozen eval ids, checkpoints, caches (never commit)

## Rules that never expire

1. **Tools own numbers; models narrate.** No model output may invent a
   measurement. `demo/report.py::check_narration` enforces this.
2. **Withhold instead of inventing.** `withheld_*`/`not_determined` are
   correct outputs, not bugs.
3. **No training/tuning on eval/test ids.** Ids in
   `gates/_cache/eval_ids/` are frozen before any preds. Assert zero
   overlap before any training run.
4. **No commercial vision APIs at runtime; product stays open-weight +
   offline-capable** — a **team rule**, not a PS clause (the PS mandates
   only open-source *training data*). Chosen for demo resilience
   (airplane-mode proof) and weight ownership.
5. **Hash pins:** `demo/test_judge_kit.py` pins SHA256 of shared `demo/`
   files. An intentional change re-pins with
   `# <date> authorized <TASK>` — never silently.
6. **Commits:** repo owner is the sole contributor — no co-author or
   agent trailers. Never commit weights, `.env`, `ops/`,
   `gates/_cache/`. Never push without explicit approval.
7. **Input binding:** a run requires a bound `upload` or `scene`. Do not
   reintroduce silent input defaults in any client path.

## How to read verdicts in this repo

Any "dead / killed / never / do not" line you meet — in `ops/`,
`strategy/`, code comments, or a spec — is **evidence-scoped to the
role, column, and date it was written under**. It records an experiment
outcome, not a universal ban. `ops/DECISIONS.md` is the registry: each
verdict carries scope, evidence, and a revisit condition. If a task
depends on a verdict that looks stale or mis-scoped, flag it — don't
obey blindly and don't route around silently.

The symmetric rule: an **assumed constraint** is a hypothesis, not a
blocker. If a design choice is being gated by a cost nobody has
measured (latency, memory, "too heavy", "the PS forbids it"), measure
it or check the PS text before letting it steer the design. Prefer the
stronger architecture and verify the cost — do not pre-emptively pick
the weaker one to avoid a phantom penalty. (2026-09-24: model-primary
routing shipped over an unmeasured latency fear; the plan call turned
out near-instant. The fear was the only real cost.)

## Deeper context (read only what your task needs)

- Orchestrator/resurrection context: `ops/ORCHESTRATOR_BOOTSTRAP.md`
- Executor contract (spec lanes): `ops/EXECUTOR_BOOTSTRAP.md`
- Live queue + ruled verdicts: `ops/NEXT_TASKS.md` / `ops/PROJECT_LOG.md`
- Strategy: `strategy/MASTER_PLAN_V2.md`
- Current ground truth: `python ops/state.py`

## Verify before you claim

- Demo suite: `..\.venv\Scripts\python.exe -m unittest discover -s demo -p "test_*.py"`
- Planner routing: `..\.venv\Scripts\python.exe demo/planner.py`
- API tests: `..\.venv\Scripts\python.exe -m unittest discover -s api -p "test_*.py"` (pytest is not installed)
- Web: `cd web && npx vitest run && npm run build && npm run lint`
- Stack health: `curl -s http://127.0.0.1:8000/health`
