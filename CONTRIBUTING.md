# Contributing to SatQuery AI

## The idea in one paragraph

`main` is the protected product branch, controlled by the repo owner. **Everyone else works on their own branch and never pushes to `main`.** You can create as many branches and sub-branches as you want, commit and push to them as often as you like, and experiment freely — but the only way code reaches `main` is a **pull request that the owner reviews and squash-merges**. That gives you maximum freedom to try things, and keeps `main` clean, readable, and always shippable for the demo.

---

## 1. The workflow (TL;DR)

1. **Sync first.** Update your branch from `main` before starting work (see §3).
2. **Branch** from up-to-date `main` with a descriptive name (§4).
3. **Commit freely** on your branch — it is your sandbox.
4. **Re-sync** before opening the PR (§2).
5. **Open a Pull Request** into `main`, with a description that answers *what / why / how tested*.
6. **Owner reviews** → **squash-merges** → your branch is auto-deleted. Repeat.

> You never push to `main` directly. GitHub will reject it, and that is intentional.

---

## 2. Why `main` is locked the way it is

The protection settings are not bureaucracy — they make one person accountable for what ships:

| Setting | Effect |
|---|---|
| Require a pull request before merging | No direct pushes to `main` by anyone but the owner |
| Squash merging **only** | Your 30 messy experiment commits become **one clean commit** on `main` |
| Required approvals: 1 | The owner reviews every PR before it lands |
| Always suggest updating branches | GitHub offers you the "sync" button when your branch is stale |
| Auto-delete head branches | Merged branches disappear — no stale-branch graveyard |
| Restrict pushes / force pushes / deletions on `main` | Nobody can rewrite or destroy the product line |
| Owner bypass | Emergencies and hotfixes stay possible |

Practical result: **you** open the PR → **owner** approves → **owner** squash-merges. Your own PRs need either a teammate's approval or the owner's admin bypass.

---

## 3. Sync before you open a PR (mandatory)

Stale branches cause conflicts and wasted reviews. Before opening (or updating) a PR:

- **Easiest:** on the PR page, click **"Update branch"** when GitHub suggests it.
- Or locally:

```bash
git fetch origin
git rebase origin/main      # or: git merge origin/main
# resolve conflicts, then:
git push --force-with-lease  # to YOUR branch only — this is allowed
```

`--force-with-lease` on **your own feature branch** is fine. Force-pushing `main` is impossible (protection), and force-pushing shared branches is not OK.

---

## 4. Branch naming

Use one of these prefixes so the PR list stays readable:

| Prefix | For |
|---|---|
| `feat/<short-name>` | New demo/UI/tool features |
| `fix/<short-name>` | Bug fixes |
| `docs/<short-name>` | Documentation-only changes |
| `exp/<short-name>` | Experiments (eval probes, analysis, spikes) |

Examples: `feat/sar-overlay-tab`, `fix/trace-panel-crash`, `docs/faq-update`, `exp/looking-blank-recheck`.

---

## 5. Commit messages

Short, imperative, and prefixed:

```text
feat: add SAR overlay toggle to scene 3
fix: planner no longer routes two-image queries to caption
docs: clarify attach bars in Team Brief
```

Squash-merging means only the **PR title** appears on `main` — so write PR titles like the lines above.

---

## 6. PR checklist (the owner reviews against this)

- [ ] **One concern per PR.** Demo vs eval vs docs — don't mix.
- [ ] **Description answers:** what changed, why, how it was tested.
- [ ] **Evidence included** if you touched measurable paths (`python demo/test_harden.py`, `python eval/score.py` — both run from `SatQuery/`).
- [ ] **Branch is up to date** with `main` (the Update-branch button is green).
- [ ] **No new binary files** (see §8).
- [ ] **No secrets or absolute paths** (`C:\Users\...`, Wi-Fi names, tokens) anywhere in the diff.

**What the owner checks before merging:** does it break the zero-shot demo? Does it touch `serve.ps1` or the locked docs? Is any evidence artifact missing? Does it leak the eval ids or invent training data?

---

## 7. Project-specific rules (these are not bureaucracy either)

- **No commercial/proprietary vision APIs.** Open-weight models only, everything runs locally.
- **Never train on the frozen eval ids** — `gates/baseline_eval_ids.json` and `gates/cdvqa_eval_ids.json`. They are deliberately public and permanently quarantined; training on them poisons every score.
- **Do not attach any adapter to `demo/serve.ps1`** unless the attach bars have passed **on disk**. The narrator stays zero-shot until then (see `docs/SIH26167_Final_Plan.md` §2.3).
- **Do not start a second large training job** on your own Modal account. GPU strategy is coordinated with the owner — parallel duplicate runs waste both wallets.
- **Do not `modal run` anything from `hunt/` or `train10/`** in a clone — those are owner-laptop jobs.
- **Public docs (`docs/`, `README.md`) are locked product facts.** Improving their *clarity* is welcome (PRs do get merged); changing *facts* (dates, scores, attach bars) requires the owner's explicit go.

---

## 8. Never commit these

The `.gitignore` already blocks most of this — if `git status` shows any of it, stop and check you did not disable the ignore:

- Model weights / GGUF / safetensors / checkpoints, `gates/_cache/`, `gates/qwen3vl/`
- `.env`, `.secrets/`, tokens, `~/.modal.toml`
- `demo/data/`, `demo/traces/`, `demo/reports/`, `archive/`, `teammate_pack/`
- `PROJECT_LOG.md`, `ORCHESTRATOR_*`, `EXECUTOR_*` (owner-laptop IPC)
- Logs (`*.log`), PIDs, zips

Rule of thumb: **if it is over ~1 MB or was generated by a run, it does not belong in git.**

---

## 9. Quick answers

**"Can I push to `main` directly?"** No — protection blocks it. Open a PR; it takes one extra minute and keeps `main` auditable.

**"Can I approve my own PR?"** No. As a teammate you open and wait; the owner approves. The owner's own PRs get a teammate approval or the admin bypass.

**"My PR has conflicts."** Sync first (§3), resolve locally, `git push --force-with-lease` to your branch, comment that you re-based.

**"I want to rewrite history on my feature branch."** Fine — your branch, your rules, `--force-with-lease` it. Just never on `main`.

**"I found a secret in an old commit."** Tell the owner immediately — do not open a PR that quotes it.
