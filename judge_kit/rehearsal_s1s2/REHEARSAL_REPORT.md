# REHEARSAL — s1s2_india cross-modal dataset

Date: 2026-09-15 (UTC run timestamps in artifact names). Lane: existing demo
lane. Cost: $0 (local CPU + two local llama-servers). No commits.

Proved the real upload path end-to-end on real Cartosat/RISAT-like proxy
GeoTIFFs: **upload → ingest → plan → tools → packet → narrate**.

## Rehearsal set (14 pairs, stratified)

One patch per parent cell, even (lon+lat) stride within each split quota
(train 6 / val 4 / test 4). Selection in `patches.json`; reproducer in
`run_rehearsal.py::pick_patches`.

| patch_id | split | lon | lat | s1/s2 dates |
|---|---|---|---|---|
| cell_25725_p0015 | train | 77.93 | 8.81 | same-day |
| cell_07835_p0015 | train | 71.13 | 21.81 | ~1 d apart |
| cell_11582_p0015 | train | 72.53 | 25.51 | ~0.5 d |
| cell_32465_p0015 | train | 80.43 | 20.31 | ~1 d |
| cell_21156_p0011 | train | 76.13 | 28.91 | ~3.5 d |
| cell_30439_p0011 | train | 79.63 | 29.71 | same-day |
| cell_23359_p0015 | val | 77.03 | 10.71 | same-day |
| cell_17093_p0015 | val | 74.63 | 20.11 | same-day |
| cell_30332_p0015 | val | 79.63 | 19.01 | ~1 d |
| cell_48387_p0015 | val | 86.43 | 22.51 | ~1.5 d |
| cell_18886_p0015 | test | 75.33 | 13.91 | ~1 d |
| cell_19972_p0015 | test | 75.73 | 16.51 | ~1 d |
| cell_16356_p0011 | test | 74.33 | 25.91 | ~0.5 d |
| cell_30118_p0015 | test | 79.53 | 24.11 | ~4 d |

s1_date ≈ s2_date on every row → **same-date pairs, never fed to
`bi-temporal`** (held — see bars).

## Per-check results

| # | check | result | evidence |
|---|---|---|---|
| 1 | GSD `source=geotransform`, 10.0 m | **PASS 14/14** | `ingest_checks.json` — every optical and sar bind |
| 2 | CRS recorded | **PASS 14/14** | EPSG:32642/32643/32644/32645 read per-file (see surprise S1) |
| 3 | `crs_note` is `None` on proper EPSG | **PASS 14/14** | TIFF-GSD-FIX contract holds on real data |
| 4 | 4-band uint16 optical preview sane | **PASS 14/14** | PNG means 52.8–180.5, std 48–70, none all-black; bands 1–3 (R,G,B) land in PNG, NIR band 4 excluded by `_load_rgb_array` |
| 5 | float64 dB SAR ingested as calibrated | **PASS 14/14** | `calibrated=True`, dtype float32, VV ∈ [-50, +11.6] dB, VH ∈ [-51.4, -0.14] dB, `vv_eq_vh=False` (real 2-band, channel-last → [0]=VV, [1]=VH) |
| 6 | SAR VV preview sane | **PASS 14/14** | PNG means 125–209, std 47–62, none all-black |
| 7 | single-image question → `vqa` + `canonical_vqa` | **PASS 5/5** | runs s1–s5; canonical answers `yes`, `yes`, `small`, `rural`, `0`; latency 0.7–4.8 s |
| 8 | `canonical_answer` typed claim + provenance | **PASS 5/5** | claim carries model id, gguf sha, seat :8091, decode contract (excerpt below) |
| 9 | narrator stays on :8080 | **PASS 14/14** | `trace.vlm.url == http://127.0.0.1:8080` on every narrated run |
| 10 | optical+sar → `sar_read` + `sar_agreement` typed verdicts | **PASS 6/6** | x1–x6; water verdicts `disagree` ×5 / `optical_only` ×1; `built_up` always `withheld_no_tool` |
| 11 | agreement-map artifact produced | **PASS 6/6** | `agreement_map_live.png` written in each upload workdir, `trace.agreement_map_path` + `trace.images` |
| 12 | real-GSD area claim on sar pair | **PASS** | x3: `area_m2=…`, `area_km2`, basis `count(mask>0) * gsd_m^2` at GSD 10.0 |
| 13 | bi-temporal count on same-date pair → refuse | **PASS** | plan-level refusal, `supported=False`, zero tools — pair never sent as change input |
| 14 | mode/data mismatch (change q on single upload) | **PASS** | refused: "Change detection needs a before/after pair…" |
| 15 | mode mismatch (change q on optical+sar pair) | **PASS** | refused: "…the Bi-temporal tab (two optical dates)" |
| 16 | unsupported-evidence probe | **PASS (observed)** | "How many helicopters…" → `canonical_answer=1` labeled as model claim; narrator said "0" — each seat keeps its own provenance, no fabricated measurement |
| 17 | `validate_packet` clean | **PASS 14/14** | `issues=[]` on every run incl. withholds |
| 18 | `check_narration` clean | **FAIL 6/6 on sar runs** (5/5 single pass) | **Bug B1** — queued |
| 19 | packet/report states co-registration basis | **SILENT → finding F2** | `misreg_shift_px=None`, only `grid:"same_grid"` recorded |

## Packet excerpts (real data)

`canonical_answer` (s3, val patch cell_30332_p0015, "How large is the water
body in this image?"):

```json
{"predicate": "canonical_answer", "value": "small",
 "confidence": {"level": "measured",
  "basis": "adapted RSVQA model, deterministic greedy decode on 127.0.0.1:8091"},
 "provenance": {"tool": "canonical_vqa",
  "model": "canonical/Qwen3VL-8B-RSVQA-Q4_K_M.gguf",
  "gguf_sha256": "a87976860370bf657660ea3badeb59038de89aff1bb3e5b3dce4057d4e21a33f",
  "seat": "127.0.0.1:8091"}}
```

`sar_agreement` verdicts (x5, val patch cell_48387_p0015, "Highlight the
flooded region using SAR backscatter."):

```json
{"predicate": "water_agreement",   "value": "optical_only",
 "confidence": {"level": "measured", "basis": "sar_agreement verdict on common grid"}}
{"predicate": "built_up_agreement","value": "withheld_no_tool",
 "confidence": {"level": "withheld",  "basis": "no optical built-up tool and no calib…"}}
{"predicate": "agreement_iou",     "value": 0.011271,
 "confidence": {"level": "measured", "basis": "intersection/union on common grid"}}
```

Tool-side basis (x1): optical `water_fraction` 0.1890 (RGB-Otsu fallback),
SAR `water_fraction` 0.1417 (calibrated dB, VV < −16.0 + morph open),
IoU 0.1375 on the same 256×256 grid → typed `disagree`.

## Bugs / findings

- **B1 — `check_narration` false-positive on percent derivation (QUEUED, out
  of scope).** On all 6 optical+sar runs the narrator renders tool fractions
  as percentages (0.1417 → "14.17%"; 0.188995 → "18.90%"). The whitelist only
  admits tokens verbatim/decimal-compatible with the tool-JSON dump, so the
  derived percent is flagged `invented number`. The numbers are honest —
  arithmetically derived from measured tool output. Fix would touch
  `demo/report.py`, which is outside the conditional-fix allowlist
  (ingest.py only) — reported, not fixed.
- **F2 — no co-registration/alignment disclosure (FINDING, per spec).**
  `bind_inputs("optical+sar")` never runs a misregistration check;
  `sar_agreement` is invoked with `misreg_shift_px=None`, so its misreg gate
  is dormant. Packets record only `grid: "same_grid"`. Nothing states that
  pairs are pixel-aligned by construction or that alignment was checked.
  Feature not built here, as instructed.
- **O1 — `water_highlight` NIR disclosure (observed, working as designed).**
  These TIFFs carry no band descriptions → the tool says
  `NIR/green identities unknown` and falls back to RGB Otsu on (B−R).
  On weak/no-water scenes this yields `disagree`/`optical_only` verdicts —
  honest typed output, not silent agreement.

## Surprises

- **S1 — per-patch UTM zones.** The brief said EPSG:32642, but files carry
  the geographically correct zone per longitude (32642/32643/32644/32645 —
  UTM 42N–45N). The pipeline reads each file's real CRS; no action needed.
- **S2 — narrator/canonical divergence on the unsupported probe.** Narrator
  (:8080) answered "0" helicopters while canonical (:8091) answered "1".
  Each claim keeps its own seat provenance; the packet does not reconcile —
  disclosed, not merged.
- No ingest breakage: `demo/ingest.py` was **not modified**, so no re-pin and
  no suite tail is owed.

## Declared bars vs measured

| bar | status |
|---|---|
| same-date pairs never sent as bi-temporal | held — refusal at plan level; no same-date upload entered a change path |
| no fabricated numbers | held — all packet values tool- or model-owned; withheld stays withheld |
| withheld stays withheld | held — `built_up=withheld_no_tool` on all 6 sar runs; validator clean |
| servers stay up | held — `:8080` and `:8091` both HTTP 200 before and after |
| $0 local-only | held |
| no commits | held |

## Commands run

```text
curl :8080/v1/models && curl :8091/v1/models          # both 200 (before + after)
PYTHONUTF8=1 ../.venv/Scripts/python.exe judge_kit/rehearsal_s1s2/run_rehearsal.py
#   -> ingest_checks.json, patches.json, probes.json, runs_summary.json,
#      14 × {packet,trace}.json
# post-hoc: findings_header excerpts -> report_excerpts.md; sha256 -> artifact_hashes.json
```

## Artifacts

`judge_kit/rehearsal_s1s2/`: `patches.json`, `ingest_checks.json`,
`runs_summary.json`, `probes.json`, 14 `*.packet.json` + 14 `*.trace.json`,
`report_excerpts.md`, `artifact_hashes.json` (sha256 of every file),
`run_rehearsal.py` (reproducer). Upload workdirs with previews, overlays and
`agreement_map_live.png` persist under `demo/data/_uploads/20260914T*`.
