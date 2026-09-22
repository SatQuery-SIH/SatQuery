# SatQuery API — HTTP contract (API-0080)

Thin FastAPI layer over `demo.pipeline`. The API never computes remote-sensing
facts itself — `/upload` runs the real `pipeline.bind_inputs` + `ingest`
readers, `/query` calls `pipeline.run_query`. Server-down seats surface as
seat state (`up:false`), never as exceptions.

**Run** (from the SatQuery repo root):

```powershell
pip install -r api/requirements.txt
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Interactive docs at `/docs` (OpenAPI at `/openapi.json`).

## Config (env)

| var | default | meaning |
|---|---|---|
| `SATQUERY_NARRATOR_URL` | `http://127.0.0.1:8080` | narrator llama-server seat |
| `SATQUERY_CANONICAL_URL` | `http://127.0.0.1:8091` | canonical (adapted RSVQA) seat |
| `SATQUERY_SEAT_PROFILE` | `local` | active seat profile |
| `SATQUERY_DEVICE_CD` | `cpu` | device passed to `run_query` |
| `SATQUERY_CORS_ORIGINS` | localhost:5173 / 3000 | comma-separated CORS origins |
| `SATQUERY_MAX_UPLOAD_BYTES` | `524288000` (500 MB) | total per-request upload cap |
| `SATQUERY_DB` | `api/_data/satquery.sqlite3` | run-store DB path |
| `SATQUERY_DATA_DIR` | `api/_data` | uploaded originals live under `<dir>/uploads/<upload_id>/` |

## Endpoints

### `GET /health`

Probes both seats (`GET {url}/v1/models`, 2 s timeout). Always 200 while the
API process is up — `ok` is the API's own health; seat state is per-seat.

```json
{"ok": true,
 "seats": {
   "narrator":  {"url": "http://127.0.0.1:8080", "model": "<served id>|null", "up": true},
   "canonical": {"url": "http://127.0.0.1:8091", "model": "<served id>|null",
                 "gguf_sha256": "a8797686…", "up": false}}}
```

### `GET /seats`

Config view of the seat profile (no live probe — use `/health` for `up`/`model`).

```json
{"profile": "local", "seats": {"narrator": {"url": "…"}, "canonical": {"url": "…", "gguf_sha256": "…"}}}
```

### `POST /seats` — `{"profile": "local"|"cloud"}`

`local` → 200 `{"profile": "local", "seats": {…}}`.
`cloud` → **501** `{"error": "not_implemented", "detail": "cloud profile pending CLOUD-SEAT"}`
(honest stub; the cloud seat model is defined by the CLOUD-SEAT lane).

### `POST /upload` — multipart

Form fields: `mode` = `single` | `bi-temporal` | `optical+sar`, plus the files
either as **role-named fields** or an **ordered `files` list**:

| mode | roles (canonical order) |
|---|---|
| `single` | `image` |
| `bi-temporal` | `before`, `after` |
| `optical+sar` | `optical`, `sar` |

SAR accepts GeoTIFF / PNG / `.npz` (keys `vv`[, `vh`]). Runs the real
`bind_inputs` ingest on the stored files. 200:

```json
{"upload_id": "…", "workdir": "…/demo/data/_uploads/<stamp>",
 "detected": {"gsd_m": 10.0, "gsd_source": "geotransform",
              "crs": "LOCAL_CS[…]|EPSG:32645|null", "crs_note": "…|null",
              "bands": 1, "dtype": "uint8", "calibrated": null,
              "files": {"<role>": {"file","gsd_m","gsd_source","crs","crs_note",
                                   "bands","dtype","inspect_backend","provenance"}}},
 "warnings": ["…"], "ingest_note": "…"}
```

`detected` is flat per the contract; the nested `files` map is additive
per-file detail. `calibrated` is set for the SAR input (`optical+sar`),
`null` otherwise. GSD is the *used* value (native × resize factor when the
preview was downscaled) — exactly what `area_calc` will use.

Errors: `422` missing role (`incomplete_upload`) / bad `mode` (`bad_mode`) /
ingest failure (`ingest_failed`); `413` over the byte cap.

### `POST /query` — `{"query", "input_mode", "scene"?, "upload_id"?, "live"?}`

`upload_id` binds the stored upload's file paths (mode must match). With no
`upload_id`, `scene` picks a prepared scene (1–4; inferred from `input_mode`
when omitted). Returns the run bundle:

```json
{"run_id": "…", "answer": "…|null", "visible_answer": "…",
 "plan": {…}, "tool_outputs": {…}, "evidence_packet": {…}|null,
 "trace": {"query","input_mode","scene","live","first_token_s","complete_s",
           "gsd","misreg_check","misreg_shift_px","misregistration_note",
           "geo_exports","vlm","overlay_path","agreement_map_path","images",
           "metrics_badge","ingest_note","input_source","narration_check", …},
 "report": {"findings": "<md>", "measurement": "<md>", "confidence": "<md>",
            "limitations": […], "narration_check": {…}|null,
            "evidence_packet_error": "…|null"},
 "artifacts": [{"type": "image|overlay|geo_export", "path": "<abs>",
                "name": "<basename>", "url": "/artifacts/<run_id>/<name>"}]}
```

Errors: `404` unknown upload (`unknown_upload`), `410` upload files gone
(`upload_gone`), `422` mode mismatch (`mode_mismatch`), `500` pipeline
failure (`pipeline_error`). A refused/unsupported query is a **200** — check
`plan.supported` / `answer`, not the HTTP code.

### `POST /query/stream` — SSE

Same request body as `POST /query`. Response is `text/event-stream`
(`Cache-Control: no-cache`, `X-Accel-Buffering: no` — uvicorn does not buffer;
tell proxies not to). The run executes in a worker thread; stage events flow
through an `asyncio.Queue` to the stream in real time. **The bundle still
lands in the run store under `run_id`** — the stream is a live view of a
normal run, not a different kind.

Wire format — one block per event (`event:` name + `data:` JSON line):

```
event: stage
data: {"stage": "plan|bind|tool|packet|narration|done",
       "status": "start|done|fail|withheld", "ts": <perf_counter float>,
       "tool": "<plan tool name>"?, "data": {...}?}

event: done
data: {<full run bundle, identical shape to POST /query — includes run_id>}

event: error
data: {"error": "<slug>", "detail": "<message>"}
```

(`token` events are reserved — the narrator call streams chunks internally
but exposes no per-token hook, so narration currently emits start/done only.
That is a deliberate scope decision, not a bug.)

Exact sequences:

**Normal run** (live, supported):
```
stage plan/start → stage plan/done {supported, task, tools, vlm_role}
→ stage bind/start → stage bind/done {source, scene, gsd_m, gsd_source,
    misreg_shift_px, coreg_shift_px}
→ stage tool/start + tool/done {latency_s, …} per tool in plan.tools,
    in execution order (e.g. water_highlight, area_calc)
→ stage narration/start → narration/done {first_token_s, complete_s, model}
→ stage packet/start → packet/done {ok}
→ stage done/done {supported, tools, scene, input_source, complete_s,
    n_images, has_overlay, has_packet}
→ event done {bundle}
```

**Refusal run** (plan unsupported — e.g. counting on bi-temporal):
```
plan start/done{supported:false} → bind start/done → plan/withheld{refusal}
→ packet start/done → done/done → event done{bundle, plan.supported=false}
```

**Bind failure** (partial/corrupt upload): same but `bind/fail{error}`.

**Tool failure** (e.g. `change_detect` raises): `tool/start` → `tool/fail`
`{error}` then `event: error` — no `done` event, no bundle stored.

**Seat down**: seats never raise — `canonical_vqa` returns
`available:false` in its done payload; a dead narrator yields `narration/fail`
then `event: error` (the run itself fails: the narrator has no fallback).

Request-validation errors (bad `upload_id`, mode mismatch, gone files) are
ordinary JSON `{"error","detail"}` responses — they happen *before* the
stream starts.

### `GET /runs`

```json
[{"run_id": "…", "query": "…", "input_mode": "…",
  "ts": "<ISO-UTC>", "supported": true}]
```

Newest first. `?limit=` (default 200).

### `GET /runs/{run_id}`

The stored run bundle (same shape as `POST /query` response). `404` unknown id.

### `GET /artifacts/{run_id}/{name}`

Serves a file the run actually produced (previews, overlays, `*.tif` geo
exports). `name` must match a basename in the run's `artifacts` list; the
resolved path must stay inside the run's artifact root (the ingest workdir
for upload runs, the scene data dir for prepared runs). `400`/`403`/`404` on
bad name, escape, or unknown.

## Errors

All error responses share `{"error": "<slug>", "detail": "<message>"}` —
including validation failures (`422 validation_error`).

## Persistence

`store.py` defines a `RunStore` protocol (no SQL in handlers) with a stdlib
`sqlite3` backend. `run_id` → `{paths, sha256, bundle JSON}`; `upload_id` →
`{paths, sha256, detected, …}`. Files stay files; Postgres/PostGIS can
replace the backend for the deployed profile.

## Streaming notes

`POST /query/stream` ships in this pass (API-STREAM). Narrator **token**
events are intentionally not emitted yet: `tools.vqa` streams internally but
has no chunk callback, and rewiring it would touch a pinned file outside
this lane's scope. The narration stage emits start/done (with real
`first_token_s`/`complete_s`) — per-token SSE can follow if WEB-FRONTEND
needs it.
