"""FastAPI layer over demo.pipeline — thin HTTP contract surface (API-0080).

This layer never computes remote-sensing facts: /upload calls the real
pipeline.bind_inputs / ingest readers, /query calls pipeline.run_query, and
responses repackage what those return. Server-down seats surface as seat
state, not exceptions.

Run from the SatQuery repo root:  uvicorn api.main:app --port 8000
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import threading
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

try:
    # Imported as a package (uvicorn api.main:app from the repo root).
    from .schemas import (
        DetectedBlock,
        ErrorBody,
        HealthResponse,
        QueryRequest,
        RunBundle,
        RunListItem,
        SeatsResponse,
        SeatState,
        SeatSwitchRequest,
        UploadResponse,
    )
    from .store import DEFAULT_DB_PATH, SqliteStore
except ImportError:
    # Imported top-level (api/ on sys.path, e.g. `python api/test_api.py`).
    from schemas import (
        DetectedBlock,
        ErrorBody,
        HealthResponse,
        QueryRequest,
        RunBundle,
        RunListItem,
        SeatsResponse,
        SeatState,
        SeatSwitchRequest,
        UploadResponse,
    )
    from store import DEFAULT_DB_PATH, SqliteStore

API_DIR = Path(__file__).resolve().parent
SAT = API_DIR.parent
DEMO = SAT / "demo"
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))

# Seat constants are read lazily from demo/tools.py so this file stays correct
# if the pins move; fallbacks mirror the current pinned values.
NARRATOR_DEFAULT_URL = "http://127.0.0.1:8080"
CANONICAL_DEFAULT_URL = "http://127.0.0.1:8091"

ROLE_ORDER = {
    "single": ["image"],
    "bi-temporal": ["before", "after"],
    "optical+sar": ["optical", "sar"],
}
# Role -> key in bound["paths"] for the PNG bind_inputs materializes into the
# upload workdir (the same pixels the pipeline tools/VLM consume; no re-render).
PREVIEW_PATH_KEY = {
    "image": "image",
    "before": "before",
    "after": "after",
    "optical": "optical",
    "sar": "sar_vv",
}
# Role whose file anchors the flat `detected` block (mirrors which file the
# pipeline prefers for GSD: bi-temporal reads `after` first, then `before`).
PRIMARY_ROLE = {"single": "image", "bi-temporal": "after", "optical+sar": "optical"}

_ERROR_SLUGS = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "unprocessable",
    500: "internal_error",
    501: "not_implemented",
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_safe(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _seat_constants() -> dict[str, Any]:
    """Live seat pins from demo/tools.py (fallbacks match current pins)."""
    try:
        import tools

        return {
            "narrator_url": tools.DEFAULT_VLM_URL,
            "canonical_url": tools.CANONICAL_VLM_URL,
            "canonical_gguf": tools.CANONICAL_GGUF_SHA256,
        }
    except Exception:
        return {
            "narrator_url": NARRATOR_DEFAULT_URL,
            "canonical_url": CANONICAL_DEFAULT_URL,
            "canonical_gguf": None,
        }


def _config() -> dict[str, Any]:
    seats = _seat_constants()
    data_dir = Path(os.environ.get("SATQUERY_DATA_DIR", str(API_DIR / "_data")))
    return {
        "profile": os.environ.get("SATQUERY_SEAT_PROFILE", "local"),
        "narrator_url": os.environ.get("SATQUERY_NARRATOR_URL", seats["narrator_url"]),
        "canonical_url": os.environ.get("SATQUERY_CANONICAL_URL", seats["canonical_url"]),
        "canonical_gguf": seats["canonical_gguf"],
        "device_cd": os.environ.get("SATQUERY_DEVICE_CD", "cpu"),
        "max_upload_bytes": _env_int("SATQUERY_MAX_UPLOAD_BYTES", 500 * 1024 * 1024),
        "data_dir": data_dir,
        "uploads_dir": data_dir / "uploads",
        "cors_origins": [
            o.strip()
            for o in os.environ.get(
                "SATQUERY_CORS_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173,"
                "http://localhost:3000,http://127.0.0.1:3000",
            ).split(",")
            if o.strip()
        ],
    }


def _seat_views(cfg: dict[str, Any]) -> dict[str, SeatState]:
    return {
        "narrator": SeatState(url=cfg["narrator_url"]),
        "canonical": SeatState(
            url=cfg["canonical_url"], gguf_sha256=cfg["canonical_gguf"]
        ),
    }


def _probe_seat(url: str, timeout: float = 2.0) -> tuple[bool, str | None]:
    """GET {url}/v1/models. Never raises — a down seat is state, not an error."""
    try:
        with urllib.request.urlopen(
            url.rstrip("/") + "/v1/models", timeout=timeout
        ) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return True, ((body.get("data") or [{}])[0]).get("id")
    except Exception:
        return False, None


def _inspect_file(path: Path) -> dict[str, Any]:
    """File-metadata read (band count, dtype) — same class of fact as
    ingest.read_gsd's crs/native_width. No pixel math, no RS derivation."""
    suf = path.suffix.lower()
    if suf == ".npz":
        try:
            import numpy as np

            blob = np.load(path)
            keys = list(blob.files)
            first = blob[keys[0]] if keys else None
            return {
                "bands": len(keys) or None,
                "dtype": str(first.dtype) if first is not None else None,
                "backend": "numpy.npz",
            }
        except Exception as e:
            return {"bands": None, "dtype": None, "backend": f"npz failed: {e}"}
    if suf in {".tif", ".tiff"}:
        try:
            import rasterio

            with rasterio.open(path) as ds:
                return {
                    "bands": int(ds.count),
                    "dtype": str(ds.dtypes[0]) if ds.count else None,
                    "backend": "rasterio",
                }
        except Exception:
            pass
        try:
            import tifffile

            arr = tifffile.imread(str(path))
            bands = 1
            if arr.ndim == 3:
                bands = int(arr.shape[-1] if arr.shape[-1] <= 4 else arr.shape[0])
            return {"bands": bands, "dtype": str(arr.dtype), "backend": "tifffile"}
        except Exception:
            pass
    try:
        from PIL import Image

        with Image.open(path) as im:
            mode_dtype = {"I": "int32", "F": "float32", "I;16": "uint16"}
            return {
                "bands": len(im.getbands()),
                "dtype": mode_dtype.get(im.mode, "uint8"),
                "backend": "pil",
            }
    except Exception as e:
        return {"bands": None, "dtype": None, "backend": f"unreadable: {e}"}


def _safe_name(filename: str | None, role: str) -> str:
    name = Path(str(filename or "")).name.strip()
    if not name or name in {".", ".."}:
        name = role
    return name


def _error(status: int, detail: str, slug: str | None = None) -> HTTPException:
    return HTTPException(
        status_code=status,
        detail={"slug": slug or _ERROR_SLUGS.get(status, "error"), "msg": detail},
    )


def _detected_block(
    mode: str,
    bound: dict[str, Any],
    saved: dict[str, Path],
    warnings: list[str],
) -> dict[str, Any]:
    """Flat judge-facing block from real ingest output + per-file detail."""
    from ingest import read_gsd

    gsd = bound.get("gsd") or {}
    files: dict[str, Any] = {}
    for role, p in saved.items():
        g = read_gsd(p)
        meta = _inspect_file(p)
        files[role] = {
            "file": p.name,
            "gsd_m": g.get("gsd_m"),
            "gsd_source": g.get("source"),
            "crs": g.get("crs"),
            "crs_note": g.get("crs_note"),
            "bands": meta.get("bands"),
            "dtype": meta.get("dtype"),
            "inspect_backend": meta.get("backend"),
            "provenance": g.get("provenance"),
        }
        if g.get("gsd_m") is None:
            warnings.append(f"{role} `{p.name}`: {g.get('provenance')}")
        elif g.get("source") == "assumed_meters":
            warnings.append(f"{role} `{p.name}`: {g.get('provenance')}")
        if g.get("crs_note"):
            warnings.append(f"{role} `{p.name}`: {g['crs_note']}")

    if bound.get("misregistration_note"):
        warnings.append(f"misregistration: {bound['misregistration_note']}")
    coreg = bound.get("coreg") or {}
    pix = coreg.get("pixel") or {}
    if coreg and pix.get("status") != "measured":
        warnings.append(
            f"coreg pixel-shift not measured ({pix.get('reason', 'unknown')}); "
            "cross-modal overlay alignment not verified"
        )
    sar = bound.get("sar_arrays") or {}
    if mode == "optical+sar":
        if sar.get("provenance"):
            warnings.append(f"sar: {sar['provenance']}")
        if sar.get("calibrated") is False:
            warnings.append(
                "sar input is not calibrated backscatter (preview DN); "
                "dB thresholds not applied"
            )

    primary = saved.get(PRIMARY_ROLE[mode])
    meta = _inspect_file(primary) if primary else {}
    detected = DetectedBlock(
        gsd_m=gsd.get("gsd_m"),
        gsd_source=str(gsd.get("source") or "none"),
        crs=gsd.get("crs"),
        crs_note=gsd.get("crs_note"),
        bands=meta.get("bands"),
        dtype=meta.get("dtype"),
        calibrated=sar.get("calibrated") if mode == "optical+sar" else None,
    )
    return {**detected.model_dump(), "files": files}


def _collect_artifacts(trace: dict[str, Any], run_id: str) -> list[dict[str, str]]:
    items: list[tuple[str, str]] = []
    for p in trace.get("images") or []:
        items.append(("image", str(p)))
    for key in ("overlay_path", "agreement_map_path"):
        if trace.get(key):
            items.append(("overlay", str(trace[key])))
    for name, rec in (trace.get("geo_exports") or {}).items():
        if isinstance(rec, dict) and rec.get("status") == "written" and rec.get("path"):
            items.append(("geo_export", str(rec["path"])))

    seen_paths: set[str] = set()
    seen_names: dict[str, int] = {}
    out: list[dict[str, str]] = []
    for kind, p in items:
        rp = str(Path(p).resolve())
        if rp in seen_paths:
            continue
        seen_paths.add(rp)
        base = Path(rp).name
        n = seen_names.get(base, 0)
        seen_names[base] = n + 1
        name = base if n == 0 else f"{n}_{base}"
        out.append(
            {
                "type": kind,
                "path": rp,
                "name": name,
                "url": f"/artifacts/{run_id}/{name}",
            }
        )
    return out


def _artifact_root(artifacts: list[dict[str, str]]) -> str | None:
    dirs = [str(Path(a["path"]).parent) for a in artifacts]
    if not dirs:
        return None
    try:
        return os.path.commonpath(dirs)
    except ValueError:
        return None


def _report_block(trace: dict[str, Any]) -> dict[str, Any]:
    from report import confidence_markdown, findings_header, measurement_markdown

    packet = trace.get("evidence_packet") or {}
    limitations = list(packet.get("limitations") or [])
    if trace.get("evidence_packet_error"):
        limitations.append(f"evidence_packet_error: {trace['evidence_packet_error']}")
    return {
        "findings": findings_header(trace),
        "measurement": measurement_markdown(trace),
        "confidence": confidence_markdown(trace),
        "limitations": limitations,
        "narration_check": trace.get("narration_check"),
    }


def create_app(store: Any = None) -> FastAPI:
    cfg = _config()
    app = FastAPI(title="SatQuery AI API", version="0.1.0")
    app.state.cfg = cfg
    app.state.store = store if store is not None else SqliteStore(
        os.environ.get("SATQUERY_DB", str(DEFAULT_DB_PATH))
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg["cors_origins"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(HTTPException)
    async def _http_exc(_req: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict) and "msg" in detail:
            slug, msg = detail.get("slug", "error"), detail["msg"]
        else:
            slug, msg = _ERROR_SLUGS.get(exc.status_code, "error"), str(detail)
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorBody(error=slug, detail=msg).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def _val_exc(_req: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ErrorBody(
                error="validation_error", detail=str(exc)
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def _any_exc(_req: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=ErrorBody(
                error="internal_error", detail=f"{type(exc).__name__}: {exc}"
            ).model_dump(),
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> dict[str, Any]:
        seats = _seat_views(cfg)
        for name, seat in seats.items():
            up, model = _probe_seat(seat.url)
            seat.up = up
            seat.model = model
        return {"ok": True, "seats": seats}

    @app.get("/seats", response_model=SeatsResponse)
    def get_seats() -> dict[str, Any]:
        return {"profile": cfg["profile"], "seats": _seat_views(cfg)}

    @app.post("/seats", response_model=SeatsResponse)
    def switch_seats(req: SeatSwitchRequest) -> dict[str, Any]:
        if req.profile == "cloud":
            raise _error(501, "cloud profile pending CLOUD-SEAT")
        cfg["profile"] = "local"
        return {"profile": cfg["profile"], "seats": _seat_views(cfg)}

    @app.post("/upload", response_model=UploadResponse)
    async def upload(request: Request) -> dict[str, Any]:
        form = await request.form()
        mode = str(form.get("mode") or "")
        roles = ROLE_ORDER.get(mode)
        if roles is None:
            for _, f in form.multi_items():
                if isinstance(f, StarletteUploadFile):
                    await f.close()
            raise _error(
                422,
                f"mode must be one of {sorted(ROLE_ORDER)}; got {mode!r}",
                slug="bad_mode",
            )
        by_role: dict[str, StarletteUploadFile] = {}
        all_files = [v for v in form.multi_items() if isinstance(v[1], StarletteUploadFile)]
        for role in roles:
            f = form.get(role)
            if isinstance(f, StarletteUploadFile):
                by_role[role] = f
        generic = [
            f for f in form.getlist("files") if isinstance(f, StarletteUploadFile)
        ]
        for i, role in enumerate(roles):
            if role not in by_role and i < len(generic):
                by_role[role] = generic[i]
        missing = [r for r in roles if r not in by_role]
        if missing:
            for _, f in all_files:
                await f.close()
            raise _error(
                422,
                f"upload incomplete for {mode}: missing file role(s) {missing}. "
                f"Send role-named fields ({', '.join(roles)}) or an ordered "
                "`files` list.",
                slug="incomplete_upload",
            )

        upload_id = uuid.uuid4().hex
        udir = cfg["uploads_dir"] / upload_id
        udir.mkdir(parents=True, exist_ok=True)
        cap = int(cfg["max_upload_bytes"])
        saved: dict[str, Path] = {}
        sha: dict[str, str] = {}
        names: dict[str, str] = {}
        total = 0
        for role, uf in by_role.items():
            names[role] = _safe_name(uf.filename, role)
            dest = udir / f"{role}__{names[role]}"
            h = hashlib.sha256()
            ok = False
            try:
                with dest.open("wb") as fh:
                    while True:
                        chunk = await uf.read(1024 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > cap:
                            raise _error(
                                413,
                                f"uploads exceed cap ({cap} bytes; "
                                "SATQUERY_MAX_UPLOAD_BYTES)",
                            )
                        h.update(chunk)
                        fh.write(chunk)
                ok = True
            finally:
                await uf.close()
                if not ok:
                    dest.unlink(missing_ok=True)
            saved[role] = dest
            sha[role] = h.hexdigest()

        import pipeline

        bound = pipeline.bind_inputs(
            mode, uploads={r: str(p) for r, p in saved.items()}
        )
        if not bound.get("ok"):
            raise _error(
                422,
                str(bound.get("error") or "ingest failed"),
                slug="ingest_failed",
            )
        warnings: list[str] = []
        detected = _detected_block(mode, bound, saved, warnings)
        # dedupe, order-preserving
        warnings = list(dict.fromkeys(warnings))
        workdir = bound.get("workdir")
        preview_paths: dict[str, str] = {}
        if workdir:
            wd_root = Path(workdir).resolve()
            bound_paths = bound.get("paths") or {}
            for role in ROLE_ORDER[mode]:
                p = bound_paths.get(PREVIEW_PATH_KEY[role])
                if not p:
                    continue
                rp = Path(p).resolve()
                if rp.is_file() and rp.is_relative_to(wd_root):
                    preview_paths[role] = str(rp)
        record = {
            "upload_id": upload_id,
            "ts": _now(),
            "mode": mode,
            "paths": {r: str(p) for r, p in saved.items()},
            "filenames": names,
            "sha256": sha,
            "workdir": str(workdir) if workdir else None,
            "detected": detected,
            "warnings": warnings,
            "ingest_note": bound.get("ingest_note"),
            "previews": preview_paths,
        }
        app.state.store.put_upload(record)
        return {
            "upload_id": upload_id,
            "workdir": str(workdir) if workdir else None,
            "detected": detected,
            "warnings": warnings,
            "ingest_note": bound.get("ingest_note"),
            "previews": [
                {"role": r, "url": f"/uploads/{upload_id}/preview/{r}"}
                for r in preview_paths
            ],
        }

    def _resolve_uploads(req: QueryRequest) -> dict[str, str] | None:
        """upload_id -> stored role paths. HTTP errors before any run starts."""
        if not req.upload_id:
            return None
        rec = app.state.store.get_upload(req.upload_id)
        if rec is None:
            raise _error(
                404, f"unknown upload_id {req.upload_id!r}", slug="unknown_upload"
            )
        if rec.get("mode") != req.input_mode:
            raise _error(
                422,
                f"upload {req.upload_id} is mode "
                f"{rec.get('mode')!r}, not {req.input_mode!r}",
                slug="mode_mismatch",
            )
        uploads = dict(rec.get("paths") or {})
        missing = [r for r, p in uploads.items() if not Path(p).is_file()]
        if missing:
            raise _error(
                410,
                f"uploaded file(s) no longer on disk for roles {missing}",
                slug="upload_gone",
            )
        return uploads

    def _run_and_store(
        req: QueryRequest,
        uploads: dict[str, str] | None,
        on_event: Any = None,
    ) -> dict[str, Any]:
        import pipeline

        try:
            trace = pipeline.run_query(
                req.query,
                req.input_mode,
                scene=req.scene,
                live=req.live,
                vlm_url=cfg["narrator_url"],
                device_cd=cfg["device_cd"],
                uploads=uploads,
                on_event=on_event,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise _error(500, f"run_query failed: {type(e).__name__}: {e}",
                         slug="pipeline_error")

        run_id = uuid.uuid4().hex
        trace = _json_safe(trace)
        artifacts = _collect_artifacts(trace, run_id)
        bundle = {
            "run_id": run_id,
            "answer": trace.get("answer"),
            "visible_answer": trace.get("visible_answer"),
            "plan": trace.get("plan") or {},
            "tool_outputs": trace.get("tool_outputs") or {},
            "evidence_packet": trace.get("evidence_packet"),
            "trace": {
                k: v
                for k, v in trace.items()
                if k
                not in {
                    "answer",
                    "visible_answer",
                    "plan",
                    "tool_outputs",
                    "evidence_packet",
                }
            },
            "report": _report_block(trace),
            "artifacts": artifacts,
        }
        record = {
            "run_id": run_id,
            "ts": _now(),
            "query": req.query,
            "input_mode": req.input_mode,
            "scene": trace.get("scene"),
            "upload_id": req.upload_id,
            "live": req.live,
            "supported": (trace.get("plan") or {}).get("supported"),
            "artifact_root": _artifact_root(artifacts),
            "artifacts": artifacts,
            "sha256": hashlib.sha256(
                json.dumps(bundle, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest(),
            "bundle": bundle,
        }
        app.state.store.put_run(record)
        return bundle

    @app.post("/query", response_model=RunBundle)
    def query(req: QueryRequest) -> dict[str, Any]:
        uploads = _resolve_uploads(req)
        return _run_and_store(req, uploads)

    @app.post("/query/stream")
    async def query_stream(req: QueryRequest) -> StreamingResponse:
        # Validation/lookup errors are plain JSON errors before streaming starts.
        uploads = _resolve_uploads(req)
        q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        _SENTINEL = object()

        def emit(item: Any) -> None:
            try:
                loop.call_soon_threadsafe(q.put_nowait, item)
            except Exception:
                pass

        def work() -> None:
            try:
                bundle = _run_and_store(req, uploads, on_event=emit)
                emit({"_sse": "done", "data": bundle})
            except HTTPException as e:
                d = e.detail
                msg = d.get("msg") if isinstance(d, dict) else str(d)
                emit({"_sse": "error", "data": {"error": "pipeline_error", "detail": msg}})
            except Exception as e:
                emit(
                    {
                        "_sse": "error",
                        "data": {
                            "error": "pipeline_error",
                            "detail": f"{type(e).__name__}: {e}",
                        },
                    }
                )
            finally:
                emit(_SENTINEL)

        async def gen():
            threading.Thread(target=work, daemon=True).start()
            while True:
                item = await q.get()
                if item is _SENTINEL:
                    break
                if "_sse" in item:
                    name, data = item["_sse"], item["data"]
                else:
                    name = item.get("event", "stage")
                    data = {k: v for k, v in item.items() if k != "event"}
                yield f"event: {name}\ndata: {json.dumps(data, default=str)}\n\n"
                if name in ("done", "error"):
                    break

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.get("/runs", response_model=list[RunListItem])
    def list_runs(limit: int = 200) -> list[dict[str, Any]]:
        return app.state.store.list_runs(limit=limit)

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        rec = app.state.store.get_run(run_id)
        if rec is None:
            raise _error(404, f"unknown run_id {run_id!r}", slug="unknown_run")
        return rec.get("bundle") or {}

    @app.get("/artifacts/{run_id}/{name}")
    def get_artifact(run_id: str, name: str) -> FileResponse:
        if not name or name != Path(name).name or name in {".", ".."}:
            raise _error(400, f"bad artifact name {name!r}", slug="bad_name")
        rec = app.state.store.get_run(run_id)
        if rec is None:
            raise _error(404, f"unknown run_id {run_id!r}", slug="unknown_run")
        entry = next(
            (a for a in rec.get("artifacts") or [] if a.get("name") == name), None
        )
        if entry is None:
            raise _error(
                404, f"no artifact {name!r} on run {run_id}", slug="unknown_artifact"
            )
        path = Path(entry["path"]).resolve()
        root = rec.get("artifact_root") or str(Path(entry["path"]).parent)
        root_p = Path(root).resolve()
        if not path.is_relative_to(root_p):
            raise _error(403, "artifact escapes run workdir", slug="escape")
        if not path.is_file():
            raise _error(
                404, f"artifact {name!r} missing on disk", slug="artifact_gone"
            )
        return FileResponse(path)

    @app.get("/uploads/{upload_id}/preview/{role}")
    def get_upload_preview(upload_id: str, role: str) -> FileResponse:
        rec = app.state.store.get_upload(upload_id)
        if rec is None:
            raise _error(
                404, f"unknown upload_id {upload_id!r}", slug="unknown_upload"
            )
        previews = rec.get("previews") or {}
        if role not in previews:
            raise _error(
                404,
                f"no preview {role!r} on upload {upload_id}",
                slug="unknown_preview",
            )
        path = Path(previews[role]).resolve()
        workdir = rec.get("workdir")
        if not workdir or not path.is_relative_to(Path(workdir).resolve()):
            raise _error(403, "preview escapes upload workdir", slug="escape")
        if not path.is_file():
            raise _error(
                404, f"preview {role!r} missing on disk", slug="preview_gone"
            )
        return FileResponse(path, media_type="image/png")

    return app


app = create_app()
