"""Run one demo query through planner -> tools -> VLM narration."""
from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ingest import (
    UPLOAD_DIR,
    _path as ingest_path,
    materialize_rgb,
    pair_misreg_fields,
    read_gsd,
    read_sar_arrays,
    scale_gsd_for_resize,
)
from planner import MODEL_FIRST, needs_fallback, plan, plan_model
from tools import (
    LEVIR_GSD_M,
    S2_GSD_M,
    SEMANTIC_UPLOAD_LIMITATION,
    UPLOAD_DOMAIN_LIMITATION,
    WATER_OVERLAY_RGB,
    _find_second_semantic_ckpt,
    area_calc,
    ask_vlm,
    attach_rung_display,
    canonical_vqa,
    cdvqa_map,
    change_detect,
    change_direction_proxy,
    coreg_check,
    export_mask_geotiff,
    load_mask,
    load_rgb,
    narration_prompt,
    overlay_mask,
    sar_agreement,
    sar_read,
    semantic_live_forward,
    vv_to_preview,
    water_highlight,
)

DEMO = Path(__file__).resolve().parent
DATA = DEMO / "data"

PREPARED_NOTE = (
    "Built-in examples below. Attach your own files on any tab to analyze them instead."
)
UPLOAD_NOTE = (
    "Analyzing your uploaded files — not the built-in examples."
)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: _jsonable(v)
            for k, v in obj.items()
            if k not in {"mask", "water_mask", "direction", "optical_mask", "sar_mask"}
        }
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def _with_visible(trace: dict[str, Any]) -> dict[str, Any]:
    # Lazy import: report.py imports PREPARED_NOTE from this module.
    from report import compose_visible_answer

    trace["visible_answer"] = compose_visible_answer(trace)
    return trace


def _attach_packet(
    trace: dict[str, Any],
    the_plan: dict[str, Any] | None,
    bound: dict[str, Any] | None,
) -> None:
    """Additive packet attach. On failure, set evidence_packet_error; do not raise."""
    try:
        from evidence_packet import build_packet_for_run

        trace["evidence_packet"] = build_packet_for_run(trace, the_plan, bound)
    except Exception as e:
        trace["evidence_packet_error"] = f"{type(e).__name__}: {e}"


def scene_paths(scene: int) -> dict[str, Path]:
    if scene == 1:
        man = json.loads((DATA / "scene1" / "manifest.json").read_text(encoding="utf-8"))
        img = DATA / "scene1" / man["primary"]
        return {"image": img}
    if scene == 2:
        return {
            "before": DATA / "scene2" / "before.png",
            "after": DATA / "scene2" / "after.png",
            "gt": DATA / "scene2" / "gt_mask.png",
            "pred": DATA / "scene2" / "pred_mask.png",
        }
    if scene == 3:
        return {
            "optical": DATA / "scene3" / "optical.png",
            "sar_vv": DATA / "scene3" / "sar_vv.png",
            "sar_vh": DATA / "scene3" / "sar_vh.png",
            "sar_npz": DATA / "scene3" / "sar_arrays.npz",
            "water": DATA / "scene3" / "water_mask.png",
        }
    if scene == 4:
        return {
            "before": DATA / "scene4" / "before.png",
            "after": DATA / "scene4" / "after.png",
            "gt": DATA / "scene4" / "gt_mask.png",
        }
    raise ValueError(f"unknown scene {scene}")


def _scene_manifest_domain(scene: int) -> str:
    """Router is the scene manifest domain field only. No ML classifier."""
    man_path = DATA / f"scene{int(scene)}" / "manifest.json"
    if man_path.is_file():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        d = man.get("domain")
        if d:
            return str(d).strip().lower()
    if int(scene) == 4:
        return "second"
    return "levir"


def _metrics_badge() -> dict[str, Any] | None:
    p = DATA / "scene2" / "levir_score.json"
    if not p.is_file():
        return None
    s = json.loads(p.read_text(encoding="utf-8"))
    return {
        "n": s.get("n"),
        "mean_iou": s.get("mean_iou"),
        "mean_f1": s.get("mean_f1"),
        "rung": s.get("rung"),
        "provenance": "tools.mask_metrics on fixed LEVIR-CD subset (not VLM)",
    }


def _present(val: Any) -> bool:
    p = ingest_path(val)
    return p is not None and p.is_file()


def _apply_misreg(bound: dict[str, Any], before: Any, after: Any) -> dict[str, Any]:
    """Additive: attach shift-check fields and append a note. No warping."""
    extra = pair_misreg_fields(before, after)
    bound.update(extra)
    note = extra.get("misregistration_note")
    if note:
        prev = (bound.get("ingest_note") or "").rstrip()
        bound["ingest_note"] = (prev + " " + note).strip()
    return bound


def _apply_coreg(
    bound: dict[str, Any],
    optical_src: Any,
    sar_src: Any,
    *,
    optical_rgb: Any = None,
    sar_vv: Any = None,
) -> dict[str, Any]:
    """Additive measured co-registration basis for optical+sar binds.

    Feeds the measured pixel-level shift into sar_agreement's misreg gate.
    Never realigns, resizes, or alters tool rasters.
    """
    cg = coreg_check(
        optical_src, sar_src, optical_rgb=optical_rgb, sar_vv=sar_vv
    )
    bound["coreg"] = cg
    bound["misreg_shift_px"] = cg.get("shift_px")
    return bound


def _geo_export(
    trace: dict[str, Any],
    bound: dict[str, Any],
    mask: Any,
    name: str,
    src_key: str,
    out_dir: Path,
) -> None:
    """Write <name> beside the PNG artifacts when the source raster is real.

    Falls back to the materialized path (PNG/npz) when no original exists —
    that exercises the recorded skip marker, never silence.
    """
    import numpy as np

    paths = bound.get("paths") or {}
    grids = bound.get("grids") or {}
    fallback = {
        "source_original": ("image", "optical"),
        "sar_original": ("sar_vv", "sar_npz"),
        "before_original": ("before",),
    }
    src = paths.get(src_key)
    dims_key = src_key
    if src is None:
        for alt in fallback.get(src_key, ()):
            if paths.get(alt) is not None:
                src = paths[alt]
                dims_key = alt
                break
    if src is None:
        rec = {"status": "skipped_no_source"}
    else:
        dims = grids.get(dims_key)
        m = np.asarray(mask)
        h, w = int(m.shape[-2]), int(m.shape[-1])
        sx = float(dims[1]) / w if dims else 1.0
        sy = float(dims[0]) / h if dims else 1.0
        rec = export_mask_geotiff(
            m, src, out_dir / name, scale_x=sx, scale_y=sy
        )
    rec = {"artifact": name, **rec}
    # Trace-level artifact metadata (like agreement_map_path) — NOT a tool
    # measurement, so it stays out of tool_outputs and its frozen hashes.
    trace.setdefault("geo_exports", {})[name] = rec


def _scene_for_mode(input_mode: str, scene: int | None) -> int:
    if scene in (1, 2, 3, 4):
        return scene
    if input_mode == "single":
        return 1
    if input_mode == "bi-temporal":
        return 2
    if input_mode == "optical+sar":
        return 3
    raise ValueError(f"cannot infer scene from mode {input_mode!r}")


def _workdir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    d = UPLOAD_DIR / stamp
    d.mkdir(parents=True, exist_ok=True)
    return d


def _gsd_pack(info: dict[str, Any], native_w: int | None, used_w: int | None) -> dict[str, Any]:
    if native_w and used_w:
        info = scale_gsd_for_resize(info, native_w, used_w)
    gsd_used = info.get("gsd_m_used", info.get("gsd_m"))
    return {
        **info,
        "gsd_m": gsd_used,
        "gsd_m_used": gsd_used,
    }


def bind_inputs(
    input_mode: str,
    scene: int | None = None,
    uploads: dict[str, Any] | None = None,
    sensor_profile: str | None = None,
) -> dict[str, Any]:
    """Resolve prepared scenes vs uploads. Partial uploads are an error, not a mix."""
    uploads = uploads or {}
    scene_n = _scene_for_mode(input_mode, scene)
    if input_mode == "single":
        keys = ("image",)
    elif input_mode == "bi-temporal":
        keys = ("before", "after")
    else:
        keys = ("optical", "sar")

    provided = {k: uploads.get(k) for k in keys if _present(uploads.get(k))}
    # Scene 3 also accepts sar_vv / sar_npz aliases.
    if input_mode == "optical+sar":
        if "sar" not in provided:
            for alt in ("sar_vv", "sar_npz"):
                if _present(uploads.get(alt)):
                    provided["sar"] = uploads.get(alt)
                    break

    if provided and len(provided) < len(keys):
        missing = [k for k in keys if k not in provided]
        return {
            "ok": False,
            "error": (
                f"Upload incomplete for {input_mode}: missing {missing}. "
                "Attach every file the tab needs, or clear uploads to use prepared scenes."
            ),
            "source": "upload",
            "scene": scene_n,
            "paths": {},
            "gsd": {},
            "use_prepared_cd_cache": False,
            "workdir": None,
            "ingest_note": UPLOAD_NOTE,
            "sar_arrays": None,
        }

    if not provided:
        paths = scene_paths(scene_n)
        domain = _scene_manifest_domain(scene_n)
        if scene_n == 2:
            gsd = {
                "gsd_m": LEVIR_GSD_M,
                "source": "benchmark_constant",
                "provenance": "LEVIR-CD documented 0.5 m GSD (prepared Scene 2 PNG).",
            }
            cache = True
        elif scene_n == 4:
            gsd = {
                "gsd_m": None,
                "source": "none",
                "provenance": "SECOND RGB PNG; GSD not claimed for this demo pair.",
            }
            cache = False
        elif scene_n == 3:
            gsd = {
                "gsd_m": S2_GSD_M,
                "source": "benchmark_constant",
                "provenance": "Sentinel-2 10 m GSD (prepared Scene 3 PNG).",
            }
            cache = False
        else:
            gsd = {
                "gsd_m": None,
                "source": "none",
                "provenance": "Scene 1 has no area_calc.",
            }
            cache = False
        out = {
            "ok": True,
            "error": None,
            "source": "prepared",
            "scene": scene_n,
            "paths": paths,
            "gsd": gsd,
            "use_prepared_cd_cache": cache,
            "cd_domain": domain,
            "workdir": None,
            "ingest_note": PREPARED_NOTE,
            "sar_arrays": None,
            "sensor_profile": sensor_profile,
        }
        if "before" in paths and "after" in paths:
            _apply_misreg(out, paths["before"], paths["after"])
        elif input_mode == "optical+sar" and "optical" in paths:
            import numpy as np

            vv = None
            try:
                if paths.get("sar_npz"):
                    vv = np.load(paths["sar_npz"])["vv"]
            except Exception:
                vv = None
            _apply_coreg(
                out,
                paths["optical"],
                paths.get("sar_npz") or paths.get("sar_vv"),
                optical_rgb=load_rgb(paths["optical"]),
                sar_vv=vv,
            )
        return out

    wd = _workdir()
    if input_mode == "single":
        mat = materialize_rgb(
            provided["image"], wd / "image.png", sensor_profile=sensor_profile
        )
        if not mat["ok"]:
            return {
                "ok": False,
                "error": mat["error"],
                "source": "upload",
                "scene": scene_n,
                "paths": {},
                "gsd": {},
                "use_prepared_cd_cache": False,
                "workdir": wd,
                "ingest_note": UPLOAD_NOTE,
                "sar_arrays": None,
            }
        gsd = _gsd_pack(read_gsd(provided["image"]), mat["native_width"], mat["used_width"])
        orig = ingest_path(provided["image"])
        return {
            "ok": True,
            "error": None,
            "source": "upload",
            "scene": scene_n,
            "paths": {"image": mat["path"], "source_original": orig},
            "grids": {
                "source_original": (mat["native_height"], mat["native_width"]),
            },
            "gsd": gsd,
            "use_prepared_cd_cache": False,
            "workdir": wd,
            "ingest_note": UPLOAD_NOTE + " " + mat["note"],
            "sar_arrays": None,
            "sensor_profile": sensor_profile,
        }

    if input_mode == "bi-temporal":
        b = materialize_rgb(
            provided["before"], wd / "before.png", sensor_profile=sensor_profile
        )
        a = materialize_rgb(
            provided["after"], wd / "after.png", sensor_profile=sensor_profile
        )
        if not b["ok"] or not a["ok"]:
            return {
                "ok": False,
                "error": b.get("error") or a.get("error"),
                "source": "upload",
                "scene": scene_n,
                "paths": {},
                "gsd": {},
                "use_prepared_cd_cache": False,
                "workdir": wd,
                "ingest_note": UPLOAD_NOTE,
                "sar_arrays": None,
            }
        if (b.get("used_height"), b.get("used_width")) != (
            a.get("used_height"),
            a.get("used_width"),
        ):
            return {
                "ok": False,
                "error": (
                    f"Bi-temporal dimensions differ "
                    f"(before {b.get('used_height')}x{b.get('used_width')} vs "
                    f"after {a.get('used_height')}x{a.get('used_width')}); "
                    "not co-registered. Upload a matching pair or clear uploads "
                    "to use prepared Scene 2."
                ),
                "source": "upload",
                "scene": scene_n,
                "paths": {},
                "gsd": {},
                "use_prepared_cd_cache": False,
                "workdir": wd,
                "ingest_note": UPLOAD_NOTE,
                "sar_arrays": None,
            }
        gsd_raw = read_gsd(provided["after"])
        if gsd_raw.get("gsd_m") is None:
            gsd_raw = read_gsd(provided["before"])
        gsd = _gsd_pack(gsd_raw, a["native_width"], a["used_width"])
        out = {
            "ok": True,
            "error": None,
            "source": "upload",
            "scene": scene_n,
            "paths": {
                "before": b["path"],
                "after": a["path"],
                "before_original": ingest_path(provided["before"]),
                "after_original": ingest_path(provided["after"]),
            },
            "grids": {
                "before_original": (b["native_height"], b["native_width"]),
                "after_original": (a["native_height"], a["native_width"]),
            },
            "gsd": gsd,
            "use_prepared_cd_cache": False,
            "workdir": wd,
            "ingest_note": UPLOAD_NOTE + " " + a["note"] + " " + UPLOAD_DOMAIN_LIMITATION + " " + SEMANTIC_UPLOAD_LIMITATION,
            "sar_arrays": None,
            "cd_domain": "levir",
            "sensor_profile": sensor_profile,
        }
        return _apply_misreg(out, b["path"], a["path"])

    # optical+sar
    opt = materialize_rgb(
        provided["optical"], wd / "optical.png", sensor_profile=sensor_profile
    )
    sar = read_sar_arrays(provided["sar"])
    if not opt["ok"] or not sar.get("ok"):
        return {
            "ok": False,
            "error": opt.get("error") or sar.get("error"),
            "source": "upload",
            "scene": scene_n,
            "paths": {},
            "gsd": {},
            "use_prepared_cd_cache": False,
            "workdir": wd,
            "ingest_note": UPLOAD_NOTE,
            "sar_arrays": None,
        }
    sar_preview = wd / "sar_vv.png"
    vv_to_preview(sar["vv"]).save(sar_preview)
    gsd_raw = read_gsd(provided["optical"])
    if gsd_raw.get("gsd_m") is None:
        gsd_raw = read_gsd(provided["sar"])
    gsd = _gsd_pack(gsd_raw, opt["native_width"], opt["used_width"])
    out = {
        "ok": True,
        "error": None,
        "source": "upload",
        "scene": scene_n,
        "paths": {
            "optical": opt["path"],
            "sar_vv": sar_preview,
            "source_original": ingest_path(provided["optical"]),
            "sar_original": ingest_path(provided["sar"]),
        },
        "grids": {
            "source_original": (opt["native_height"], opt["native_width"]),
            "sar_original": tuple(int(x) for x in sar["vv"].shape[:2]),
        },
        "gsd": gsd,
        "use_prepared_cd_cache": False,
        "workdir": wd,
        "ingest_note": UPLOAD_NOTE + " " + opt["note"] + " " + sar.get("provenance", ""),
        "sar_arrays": {
            "vv": sar["vv"],
            "vh": sar["vh"],
            "provenance": sar.get("provenance"),
            "calibrated": sar.get("calibrated", True),
            "pol_verified": sar.get("pol_verified"),
        },
        "sensor_profile": sensor_profile,
    }
    return _apply_coreg(
        out,
        provided["optical"],
        provided["sar"],
        optical_rgb=load_rgb(opt["path"]),
        sar_vv=sar["vv"],
    )


def _classify_semantic_family(query: str) -> str | None:
    # _cdvqa_family covers the six compiler families plus per-class binary
    # and the global with/without-change ratio so those questions still get
    # second_semantic evidence for cdvqa_map.
    from tools import _cdvqa_family

    return _cdvqa_family(query)


def _maybe_attach_scene4_semantic(
    *,
    query: str,
    bound: dict[str, Any],
    cd: dict[str, Any],
    paths: dict[str, Any],
    device_cd: str,
    numbers: dict[str, Any],
    trace: dict[str, Any],
) -> None:
    """Additive: second_semantic live only on prepared scene4 type-family after approval."""
    if int(bound.get("scene") or 0) != 4:
        return
    if bound.get("source") != "prepared":
        return
    if str(bound.get("cd_domain") or "").strip().lower() != "second":
        return
    family = _classify_semantic_family(query)
    if family is None:
        return
    ckpt = _find_second_semantic_ckpt(require_approved=True)
    if ckpt is None:
        return
    sem = semantic_live_forward(
        paths["before"],
        paths["after"],
        cd["mask"],
        device=device_cd,
        ckpt_path=ckpt,
        require_approved=True,
    )
    slim = {k: v for k, v in sem.items() if k not in {"cls_a", "cls_b", "mask"}}
    slim["family"] = family
    if family in (
        "change_to_what",
        "change_ratio_types",
        "largest_change",
        "smallest_change",
        "increase_or_not",
        "decrease_or_not",
    ):
        sat = DEMO.parent
        if str(sat) not in sys.path:
            sys.path.insert(0, str(sat))
        from cf_ft.semantic import compile_from_features, protocol_dict

        proto = protocol_dict()
        bins_path = sat / "gates" / "_cache" / "cf_ft" / "semantic_ratio_bins_v2.json"
        scores_path = sat / "gates" / "_cache" / "cf_ft" / "cdvqa_scores_sem_v2.json"
        if scores_path.is_file() and bins_path.is_file():
            scores = json.loads(scores_path.read_text(encoding="utf-8"))
            if scores.get("configured") == "v2":
                proto["ratio_bins"] = json.loads(bins_path.read_text(encoding="utf-8"))["ratio_bins"]
        pred, err = compile_from_features(sem.get("features") or {}, family, query, protocol=proto)
        slim["pred"] = pred
        slim["pred_error"] = err
    attach_rung_display(slim)
    jsonable = _jsonable(slim)
    cd["built_up_direction"] = sem.get("built_up_direction")
    cd["semantic_rung"] = "semantic_live"
    cd["semantic_checkpoint_sha256"] = sem.get("checkpoint_sha256")
    cd["buildings_a"] = sem.get("buildings_a")
    cd["buildings_b"] = sem.get("buildings_b")
    cd["dominant_transition"] = _jsonable(sem.get("dominant_transition"))
    numbers["semantic"] = jsonable
    trace["tool_outputs"]["semantic"] = jsonable


def _agreement_map_image(ag: dict[str, Any]):
    """RGB agreement map on sar_agreement's common grid.

    green=both masks, blue=optical only, orange=SAR only, dim gray=neither;
    when the water verdict is "disagree" the conflicting pixels render red.
    """
    import numpy as np
    from PIL import Image

    om = np.asarray(ag["optical_mask"]) > 0
    sm = np.asarray(ag["sar_mask"]) > 0
    both = om & sm
    only_o = om & ~sm
    only_s = sm & ~om
    rgb = np.full(om.shape + (3,), 45, dtype=np.uint8)
    rgb[both] = (30, 180, 60)
    rgb[only_o] = (60, 120, 230)
    rgb[only_s] = (240, 150, 30)
    if (ag.get("verdicts") or {}).get("water") == "disagree":
        rgb[only_o | only_s] = (220, 40, 40)
    return Image.fromarray(rgb, mode="RGB")


def run_query(
    query: str,
    input_mode: str,
    scene: int | None = None,
    live: bool = True,
    vlm_url: str = "http://127.0.0.1:8080",
    device_cd: str = "cpu",
    uploads: dict[str, Any] | None = None,
    cd_prefer: str = "changeformer",
    on_event: Callable[[dict], None] | None = None,
    sensor_profile: str | None = None,
) -> dict[str, Any]:
    t_all = time.perf_counter()

    # API-STREAM: observational stage events for the live pipeline view.
    # A raising callback can never break a run; with on_event=None every
    # _emit is a no-op and behavior is unchanged.
    def _emit(
        stage: str,
        status: str = "done",
        tool: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        if on_event is None:
            return
        ev: dict[str, Any] = {
            "event": "stage",
            "stage": stage,
            "status": status,
            "ts": time.perf_counter(),
        }
        if tool is not None:
            ev["tool"] = tool
        if data is not None:
            ev["data"] = data
        try:
            on_event(ev)
        except Exception:
            pass

    def _tool_start(tool: str) -> float:
        _emit("tool", "start", tool=tool)
        return time.perf_counter()

    def _tool_done(
        tool: str, t0: float, data: dict[str, Any] | None = None
    ) -> None:
        _emit(
            "tool",
            "done",
            tool=tool,
            data={"latency_s": round(time.perf_counter() - t0, 3), **(data or {})},
        )

    def _call(
        stage: str, tool: str | None, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Emit stage fail on a tool/narration exception, then re-raise."""
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            _emit(
                stage,
                "fail",
                tool=tool,
                data={"error": f"{type(e).__name__}: {e}"},
            )
            raise

    def _packet() -> None:
        _emit("packet", "start")
        _attach_packet(trace, the_plan, bound)
        _emit(
            "packet",
            "done",
            data={
                "ok": "evidence_packet" in trace,
                "error": trace.get("evidence_packet_error"),
            },
        )

    def _finish(tr: dict[str, Any]) -> dict[str, Any]:
        out = _with_visible(tr)
        _emit(
            "done",
            "done",
            data={
                "supported": (tr.get("plan") or {}).get("supported"),
                "tools": (tr.get("plan") or {}).get("tools"),
                "scene": tr.get("scene"),
                "input_source": tr.get("input_source"),
                "complete_s": tr.get("complete_s"),
                "n_images": len(tr.get("images") or []),
                "has_overlay": bool(tr.get("overlay_path")),
                "has_packet": "evidence_packet" in tr,
            },
        )
        return out

    _emit("plan", "start")
    the_plan = plan(query, input_mode)
    # MODEL-FIRST-ROUTER: the regex plan runs first as the deterministic
    # safety layer — refusals/mode guards never reach the model. For live
    # runs the narrator seat is the primary router for every non-refusal
    # query (validator still gates); the regex plan is the routing
    # fallback. MODEL_FIRST=0 restores regex-primary with bare-plan
    # fallback. live=False never calls the seat.
    router = "regex"
    if live and the_plan.get("supported"):
        if MODEL_FIRST:
            why: dict[str, str] = {}
            fb = plan_model(query, input_mode, url=vlm_url, _reason=why)
            if fb is not None:
                the_plan = fb
                router = "model"
            else:
                router = "regex-fallback"
                if why.get("why"):
                    the_plan["model_fallback_reason"] = why["why"]
        elif needs_fallback(the_plan):
            fb = plan_model(query, input_mode, url=vlm_url)
            if fb is not None:
                the_plan = fb
                router = "model-fallback"
    the_plan["router"] = router
    _emit(
        "plan",
        "done",
        data={
            "supported": the_plan.get("supported"),
            "task": the_plan.get("task"),
            "tools": the_plan.get("tools"),
            "vlm_role": the_plan.get("vlm_role"),
            "router": router,
        },
    )
    _emit("bind", "start")
    bound = bind_inputs(input_mode, scene, uploads, sensor_profile=sensor_profile)
    _emit(
        "bind",
        "done" if bound.get("ok") else "fail",
        data={
            "source": bound.get("source"),
            "scene": bound.get("scene"),
            "gsd_m": (bound.get("gsd") or {}).get("gsd_m"),
            "gsd_source": (bound.get("gsd") or {}).get("source"),
            "misreg_shift_px": bound.get("misreg_shift_px"),
            "coreg_shift_px": (bound.get("coreg") or {}).get("shift_px"),
            "error": bound.get("error"),
        },
    )
    trace: dict[str, Any] = {
        "query": query,
        "input_mode": input_mode,
        "scene": bound.get("scene", scene),
        "live": live,
        "plan": the_plan,
        "tool_outputs": {},
        "answer": None,
        "first_token_s": None,
        "complete_s": None,
        "overlay_path": None,
        "images": [],
        "metrics_badge": _metrics_badge() if bound.get("source") == "prepared" else None,
        "ingest_note": bound.get("ingest_note"),
        "input_source": bound.get("source"),
        "gsd": bound.get("gsd"),
        "misreg_check": bound.get("misreg_check"),
        "misregistration_note": bound.get("misregistration_note"),
        "misreg_shift_px": bound.get("misreg_shift_px"),
    }
    if not bound.get("ok"):
        trace["answer"] = bound.get("error")
        trace["complete_s"] = round(time.perf_counter() - t_all, 3)
        _packet()
        return _finish(trace)
    if not the_plan["supported"]:
        trace["answer"] = the_plan["refusal"]
        trace["complete_s"] = round(time.perf_counter() - t_all, 3)
        _emit("plan", "withheld", data={"refusal": the_plan.get("refusal")})
        _packet()
        return _finish(trace)

    tools_needed = the_plan["tools"]
    numbers: dict[str, Any] = {}
    paths = bound["paths"]
    gsd_info = bound.get("gsd") or {}
    use_cache = bool(bound.get("use_prepared_cd_cache"))
    workdir: Path | None = bound.get("workdir")

    if input_mode == "single" or bound["scene"] == 1:
        image_paths = [paths["image"]]
    elif input_mode == "bi-temporal" or bound["scene"] in (2, 4):
        image_paths = [paths["before"], paths["after"]]
    else:
        image_paths = [paths["optical"], paths["sar_vv"]]

    if "change_detect" in tools_needed:
        _t0 = _tool_start("change_detect")
        pred_path = paths.get("pred")
        gt = load_mask(paths["gt"]) if paths.get("gt") and Path(paths["gt"]).is_file() else None
        if use_cache and pred_path is not None and Path(pred_path).is_file() and not live:
            mask = load_mask(pred_path)
            cd = {
                "rung": "cached_mask",
                "rung_label": "cached ChangeFormer mask (cached mode)",
                "mask": mask,
                "changed_pixels": int(mask.sum()),
                "provenance": "demo/data/scene2/pred_mask.png",
            }
            if gt is not None:
                from tools import mask_metrics

                cd["vs_gt"] = mask_metrics(mask, gt)
        elif use_cache and pred_path is not None and Path(pred_path).is_file():
            # Prepared Scene 2: reuse on-disk mask (ChangeFormer is deterministic).
            mask = load_mask(pred_path)
            meta_path = DATA / "scene2" / "pred_trace.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
            cd = {
                "rung": meta.get("rung", "changeformer_v6"),
                "rung_label": meta.get("rung_label", "ChangeFormerV6 (deterministic mask on disk)"),
                "checkpoint": meta.get("checkpoint"),
                "mask": mask,
                "changed_pixels": int(mask.sum()),
                "vs_gt": meta.get("vs_gt"),
                "note": "mask loaded from pred_mask.png (same weights; not a fake live VLM path)",
                "provenance": "tools.change_detect / pred_mask.png",
            }
        else:
            cd = _call(
                "tool",
                "change_detect",
                change_detect,
                paths["before"],
                paths["after"],
                prefer=cd_prefer,
                device=device_cd,
                gt_mask=gt,
                domain=bound.get("cd_domain"),
            )
            mask = cd["mask"]
            # Never overwrite prepared Scene 2 pred_mask.png (uploads or cache miss).
        dir_proxy = change_direction_proxy(
            load_rgb(paths["before"]), load_rgb(paths["after"]), cd["mask"]
        )
        cd.update(dir_proxy)
        _maybe_attach_scene4_semantic(
            query=query,
            bound=bound,
            cd=cd,
            paths=paths,
            device_cd=device_cd,
            numbers=numbers,
            trace=trace,
        )
        _tool_done(
            "change_detect",
            _t0,
            {
                "rung": cd.get("rung"),
                "changed_pixels": cd.get("changed_pixels"),
                "semantic_rung": cd.get("semantic_rung"),
            },
        )
        if "cdvqa_map" in tools_needed:
            _t0 = _tool_start("cdvqa_map")
            cm = _call("tool", "cdvqa_map", cdvqa_map, query, semantic=numbers.get("semantic"))
            numbers["cdvqa_map"] = _jsonable(cm)
            trace["tool_outputs"]["cdvqa_map"] = numbers["cdvqa_map"]
            _tool_done(
                "cdvqa_map",
                _t0,
                {"answer": cm.get("answer"), "withheld_reason": cm.get("withheld_reason")},
            )
        attach_rung_display(cd)
        overlay = overlay_mask(load_rgb(paths["after"]), cd["mask"])
        if workdir is not None:
            overlay_dir = workdir
        elif int(bound.get("scene") or 0) == 4:
            overlay_dir = DATA / "scene4"
        else:
            overlay_dir = DATA / "scene2"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = overlay_dir / "overlay_live.png"
        overlay.save(overlay_path)
        trace["overlay_path"] = str(overlay_path)
        _geo_export(
            trace, bound, cd["mask"], "change_mask.tif",
            "before_original", overlay_dir,
        )
        numbers["change_detect"] = _jsonable(cd)
        trace["tool_outputs"]["change_detect"] = numbers["change_detect"]
        if "area_calc" in tools_needed:
            _t0 = _tool_start("area_calc")
            area = _call(
                "tool",
                "area_calc",
                area_calc,
                cd["mask"],
                gsd_m=gsd_info.get("gsd_m"),
                label="built-up_change",
                gsd_meta=gsd_info,
            )
            numbers["area_calc"] = area
            trace["tool_outputs"]["area_calc"] = area
            _tool_done(
                "area_calc",
                _t0,
                {"gsd_m": area.get("gsd_m"), "area_m2": area.get("area_m2")},
            )

    if "water_highlight" in tools_needed:
        _t0 = _tool_start("water_highlight")
        src = paths["image"]
        orig = paths.get("source_original") or src
        wh = _call(
            "tool",
            "water_highlight",
            water_highlight,
            src,
            source_path=orig,
            sensor_profile=bound.get("sensor_profile"),
        )
        if not wh.get("withheld"):
            overlay = overlay_mask(
                load_rgb(src), wh["mask"], color=WATER_OVERLAY_RGB, alpha=0.5
            )
            overlay_dir = workdir if workdir is not None else (DATA / "scene1")
            overlay_dir.mkdir(parents=True, exist_ok=True)
            overlay_path = overlay_dir / "water_overlay_live.png"
            overlay.save(overlay_path)
            trace["overlay_path"] = str(overlay_path)
            _geo_export(
                trace, bound, wh["mask"], "water_mask.tif",
                "source_original", overlay_dir,
            )
        numbers["water_highlight"] = _jsonable(wh)
        trace["tool_outputs"]["water_highlight"] = numbers["water_highlight"]
        _tool_done(
            "water_highlight",
            _t0,
            {"water_pixels": wh.get("water_pixels"), "method": wh.get("method")},
        )
        if "area_calc" in tools_needed:
            _t0 = _tool_start("area_calc")
            if wh.get("withheld"):
                area = {
                    "withheld": True,
                    "withheld_reason": "water_highlight withheld "
                    f"({wh.get('withheld_reason')})",
                    "label": "water",
                }
            else:
                area = _call(
                    "tool",
                    "area_calc",
                    area_calc,
                    wh["mask"],
                    gsd_m=gsd_info.get("gsd_m"),
                    label="water",
                    gsd_meta=gsd_info,
                )
            numbers["area_calc"] = area
            trace["tool_outputs"]["area_calc"] = area
            _tool_done(
                "area_calc",
                _t0,
                {"gsd_m": area.get("gsd_m"), "area_m2": area.get("area_m2")},
            )

    if "sar_read" in tools_needed:
        import numpy as np

        _t0 = _tool_start("sar_read")
        sar_pack = bound.get("sar_arrays")
        if sar_pack is not None:
            sr = _call(
                "tool",
                "sar_read",
                sar_read,
                sar_pack["vv"],
                sar_pack["vh"],
                calibrated=bool(sar_pack.get("calibrated", True)),
            )
            sr["ingest_provenance"] = sar_pack.get("provenance")
        elif input_mode == "single":
            # Single SAR is PS scope: the bound image IS the SAR candidate.
            # Read the ORIGINAL file — materialize_rgb's PNG would destroy
            # dtype/calibration and band names.
            sar_src = paths.get("source_original") or paths.get("image")
            sr_ing = read_sar_arrays(sar_src) if sar_src else {"ok": False, "error": "no bound image"}
            if sr_ing.get("ok"):
                sr = _call(
                    "tool",
                    "sar_read",
                    sar_read,
                    sr_ing["vv"],
                    sr_ing["vh"],
                    calibrated=bool(sr_ing.get("calibrated", True)),
                )
                sr["ingest_provenance"] = sr_ing.get("provenance")
                sr["pol_verified"] = sr_ing.get("pol_verified")
            else:
                sr = {
                    "withheld": True,
                    "withheld_reason": sr_ing.get("error"),
                    "provenance": (
                        "tools.sar_read (withheld — ingest refused). "
                        + str(sr_ing.get("error"))
                    ),
                }
        else:
            npz = np.load(paths["sar_npz"])
            sr = _call(
                "tool", "sar_read", sar_read, npz["vv"], npz["vh"], calibrated=True
            )
        if (
            "area_calc" in tools_needed
            and not sr.get("withheld")
            and sr.get("water_calibrated") is not False
        ):
            _ta = _tool_start("area_calc")
            area = _call(
                "tool",
                "area_calc",
                area_calc,
                sr["water_mask"],
                gsd_m=gsd_info.get("gsd_m"),
                label="water",
                gsd_meta=gsd_info,
            )
            numbers["area_calc"] = area
            trace["tool_outputs"]["area_calc"] = area
            _tool_done(
                "area_calc",
                _ta,
                {"gsd_m": area.get("gsd_m"), "area_m2": area.get("area_m2")},
            )
        if not sr.get("withheld") and sr.get("water_mask") is not None:
            # single SAR overlays on the materialized image the VLM sees;
            # optical+sar overlays on the optical.
            base_key = "image" if input_mode == "single" else "optical"
            overlay = overlay_mask(load_rgb(paths[base_key]), sr["water_mask"], color=(30, 90, 220), alpha=0.5)
            overlay_dir = workdir if workdir is not None else (DATA / "scene3")
            overlay_dir.mkdir(parents=True, exist_ok=True)
            overlay_path = overlay_dir / "overlay_live.png"
            overlay.save(overlay_path)
            trace["overlay_path"] = str(overlay_path)
            _geo_export(
                trace, bound, sr["water_mask"], "water_sar_mask.tif",
                "source_original" if input_mode == "single" else "sar_original",
                overlay_dir,
            )
        numbers["sar_read"] = _jsonable(sr)
        trace["tool_outputs"]["sar_read"] = numbers["sar_read"]
        _tool_done(
            "sar_read",
            _t0,
            {
                "water_pixels": sr.get("water_pixels"),
                "water_calibrated": sr.get("water_calibrated"),
            },
        )
        if "sar_agreement" in tools_needed:
            _t0 = _tool_start("sar_agreement")
            wh = _call(
                "tool",
                "water_highlight",
                water_highlight,
                paths["optical"],
                source_path=paths.get("source_original"),
                sensor_profile=bound.get("sensor_profile"),
            )
            numbers["water_highlight"] = _jsonable(wh)
            trace["tool_outputs"]["water_highlight"] = numbers["water_highlight"]
            if wh.get("withheld"):
                ag = {
                    "withheld": True,
                    "withheld_reason": "optical water mask withheld "
                    f"({wh.get('withheld_reason')})",
                }
                numbers["sar_agreement"] = _jsonable(ag)
                trace["tool_outputs"]["sar_agreement"] = numbers["sar_agreement"]
                _tool_done(
                    "sar_agreement",
                    _t0,
                    {"withheld": ag["withheld_reason"]},
                )
            else:
                ag = _call(
                    "tool",
                    "sar_agreement",
                    sar_agreement,
                    wh["mask"],
                    sr["water_mask"],
                    sar_calibrated=sr.get("water_calibrated", True),
                    misreg_shift_px=bound.get("misreg_shift_px"),
                )
                amap = _agreement_map_image(ag)
                amap_dir = workdir if workdir is not None else (DATA / "scene3")
                amap_dir.mkdir(parents=True, exist_ok=True)
                amap_path = amap_dir / "agreement_map_live.png"
                amap.save(amap_path)
                trace["agreement_map_path"] = str(amap_path)
                _geo_export(
                    trace, bound, wh["mask"], "water_optical_mask.tif",
                    "source_original", amap_dir,
                )
                # Categorical agreement grid: 0=neither 1=optical 2=sar 3=both.
                cat = ag["optical_mask"].astype(np.uint8) + 2 * ag[
                    "sar_mask"
                ].astype(np.uint8)
                _geo_export(
                    trace, bound, cat, "agreement_map_live.tif",
                    "source_original", amap_dir,
                )
                numbers["sar_agreement"] = _jsonable(ag)
                trace["tool_outputs"]["sar_agreement"] = numbers["sar_agreement"]
                _tool_done(
                    "sar_agreement",
                    _t0,
                    {
                        "verdicts": _jsonable(ag.get("verdicts")),
                        "agreement_iou": ag.get("agreement_iou"),
                    },
                )

    if isinstance(bound, dict) and bound.get("coreg"):
        numbers["coreg_check"] = _jsonable(bound["coreg"])
        trace["tool_outputs"]["coreg_check"] = numbers["coreg_check"]

    if "canonical_vqa" in tools_needed:
        # Adapted RSVQA answer seat on :8091 — a specialist claim, not
        # narration. Runs whenever routed (single-image question plans only);
        # a down server yields a structured unavailable record, never a
        # silent :8080 narrator substitution.
        _t0 = _tool_start("canonical_vqa")
        cv = _call("tool", "canonical_vqa", canonical_vqa, query, image_paths[:1])
        numbers["canonical_vqa"] = _jsonable(cv)
        trace["tool_outputs"]["canonical_vqa"] = numbers["canonical_vqa"]
        _tool_done(
            "canonical_vqa",
            _t0,
            {
                "available": cv.get("available"),
                "answer": cv.get("answer"),
                "seat": cv.get("seat"),
            },
        )

    if "vqa" in tools_needed and the_plan["vlm_role"] != "none":
        if the_plan["vlm_role"] in {"narrate"} and numbers:
            prompt = narration_prompt(query, numbers, the_plan["task"])
        elif the_plan["task"] == "caption":
            prompt = (
                f"{query}\n\nDescribe land cover and major objects. "
                "Do not invent numeric measurements (no areas, counts, or distances)."
            )
        else:
            prompt = (
                f"{query}\n\nAnswer from the image. Do not invent numeric measurements "
                "that a tool did not produce."
            )
        if live:
            _emit(
                "narration",
                "start",
                data={
                    "vlm_role": the_plan["vlm_role"],
                    "n_images": len(image_paths),
                    "url": vlm_url,
                },
            )
            vlm = _call("narration", None, ask_vlm, prompt, image_paths, url=vlm_url)
            trace["answer"] = vlm["text"]
            trace["first_token_s"] = vlm.get("first_token_s")
            trace["vlm"] = _jsonable(vlm)
            _emit(
                "narration",
                "done",
                data={
                    "first_token_s": vlm.get("first_token_s"),
                    "complete_s": vlm.get("complete_s"),
                    "model": vlm.get("model"),
                    "url": vlm.get("url"),
                },
            )
        else:
            trace["answer"] = None  # filled by cache
            trace["vlm"] = {"skipped": "cached mode"}
            _emit(
                "narration",
                "withheld",
                data={"reason": "live=false (cached mode)"},
            )
    elif the_plan["vlm_role"] == "none":
        trace["answer"] = (
            "Change mask produced by the change_detect tool. "
            + json.dumps(numbers.get("change_detect", {}), indent=2)
        )
        _emit("narration", "withheld", data={"reason": "vlm_role=none"})

    trace["images"] = [str(p) for p in image_paths]
    if trace.get("agreement_map_path"):
        trace["images"].append(trace["agreement_map_path"])
    trace["complete_s"] = round(time.perf_counter() - t_all, 3)
    _packet()
    return _finish(trace)
