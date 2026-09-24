"""Typed evidence packet + two renderers (Lane P). Qwen does not own numbers.

Laptop only. Builds a packet from existing tool_outputs plus a planner task id.
pipeline/report/app may attach and download it; planner/tools stay untouched.

- benchmark: short answer / mask path / box only
- product: findings header from report.py first; Qwen explains claims, still
  gated by report.check_narration
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from report import (
    INTERPRETATION_NOTE,
    check_narration,
    findings_header,
    tool_outputs_sha256,
)

DEMO = Path(__file__).resolve().parent
SAT = DEMO.parent
CACHE = SAT / "gates" / "_cache" / "evidence_packet"

SCHEMA_VERSION = "1.0"
CLAIM_FIELDS = ("id", "predicate", "value", "region", "confidence", "provenance")
BENCHMARK_KEYS = ("answer", "mask", "box")
CANONICAL_NOT_DETERMINED = "not_determined"

_CLAIM_ID_RE = re.compile(r"\bE\d+\b")
_DIRECTION_CLAIM_RE = re.compile(
    r"\b(?:increase[sd]?|decrease[sd]?|grew|growth)\b",
    flags=re.I,
)


def _as_conf(level: str, basis: str) -> dict[str, str]:
    return {"level": level, "basis": basis}


def _as_prov(tool: str, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"tool": tool}
    for k, v in extra.items():
        if v is not None:
            out[k] = v
    return out


def make_claim(
    *,
    cid: str,
    predicate: str,
    value: Any,
    region: str | None,
    confidence: dict[str, str],
    provenance: dict[str, Any],
    unit: str | None = None,
) -> dict[str, Any]:
    claim: dict[str, Any] = {
        "id": cid,
        "predicate": predicate,
        "value": value,
        "region": region,
        "confidence": confidence,
        "provenance": provenance,
    }
    if unit is not None:
        claim["unit"] = unit
    return claim


def mask_box_xyxy(mask_path: str | Path | None) -> list[int] | None:
    """Native-grid [x0, y0, x1, y1) of nonzero pixels, or None."""
    if not mask_path:
        return None
    path = Path(mask_path)
    if not path.is_file():
        return None
    arr = Image.open(path)
    try:
        import numpy as np

        m = np.asarray(arr)
    finally:
        arr.close()
    if m.ndim == 3:
        m = m[..., 0]
    ys, xs = (m > 0).nonzero()
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(int(xs.max()) + 1), int(int(ys.max()) + 1)]


def _region_for(tool_outputs: dict[str, Any], mask_path: str | None, overlay_path: str | None) -> str | None:
    cd = tool_outputs.get("change_detect") or {}
    for cand in (
        mask_path,
        overlay_path,
        cd.get("provenance") if isinstance(cd.get("provenance"), str) else None,
    ):
        if cand:
            return str(cand)
    wh = tool_outputs.get("water_highlight") or {}
    if isinstance(wh.get("provenance"), str) and wh.get("provenance"):
        return str(wh["provenance"])
    return None


def _ids() -> Any:
    n = 0

    def next_id() -> str:
        nonlocal n
        n += 1
        return f"E{n}"

    return next_id


def _change_canonical(_cd: dict[str, Any]) -> str:
    """Tools own this. Never Qwen. No class map → not_determined."""
    return CANONICAL_NOT_DETERMINED


def _claims_change(
    cd: dict[str, Any],
    area: dict[str, Any] | None,
    region: str | None,
    nid,
    limitations: list[str],
) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    px = cd.get("changed_pixels")
    rung = cd.get("rung")
    model = cd.get("checkpoint") or rung
    if px is not None:
        n_px = int(px)
        claims.append(
            make_claim(
                cid=nid(),
                predicate="change_detected",
                value=n_px > 0,
                region=region,
                confidence=_as_conf("measured", "changed_pixels from specialist mask"),
                provenance=_as_prov("change_detect", model=model, rung=rung),
            )
        )
        claims.append(
            make_claim(
                cid=nid(),
                predicate="changed_pixels",
                value=n_px,
                region=region,
                confidence=_as_conf("measured", "count(mask>0)"),
                provenance=_as_prov("change_detect", model=model, rung=rung),
            )
        )
    delta = cd.get("radiometric_delta", cd.get("delta_mean"))
    if delta is not None:
        claims.append(
            make_claim(
                cid=nid(),
                predicate="radiometric_delta",
                value=delta,
                region=region,
                confidence=_as_conf("measured", "deterministic masked mean RGB"),
                provenance=_as_prov("change_direction_proxy"),
                unit="mean_rgb_level",
            )
        )
    radio = cd.get("radiometric_label")
    if radio is not None:
        claims.append(
            make_claim(
                cid=nid(),
                predicate="radiometric_label",
                value=radio,
                region=region,
                confidence=_as_conf(
                    "measured",
                    f"eps={cd.get('direction_epsilon')} on radiometric_delta",
                ),
                provenance=_as_prov("change_direction_proxy"),
            )
        )
    claims.append(
        make_claim(
            cid=nid(),
            predicate="built_up_direction",
            value=CANONICAL_NOT_DETERMINED,
            region=region,
            confidence=_as_conf(
                "withheld",
                "no class-aware before/after land-cover tool",
            ),
            provenance=_as_prov("change_direction_proxy"),
        )
    )
    vs = cd.get("vs_gt") or {}
    if vs.get("iou") is not None:
        claims.append(
            make_claim(
                cid=nid(),
                predicate="iou_vs_gt",
                value=vs.get("iou"),
                region=region,
                confidence=_as_conf("measured", "tools.mask_metrics pixel confusion"),
                provenance=_as_prov("mask_metrics"),
            )
        )
    if area:
        gsd = area.get("gsd_m")
        if area.get("area_m2") is not None and gsd is not None:
            claims.append(
                make_claim(
                    cid=nid(),
                    predicate="area_m2",
                    value=area.get("area_m2"),
                    region=region,
                    confidence=_as_conf("measured", str(area.get("formula") or "area_calc")),
                    provenance=_as_prov("area_calc"),
                    unit="m2",
                )
            )
            if area.get("area_km2") is not None:
                claims.append(
                    make_claim(
                        cid=nid(),
                        predicate="area_km2",
                        value=area.get("area_km2"),
                        region=region,
                        confidence=_as_conf("measured", str(area.get("formula") or "area_calc")),
                        provenance=_as_prov("area_calc"),
                        unit="km2",
                    )
                )
        else:
            limitations.append("square metres withheld (no validated GSD).")
        if area.get("dominant_quadrant"):
            claims.append(
                make_claim(
                    cid=nid(),
                    predicate="dominant_quadrant",
                    value=area.get("dominant_quadrant"),
                    region=region,
                    confidence=_as_conf("measured", "pixel counts per image quadrant"),
                    provenance=_as_prov("area_calc"),
                )
            )
    limitations.append(
        "The current evidence does not identify the changed land-cover class."
    )
    if radio is not None:
        limitations.append(
            f"radiometric_label={radio!r} is brightness inside the change mask, "
            "not a built-up direction."
        )
    return claims


def _claims_water(
    wh: dict[str, Any],
    area: dict[str, Any] | None,
    region: str | None,
    nid,
    limitations: list[str],
) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    method = wh.get("method")
    if wh.get("withheld"):
        limitations.append(
            f"water mask withheld ({wh.get('withheld_reason')}): band order "
            "unidentified — declare a sensor_profile or supply named bands."
        )
    if method is not None:
        claims.append(
            make_claim(
                cid=nid(),
                predicate="water_method",
                value=method,
                region=region,
                confidence=_as_conf("measured", "optical water tool metadata"),
                provenance=_as_prov("water_highlight"),
            )
        )
    wp = wh.get("water_pixels")
    if wp is None:
        limitations.append("water_pixels withheld by the optical water tool.")
    else:
        claims.append(
            make_claim(
                cid=nid(),
                predicate="water_pixels",
                value=wp,
                region=region,
                confidence=_as_conf("measured", str(method or "water_highlight")),
                provenance=_as_prov("water_highlight"),
            )
        )
    if area and area.get("area_m2") is not None and area.get("gsd_m") is not None:
        claims.append(
            make_claim(
                cid=nid(),
                predicate="area_m2",
                value=area.get("area_m2"),
                region=region,
                confidence=_as_conf("measured", str(area.get("formula") or "area_calc")),
                provenance=_as_prov("area_calc"),
                unit="m2",
            )
        )
    elif area and (area.get("gsd_m") is None or area.get("area_m2") is None):
        limitations.append("square metres withheld (no validated GSD).")
    return claims


def _claims_sar(
    sr: dict[str, Any],
    area: dict[str, Any] | None,
    region: str | None,
    nid,
    limitations: list[str],
) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    cal = sr.get("water_calibrated")
    claims.append(
        make_claim(
            cid=nid(),
            predicate="water_calibrated",
            value=cal,
            region=region,
            confidence=_as_conf("measured", "ingest calibration flag"),
            provenance=_as_prov("sar_read"),
        )
    )
    if sr.get("sdwi_label") is not None:
        claims.append(
            make_claim(
                cid=nid(),
                predicate="sdwi_label",
                value=sr.get("sdwi_label"),
                region=region,
                confidence=_as_conf("measured", "tool metadata; render as SDWI"),
                provenance=_as_prov("sar_read"),
            )
        )
    if cal is False:
        limitations.append(
            "Unknown/8-bit SAR calibration: calibrated threshold claims withheld."
        )
        if sr.get("water_pixels") is None:
            limitations.append("water_pixels withheld (preview DN, threshold not applied).")
    elif sr.get("water_pixels") is not None:
        claims.append(
            make_claim(
                cid=nid(),
                predicate="water_pixels",
                value=sr.get("water_pixels"),
                region=region,
                confidence=_as_conf("measured", str(sr.get("method") or "sar_read")),
                provenance=_as_prov("sar_read"),
            )
        )
    if area and area.get("area_m2") is not None and area.get("gsd_m") is not None and cal is not False:
        claims.append(
            make_claim(
                cid=nid(),
                predicate="area_m2",
                value=area.get("area_m2"),
                region=region,
                confidence=_as_conf("measured", str(area.get("formula") or "area_calc")),
                provenance=_as_prov("area_calc"),
                unit="m2",
            )
        )
    return claims


def _claims_agreement(
    ag: dict[str, Any],
    region: str | None,
    nid,
    limitations: list[str],
) -> list[dict[str, Any]]:
    """sar_agreement claims: verdict, iou, grid. Withheld verdicts are reasons."""
    claims: list[dict[str, Any]] = []
    verdicts = ag.get("verdicts") or {}
    water_v = verdicts.get("water")
    withheld = isinstance(water_v, str) and water_v.startswith("withheld")
    claims.append(
        make_claim(
            cid=nid(),
            predicate="water_agreement",
            value=water_v,
            region=region,
            confidence=_as_conf(
                "withheld" if withheld else "measured",
                "sar_agreement verdict on common grid"
                if not withheld
                else f"verdict withheld: {water_v}",
            ),
            provenance=_as_prov("sar_agreement", parents="water_highlight+sar_read"),
        )
    )
    claims.append(
        make_claim(
            cid=nid(),
            predicate="built_up_agreement",
            value="withheld_no_tool",
            region=region,
            confidence=_as_conf(
                "withheld",
                "no optical built-up tool and no calibrated SAR built-up mask",
            ),
            provenance=_as_prov("sar_agreement"),
        )
    )
    if ag.get("iou") is not None:
        claims.append(
            make_claim(
                cid=nid(),
                predicate="agreement_iou",
                value=ag.get("iou"),
                region=region,
                confidence=_as_conf("measured", "intersection/union on common grid"),
                provenance=_as_prov("sar_agreement"),
                unit="iou",
            )
        )
    if ag.get("grid"):
        claims.append(
            make_claim(
                cid=nid(),
                predicate="agreement_grid",
                value=ag.get("grid"),
                region=region,
                confidence=_as_conf("measured", "comparison grid disclosure"),
                provenance=_as_prov("sar_agreement"),
            )
        )
    if withheld:
        limitations.append(str(water_v))
    limitations.append(
        "water agreement compares two independent deterministic masks on a "
        "common grid; it is not pixel-level optical/SAR fusion."
    )
    return claims


def _claims_coreg(
    cg: dict[str, Any],
    nid,
    limitations: list[str],
) -> list[dict[str, Any]]:
    """coreg_check claims: measured co-registration basis for optical+SAR."""
    claims: list[dict[str, Any]] = []
    tr = cg.get("transform") or {}
    px = cg.get("pixel") or {}
    if tr.get("status") == "measured":
        claims.append(
            make_claim(
                cid=nid(),
                predicate="coreg_transform_offset_m",
                value=tr.get("offset_m"),
                region=None,
                confidence=_as_conf("measured", "affine transform comparison"),
                provenance=_as_prov("coreg_check"),
                unit="m",
            )
        )
        claims.append(
            make_claim(
                cid=nid(),
                predicate="coreg_same_res",
                value=tr.get("same_res"),
                region=None,
                confidence=_as_conf("measured", "affine transform comparison"),
                provenance=_as_prov("coreg_check"),
            )
        )
    else:
        claims.append(
            make_claim(
                cid=nid(),
                predicate="coreg_transform_offset_m",
                value="withheld",
                region=None,
                confidence=_as_conf(
                    "withheld",
                    f"transform check {tr.get('status')}: {tr.get('reason')}",
                ),
                provenance=_as_prov("coreg_check"),
            )
        )
    if px.get("status") == "measured":
        claims.append(
            make_claim(
                cid=nid(),
                predicate="coreg_shift_px",
                value=px.get("shift_px"),
                region=None,
                confidence=_as_conf(
                    "measured",
                    "phase cross-correlation on common grid; cross-modal "
                    "estimate is noisy — measured-with-caveat, not ground truth",
                ),
                provenance=_as_prov("coreg_check"),
                unit="px",
            )
        )
    else:
        claims.append(
            make_claim(
                cid=nid(),
                predicate="coreg_shift_px",
                value="withheld",
                region=None,
                confidence=_as_conf(
                    "withheld",
                    f"pixel shift inconclusive: {px.get('reason')}",
                ),
                provenance=_as_prov("coreg_check"),
            )
        )
    limitations.append(
        "co-registration is measured (transform + phase-correlation levels), "
        "not assumed; the pair is not realigned."
    )
    return claims


# second_semantic internal tokens -> CDVQA gold casing (mirrors tools.py).
_SEM_CLASS_TOKENS = (
    "water",
    "nvg_surface",
    "low_vegetation",
    "trees",
    "buildings",
    "playgrounds",
)
_CDVQA_GOLD_CASE = {"nvg_surface": "NVG_surface"}


def _claims_semantic(
    sem: dict[str, Any],
    region: str | None,
    nid,
    limitations: list[str],
) -> list[dict[str, Any]]:
    """second_semantic record -> typed transition/dominance/trend claims."""
    claims: list[dict[str, Any]] = []
    feat = sem.get("features") or {}
    if not feat:
        return claims
    prov = _as_prov(
        "second_semantic",
        parents="change_detect",
        checkpoint_sha256=sem.get("checkpoint_sha256"),
    )
    dom = sem.get("dominant_transition") or {}
    if dom.get("token"):
        fr = _CDVQA_GOLD_CASE.get(dom.get("from_token"), dom.get("from_token"))
        to = _CDVQA_GOLD_CASE.get(dom.get("to_token"), dom.get("to_token"))
        claims.append(
            make_claim(
                cid=nid(),
                predicate="primary_transition",
                value=f"{fr}->{to}",
                region=region,
                confidence=_as_conf(
                    "measured", "largest from->to cell under predicted change mask"
                ),
                provenance=prov,
            )
        )
    else:
        claims.append(
            make_claim(
                cid=nid(),
                predicate="primary_transition",
                value=None,
                region=region,
                confidence=_as_conf(
                    "withheld", "no predicted change transitions in evidence"
                ),
                provenance=prov,
            )
        )
    from_hist = [int(x) for x in (feat.get("from_hist") or [])]
    to_hist = [int(x) for x in (feat.get("to_hist") or [])]
    combo = [a + b for a, b in zip(from_hist, to_hist)]
    if combo and max(combo) > 0:
        i = combo.index(max(combo))
        tok = _SEM_CLASS_TOKENS[i]
        claims.append(
            make_claim(
                cid=nid(),
                predicate="largest_change_class",
                value=_CDVQA_GOLD_CASE.get(tok, tok),
                region=region,
                confidence=_as_conf(
                    "measured",
                    "argmax of combined from+to change histogram (unqualified)",
                ),
                provenance=prov,
            )
        )
    else:
        claims.append(
            make_claim(
                cid=nid(),
                predicate="largest_change_class",
                value=None,
                region=region,
                confidence=_as_conf(
                    "withheld", "no changed pixels in semantic evidence"
                ),
                provenance=prov,
            )
        )
    direction = sem.get("built_up_direction")
    claims.append(
        make_claim(
            cid=nid(),
            predicate="trend",
            value=direction if direction else None,
            region=region,
            confidence=_as_conf(
                "measured" if direction else "withheld",
                "buildings count_b vs count_a (named-class trend)"
                if direction
                else "no direction evidence in packet",
            ),
            provenance=prov,
        )
    )
    limitations.append(
        "semantic claims inherit the second_semantic backbone (val mIoU 0.417); "
        "they are measured over the predicted-change mask, not pixel truth."
    )
    return claims


def _claims_cdvqa(
    cm: dict[str, Any],
    region: str | None,
    nid,
    limitations: list[str],
) -> list[dict[str, Any]]:
    """cdvqa_map output -> canonical answer + family claims."""
    claims: list[dict[str, Any]] = []
    ok = bool(cm.get("answer_vocab_ok")) and cm.get("answer") is not None
    claims.append(
        make_claim(
            cid=nid(),
            predicate="cdvqa_answer",
            value=cm.get("answer") if ok else None,
            region=region,
            confidence=_as_conf(
                "measured" if ok else "withheld",
                f"cdvqa_map {cm.get('official_type')} -> {cm.get('claim')}"
                if ok
                else f"cdvqa_map withheld: {cm.get('withheld_reason')}",
            ),
            provenance=_as_prov("cdvqa_map", parents="second_semantic"),
        )
    )
    claims.append(
        make_claim(
            cid=nid(),
            predicate="cdvqa_family",
            value=cm.get("official_type") or cm.get("withheld_reason"),
            region=region,
            confidence=_as_conf(
                "measured", "deterministic family detector on question text"
            ),
            provenance=_as_prov("cdvqa_map"),
        )
    )
    if not ok:
        limitations.append(f"cdvqa_map withheld: {cm.get('withheld_reason')}")
    limitations.append(
        "cdvqa_map is a deterministic packet->vocabulary map over "
        "second_semantic features (backbone val mIoU 0.417); it is not a "
        "learned answer model."
    )
    return claims


def _claims_canonical_vqa(
    cv: dict[str, Any],
    nid,
    limitations: list[str],
) -> list[dict[str, Any]]:
    """canonical_vqa output -> canonical_answer claim (:8091 adapted seat)."""
    ok = bool(cv.get("available")) and cv.get("answer") is not None
    prov = _as_prov(
        "canonical_vqa",
        model=cv.get("model"),
        gguf_sha256=cv.get("gguf_sha256"),
        seat=cv.get("seat"),
        decode="greedy temp=0 repeat_penalty=1.08 max_tokens=16",
    )
    claims = [
        make_claim(
            cid=nid(),
            predicate="canonical_answer",
            value=cv.get("answer") if ok else None,
            region=None,
            confidence=_as_conf(
                "measured" if ok else "withheld",
                "adapted RSVQA model, deterministic greedy decode on 127.0.0.1:8091"
                if ok
                else "canonical_vqa_unavailable",
            ),
            provenance=prov,
        )
    ]
    if not ok:
        err = cv.get("error") or "empty answer"
        limitations.append(
            f"canonical_vqa unavailable ({err}); no narrator substitution — "
            "different model, different provenance."
        )
    limitations.append(
        "canonical_answer is the adapted Qwen3VL-8B-RSVQA model's short answer "
        "(greedy decode, seat 127.0.0.1:8091); it is a model claim, not a "
        "measured number."
    )
    return claims


def build_packet(
    tool_outputs: dict[str, Any] | None,
    task: str,
    *,
    overlay_path: str | None = None,
    mask_path: str | None = None,
    box: list[int] | None = None,
    input_contract: dict[str, Any] | None = None,
    run_id: str | None = None,
    gsd_m: float | None = None,
    modality: str | None = None,
    trace: list[Any] | None = None,
    agreement_map_path: str | None = None,
    geo_exports: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile a typed packet. Qwen does not author canonical_answer or claims."""
    tout = dict(tool_outputs or {})
    nid = _ids()
    claims: list[dict[str, Any]] = []
    limitations: list[str] = []
    region = _region_for(tout, mask_path, overlay_path)
    cd = tout.get("change_detect") or {}
    area = tout.get("area_calc") or None
    wh = tout.get("water_highlight") or {}
    sr = tout.get("sar_read") or {}
    ag = tout.get("sar_agreement") or {}
    sem = tout.get("semantic") or {}
    cm = tout.get("cdvqa_map") or {}
    cv = tout.get("canonical_vqa") or {}
    cg = tout.get("coreg_check") or {}

    canonical: Any = None
    if cd or (task or "").startswith("change"):
        canonical = _change_canonical(cd)
        claims.extend(_claims_change(cd, area, region, nid, limitations))
    else:
        # water_highlight and sar_read co-occur under sar_agreement runs.
        if wh:
            claims.extend(_claims_water(wh, area, region, nid, limitations))
        if sr:
            claims.extend(_claims_sar(sr, area, region, nid, limitations))
        if ag:
            claims.extend(_claims_agreement(ag, region, nid, limitations))
        if cg:
            claims.extend(_claims_coreg(cg, nid, limitations))
        if not (wh or sr or ag or sem or cm or cv):
            limitations.append(
                "No tool-owned canonical answer; Qwen text is not canonical."
            )
    if sem:
        claims.extend(_claims_semantic(sem, region, nid, limitations))
    if cm:
        claims.extend(_claims_cdvqa(cm, region, nid, limitations))
        # A vocab-checked cdvqa answer is the tool-owned canonical answer.
        if cm.get("answer_vocab_ok") and cm.get("answer") is not None:
            canonical = cm["answer"]
    if cv:
        claims.extend(_claims_canonical_vqa(cv, nid, limitations))
        # An available canonical_vqa answer is the tool-owned canonical
        # answer for single-image question plans.
        if cv.get("available") and cv.get("answer") is not None:
            canonical = cv["answer"]

    if region is None and claims:
        limitations.append("Spatial claims have no on-disk region artifact path.")

    gsd = gsd_m
    if gsd is None and area:
        gsd = area.get("gsd_m")
    contract = input_contract or {
        "valid": True,
        "modality": modality,
        "gsd_m": gsd,
    }

    mask_art = mask_path or region
    box_val = box if box is not None else mask_box_xyxy(mask_art)
    artifacts: list[dict[str, Any]] = []
    if mask_art:
        artifacts.append({"type": "mask", "path": mask_art, "grid": "native"})
    if overlay_path:
        artifacts.append({"type": "overlay", "path": overlay_path, "grid": "native"})
    if agreement_map_path:
        artifacts.append(
            {"type": "agreement_map", "path": str(agreement_map_path), "grid": "native"}
        )
    if box_val is not None:
        artifacts.append({"type": "box", "xyxy": box_val, "grid": "native"})
    for name, rec in (geo_exports or {}).items():
        if rec.get("status") == "written" and rec.get("path"):
            artifacts.append(
                {
                    "type": "geotiff",
                    "path": str(rec["path"]),
                    "grid": "native",
                    "crs": rec.get("crs"),
                    "source": rec.get("source"),
                }
            )

    seen: set[str] = set()
    uniq_lim: list[str] = []
    for item in limitations:
        if item not in seen:
            seen.add(item)
            uniq_lim.append(item)

    packet = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id or tool_outputs_sha256(tout),
        "task": task,
        "input_contract": contract,
        "canonical_answer": canonical,
        "claims": claims,
        "artifacts": artifacts,
        "limitations": uniq_lim,
        "trace": list(trace or []),
        "tool_outputs": tout,
    }
    if geo_exports:
        # Recorded export events — written paths AND honest skips (e.g.
        # geo_export=skipped_no_transform), never a silent omission.
        packet["geo_exports"] = geo_exports
    return packet


def build_packet_for_run(
    run_trace: dict[str, Any],
    the_plan: dict[str, Any] | None = None,
    bound: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Packet from a pipeline/app run dict. Does not mutate tool_outputs."""
    plan = the_plan or run_trace.get("plan") or {}
    bound = bound or {}
    paths = bound.get("paths") or {}
    pred = paths.get("pred")
    mask_path = str(pred) if pred else None
    gsd_info = bound.get("gsd") if bound.get("gsd") is not None else run_trace.get("gsd")
    gsd_m = gsd_info.get("gsd_m") if isinstance(gsd_info, dict) else None
    modality = run_trace.get("input_mode")
    ok = bound.get("ok")
    return build_packet(
        run_trace.get("tool_outputs") or {},
        plan.get("task") or "unsupported",
        overlay_path=run_trace.get("overlay_path"),
        mask_path=mask_path,
        input_contract={
            "valid": True if ok is None else bool(ok),
            "modality": modality,
            "gsd_m": gsd_m,
        },
        modality=modality,
        gsd_m=gsd_m,
        trace=[{"plan_task": plan.get("task"), "live": run_trace.get("live")}],
        agreement_map_path=run_trace.get("agreement_map_path"),
        geo_exports=run_trace.get("geo_exports"),
    )


def allowed_rendering(packet: dict[str, Any]) -> dict[str, Any]:
    """JSON the narrator may cite. Qwen may not add numbers beyond this."""
    return {
        "canonical_answer": packet.get("canonical_answer"),
        "claims": packet.get("claims") or [],
        "limitations": packet.get("limitations") or [],
        "tool_outputs": packet.get("tool_outputs") or {},
    }


def check_packet_narration(text: str | None, packet: dict[str, Any]) -> dict[str, Any]:
    """check_narration plus claim-id and unsupported-direction gates."""
    base = check_narration(text, allowed_rendering(packet))
    issues = list(base.get("issues") or [])
    raw = text or ""
    known = {c.get("id") for c in (packet.get("claims") or []) if c.get("id")}
    for m in _CLAIM_ID_RE.finditer(raw):
        cid = m.group(0)
        if cid not in known:
            issues.append(f"unknown claim id {cid}")
    if packet.get("canonical_answer") == CANONICAL_NOT_DETERMINED and _DIRECTION_CLAIM_RE.search(raw):
        issues.append("unsupported increase/decrease claim")
    return {"ok": not issues, "issues": issues}


def render_benchmark(packet: dict[str, Any]) -> dict[str, Any]:
    """Short answer / mask path / box only. Same packet as the product renderer."""
    mask = None
    box = None
    for art in packet.get("artifacts") or []:
        if art.get("type") == "mask" and mask is None:
            mask = art.get("path")
        if art.get("type") == "box" and box is None:
            box = art.get("xyxy")
    return {
        "answer": packet.get("canonical_answer"),
        "mask": mask,
        "box": box,
    }


def _claims_markdown(packet: dict[str, Any]) -> str:
    lines = ["### Claims (from tools)", ""]
    claims = packet.get("claims") or []
    if not claims:
        lines.append("No tool claims this run.")
        return "\n".join(lines)
    for c in claims:
        unit = f" {c['unit']}" if c.get("unit") else ""
        lines.append(
            f"- `{c.get('id')}` {c.get('predicate')}={c.get('value')!r}{unit} "
            f"(region={c.get('region')!r})"
        )
    return "\n".join(lines)


def _limitations_markdown(packet: dict[str, Any]) -> str:
    lines = ["### Limitations", ""]
    lim = packet.get("limitations") or []
    if not lim:
        lines.append("None recorded.")
        return "\n".join(lines)
    for item in lim:
        lines.append(f"- {item}")
    return "\n".join(lines)


def render_product(
    packet: dict[str, Any],
    qwen_text: str | None = None,
) -> str:
    """Findings header first; Qwen explains claims; still gated by check_narration."""
    header = findings_header({"tool_outputs": packet.get("tool_outputs") or {}})
    body_mid = _claims_markdown(packet) + "\n\n" + _limitations_markdown(packet)
    raw = (qwen_text or "").strip()
    if not raw:
        return header + "\n\n" + body_mid
    check = check_packet_narration(raw, packet)
    if not check["ok"]:
        return header + "\n\n" + body_mid + INTERPRETATION_NOTE + raw
    return header + "\n\n" + body_mid + "\n\n" + raw


def validate_packet(packet: dict[str, Any]) -> list[str]:
    """Structural issues (empty list = packet shape ok). Not a PASS verdict."""
    issues: list[str] = []
    for key in ("task", "canonical_answer", "claims", "limitations"):
        if key not in packet:
            issues.append(f"missing field {key}")
    for c in packet.get("claims") or []:
        for f in CLAIM_FIELDS:
            if f not in c:
                issues.append(f"claim {c.get('id')} missing {f}")
        pred = c.get("predicate")
        if pred in {"change_detected", "changed_pixels", "radiometric_delta", "built_up_direction"}:
            if not c.get("region"):
                issues.append(f"spatial claim {c.get('id')} missing region")
        val = c.get("value")
        if pred == "built_up_direction" and val != CANONICAL_NOT_DETERMINED:
            issues.append(f"built_up_direction claim is {val!r}")
    if (packet.get("task") or "").startswith("change"):
        canon = packet.get("canonical_answer")
        if canon != CANONICAL_NOT_DETERMINED:
            cdvqa_vals = {
                c.get("value")
                for c in packet.get("claims") or []
                if c.get("predicate") == "cdvqa_answer"
            }
            if canon not in cdvqa_vals:
                issues.append(
                    f"change canonical_answer is {canon!r}"
                )
    return issues


def dump_prepared_scene2(
    cache_dir: Path | None = None,
    query: str = "What changed between these two dates, and where?",
) -> dict[str, Any]:
    """Write packet/renderer artifacts for prepared Scene 2. live=False, no serve."""
    from planner import plan
    from pipeline import run_query

    out_dir = cache_dir or CACHE
    out_dir.mkdir(parents=True, exist_ok=True)
    the_plan = plan(query, "bi-temporal")
    pred = DEMO / "data" / "scene2" / "pred_mask.png"
    mtime_before = pred.stat().st_mtime if pred.is_file() else None
    trace = run_query(
        query,
        "bi-temporal",
        scene=2,
        live=False,
        cd_prefer="classical",
    )
    if mtime_before is not None:
        mtime_after = pred.stat().st_mtime
        if mtime_after != mtime_before:
            raise RuntimeError("dump_prepared_scene2 rewrote pred_mask.png")
    mask_path = str(pred) if pred.is_file() else None
    gsd = None
    area = (trace.get("tool_outputs") or {}).get("area_calc") or {}
    if area.get("gsd_m") is not None:
        gsd = area.get("gsd_m")
    packet = build_packet(
        trace.get("tool_outputs") or {},
        the_plan.get("task") or "change_description",
        overlay_path=trace.get("overlay_path"),
        mask_path=mask_path,
        input_contract={
            "valid": True,
            "modality": "bi-temporal",
            "gsd_m": gsd,
        },
        modality="bi-temporal",
        gsd_m=gsd,
        trace=[{"plan_task": the_plan.get("task"), "live": False}],
    )
    bench = render_benchmark(packet)
    product = render_product(packet, qwen_text=None)
    invented = check_narration(
        "Pair IoU is 0.99 versus the ground-truth mask.",
        packet.get("tool_outputs") or {},
    )
    invented_pkt = check_packet_narration(
        "Pair IoU is 0.99 versus the ground-truth mask.",
        packet,
    )
    product_bad = render_product(
        packet,
        qwen_text="Pair IoU is 0.99 versus the ground-truth mask.",
    )
    packet_path = out_dir / "scene2_packet.json"
    bench_path = out_dir / "scene2_benchmark.json"
    product_path = out_dir / "scene2_product.md"
    packet_path.write_text(json.dumps(packet, indent=2, default=str) + "\n", encoding="utf-8")
    bench_path.write_text(json.dumps(bench, indent=2, default=str) + "\n", encoding="utf-8")
    product_path.write_text(product + "\n", encoding="utf-8")
    (out_dir / "scene2_product_invented_099.md").write_text(product_bad + "\n", encoding="utf-8")
    report = {
        "task": "EVIDENCE-PACKET",
        "scene": 2,
        "planner_task": the_plan.get("task"),
        "canonical_answer": packet.get("canonical_answer"),
        "claim_predicates": [c.get("predicate") for c in packet.get("claims") or []],
        "benchmark": bench,
        "findings_first": product.startswith("### Findings (from tools)"),
        "invented_0.99_check_narration": invented,
        "invented_0.99_packet_narration": invented_pkt,
        "pred_mask_mtime_unchanged": True,
        "paths": {
            "packet": str(packet_path),
            "benchmark": str(bench_path),
            "product": str(product_path),
        },
        "validate_packet": validate_packet(packet),
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    md = [
        "# EVIDENCE-PACKET Scene 2 dump",
        "",
        f"- planner task: `{the_plan.get('task')}`",
        f"- canonical_answer: `{packet.get('canonical_answer')}`",
        f"- claims: {', '.join(report['claim_predicates'])}",
        f"- benchmark: `{json.dumps(bench, default=str)}`",
        f"- findings_first: `{report['findings_first']}`",
        f"- check_narration invented 0.99: `{invented}`",
        f"- packet narration invented 0.99: `{invented_pkt}`",
        f"- validate_packet: `{report['validate_packet']}`",
        "",
        "Product markdown is findings-first (report.findings_header) then claims.",
        "Qwen does not own canonical_answer, numbers, masks, or confidence.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    report["report_md"] = str(out_dir / "report.md")
    return report


if __name__ == "__main__":
    dumped = dump_prepared_scene2()
    print(json.dumps({k: dumped[k] for k in ("canonical_answer", "benchmark", "invented_0.99_check_narration", "paths") if k in dumped}, indent=2, default=str))
