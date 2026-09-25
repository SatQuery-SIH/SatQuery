"""Downloadable markdown report, TIFF ingest, measurement card.

Prepared scenes remain the default. Uploads, when attached on a tab, are
inference inputs (see pipeline.bind_inputs) — not preview-only.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from PIL import Image

from ingest import preview_file
from pipeline import PREPARED_NOTE
from tools import display_name_for_rung

DEMO = Path(__file__).resolve().parent
REPORTS = DEMO / "reports"
HONESTY = PREPARED_NOTE


def tool_outputs_sha256(tool_outputs: Any) -> str:
    blob = json.dumps(tool_outputs if tool_outputs is not None else {}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


_BANNED_SDWI = (
    "Synthetic Dual-Wideband",
    "Synthetic Dual Wideband",
    "Dual-Wideband Index",
)
_NUM_RE = re.compile(r"(?<![A-Za-z_])[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _frac_places(num: str) -> int | None:
    if "." not in num:
        return None
    body = num.lstrip("+-")
    if "e" in body.lower():
        return None
    return len(body.split(".", 1)[1])


def _decimal_compatible(tok: str, dump_tok: str) -> bool:
    """True if tok is a decimal prefix or same-precision rounding of dump_tok.

    Both tokens must contain `.`. Integers never match floats this way
    (so dump `0` from `count(mask>0)` cannot excuse narration `0.99`).
    """
    if "." not in tok or "." not in dump_tok:
        return False
    if dump_tok.startswith(tok) or tok.startswith(dump_tok):
        return True
    places = _frac_places(tok)
    if places is None or places < 1:
        return False
    try:
        q = Decimal(1).scaleb(-places)
        rounded = Decimal(dump_tok).quantize(q, rounding=ROUND_HALF_UP)
        return rounded == Decimal(tok)
    except (InvalidOperation, ValueError):
        return False


def _percent_compatible(tok: str, dump_nums: list[str]) -> bool:
    """True if a %-marked token equals a tool value x100 at the token's own
    displayed precision (0.1417 -> "14.17%" passes; "1.42%" / "28.3%" fail).
    """
    try:
        t = Decimal(tok)
    except InvalidOperation:
        return False
    places = _frac_places(tok)
    q = Decimal(1).scaleb(-(places if places is not None else 0))
    for d in dump_nums:
        try:
            if (Decimal(d) * 100).quantize(q, rounding=ROUND_HALF_UP) == t:
                return True
        except (InvalidOperation, ValueError):
            continue
    return False


def check_narration(text: str | None, tool_json: Any) -> dict[str, Any]:
    """CPU check: invented numbers / banned SDWI expansions. No llama-server."""
    raw = text or ""
    dump = json.dumps(tool_json if tool_json is not None else {}, sort_keys=True, default=str)
    dump_nums = [m.group(0) for m in _NUM_RE.finditer(dump)]
    # The canonical_vqa answer is narratable evidence: its number tokens are
    # whitelisted even when formatting differs from the JSON dump (e.g. an
    # answer of "617m2" may be narrated as "617 m2").
    cv = {}
    if isinstance(tool_json, dict):
        cv = tool_json.get("canonical_vqa") or (
            (tool_json.get("tool_outputs") or {}).get("canonical_vqa") or {}
        )
    for key in ("answer", "text"):
        dump_nums.extend(
            m.group(0) for m in _NUM_RE.finditer(str(cv.get(key) or ""))
        )
    issues: list[str] = []
    low = raw.lower()
    for phrase in _BANNED_SDWI:
        if phrase.lower() in low:
            issues.append(f"banned SDWI expansion: {phrase}")
    for m in _NUM_RE.finditer(raw):
        tok = m.group(0)
        if re.fullmatch(r"(?:19|20)\d{2}", tok):
            continue
        if tok in dump or tok in dump_nums:
            continue
        if any(_decimal_compatible(tok, d) for d in dump_nums):
            continue
        tail = raw[m.end():]
        # %-marked tokens only: a derivation check (value x100 at the token's
        # precision), not a percent free-pass — wrong percents still flag.
        if (
            tail.lstrip().startswith("%")
            or re.match(r"(?i)\s*percent\b", tail)
        ) and _percent_compatible(tok, dump_nums):
            continue
        issues.append(f"invented number {tok}")
    # evidence_class qualifier: when every evidence-bearing output this run
    # is an estimate (no measured_index present), the narration may not call
    # anything "measured" or claim exactness — estimates are not measurements.
    est_only = (
        re.search(r'"evidence_class":\s*"(heuristic|learned)_estimate"', dump)
        and '"evidence_class": "measured_index"' not in dump
    )
    if est_only and re.search(r"\b(measur\w*|exactly|precisely)\b", low):
        issues.append("estimate-class output narrated as a measurement")
    return {"ok": not issues, "issues": issues}


def _rung_with_display(
    record: dict[str, Any],
    key: str = "rung",
    display_key: str = "rung_display",
) -> str:
    """Keep the evidence key; show the human label alongside when it differs."""
    rung = record.get(key)
    disp = record.get(display_key) or display_name_for_rung(rung)
    if disp and str(disp) != str(rung):
        return f"`{rung}` ({disp})"
    return f"`{rung}`"


def findings_header(trace: dict[str, Any] | None) -> str:
    """Deterministic Findings block from tool_outputs only (not Qwen)."""
    lines = ["### Findings (from tools)", ""]
    if not trace:
        lines.append("No run yet.")
        return "\n".join(lines)
    tout = trace.get("tool_outputs") or {}
    if not tout:
        lines.append("No tool measurements this run.")
        return "\n".join(lines)
    cd = tout.get("change_detect") or {}
    if cd:
        line = (
            f"- change_detect: rung {_rung_with_display(cd)}; changed_pixels={cd.get('changed_pixels')}; "
            f"radiometric_label=`{cd.get('radiometric_label')}` "
            f"(delta={cd.get('radiometric_delta', cd.get('delta_mean'))}, "
            f"eps={cd.get('direction_epsilon')}); "
            f"built_up_direction=`{cd.get('built_up_direction', 'not_determined')}` "
            "(not a class map)."
        )
        if cd.get("semantic_rung") is not None:
            line = (
                line[:-1]
                + f" semantic_rung {_rung_with_display(cd, 'semantic_rung', 'semantic_rung_display')}."
            )
        lines.append(line)
    area = tout.get("area_calc") or {}
    if area:
        lines.append(
            f"- area_calc: {area.get('changed_pixels')} px; area_m2={area.get('area_m2')}; "
            f"area_km2={area.get('area_km2')}; GSD={area.get('gsd_m')} "
            f"({area.get('formula')})."
        )
    wh = tout.get("water_highlight") or {}
    if wh:
        lines.append(
            f"- water_highlight: water_pixels={wh.get('water_pixels')}; method={wh.get('method')}."
        )
    sr = tout.get("sar_read") or {}
    if sr:
        sdwi = sr.get("sdwi_stats") or {}
        cal = sr.get("water_calibrated")
        lines.append(
            f"- sar_read: water_pixels={sr.get('water_pixels')}; "
            f"water_calibrated={cal}; "
            f"SDWI mean={sdwi.get('mean')} (write SDWI, do not expand)."
        )
    ag = tout.get("sar_agreement") or {}
    if ag:
        verdicts = ag.get("verdicts") or {}
        lines.append(
            f"- sar_agreement: water={verdicts.get('water')} "
            f"built_up={verdicts.get('built_up', 'withheld_no_tool')}"
        )
    cg = tout.get("coreg_check") or {}
    if cg:
        tr = cg.get("transform") or {}
        px = cg.get("pixel") or {}
        off = tr.get("offset_m")
        sh = px.get("shift_px")
        shift_s = (
            f"{sh}px (phase-corr)"
            if sh is not None
            else f"inconclusive({px.get('reason')})"
        )
        lines.append(
            f"- coreg: shift={shift_s}; transform_offset_m="
            f"{off if off is not None else tr.get('status')}; "
            f"same_res={tr.get('same_res')}"
        )
    sem = tout.get("semantic") or {}
    if sem:
        dom = sem.get("dominant_transition") or {}
        lines.append(
            f"- second_semantic: dominant_transition={dom.get('token')} "
            f"built_up_direction={sem.get('built_up_direction')} "
            "(backbone val mIoU 0.417)"
        )
    cm = tout.get("cdvqa_map") or {}
    if cm:
        ans = cm.get("answer")
        if ans is None:
            ans = f"withheld({cm.get('withheld_reason')})"
        lines.append(
            f"- cdvqa_map: answer={ans} type={cm.get('official_type')} "
            f"claim={cm.get('claim')}"
        )
    cv = tout.get("canonical_vqa") or {}
    if cv:
        ans = cv.get("answer")
        if not cv.get("available") or ans is None:
            ans = f"withheld({cv.get('error') or 'empty answer'})"
        lines.append(
            f"- canonical_vqa: answer={ans} model={cv.get('model')} "
            f"seat={cv.get('seat', '127.0.0.1:8091')}"
        )
    gr = tout.get("ground") or {}
    if gr:
        pres = gr.get("presence") or {}
        if gr.get("withheld") or not gr.get("box01"):
            lines.append(
                f"- ground: withheld ({gr.get('withheld_reason')}) "
                f"target={gr.get('target')!r} "
                f"presence={pres.get('answer')!r}"
            )
        else:
            lines.append(
                f"- ground: target={gr.get('target')!r} "
                f"box01={gr.get('box01')} frame={gr.get('frame_tag')} "
                f"presence={pres.get('answer')!r}@{pres.get('seat')} "
                f"evidence={gr.get('evidence_class')} (estimate, not a "
                "measurement)"
            )
    ge = trace.get("geo_exports") or {}
    if ge:
        parts = []
        for name, rec in ge.items():
            if rec.get("status") == "written":
                parts.append(
                    f"{name} -> {Path(str(rec.get('path'))).name} [{rec.get('crs')}]"
                )
            else:
                parts.append(f"{name}: {rec.get('status')}")
        lines.append("- geo_export: " + "; ".join(parts))
    return "\n".join(lines)


# Inserted between the findings header and the written answer when the
# narration check fails — product-voice marker that the summary is prose.
INTERPRETATION_NOTE = (
    "\n\n### Interpretation\n\n"
    "Measured values are computed by the tools; the written summary may "
    "restate them approximately — cite the measurement card.\n\n"
)


def compose_visible_answer(trace: dict[str, Any] | None) -> str:
    """Findings header above the VLM paragraph. Flag unverified interpretation."""
    header = findings_header(trace)
    if not trace:
        return header
    raw = (trace.get("answer") or "").strip()
    if raw.startswith("### Findings (from tools)"):
        return raw
    tout = trace.get("tool_outputs") or {}
    should_check = bool(tout) or bool(trace.get("vlm"))
    if should_check:
        check = check_narration(raw, tout)
        trace["narration_check"] = check
        if not raw:
            return header
        if not check["ok"]:
            return header + INTERPRETATION_NOTE + raw
        return header + "\n\n" + raw
    if not raw:
        return header
    return header + "\n\n" + raw


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def measurement_markdown(trace: dict[str, Any] | None) -> str:
    """Judge-pointable area_calc card. Whole-image pixels ≠ NE quadrant pixels."""
    if not trace:
        return "_No run yet. Run a scene to fill this card._"
    area = (trace.get("tool_outputs") or {}).get("area_calc")
    lines = [
        "### Measurement card (tool `area_calc` — deterministic tool measurement)",
        "",
    ]
    if not area:
        lines.append("No `area_calc` on this run (caption/VQA/refusal).")
        return "\n".join(lines)
    quads = area.get("quadrants") or {}
    ne = int(quads.get("NE") or 0)
    whole = int(area.get("changed_pixels") or 0)
    gsd = area.get("gsd_m")
    px_m2 = area.get("pixel_area_m2")
    lines += [
        f"- **label:** `{area.get('label')}`",
        f"- **changed_pixels (WHOLE IMAGE):** {whole:,}",
        f"- **GSD:** {gsd if gsd is not None else 'withheld'} "
        f"({area.get('gsd_source') or 'from tool record'})",
        f"- **formula:** `{area.get('formula')}`",
        f"- **area_m2 (whole):** {area.get('area_m2')}",
        f"- **area_km2 (whole):** {area.get('area_km2')}",
        f"- **percent_of_image (whole):** {area.get('percent_of_image')}",
        f"- **quadrants (pixels):** NW={quads.get('NW')} NE={quads.get('NE')} "
        f"SW={quads.get('SW')} SE={quads.get('SE')}",
        f"- **dominant_quadrant:** {area.get('dominant_quadrant')}",
        f"- **provenance:** {area.get('provenance')}",
        "",
        "**Reading these numbers:**",
    ]
    if gsd is None or px_m2 is None:
        lines += [
            f"- Whole-image `{whole:,}` px; **m² withheld** (no geotransform / GSD).",
            f"- NE quadrant `{ne:,}` px (still not a whole-image percent).",
            "- Area in km² is withheld; cite this card, not the written summary.",
        ]
        return "\n".join(lines)
    ne_m2 = ne * float(px_m2)
    lines += [
        f"- Whole-image `{whole:,}` px × {px_m2} m²/px = **{area.get('area_m2')} m²** "
        f"({float(area.get('percent_of_image') or 0):.4f}% of image).",
        f"- NE quadrant `{ne:,}` px × {px_m2} m²/px = **{ne_m2} m²** "
        f"(not the whole-image km² / percent).",
        "- The NE figure is quadrant-only; cite this card for whole-image values.",
    ]
    return "\n".join(lines)


def confidence_markdown(trace: dict[str, Any] | None) -> str:
    if not trace:
        return "_No run yet._"
    plan = trace.get("plan") or {}
    tools = plan.get("tools") or []
    vlm_role = plan.get("vlm_role")
    lines = [
        "### Confidence",
        "",
        "**Tool measurements** — raster math / ChangeFormer / SAR threshold. "
        "These are the values to cite.",
    ]
    tout = trace.get("tool_outputs") or {}
    if "area_calc" in tout:
        a = tout["area_calc"]
        lines.append(
            f"- `area_calc`: {a.get('changed_pixels')} px → {a.get('area_m2')} m² "
            f"@ {a.get('gsd_m')} m GSD ({a.get('provenance')})."
        )
    if "change_detect" in tout:
        c = tout["change_detect"]
        vs = c.get("vs_gt") or {}
        iou = vs.get("iou")
        lines.append(
            f"- `change_detect`: rung {_rung_with_display(c)}; "
            + (f"IoU vs GT {iou} (mask_metrics)." if iou is not None else "no GT IoU this call.")
            + f" radiometric_label=`{c.get('radiometric_label')}`;"
            f" built_up_direction=`{c.get('built_up_direction', 'not_determined')}`."
            + (
                f" semantic_rung {_rung_with_display(c, 'semantic_rung', 'semantic_rung_display')}."
                if c.get("semantic_rung") is not None
                else ""
            )
        )
    if "water_highlight" in tout:
        w = tout["water_highlight"]
        lines.append(
            f"- `water_highlight`: water_pixels={w.get('water_pixels')} "
            f"({w.get('method')})."
        )
    if "sar_read" in tout:
        s = tout["sar_read"]
        lines.append(
            f"- `sar_read`: water_pixels={s.get('water_pixels')} "
            f"({s.get('provenance', 'tools.sar_read')})."
        )
    if not tout:
        lines.append("- None this run (planner refused or caption-only).")
    lines += [
        "",
        f"**Interpretation (written summary)** — model role `{vlm_role}`; tools selected: {tools}. "
        "The summary may restate measured values approximately; cite the tool measurements above.",
    ]
    if not plan.get("supported"):
        lines.append(f"- Refusal (not a count/area): {plan.get('refusal')}")
    return "\n".join(lines)


def write_report(trace: dict[str, Any], extra_note: str = "") -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = REPORTS / f"satquery_run_{stamp}.md"
    area = (trace.get("tool_outputs") or {}).get("area_calc") or {}
    trace_sha = tool_outputs_sha256(trace.get("tool_outputs") or {})
    body = [
        "# SatQuery run report",
        "",
        f"- UTC: `{_now()}`",
        f"- trace_sha256: `{trace_sha}`",
        f"- mode: `{'live' if trace.get('live') else 'cached'}`",
        f"- query: {trace.get('query')!r}",
        f"- input_mode: `{trace.get('input_mode')}` scene=`{trace.get('scene')}`",
        f"- source: `{trace.get('input_source') or 'prepared'}`",
        f"- {trace.get('ingest_note') or HONESTY}",
        "",
        extra_note,
        "",
        findings_header(trace),
        "",
        "## Plan",
        "",
        "```json",
        json.dumps(trace.get("plan"), indent=2, default=str),
        "```",
        "",
        measurement_markdown(trace),
        "",
        confidence_markdown(trace),
        "",
        "## Answer (interpretation)",
        "",
        trace.get("answer") or "_empty_",
        "",
        "## Image / mask paths",
        "",
    ]
    for p in trace.get("images") or []:
        body.append(f"- image: `{p}`")
    if trace.get("overlay_path"):
        body.append(f"- overlay/mask: `{trace.get('overlay_path')}`")
    if area:
        body.append(
            f"- area_calc snapshot: changed_pixels={area.get('changed_pixels')} "
            f"GSD={area.get('gsd_m')} area_m2={area.get('area_m2')} "
            f"quadrants={area.get('quadrants')}"
        )
    body += [
        "",
        "## Full tool_outputs",
        "",
        "```json",
        json.dumps(trace.get("tool_outputs"), indent=2, default=str),
        "```",
        "",
        f"- first_token_s: {trace.get('first_token_s')} complete_s: {trace.get('complete_s')}",
    ]
    pkt = trace.get("evidence_packet")
    err = trace.get("evidence_packet_error")
    if pkt or err:
        body += ["", "## Evidence packet", ""]
        if pkt:
            body.append(f"- canonical_answer: `{pkt.get('canonical_answer')}`")
            body.append(f"- packet: `satquery_run_{stamp}_packet.json`")
            body.append(f"- benchmark: `satquery_run_{stamp}_benchmark.json`")
            body.append(f"- product: `satquery_run_{stamp}_product.md`")
        if err:
            body.append(f"- evidence_packet_error: `{err}`")
    path.write_text("\n".join(body), encoding="utf-8")
    return path


def _stamp_from_report(md_path: Path) -> str:
    stem = md_path.stem
    prefix = "satquery_run_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem


def write_packet_artifacts(trace: dict[str, Any], stamp: str) -> dict[str, Path]:
    """Write sidecar packet/benchmark/product files for satquery_run_<stamp>.md."""
    from evidence_packet import (
        build_packet_for_run,
        render_benchmark,
        render_product,
    )

    REPORTS.mkdir(parents=True, exist_ok=True)
    packet = trace.get("evidence_packet")
    if packet is None:
        try:
            packet = build_packet_for_run(trace, trace.get("plan"), None)
        except Exception as e:
            packet = {
                "schema_version": "1.0",
                "task": (trace.get("plan") or {}).get("task") or "unsupported",
                "canonical_answer": None,
                "claims": [],
                "limitations": [f"packet build failed: {type(e).__name__}: {e}"],
                "artifacts": [],
                "tool_outputs": trace.get("tool_outputs") or {},
            }
    bench = render_benchmark(packet)
    product = render_product(packet, qwen_text=trace.get("answer"))
    packet_path = REPORTS / f"satquery_run_{stamp}_packet.json"
    bench_path = REPORTS / f"satquery_run_{stamp}_benchmark.json"
    product_path = REPORTS / f"satquery_run_{stamp}_product.md"
    packet_path.write_text(json.dumps(packet, indent=2, default=str) + "\n", encoding="utf-8")
    bench_path.write_text(json.dumps(bench, indent=2, default=str) + "\n", encoding="utf-8")
    product_path.write_text(product + "\n", encoding="utf-8")
    return {"packet": packet_path, "benchmark": bench_path, "product": product_path}


def write_report_bundle(trace: dict[str, Any], extra_note: str = "") -> dict[str, Path]:
    """Markdown report plus packet/benchmark/product sidecars. Same stamp."""
    md = write_report(trace, extra_note=extra_note)
    stamp = _stamp_from_report(md)
    arts = write_packet_artifacts(trace, stamp)
    return {"md": md, **arts}


def ingest_preview(src: str | Path) -> tuple[Image.Image | None, str]:
    """RGB preview + GSD note. Inference use is decided by pipeline.bind_inputs."""
    return preview_file(src)
