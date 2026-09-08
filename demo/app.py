"""Gradio UI — 3 input-mode tabs, agent-trace panel, mask overlay (DEMO-SPEC-05)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
from PIL import Image

DEMO = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO))

from cache import match_cached  # noqa: E402
from pipeline import run_query, scene_paths  # noqa: E402
from report import (  # noqa: E402
    HONESTY,
    confidence_markdown,
    ingest_preview,
    measurement_markdown,
    write_report,
)

MODE = "live"
LAST_TRACE: dict | None = None


def _banner() -> str:
    if MODE == "cached":
        return (
            "### CACHED MODE — trapdoor. Traces are pre-computed. Zero GPU. "
            "This banner is required when cache is active (honesty is the brand). "
            "Adapter off. Narrator is still the zero-shot Qwen3-VL-8B voice (replay)."
        )
    return (
        "### LIVE = Qwen3-VL-8B zero-shot, adapter off. "
        "`serve.ps1` loads **base** Q4 + **base** mmproj on port **8080**. "
        "Run 06/08 adapters are parked and not in this UI."
    )


def _load_preview(scene: int):
    try:
        p = scene_paths(scene)
    except Exception:
        return None, None, None
    if scene == 1:
        img = Image.open(p["image"])
        return img, None, None
    if scene == 2:
        return Image.open(p["before"]), Image.open(p["after"]), None
    return Image.open(p["optical"]), Image.open(p["sar_vv"]), None


def _format_trace(trace: dict) -> str:
    slim = {
        "plan": trace.get("plan"),
        "tool_outputs": trace.get("tool_outputs"),
        "metrics_badge": trace.get("metrics_badge"),
        "first_token_s": trace.get("first_token_s"),
        "complete_s": trace.get("complete_s"),
        "live": trace.get("live"),
    }
    return json.dumps(slim, indent=2, default=str)


def _badge(trace: dict) -> str:
    mb = trace.get("metrics_badge") or {}
    area = (trace.get("tool_outputs") or {}).get("area_calc") or {}
    bits = []
    if MODE == "cached":
        bits.append("MODE=cached")
    else:
        bits.append("MODE=live zero-shot adapter-off")
    if mb.get("mean_iou") is not None:
        bits.append(f"LEVIR n={mb.get('n')} IoU={mb['mean_iou']:.3f} F1={mb['mean_f1']:.3f} ({mb.get('rung')})")
    if area:
        quads = area.get("quadrants") or {}
        pct = area.get("percent_of_image")
        pct_s = f"{float(pct):.2f}" if pct is not None else "n/a"
        bits.append(
            f"WHOLE px={area.get('changed_pixels')} "
            f"m2={area.get('area_m2')} km2={area.get('area_km2')} "
            f"pct={pct_s} GSD={area.get('gsd_m')}m "
            f"NE_px={quads.get('NE')} (≠ whole) tool=area_calc"
        )
    ft = trace.get("first_token_s")
    if ft is not None:
        bits.append(f"first_token={ft}s complete={trace.get('complete_s')}s")
    return " | ".join(bits) if bits else "no metrics yet"


def _pack(trace: dict, overlay):
    global LAST_TRACE
    LAST_TRACE = trace
    report = write_report(trace)
    return (
        trace.get("answer") or "",
        _format_trace(trace),
        _badge(trace),
        overlay,
        measurement_markdown(trace),
        confidence_markdown(trace),
        str(report),
    )


def _run(query: str, input_mode: str, scene: int):
    if MODE == "cached":
        hit = match_cached(query, input_mode)
        if hit is None:
            # still route through the planner for OOS / free-ask without a cached answer
            from planner import plan

            p = plan(query, input_mode)
            dummy = {
                "plan": p,
                "tool_outputs": {},
                "answer": p.get("refusal")
                or "No cached trace for this free-ask. Re-run with --mode live, or pick a prepared question.",
                "live": False,
                "first_token_s": 0.0,
                "complete_s": 0.0,
                "metrics_badge": None,
                "overlay_path": None,
                "images": [],
            }
            return _pack(dummy, None)
        overlay = None
        if hit.get("overlay_path") and Path(hit["overlay_path"]).is_file():
            overlay = Image.open(hit["overlay_path"])
        elif scene == 2 and (DEMO / "data" / "scene2" / "overlay.png").is_file():
            overlay = Image.open(DEMO / "data" / "scene2" / "overlay.png")
        elif scene == 3 and (DEMO / "data" / "scene3" / "water_overlay.png").is_file():
            overlay = Image.open(DEMO / "data" / "scene3" / "water_overlay.png")
        hit["live"] = False
        return _pack(hit, overlay)

    trace = run_query(query, input_mode, scene=scene, live=True)
    overlay = None
    if trace.get("overlay_path") and Path(trace["overlay_path"]).is_file():
        overlay = Image.open(trace["overlay_path"])
    return _pack(trace, overlay)


def _preview_upload(file_obj):
    if file_obj is None:
        return None, HONESTY
    path = file_obj
    if isinstance(file_obj, dict):
        path = file_obj.get("path") or file_obj.get("name")
    im, note = ingest_preview(path)
    return im, note


def build_app() -> gr.Blocks:
    s1a, _, _ = _load_preview(1)
    s2a, s2b, _ = _load_preview(2)
    s3a, s3b, _ = _load_preview(3)

    with gr.Blocks(title="SatQuery AI") as app:
        gr.Markdown("# SatQuery AI — internal demo (SIH26167)")
        banner = gr.Markdown(_banner())
        badge = gr.Textbox(label="Metric badge (every number from a tool)", interactive=False)
        meas = gr.Markdown(measurement_markdown(None), elem_id="measurement-card")
        conf = gr.Markdown(confidence_markdown(None), elem_id="confidence-card")
        report_file = gr.File(label="Download last run (markdown)", interactive=False)

        gr.Markdown(
            "**" + HONESTY + "**  \n"
            "PNG/JPEG remain OK. GeoTIFF/TIFF is previewed here (rasterio if installed, else first-page RGB fallback)."
        )
        with gr.Row():
            up = gr.File(
                label="Upload GeoTIFF / TIFF / PNG / JPEG (preview only)",
                file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"],
                type="filepath",
            )
            prev = gr.Image(label="Ingest preview (not used for inference)", type="pil")
        ingest_note = gr.Markdown(HONESTY)
        up.change(_preview_upload, inputs=[up], outputs=[prev, ingest_note])

        with gr.Tabs():
            with gr.Tab("1 · Single image"):
                with gr.Row():
                    img1 = gr.Image(value=s1a, label="Scene 1 (VRSBench)", type="pil")
                    ov1 = gr.Image(label="Overlay (none for single-image caption)", type="pil")
                q1 = gr.Textbox(
                    value="Describe the land cover and major objects.",
                    label="Query",
                )
                b1 = gr.Button("Run Scene 1", variant="primary")
                a1 = gr.Textbox(label="Answer (VLM interpretation — not a measurement)", lines=8)
                t1 = gr.Code(label="Agent trace (JSON plan + tool outputs)", language="json")
                b1.click(
                    lambda q: _run(q, "single", 1),
                    inputs=[q1],
                    outputs=[a1, t1, badge, ov1, meas, conf, report_file],
                    api_name="scene1",
                )

            with gr.Tab("2 · Bi-temporal (money scene)"):
                with gr.Row():
                    img2a = gr.Image(value=s2a, label="Before (LEVIR-CD T1 / test_45)", type="pil")
                    img2b = gr.Image(value=s2b, label="After (LEVIR-CD T2 / test_45)", type="pil")
                    ov2 = gr.Image(label="Change overlay (red = new built-up)", type="pil")
                q2 = gr.Textbox(
                    value="What changed between these two dates, and where?",
                    label="Query",
                )
                b2 = gr.Button("Run Scene 2", variant="primary")
                a2 = gr.Textbox(label="Answer (VLM interpretation — not a measurement)", lines=8)
                t2 = gr.Code(label="Agent trace (JSON plan + tool outputs)", language="json")
                b2.click(
                    lambda q: _run(q, "bi-temporal", 2),
                    inputs=[q2],
                    outputs=[a2, t2, badge, ov2, meas, conf, report_file],
                    api_name="scene2",
                )

            with gr.Tab("3 · Optical + SAR"):
                with gr.Row():
                    img3a = gr.Image(value=s3a, label="Optical (Sentinel-2 RGB, Lithuania BEN)", type="pil")
                    img3b = gr.Image(value=s3b, label="SAR VV (Sentinel-1)", type="pil")
                    ov3 = gr.Image(label="Water overlay (blue, from SAR threshold)", type="pil")
                q3 = gr.Textbox(
                    value="Identify water-covered regions.",
                    label="Query",
                )
                b3 = gr.Button("Run Scene 3", variant="primary")
                a3 = gr.Textbox(label="Answer (VLM interpretation — not a measurement)", lines=8)
                t3 = gr.Code(label="Agent trace (JSON plan + tool outputs)", language="json")
                b3.click(
                    lambda q: _run(q, "optical+sar", 3),
                    inputs=[q3],
                    outputs=[a3, t3, badge, ov3, meas, conf, report_file],
                    api_name="scene3",
                )

        gr.Markdown(
            "Free-ask uses the same planner. Out-of-scope queries (including **how many buildings**) "
            "get a refusal card — no invented counts. "
            "Numbers (area, IoU, backscatter) come from tools, never from the VLM. "
            "Whole-image changed_pixels (e.g. 270,611) is **not** the NE quadrant (81,882)."
        )
        _ = (banner, img1, img2a, img2b, img3a, img3b)
    return app


def main() -> None:
    global MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["live", "cached"], default="live")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    MODE = args.mode
    app = build_app()
    # Gradio must run fully offline (spec row A). Prevent phone-home.
    app.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        inbrowser=False,
    )


if __name__ == "__main__":
    main()
