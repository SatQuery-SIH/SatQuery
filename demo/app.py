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
from pipeline import PREPARED_NOTE, run_query, scene_paths  # noqa: E402
from report import (  # noqa: E402
    HONESTY,
    compose_visible_answer,
    confidence_markdown,
    ingest_preview,
    measurement_markdown,
    write_report_bundle,
)

MODE = "live"
LAST_TRACE: dict | None = None


def _banner() -> str:
    if MODE == "cached":
        return (
            "### Demo replay — pre-computed results. No live AI running. "
            " switch to Live mode for real inference."
        )
    return (
        "### Live demo — AI narrator + measurement tools running on this laptop. "
        "No internet needed. All numbers come from measurement tools, not the narrator."
    )


def _load_preview(scene: int):
    try:
        p = scene_paths(scene)
    except Exception:
        return None, None, None
    if scene == 1:
        img = Image.open(p["image"])
        return img, None, None
    if scene in (2, 4):
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
    bundle = write_report_bundle(trace)
    extras = []
    for key in ("packet", "benchmark", "product"):
        p = bundle.get(key)
        if p is not None:
            extras.append(str(p))
    return (
        compose_visible_answer(trace),
        _format_trace(trace),
        _badge(trace),
        overlay,
        measurement_markdown(trace),
        confidence_markdown(trace),
        str(bundle["md"]),
        extras,
    )


def _uploads_for(input_mode: str, file_a, file_b):
    if input_mode == "single":
        return {"image": file_a} if file_a else None
    if input_mode == "bi-temporal":
        if file_a or file_b:
            return {"before": file_a, "after": file_b}
        return None
    if file_a or file_b:
        return {"optical": file_a, "sar": file_b}
    return None


def _run(query: str, input_mode: str, scene: int, file_a=None, file_b=None):
    uploads = _uploads_for(input_mode, file_a, file_b)
    if MODE == "cached":
        if uploads:
            dummy = {
                "plan": {"supported": False, "tools": [], "refusal": "cached+upload"},
                "tool_outputs": {},
                "answer": (
                    "Cached trapdoor uses prepared scenes only. "
                    "Clear the upload boxes, or re-run with --mode live to infer on your files."
                ),
                "live": False,
                "first_token_s": 0.0,
                "complete_s": 0.0,
                "metrics_badge": None,
                "overlay_path": None,
                "images": [],
                "ingest_note": PREPARED_NOTE,
                "input_source": "cached",
            }
            return _pack(dummy, None)
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

    trace = run_query(query, input_mode, scene=scene, live=True, uploads=uploads)
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
        gr.Markdown("# SatQuery AI — ask questions about satellite images")
        banner = gr.Markdown(_banner())

        gr.Markdown(
            "**" + HONESTY + "**  \n"
            "Empty file boxes = built-in examples. "
            "Attach your own GeoTIFF/PNG/JPEG on a tab to analyze **your** files. "
        )

        with gr.Tabs():
            with gr.Tab("1 · Single image"):
                with gr.Row():
                    img1 = gr.Image(value=s1a, label="Satellite image", type="pil")
                    ov1 = gr.Image(label="Highlighted regions", type="pil")
                up1 = gr.File(
                    label="Optional: try your own image (GeoTIFF / PNG / JPEG)",
                    file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"],
                    type="filepath",
                )
                note1 = gr.Markdown(HONESTY)
                up1.change(_preview_upload, inputs=[up1], outputs=[img1, note1])
                q1 = gr.Textbox(
                    value="Describe the land cover and major objects.",
                    label="Ask a question",
                )
                b1 = gr.Button("Analyze image", variant="primary")
                a1 = gr.Textbox(label="Answer", lines=8)
                t1 = gr.Code(label="How this answer was produced (tools used, settings)", language="json")
                with gr.Accordion("Measurements, confidence & downloads", open=False):
                    badge1 = gr.Textbox(label="Key measurements", interactive=False)
                    meas1 = gr.Markdown(measurement_markdown(None))
                    conf1 = gr.Markdown(confidence_markdown(None))
                    report_file1 = gr.File(label="Download report", interactive=False)
                    packet_files1 = gr.File(
                        label="Download evidence files",
                        interactive=False,
                        file_count="multiple",
                    )
                b1.click(
                    lambda q, f: _run(q, "single", 1, f, None),
                    inputs=[q1, up1],
                    outputs=[a1, t1, badge1, ov1, meas1, conf1, report_file1, packet_files1],
                    api_name="scene1",
                )

            with gr.Tab("2 · Compare two dates"):
                scene2_choice = gr.Radio(
                    choices=[
                        ("Housing construction (high-res)", 2),
                        ("Mixed land change (general purpose)", 4),
                    ],
                    value=2,
                    label="Example image pair",
                )
                gr.Markdown(
                    "**Two change-finding models run here, picked automatically:**\n\n"
                    "- **Housing Change Specialist (SECOND-tuned)** for detailed housing imagery, like the built-up example.\n"
                    "- **General Change Model (imported)** for general land change. **Your uploads use this one.**\n\n"
                    "**What kind of land changed?** On the *Mixed land change* example (SECOND-domain type-family only), "
                    "the **Land-Cover Change Describer (6-class)** answers land-type questions: "
                    "frozen-val mIoU **0.417** vs **0.057** for always-majority. **Uploaded pairs skip this step.**\n\n"
                    "**Built-up increase or decrease?** A fixed counting rule decides on this pair: "
                    "*increase iff buildings_B \u2212 buildings_A > 0.005 \u00d7 pixels*. "
                    "Scene 4 live counts 86104 \u2192 176608 (increase). The housing pair stays `not_determined`.\n\n"
                    "<details>\n<summary><i>Fit details (for the curious)</i></summary>\n\n"
                    "Ratio bins re-fit from TRAIN-share deciles: configured=v2 combined 0.554 vs prior 0.506; "
                    "change_ratio_types 0.570 vs prior 0.437 vs majority 0.477. "
                    "SECOND-style detailed pairs: SECOND = Semantic Change Detection dataset.\n"
                    "</details>"
                )
                with gr.Row():
                    img2a = gr.Image(value=s2a, label="Before", type="pil")
                    img2b = gr.Image(value=s2b, label="After", type="pil")
                    ov2 = gr.Image(label="Change overlay (red = change mask)", type="pil")
                with gr.Row():
                    up2a = gr.File(
                        label="Optional: before (T1)",
                        file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"],
                        type="filepath",
                    )
                    up2b = gr.File(
                        label="Optional: after (T2)",
                        file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"],
                        type="filepath",
                    )
                note2 = gr.Markdown(HONESTY)
                up2a.change(_preview_upload, inputs=[up2a], outputs=[img2a, note2])
                up2b.change(_preview_upload, inputs=[up2b], outputs=[img2b, note2])

                def _swap_prepared(sc):
                    a, b, _ = _load_preview(int(sc))
                    return a, b

                scene2_choice.change(_swap_prepared, inputs=[scene2_choice], outputs=[img2a, img2b])
                q2 = gr.Textbox(
                    value="What changed between these two dates, and where?",
                    label="Ask a question",
                )
                b2 = gr.Button("Compare images", variant="primary")
                a2 = gr.Textbox(label="Answer", lines=8)
                t2 = gr.Code(label="How this answer was produced (tools used, settings)", language="json")
                with gr.Accordion("Measurements, confidence & downloads", open=False):
                    badge2 = gr.Textbox(label="Key measurements", interactive=False)
                    meas2 = gr.Markdown(measurement_markdown(None))
                    conf2 = gr.Markdown(confidence_markdown(None))
                    report_file2 = gr.File(label="Download report", interactive=False)
                    packet_files2 = gr.File(
                        label="Download evidence files",
                        interactive=False,
                        file_count="multiple",
                    )
                b2.click(
                    lambda q, a, b, sc: _run(q, "bi-temporal", int(sc), a, b),
                    inputs=[q2, up2a, up2b, scene2_choice],
                    outputs=[a2, t2, badge2, ov2, meas2, conf2, report_file2, packet_files2],
                    api_name="scene2",
                )

            with gr.Tab("3 · Optical + radar"):
                with gr.Row():
                    img3a = gr.Image(value=s3a, label="Optical image", type="pil")
                    img3b = gr.Image(value=s3b, label="Radar image", type="pil")
                    ov3 = gr.Image(label="Water regions found", type="pil")
                with gr.Row():
                    up3a = gr.File(
                        label="Optional: optical",
                        file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"],
                        type="filepath",
                    )
                    up3b = gr.File(
                        label="Optional: SAR (GeoTIFF / PNG / npz)",
                        file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg", ".npz"],
                        type="filepath",
                    )
                note3 = gr.Markdown(HONESTY)
                up3a.change(_preview_upload, inputs=[up3a], outputs=[img3a, note3])
                up3b.change(_preview_upload, inputs=[up3b], outputs=[img3b, note3])
                q3 = gr.Textbox(
                    value="Identify water-covered regions.",
                    label="Ask a question",
                )
                b3 = gr.Button("Analyze pair", variant="primary")
                a3 = gr.Textbox(label="Answer", lines=8)
                t3 = gr.Code(label="How this answer was produced (tools used, settings)", language="json")
                with gr.Accordion("Measurements, confidence & downloads", open=False):
                    badge3 = gr.Textbox(label="Key measurements", interactive=False)
                    meas3 = gr.Markdown(measurement_markdown(None))
                    conf3 = gr.Markdown(confidence_markdown(None))
                    report_file3 = gr.File(label="Download report", interactive=False)
                    packet_files3 = gr.File(
                        label="Download evidence files",
                        interactive=False,
                        file_count="multiple",
                    )
                b3.click(
                    lambda q, a, b: _run(q, "optical+sar", 3, a, b),
                    inputs=[q3, up3a, up3b],
                    outputs=[a3, t3, badge3, ov3, meas3, conf3, report_file3, packet_files3],
                    api_name="scene3",
                )

        gr.Markdown(
            "Ask anything about the images above — the system picks the right tools automatically. "
            "Counts and areas come from measurement tools, never guessed. "
            "Unclear questions get a clear refusal instead of a made-up answer."
        )
        _ = (banner, img1, img2a, img2b, img3a, img3b)
        _ = (badge1, meas1, conf1, report_file1, packet_files1)
        _ = (badge2, meas2, conf2, report_file2, packet_files2)
        _ = (badge3, meas3, conf3, report_file3, packet_files3)
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
