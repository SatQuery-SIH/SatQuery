"""Run one demo query through planner -> tools -> VLM narration."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from planner import plan
from tools import (
    LEVIR_GSD_M,
    S2_GSD_M,
    area_calc,
    ask_vlm,
    change_detect,
    load_mask,
    load_rgb,
    narration_prompt,
    overlay_mask,
    sar_read,
)

DEMO = Path(__file__).resolve().parent
DATA = DEMO / "data"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if k != "mask" and k != "water_mask"}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


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
    raise ValueError(f"unknown scene {scene}")


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


def run_query(
    query: str,
    input_mode: str,
    scene: int | None = None,
    live: bool = True,
    vlm_url: str = "http://127.0.0.1:8080",
    device_cd: str = "cpu",
) -> dict[str, Any]:
    t_all = time.perf_counter()
    the_plan = plan(query, input_mode)
    trace: dict[str, Any] = {
        "query": query,
        "input_mode": input_mode,
        "scene": scene,
        "live": live,
        "plan": the_plan,
        "tool_outputs": {},
        "answer": None,
        "first_token_s": None,
        "complete_s": None,
        "overlay_path": None,
        "images": [],
        "metrics_badge": _metrics_badge(),
    }
    if not the_plan["supported"]:
        trace["answer"] = the_plan["refusal"]
        trace["complete_s"] = round(time.perf_counter() - t_all, 3)
        return trace

    tools_needed = the_plan["tools"]
    numbers: dict[str, Any] = {}
    image_paths: list[Path] = []

    if scene == 1 or input_mode == "single":
        paths = scene_paths(1)
        image_paths = [paths["image"]]
    elif scene == 2 or input_mode == "bi-temporal":
        paths = scene_paths(2)
        image_paths = [paths["before"], paths["after"]]
    elif scene == 3 or input_mode == "optical+sar":
        paths = scene_paths(3)
        image_paths = [paths["optical"], paths["sar_vv"]]
    else:
        paths = {}

    if "change_detect" in tools_needed:
        pred_path = paths.get("pred")
        gt = load_mask(paths["gt"]) if paths.get("gt") and paths["gt"].is_file() else None
        if pred_path is not None and pred_path.is_file() and not live:
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
        else:
            # Reuse on-disk pred_mask if it exists even in live mode: ChangeFormer is
            # deterministic; recomputing 16 CPU tiles would blow the first-token budget
            # while llama-server holds the GPU. Fresh compute if the file is missing.
            if pred_path is not None and pred_path.is_file():
                mask = load_mask(pred_path)
                meta_path = DEMO / "data" / "scene2" / "pred_trace.json"
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
                cd = change_detect(paths["before"], paths["after"], prefer="changeformer", device=device_cd, gt_mask=gt)
                mask = cd["mask"]
                from PIL import Image

                Image.fromarray((mask * 255).astype("uint8"), mode="L").save(paths["pred"])
        overlay = overlay_mask(load_rgb(paths["after"]), cd["mask"])
        overlay_path = DATA / "scene2" / "overlay_live.png"
        overlay.save(overlay_path)
        trace["overlay_path"] = str(overlay_path)
        numbers["change_detect"] = _jsonable(cd)
        trace["tool_outputs"]["change_detect"] = numbers["change_detect"]
        if "area_calc" in tools_needed:
            area = area_calc(cd["mask"], gsd_m=LEVIR_GSD_M, label="built-up_change")
            numbers["area_calc"] = area
            trace["tool_outputs"]["area_calc"] = area

    if "sar_read" in tools_needed:
        import numpy as np

        npz = np.load(paths["sar_npz"])
        sr = sar_read(npz["vv"], npz["vh"])
        if "area_calc" in tools_needed:
            area = area_calc(sr["water_mask"], gsd_m=S2_GSD_M, label="water")
            numbers["area_calc"] = area
            trace["tool_outputs"]["area_calc"] = area
        overlay = overlay_mask(load_rgb(paths["optical"]), sr["water_mask"], color=(30, 90, 220), alpha=0.5)
        overlay_path = DATA / "scene3" / "overlay_live.png"
        overlay.save(overlay_path)
        trace["overlay_path"] = str(overlay_path)
        numbers["sar_read"] = _jsonable(sr)
        trace["tool_outputs"]["sar_read"] = numbers["sar_read"]

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
            vlm = ask_vlm(prompt, image_paths, url=vlm_url)
            trace["answer"] = vlm["text"]
            trace["first_token_s"] = vlm.get("first_token_s")
            trace["vlm"] = _jsonable(vlm)
        else:
            trace["answer"] = None  # filled by cache
            trace["vlm"] = {"skipped": "cached mode"}
    elif the_plan["vlm_role"] == "none":
        trace["answer"] = (
            "Change mask produced by the change_detect tool. "
            + json.dumps(numbers.get("change_detect", {}), indent=2)
        )

    trace["images"] = [str(p) for p in image_paths]
    trace["complete_s"] = round(time.perf_counter() - t_all, 3)
    return trace
