"""Compose fallback_recording.mp4 from the 3 live scene outputs (demo trapdoor clip)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DEMO = Path(__file__).resolve().parent
DATA = DEMO / "data"
TRACES = DEMO / "traces"
FRAMES = DEMO / "traces" / "demo_frames"
FRAMES.mkdir(parents=True, exist_ok=True)
OUT = DEMO / "fallback_recording.mp4"
W, H = 1280, 720


def _font(size: int):
    for p in (
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
    ):
        if Path(p).is_file():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _fit(im: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    im = im.convert("RGB")
    im.thumbnail((bw, bh))
    return im


def _paste(canvas, im, box):
    fitted = _fit(im, box)
    x0, y0, x1, y1 = box
    canvas.paste(fitted, (x0 + (x1 - x0 - fitted.size[0]) // 2, y0 + (y1 - y0 - fitted.size[1]) // 2))


def _wrap(draw, text, font, width):
    words = (text or "").replace("\n", " ").split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines[:8]


def slide(title: str, images: list[tuple[str, Path]], answer: str, badge: str) -> Image.Image:
    canvas = Image.new("RGB", (W, H), (12, 16, 24))
    draw = ImageDraw.Draw(canvas)
    f_title = _font(28)
    f_small = _font(16)
    f_body = _font(18)
    draw.text((24, 16), "SatQuery AI — 3-scene offline demo (SIH26167)", font=f_title, fill=(240, 240, 240))
    draw.text((24, 56), title, font=f_small, fill=(120, 200, 255))
    n = max(1, len(images))
    slot_w = (W - 48) // n
    for i, (label, path) in enumerate(images):
        box = (24 + i * slot_w, 88, 24 + (i + 1) * slot_w - 12, 430)
        if path.is_file():
            _paste(canvas, Image.open(path), box)
        draw.text((box[0], 434), label, font=f_small, fill=(180, 180, 180))
    draw.rectangle((24, 470, W - 24, H - 24), outline=(50, 70, 90), fill=(20, 26, 36))
    y = 478
    draw.text((36, y), badge[:140], font=f_small, fill=(255, 196, 90))
    y += 28
    for line in _wrap(draw, answer, f_body, W - 80):
        draw.text((36, y), line, font=f_body, fill=(230, 230, 230))
        y += 24
    return canvas


def main() -> None:
    s1 = json.loads((TRACES / "scene1.json").read_text(encoding="utf-8"))
    s2 = json.loads((TRACES / "scene2.json").read_text(encoding="utf-8"))
    s3 = json.loads((TRACES / "scene3.json").read_text(encoding="utf-8"))
    badge2 = (
        f"area_calc: {s2['tool_outputs']['area_calc']['area_km2']:.5f} km^2 "
        f"({s2['tool_outputs']['area_calc']['percent_of_image']:.1f}%), "
        f"ChangeFormer IoU vs GT={s2['tool_outputs']['change_detect']['vs_gt']['iou']:.3f}, "
        f"LEVIR n=20 mean IoU={s2['metrics_badge']['mean_iou']:.3f}"
    )
    slides = [
        slide(
            "Scene 1 — Single image (VRSBench)",
            [("VRSBench val PNG", DATA / "scene1" / "05867_0000.png")],
            s1["answer"],
            f"first_token={s1['first_token_s']}s  complete={s1['complete_s']}s  tool=vqa",
        ),
        slide(
            "Scene 2 — Bi-temporal LEVIR-CD  (red = new built-up; numbers from tools)",
            [
                ("Before T1", DATA / "scene2" / "before.png"),
                ("After T2", DATA / "scene2" / "after.png"),
                ("Change overlay", DATA / "scene2" / "overlay.png"),
            ],
            s2["answer"],
            badge2,
        ),
        slide(
            "Scene 3 — Optical + SAR late fusion  (blue = SAR water threshold)",
            [
                ("Sentinel-2 RGB", DATA / "scene3" / "optical.png"),
                ("Sentinel-1 VV", DATA / "scene3" / "sar_vv.png"),
                ("Water overlay", DATA / "scene3" / "overlay_live.png"),
            ],
            s3["answer"],
            f"VV mean={s3['tool_outputs']['sar_read']['vv_db_stats']['mean']:.2f} dB  "
            f"water={s3['tool_outputs']['sar_read']['water_fraction']*100:.1f}%  "
            f"first_token={s3['first_token_s']}s",
        ),
    ]
    idx = 0
    hold = [8, 14, 10]  # frames at 2 fps
    for sl, n in zip(slides, hold):
        for _ in range(n):
            sl.save(FRAMES / f"d{idx:03d}.png")
            idx += 1
    cmd = [
        "ffmpeg", "-y", "-framerate", "2",
        "-i", str(FRAMES / "d%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(OUT),
    ]
    subprocess.check_call(cmd)
    print("wrote", OUT, OUT.stat().st_size)


if __name__ == "__main__":
    main()
