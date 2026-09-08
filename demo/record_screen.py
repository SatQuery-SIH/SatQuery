"""Grab the desktop while the Gradio UI is open and encode fallback_recording.mp4."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from PIL import ImageGrab

DEMO = Path(__file__).resolve().parent
FRAMES = DEMO / "traces" / "frames"
FRAMES.mkdir(parents=True, exist_ok=True)
OUT = DEMO / "fallback_recording.mp4"


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 36
    delay = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
    print(f"capturing {n} frames delay={delay}", flush=True)
    for i in range(n):
        img = ImageGrab.grab()
        # downscale for size
        img = img.resize((1280, 720))
        img.save(FRAMES / f"f{i:03d}.png")
        print(i, flush=True)
        time.sleep(delay)
    cmd = [
        "ffmpeg", "-y", "-framerate", "2",
        "-i", str(FRAMES / "f%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(OUT),
    ]
    print("ffmpeg", cmd, flush=True)
    subprocess.check_call(cmd)
    print("wrote", OUT, "size", OUT.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
