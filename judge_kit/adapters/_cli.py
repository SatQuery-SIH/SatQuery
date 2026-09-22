"""Shared --from / --to CLI for named adapters."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable


def run_convert(convert: Callable[..., list[dict]], *, description: str) -> None:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--from", dest="src", required=True, help="Source file or folder")
    p.add_argument("--to", dest="dst", required=True, help="Destination questions.jsonl")
    p.add_argument("--images", dest="images", default="", help="Optional images directory")
    args = p.parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    images = Path(args.images) if args.images else None
    rows = convert(src, images_dir=images)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {dst}")
