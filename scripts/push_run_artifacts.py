"""One-off: upload local dir to the satquery-data volume via batch_upload.

Usage: python push_run_artifacts.py <local_dir> <remote_dir>
"""
import sys
from pathlib import Path

import modal

vol = modal.Volume.from_name("satquery-data")
LOCAL = Path(sys.argv[1])
REMOTE = sys.argv[2]

files = [p for p in LOCAL.rglob("*") if p.is_file()]
with vol.batch_upload(force=True) as up:
    for p in files:
        rp = REMOTE + "/" + p.relative_to(LOCAL).as_posix()
        up.put_file(str(p), rp)
        print(f"{p} -> {rp}", flush=True)
print(f"done: {len(files)} files")
