"""One-off: mirror small run artifacts from the satquery-data volume to local.

Usage: python pull_run_artifacts.py [remote_subdir] [local_dir]
"""
import sys
from pathlib import Path

import modal

vol = modal.Volume.from_name("satquery-data")
REMOTE = sys.argv[1] if len(sys.argv) > 1 else "runs/rsvqa_adapt"
LOCAL = Path(sys.argv[2] if len(sys.argv) > 2 else "modal_volume_backup/rsvqa_adapt")

n = 0
for ent in vol.listdir(REMOTE, recursive=True):
    if ent.type == 2:  # directory
        continue
    lp = LOCAL / Path(ent.path).relative_to(REMOTE)
    lp.parent.mkdir(parents=True, exist_ok=True)
    with lp.open("wb") as f:
        vol.read_file_into_fileobj(ent.path, f)
    n += 1
    print(f"{ent.path} -> {lp} ({lp.stat().st_size} B)", flush=True)
print(f"done: {n} files")
