"""Resumable part puller with stall watchdog — pulls
runs/lr_fold/gguf/parts/<name>.partNNN via SDK read_file, restarting any
stream that delivers no bytes for STALL_S seconds. Concatenates verified
parts into the final file and checks the sha256.

Usage: PYTHONUTF8=1 python pull_parts.py <remote_name> <local_out> <want_sha>
"""
import hashlib
import json
import queue
import sys
import threading
import time
from pathlib import Path

import modal

vol = modal.Volume.from_name("satquery-data")
PARTS = "runs/lr_fold/gguf/parts"
STALL_S = 120
MAX_ATTEMPT = 15


def stream(remote, local: Path):
    """One attempt: write stream chunks to local; raise TimeoutError on stall."""
    q: queue.Queue = queue.Queue()

    def feed():
        try:
            for c in vol.read_file(remote):
                q.put(c)
            q.put(None)
        except Exception as e:  # noqa: BLE001
            q.put(e)

    threading.Thread(target=feed, daemon=True).start()
    n = 0
    with local.open("wb") as f:
        while True:
            c = q.get(timeout=STALL_S)
            if c is None:
                break
            if isinstance(c, Exception):
                raise c
            f.write(c)
            n += len(c)
    return n


def pull_part(remote, local: Path):
    for attempt in range(1, MAX_ATTEMPT + 1):
        try:
            n = stream(remote, local)
            return n
        except Exception as e:  # noqa: BLE001
            print(f"    {local.name} attempt {attempt}: {type(e).__name__} "
                  f"{e} — retrying", flush=True)
            time.sleep(3)
    raise RuntimeError(f"{remote}: {MAX_ATTEMPT} failed attempts")


def main():
    name, out, want = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    man = json.loads(vol.read_file(f"{PARTS}/parts_manifest.json")
                     .__iter__().__next__() if False else
                     b"".join(vol.read_file(f"{PARTS}/parts_manifest.json")))
    nparts = man["parts"][name]["n_parts"]
    total = man["parts"][name]["bytes"]
    print(f"{name}: {nparts} parts, {total/1e9:.2f} GB", flush=True)

    pdir = out.parent / (out.name + ".parts")
    pdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done_bytes = 0
    for i in range(nparts):
        pt = pdir / f"part{i:03d}"
        remote = f"{PARTS}/{name}.part{i:03d}"
        if pt.exists() and pt.stat().st_size > 0:
            done_bytes += pt.stat().st_size
            print(f"  part{i:03d} cached ({pt.stat().st_size/1e6:.0f} MB)",
                  flush=True)
            continue
        n = pull_part(remote, pt)
        done_bytes += n
        rate = done_bytes / 1e6 / max(time.time() - t0, 1)
        print(f"  part{i:03d} ok ({n/1e6:.0f} MB) — "
              f"{done_bytes/1e6:.0f}/{total/1e6:.0f} MB @ {rate:.1f} MB/s",
              flush=True)

    # concat + verify
    h = hashlib.sha256()
    with out.open("wb") as fo:
        for i in range(nparts):
            data = (pdir / f"part{i:03d}").read_bytes()
            fo.write(data)
            h.update(data)
    got = h.hexdigest()
    ok = got == want and out.stat().st_size == total
    print(f"CONCAT {out.name}: {out.stat().st_size} bytes")
    print(f"sha256 {got}\nwant   {want}\n{'MATCH' if ok else 'MISMATCH'}")
    if ok:
        import shutil
        shutil.rmtree(pdir)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
