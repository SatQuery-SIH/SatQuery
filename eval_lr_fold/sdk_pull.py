"""Sequential volume pull via modal SDK read_file — fallback for the CLI's
chunked writer, which corrupted a 5GB download on a slow link (observed:
preallocated file, out-of-order chunk writes, final sha != manifest).

Usage: PYTHONUTF8=1 python sdk_pull.py <remote_path> <local_path> [expect_sha256]
"""
import hashlib
import sys
import time
from pathlib import Path

import modal

vol = modal.Volume.from_name("satquery-data")


def main():
    remote, local = sys.argv[1], Path(sys.argv[2])
    want = sys.argv[3] if len(sys.argv) > 3 else None
    local.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    h = hashlib.sha256()
    n = 0
    last = t0
    with local.open("wb") as f:
        for chunk in vol.read_file(remote):
            f.write(chunk)
            h.update(chunk)
            n += len(chunk)
            if time.time() - last > 30:
                mb = n / 1e6
                print(f"  {mb:.0f} MB  {mb/max(time.time()-t0,1):.1f} MB/s",
                      flush=True)
                last = time.time()
    got = h.hexdigest()
    wall = time.time() - t0
    ok = (want is None) or (got == want)
    print(f"DONE {n} bytes in {wall:.0f}s ({n/1e6/wall:.1f} MB/s)")
    print(f"sha256 {got}")
    if want:
        print(f"want   {want}\n{'MATCH' if ok else 'MISMATCH'}")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
