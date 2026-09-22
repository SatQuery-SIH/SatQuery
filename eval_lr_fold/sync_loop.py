"""LR-FOLD laptop-side sync loop — the "a checkpoint that only exists on the
studio disk is not a checkpoint" rule.

Every INTERVAL seconds:
  1. pull STATUS.json + train_log.jsonl (tiny, always)
  2. if ckpt_last/step.json moved -> pull whole ckpt_last/ over the SAME local
     dir (rolling overwrite; never timestamped copies — disk rule)
  3. if run/report.json exists -> pull it + ckpt_final/ (adapter + merger +
     configs only ~500MB; there is NO merged tree on the studio — merge is a
     Modal-side step, so nothing 16GB can be pulled by accident)
  4. free-disk guard: local free < ~5GB -> pause pulls, write SYNC_PAUSED,
     keep trying STATUS only
  5. studio unreachable (credits died / stopped): count consecutive failures,
     write SYNC_UNREACHABLE with the last synced step — artifacts already
     local stay local; training continues on the studio regardless

Run detached:  start /b python sync_loop.py   (or just leave this session up)
Stop: create eval_lr_fold/STOP_SYNC, or kill the process.
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

STUDIO_NAME = "satquery-lr-fold"
TEAMSPACE = "default-project"
ORG = "hackathon-wzyvm"
REMOTE_RUN = "/teamspace/studios/this_studio/lr_fold/run"

LOCAL_RUN = HERE / "run"           # mirror of remote run dir
LOCAL_CKPT_LAST = LOCAL_RUN / "ckpt_last"
LOCAL_FINAL = HERE / "ckpt_final"

INTERVAL_S = int(os.environ.get("SYNC_INTERVAL_S", "600"))
MIN_FREE_GB = float(os.environ.get("MIN_FREE_GB", "5"))
FAIL_QUIT_AFTER = int(os.environ.get("SYNC_FAIL_QUIT", "30"))  # ~5h at 600s

LOG = HERE / "sync_log.txt"


def _log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def _load_env() -> None:
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def _free_gb() -> float:
    return shutil.disk_usage(HERE.anchor).free / (1 << 30)


def _remote_step(st) -> int | None:
    try:
        out = st.run(f"cat {REMOTE_RUN}/ckpt_last/step.json 2>/dev/null")
        txt = out if isinstance(out, str) else getattr(out, "output", str(out))
        return json.loads(txt)["opt_step"]
    except Exception:
        return None


def _dl(st, remote: str, local: Path) -> bool:
    """remote is a posix path relative to the studio content root
    (/teamspace/studios/this_studio) — Studio.download_file passes it
    verbatim into the blob URL; download_folder is avoided (os.path.join
    builds backslash URLs on Windows)."""
    try:
        local.parent.mkdir(parents=True, exist_ok=True)
        st.download_file(remote.strip("/"), str(local))
        return True
    except Exception as e:
        _log(f"dl fail {remote}: {type(e).__name__}: {str(e)[:120]}")
        return False


def _dl_dir(st, remote_abs_dir: str, local_dir: Path) -> int:
    out = st.run(f"find {remote_abs_dir} -type f 2>/dev/null")
    txt = out if isinstance(out, str) else getattr(out, "output", str(out))
    n = 0
    for line in txt.splitlines():
        rp = line.strip()
        if not rp.startswith(REMOTE_RUN.rsplit("/run", 1)[0] + "/"):
            continue
        rel = rp[len("/teamspace/studios/this_studio") + 1:]   # content-root rel
        sub = rp[len(remote_abs_dir) + 1:]
        if _dl(st, rel, local_dir / sub):
            n += 1
    return n


def main() -> int:
    _load_env()
    from lightning_sdk import Studio

    st = Studio(STUDIO_NAME, teamspace=TEAMSPACE, org=ORG, create_ok=False)
    _log(f"sync loop start interval={INTERVAL_S}s min_free={MIN_FREE_GB}GB")
    last_step = -1
    fails = 0
    got_final = False
    while not (HERE / "STOP_SYNC").exists():
        free = _free_gb()
        try:
            status_ok = _dl(st, "lr_fold/run/STATUS.json", LOCAL_RUN / "STATUS.json")
            _dl(st, "lr_fold/run/train_log.jsonl", LOCAL_RUN / "train_log.jsonl")
            if not status_ok:
                raise RuntimeError("STATUS pull failed")
            fails = 0
        except Exception as e:
            fails += 1
            _log(f"UNREACHABLE {fails}/{FAIL_QUIT_AFTER}: {e}")
            (HERE / "SYNC_UNREACHABLE").write_text(
                f"last synced opt_step={last_step} fails={fails}\n")
            if fails >= FAIL_QUIT_AFTER:
                _log("giving up — studio unreachable too long")
                return 2
            time.sleep(INTERVAL_S)
            continue

        (HERE / "SYNC_UNREACHABLE").unlink(missing_ok=True)
        step = _remote_step(st)
        if free < MIN_FREE_GB:
            (HERE / "SYNC_PAUSED").write_text(
                f"free={free:.1f}GB < {MIN_FREE_GB}GB; ckpt pulls paused; "
                f"remote step={step} local step={last_step}\n")
            _log(f"PAUSED free={free:.1f}GB remote_step={step}")
            time.sleep(INTERVAL_S)
            continue
        (HERE / "SYNC_PAUSED").unlink(missing_ok=True)

        if step is not None and step != last_step:
            t = time.time()
            n = _dl_dir(st, f"{REMOTE_RUN}/ckpt_last", LOCAL_CKPT_LAST)
            if n:
                last_step = step
                _log(f"ckpt_last synced opt_step={step} files={n} "
                     f"free={free:.1f}GB ({time.time() - t:.0f}s)")
            else:
                _log(f"ckpt_last pull found no files (step={step})")

        if not got_final:
            if _dl(st, "lr_fold/run/report.json", LOCAL_RUN / "report.json"):
                t = time.time()
                n = _dl_dir(st, f"{REMOTE_RUN}/ckpt_final", LOCAL_FINAL)
                if n:
                    got_final = True
                    _log(f"ckpt_final synced files={n} ({time.time() - t:.0f}s)")
        if got_final:
            _log("final artifacts local — sync loop complete")
            return 0
        time.sleep(INTERVAL_S)
    _log("STOP_SYNC seen — exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
