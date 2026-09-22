"""LR-FOLD Lightning driver — provision, stage, launch, and file ops for the
ONE Studio the spec authorizes (GPU >=40GB; we use A100-40GB — the same class
the original run trained on).

Usage (each subcommand is one bounded action):
    python lightning_drive.py provision     # create+start studio, print facts
    python lightning_drive.py setup         # pip deps + HF base snapshot
    python lightning_drive.py upload        # staged tarballs + code + mix + ckpt
    python lightning_drive.py untar         # extract on studio, verify counts
    python lightning_drive.py launch_smoke  # detached MODE=smoke run
    python lightning_drive.py launch_train  # detached MODE=train run
    python lightning_drive.py status        # studio status + STATUS.json tail
    python lightning_drive.py pull          # one-shot pull of sync targets
    python lightning_drive.py stop          # stop the studio (manual only)

Auth: LIGHTNING_USER_ID + LIGHTNING_API_KEY from SatQuery/.env — never printed.
Studio layout on the remote FS:
    /teamspace/studios/<studio>/lr_fold/
        data/rsvqa_lr/Images_LR/... data/rsvqa_hr/Data/... data/vrsbench/Images_train/...
        mix_lr_fold.jsonl  lightning_lr_fold.py
        ckpt_init/adapter+merger.pt   (the OLD ckpt_final — continuation seed)
        run/ckpt_last/  run/ckpt_final/  run/STATUS.json  run/train_log.jsonl
        run/smoke_report.json  run/report.json  run/nohup.log
"""

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOCAL_CKPT = ROOT / "modal_volume_backup" / "rsvqa_adapt" / "full_qwen3vl" / "ckpt_final"
STAGING = HERE / "staging"

STUDIO_NAME = "satquery-lr-fold"
TEAMSPACE = "default-project"
ORG = "hackathon-wzyvm"
REMOTE = "/teamspace/studios/this_studio/lr_fold"
REMOTE_DATA = REMOTE + "/data"
REMOTE_RUN = REMOTE + "/run"

PIP_DEPS = (
    "torch==2.8.0 torchvision==0.23.0 "
    "'transformers>=4.57.0,<4.60.0' 'peft>=0.15.0,<0.19.0' "
    "'accelerate>=1.6.0,<2.0.0' pillow 'numpy<2.4' safetensors hf_transfer"
)
BASE_ID = "Qwen/Qwen3-VL-8B-Instruct"
BASE_REV = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


def _load_env() -> None:
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def _studio():
    from lightning_sdk import Studio
    return Studio(STUDIO_NAME, teamspace=TEAMSPACE, org=ORG, create_ok=True)


def _run(st, cmd: str, timeout_note: str = "") -> str:
    print(f"$ {cmd}", flush=True)
    out = st.run(cmd)
    print(out if isinstance(out, str) else getattr(out, "output", out), flush=True)
    return out if isinstance(out, str) else getattr(out, "output", str(out))


# --- Windows path quirk -----------------------------------------------------
# Studio.upload_file() runs remote_path through os.path.normpath -> on
# Windows every "/" becomes "\" and the file lands as ONE literal
# backslash-named blob. Same hazard in download_folder (os.path.join into a
# URL). Workaround: call the api layer directly with posix-style relative
# paths (relative to the studio content root = /teamspace/studios/this_studio).

def _up(st, local: str, remote_rel: str) -> None:
    """Upload local file -> studio content root + remote_rel (posix)."""
    st._studio_api.upload_file(
        studio_id=st._studio.id,
        teamspace_id=st._teamspace.id,
        cloud_account=st._studio.cluster_id,
        file_path=str(local),
        remote_path=remote_rel.strip("/"),
        progress_bar=True,
    )


def _dl(st, remote_rel: str, local: Path) -> bool:
    """Download studio file (posix rel path) -> local path. Returns ok."""
    try:
        local.parent.mkdir(parents=True, exist_ok=True)
        st.download_file(remote_rel.strip("/"), str(local))
        return True
    except Exception as e:
        print(f"dl fail {remote_rel}: {type(e).__name__} {str(e)[:160]}",
              flush=True)
        return False


def _dl_dir(st, remote_rel_dir: str, local_dir: Path) -> int:
    """Pull every file under remote_rel_dir via verbatim download_file."""
    out = _run(st, f"find {REMOTE}/{remote_rel_dir} -type f 2>/dev/null")
    n = 0
    for line in out.splitlines():
        rp = line.strip()
        if not rp or not rp.startswith(REMOTE + "/"):
            continue
        rel = rp[len(REMOTE) + 1:]
        if _dl(st, rel, local_dir / Path(rel).relative_to(remote_rel_dir)):
            n += 1
    return n


def provision() -> None:
    from lightning_sdk import Machine
    st = _studio()
    print("status:", st.status, flush=True)
    if str(st.status) not in ("Status.Running", "Running"):
        st.start(Machine.A100_40GB, max_runtime=15 * 3600)
    for _ in range(60):
        s = st.status
        print("status:", s, flush=True)
        if "Running" in str(s):
            break
        time.sleep(10)
    _run(st, "hostname; pwd; nvidia-smi --query-gpu=name,memory.total --format=csv; "
              "free -g | head -2; df -h /teamspace /root 2>/dev/null | tail -3; "
              "python3 --version; which python3 pip3")


def setup() -> None:
    st = _studio()
    _run(st, f"pip install --no-cache-dir {PIP_DEPS} 2>&1 | tail -5")
    _run(st, "python -c \"import torch,transformers,peft;print(torch.__version__,"
             "transformers.__version__,peft.__version__,torch.cuda.is_available(),"
             "torch.cuda.get_device_name(0))\"")
    _run(st, f"mkdir -p {REMOTE} {REMOTE_DATA} {REMOTE_RUN} {REMOTE}/hf && "
             f"HF_HOME={REMOTE}/hf HF_HUB_ENABLE_HF_TRANSFER=1 python -c \""
             f"from huggingface_hub import snapshot_download;"
             f"p=snapshot_download('{BASE_ID}',revision='{BASE_REV}');"
             f"print('BASE',p)\"", "hf download ~16GB")


def upload() -> None:
    st = _studio()
    _run(st, f"mkdir -p {REMOTE}/staging {REMOTE}/ckpt_init/adapter {REMOTE}/code")
    for name in ("lr_images.tar", "hr_images.tar", "vrsb_images.tar"):
        p = STAGING / name
        if not p.exists():
            print(f"SKIP {name} (not staged)", flush=True)
            continue
        t = time.time()
        _up(st, str(p), f"lr_fold/staging/{name}")
        print(f"UP {name} {p.stat().st_size >> 20}MB in {time.time() - t:.0f}s",
              flush=True)
    _up(st, str(HERE / "mix_lr_fold.jsonl"), "lr_fold/mix_lr_fold.jsonl")
    _up(st, str(HERE / "lightning_lr_fold.py"), "lr_fold/code/lightning_lr_fold.py")
    _up(st, str(HERE / "mix_manifest_lr_fold.json"), "lr_fold/mix_manifest_lr_fold.json")
    for fn in ("adapter/adapter_model.safetensors", "adapter/adapter_config.json",
               "merger.pt"):
        src = LOCAL_CKPT / fn
        _up(st, str(src), f"lr_fold/ckpt_init/{fn}")
        print(f"UP ckpt_init/{fn} {src.stat().st_size >> 20}MB", flush=True)


def untar() -> None:
    st = _studio()
    ex = f"{REMOTE}/staging/extract"
    for name in ("lr_images", "hr_images", "vrsb_images"):
        _run(st, f"[ -f {REMOTE}/staging/{name}.tar ] && mkdir -p {ex}/{name} && "
                 f"tar -xf {REMOTE}/staging/{name}.tar -C {ex}/{name} && "
                 f"echo {name} OK || echo {name} MISSING")
    # tars carry data/... trees; relocate under REMOTE_DATA with the mix's
    # relative layout (rsvqa_lr/Images_LR, rsvqa_hr/Data, vrsbench/Images_train)
    _run(st, f"mkdir -p {REMOTE_DATA}/rsvqa_lr {REMOTE_DATA}/rsvqa_hr {REMOTE_DATA}/vrsbench; "
             f"[ -d {ex}/lr_images/data/rsvqa/lr/Images_LR ] && "
             f"mv {ex}/lr_images/data/rsvqa/lr/Images_LR {REMOTE_DATA}/rsvqa_lr/Images_LR; "
             f"[ -d {ex}/hr_images/data/rsvqa/hr/Data ] && "
             f"mv {ex}/hr_images/data/rsvqa/hr/Data {REMOTE_DATA}/rsvqa_hr/Data; "
             f"[ -d {ex}/vrsb_images/data/vrsbench/Images_train ] && "
             f"mv {ex}/vrsb_images/data/vrsbench/Images_train {REMOTE_DATA}/vrsbench/Images_train; "
             f"echo '-- counts --'; "
             f"find {REMOTE_DATA}/rsvqa_lr/Images_LR -type f 2>/dev/null | wc -l; "
             f"find {REMOTE_DATA}/rsvqa_hr/Data -type f 2>/dev/null | wc -l; "
             f"find {REMOTE_DATA}/vrsbench/Images_train -type f 2>/dev/null | wc -l; "
             f"du -sh {REMOTE_DATA} 2>/dev/null")


def _launch(mode: str) -> None:
    st = _studio()
    env = (f"DATA_ROOT={REMOTE_DATA} MIX={REMOTE}/mix_lr_fold.jsonl "
           f"RUN_DIR={REMOTE_RUN} INIT_CKPT={REMOTE}/ckpt_init MODE={mode} "
           f"HF_HOME={REMOTE}/hf TOKENIZERS_PARALLELISM=false PYTHONUTF8=1 "
           f"MODEL_ID={BASE_ID} BASE_REVISION={BASE_REV}")
    if mode == "train":
        env += " CKPT_EVERY=500 LOG_EVERY=50 TIME_CAP_S=50400"
    else:
        env += " LOG_EVERY=10 TIME_CAP_S=3600"
    cmd = (f"cd {REMOTE}/code && setsid nohup env {env} python lightning_lr_fold.py "
           f"> {REMOTE_RUN}/nohup_{mode}.log 2>&1 < /dev/null & echo LAUNCHED pid=$!")
    _run(st, cmd)


def status() -> None:
    st = _studio()
    print("studio:", st.status, flush=True)
    _run(st, f"cat {REMOTE_RUN}/STATUS.json 2>/dev/null; "
             f"tail -3 {REMOTE_RUN}/train_log.jsonl 2>/dev/null; "
             f"tail -5 {REMOTE_RUN}/nohup_train.log 2>/dev/null | cut -c1-400")


def pull() -> None:
    """One-shot sync of STATUS + train log + ckpt_last + final artifacts."""
    st = _studio()
    local_run = HERE / "run"
    local_run.mkdir(exist_ok=True)
    for fn in ("STATUS.json", "train_log.jsonl", "report.json", "smoke_report.json"):
        if _dl(st, f"lr_fold/run/{fn}", local_run / fn):
            print(f"PULLED {fn}", flush=True)
    # rolling ckpt_last — overwrite the same local dir, never timestamped
    n = _dl_dir(st, "run/ckpt_last", local_run / "ckpt_last")
    print(f"PULLED ckpt_last files={n}", flush=True)


def main() -> int:
    _load_env()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "provision":
        provision()
    elif cmd == "setup":
        setup()
    elif cmd == "upload":
        upload()
    elif cmd == "untar":
        untar()
    elif cmd == "launch_smoke":
        _launch("smoke")
    elif cmd == "launch_train":
        _launch("train")
    elif cmd == "status":
        status()
    elif cmd == "pull":
        pull()
    elif cmd == "stop":
        st = _studio()
        st.stop()
        print("stopped", flush=True)
    else:
        print(f"unknown cmd {cmd}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
