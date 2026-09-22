"""SWAP-8091 helper — serve gguf outputs over HTTP with Range support so
`curl -C - --retry` can resume a flaky download byte-correctly. FileResponse
handles If-Range/Range natively; paths are confined to runs/lr_fold/gguf/.

Deploy (profile proxynanmaga):
    python -m modal deploy eval_lr_fold/modal_fileserver.py
Then pull:
    curl -C - --retry 20 --retry-all-errors -o out.gguf <url>?f=<name>
"""
from pathlib import Path

import modal
from modal import fastapi_endpoint
from starlette.responses import FileResponse, PlainTextResponse

vol = modal.Volume.from_name("satquery-data", create_if_missing=False)
DATA = Path("/data")
ROOT = DATA / "runs" / "lr_fold" / "gguf"

app = modal.App("satquery-lrfold-serve")
img = modal.Image.debian_slim(python_version="3.12").pip_install("fastapi")


@app.function(image=img, volumes={str(DATA): vol})
@fastapi_endpoint(method="GET", requires_proxy_auth=False)
def file(f: str = ""):
    vol.reload()
    fp = Path(f)
    if fp.is_absolute() or ".." in fp.parts or len(fp.parts) > 1:
        return PlainTextResponse(
            f"bad path: f={f!r} parts={fp.parts!r}", status_code=400)
    p = ROOT / fp.name
    if not p.is_file():
        listing = [x.name for x in ROOT.iterdir()] if ROOT.is_dir() else \
            ["ROOT missing"]
        return PlainTextResponse(
            f"not found: {p} ; dir={listing}", status_code=404)
    return FileResponse(str(p), filename=p.name)
