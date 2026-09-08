"""RASTER-HUNT-13 local rows: LEVIR 40, SECOND href/IEEE/HF, lists for Modal.

Does NOT train. Does NOT download Images_train.zip or 155 GB BEN onto the laptop.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import ssl
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent  # SatQuery/
GATES = ROOT / "gates"
OUT = GATES / "_cache" / "prod10"
UNIQUE = OUT / "unique_images.json"
PACK_LOG = OUT / "pack_log.json"
CDVQA_EVAL = GATES / "cdvqa_eval_ids.json"
LEVIR_DEST = OUT / "png" / "levir"
REPORT_MD = OUT / "raster_hunt_report.md"
REPORT_JSON = OUT / "raster_hunt_report.json"
INPUTS = OUT / "raster_hunt_inputs.json"
HF_CACHE = GATES / "_cache" / "hf"

SEED_SPOT = 42
N_VRS_SPOT = 20
LEVIR_HF_REPO = "satellite-image-deep-learning/LEVIR-CD"
LEVIR_HF_FILE = "test.zip"
LEVIR_MD5_EXPECTED = "07d5dd89e46f5c1359e2eca746989ed9"
SCD_URL = "https://captain-whu.github.io/SCD/"
FALLBACK_DRIVE_ID = "1mN8jzCKKK27p3ODGoDgepjiRYGQpB34u"
IEEE_URLS = [
    "https://ieee-dataport.org/open-access/second-semantic-change-detection-dataset",
    "https://ieee-dataport.org/documents/second-semantic-change-detection-dataset",
    "https://dx.doi.org/10.21227/nsbb-ar76",
    "https://ieee-dataport.org/open-access/semantic-change-detection-dataset-second",
]
HF_CDVQA = "https://huggingface.co/api/datasets/YZHJessica/CDVQA"
HF_VRS = "https://huggingface.co/api/datasets/xiang709/VRSBench"
CTX = ssl.create_default_context()

BAN_TEST_45 = "test_45.png"
SPEC_NAMES = [
    "test_1.png",
    "test_102.png",
    "test_110.png",
    "test_112.png",
    "test_119.png",
    "test_124.png",
    "test_18.png",
    "test_19.png",
    "test_20.png",
    "test_26.png",
    "test_30.png",
    "test_31.png",
    "test_39.png",
    "test_54.png",
    "test_57.png",
    "test_69.png",
    "test_7.png",
    "test_72.png",
    "test_75.png",
    "test_81.png",
]


def _get(url: str, timeout: int = 45) -> tuple[int, str, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "SatQuery-raster-hunt/13"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=CTX) as resp:
            body = resp.read()
            text = body.decode("utf-8", errors="replace")
            return int(resp.status), str(resp.geturl()), text
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        return int(e.code), url, body
    except Exception as e:
        return 0, url, f"{type(e).__name__}: {e}"


def md5_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_lists() -> dict:
    unique = json.loads(UNIQUE.read_text(encoding="utf-8"))
    pack = json.loads(PACK_LOG.read_text(encoding="utf-8"))
    eval_ids = json.loads(CDVQA_EVAL.read_text(encoding="utf-8"))
    brief_pairs = list(pack["brief"]["pairs"])
    if brief_pairs != SPEC_NAMES:
        raise RuntimeError(f"STOP: pack_log brief.pairs != spec names: {brief_pairs}")
    if BAN_TEST_45 in brief_pairs:
        raise RuntimeError("STOP: test_45 in brief.pairs")
    vrs = list(unique["vrs"])
    rng = __import__("numpy").random.RandomState(SEED_SPOT)
    spot_idx = sorted(int(x) for x in rng.choice(len(vrs), size=N_VRS_SPOT, replace=False))
    vrs_spot = [vrs[i] for i in spot_idx]
    vrs_spot_basenames = [Path(p).name for p in vrs_spot]
    second_paths = list(unique["second"])
    second_basenames = sorted({Path(p).name for p in second_paths})
    eval_pairs = {str(x) for x in eval_ids.get("pair_ids_test_union") or []}
    overlap = sorted(set(second_basenames) & eval_pairs)
    ben_s2 = [int(Path(p).stem) for p in unique["ben_s2"]]
    ben_s1 = [int(Path(p).stem) for p in unique["ben_s1"]]
    if ben_s2 != ben_s1:
        raise RuntimeError("STOP: ben_s2 stems != ben_s1 stems")
    payload = {
        "n_vrs": len(vrs),
        "n_levir": len(unique["levir"]),
        "n_second_files": len(second_paths),
        "n_second_basenames": len(second_basenames),
        "n_ben": len(ben_s2),
        "vrs_spot_paths": vrs_spot,
        "vrs_spot_basenames": vrs_spot_basenames,
        "vrs_spot_indices": spot_idx,
        "levir_names": brief_pairs,
        "ben_indices": ben_s2,
        "second_basenames": second_basenames,
        "cdvqa_eval_overlap_n": len(overlap),
        "cdvqa_eval_overlap": overlap[:20],
        "test_45_in_brief_pairs": False,
    }
    INPUTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def vrs_card() -> dict:
    status, final, text = _get(HF_VRS)
    license_name = None
    siblings = []
    size = None
    try:
        card = json.loads(text) if text.strip().startswith("{") else {}
    except json.JSONDecodeError:
        card = {}
    if isinstance(card, dict):
        license_name = card.get("cardData", {}).get("license") or card.get("license")
        for s in card.get("siblings") or []:
            name = s.get("rfilename") or s.get("filename")
            if name:
                siblings.append(name)
            if str(name).endswith("Images_train.zip"):
                size = s.get("size")
    return {
        "url": "https://huggingface.co/datasets/xiang709/VRSBench",
        "api_status": status,
        "final_url": final,
        "license": license_name or "CC-BY-4.0 (dataset card; confirm on volume file)",
        "has_images_train_zip": any(str(x).endswith("Images_train.zip") for x in siblings),
        "siblings_zip": [x for x in siblings if str(x).lower().endswith(".zip")],
        "api_size": size,
        "expected_filename": "Images_train.zip",
        "run08_zip_bytes": 8359313269,
    }


def levir_copy(names: list[str]) -> dict:
    from huggingface_hub import hf_hub_download

    (LEVIR_DEST / "A").mkdir(parents=True, exist_ok=True)
    (LEVIR_DEST / "B").mkdir(parents=True, exist_ok=True)
    zip_path = Path(
        hf_hub_download(
            repo_id=LEVIR_HF_REPO,
            filename=LEVIR_HF_FILE,
            repo_type="dataset",
            cache_dir=str(HF_CACHE),
        )
    )
    digest = md5_file(zip_path)
    members_a = {}
    members_b = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for m in zf.namelist():
            base = Path(m).name
            norm = m.replace("\\", "/")
            if "/A/" in f"/{norm}" or norm.startswith("A/") or "/test/A/" in f"/{norm}":
                members_a[base] = m
            if "/B/" in f"/{norm}" or norm.startswith("B/") or "/test/B/" in f"/{norm}":
                members_b[base] = m
        missing = []
        copied = []
        for name in names:
            if name == BAN_TEST_45:
                raise RuntimeError("STOP: attempted to copy banned test_45")
            ma, mb = members_a.get(name), members_b.get(name)
            if not ma or not mb:
                missing.append({"name": name, "A": ma, "B": mb})
                continue
            for side, member in (("A", ma), ("B", mb)):
                dest = LEVIR_DEST / side / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, dest.open("wb") as dst:
                    dst.write(src.read())
                im = Image.open(dest)
                im.verify()
                copied.append(
                    {
                        "path": f"png/levir/{side}/{name}",
                        "bytes": dest.stat().st_size,
                        "member": member,
                    }
                )
    test45_on_disk = (LEVIR_DEST / "A" / BAN_TEST_45).exists() or (LEVIR_DEST / "B" / BAN_TEST_45).exists()
    n_a = len(list((LEVIR_DEST / "A").glob("*.png")))
    n_b = len(list((LEVIR_DEST / "B").glob("*.png")))
    present = sorted(p.name for p in (LEVIR_DEST / "A").glob("*.png"))
    if BAN_TEST_45 in present:
        raise RuntimeError("STOP: test_45 present under png/levir/")
    if missing:
        return {
            "ok": False,
            "blocked": True,
            "reason": "missing names in official test zip; did not substitute",
            "missing": missing,
            "url": f"https://huggingface.co/datasets/{LEVIR_HF_REPO} ({LEVIR_HF_FILE})",
            "license": "academic use; Google Earth terms; no commercial redistribution",
            "zip_path": str(zip_path),
            "zip_bytes": zip_path.stat().st_size,
            "md5": digest,
            "md5_expected": LEVIR_MD5_EXPECTED,
            "md5_match": digest == LEVIR_MD5_EXPECTED,
            "n_files": n_a + n_b,
            "test_45_present": test45_on_disk,
            "names_on_disk": present,
        }
    if present != sorted(names) or n_a != 20 or n_b != 20:
        raise RuntimeError(f"STOP: levir dest names={present} nA={n_a} nB={n_b}")
    return {
        "ok": True,
        "blocked": False,
        "url": f"https://huggingface.co/datasets/{LEVIR_HF_REPO} ({LEVIR_HF_FILE}); page https://justchenhao.github.io/LEVIR/",
        "license": "academic use; Google Earth terms; no commercial redistribution",
        "zip_path": str(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "md5": digest,
        "md5_expected": LEVIR_MD5_EXPECTED,
        "md5_match": digest == LEVIR_MD5_EXPECTED,
        "where": str(LEVIR_DEST),
        "n_files": 40,
        "n_a": n_a,
        "n_b": n_b,
        "test_45_present": False,
        "names": names,
        "copied_sample": copied[:4],
    }


def second_hunt(second_basenames: list[str], overlap_n: int) -> dict:
    status, final, html = _get(SCD_URL)
    hrefs = sorted(set(re.findall(r"https?://(?:drive\.google\.com|docs\.google\.com)[^\"'\s<>]+", html)))
    file_ids = sorted(set(re.findall(r"/file/d/([A-Za-z0-9_-]+)", html)))
    file_ids += sorted(set(re.findall(r"[?&]id=([A-Za-z0-9_-]+)", html)))
    file_ids = sorted(set(file_ids))
    official_href = hrefs[0] if hrefs else None
    used_fallback = False
    drive_id = file_ids[0] if file_ids else None
    if not official_href and not drive_id:
        drive_id = FALLBACK_DRIVE_ID
        used_fallback = True
        official_href = f"https://drive.google.com/file/d/{FALLBACK_DRIVE_ID}/view"
    elif not official_href and drive_id:
        official_href = f"https://drive.google.com/file/d/{drive_id}/view"

    ieee = []
    for url in IEEE_URLS:
        st, fin, body = _get(url)
        ieee.append(
            {
                "url": url,
                "status": st,
                "final_url": fin,
                "n_chars": len(body),
                "snippet": re.sub(r"\s+", " ", body)[:400],
                "mentions_no_files": bool(re.search(r"no files|0 files|not available", body, re.I)),
            }
        )

    hf_st, hf_fin, hf_body = _get(HF_CDVQA)
    hf_note = "401 Unauthorized / dataset not public on the Hub" if hf_st in (401, 403) else f"status={hf_st}"

    sample_names = ["10589.png", "00003.png", "00011.png"]
    return {
        "page_url": SCD_URL,
        "page_status": status,
        "page_final": final,
        "page_has_available_at_google_drive": "available at Google Drive" in html or "Google Drive" in html,
        "drive_hrefs_on_page": hrefs,
        "drive_file_ids_on_page": file_ids,
        "drive_href": official_href,
        "drive_id": drive_id,
        "used_third_party_id_fallback": used_fallback,
        "fallback_id": FALLBACK_DRIVE_ID,
        "advertised_size": None,
        "login_walled": None,
        "license_text": "not stated as a SPDX id on captain-whu.github.io/SCD/; academic SCD benchmark page",
        "ieee": ieee,
        "hf_YZHJessica_CDVQA": {"url": HF_CDVQA, "status": hf_st, "final": hf_fin, "note": hf_note, "body_head": hf_body[:300]},
        "exact_basename_note": (
            "CDVQA file_name examples include 10589.png; unique_images second keys are "
            "png/second/A|B/<file_name>. Zip/tree key is the basename. No download this cycle "
            "(Drive not fetched until Modal spare profile is active). No invented mapping."
        ),
        "sample_pack_basenames": [n for n in sample_names if n in set(second_basenames)] + second_basenames[:3],
        "n_unique_basenames": len(second_basenames),
        "n_files_listed": 2560,
        "matched_n": 0,
        "cdvqa_eval_overlap": overlap_n,
        "blocked": True,
        "blocked_reason": "Drive archive not downloaded this cycle; waiting on spare Modal account. Names not matched against rasters yet.",
    }


def write_report(payload: dict) -> None:
    REPORT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    vrs = payload["sources"]["vrs"]
    lev = payload["sources"]["levir"]
    sec = payload["sources"]["second"]
    s2 = payload["sources"]["ben_s2"]
    s1 = payload["sources"]["ben_s1"]
    lines = [
        "# RASTER-HUNT-13 report",
        "",
        "No PASS/FAIL. No train. `samples.jsonl` / `serve.ps1` not edited.",
        "",
        "| Source | URL | License | Size | Where (volume/path) | n files | blocked? |",
        "|---|---|---|---|---|---|---|",
        f"| vrs | {vrs.get('url')} | {vrs.get('license')} | {vrs.get('size')} | {vrs.get('where')} | {vrs.get('n_files')}/10939 | {vrs.get('blocked')} |",
        f"| levir | {lev.get('url')} | {lev.get('license')} | {lev.get('zip_bytes')} | `{lev.get('where')}` | {lev.get('n_files')}/40 | {lev.get('blocked')} |",
        f"| second | {sec.get('drive_href')} | {sec.get('license_text')} | {sec.get('advertised_size')} | {sec.get('where')} | {sec.get('matched_n')}/2560 | {sec.get('blocked')} |",
        f"| ben_s2 | {s2.get('url')} | {s2.get('license')} | {s2.get('size')} | {s2.get('where')} | {s2.get('n_files')}/4800 | {s2.get('blocked')} |",
        f"| ben_s1 | {s1.get('url')} | {s1.get('license')} | {s1.get('size')} | {s1.get('where')} | {s1.get('n_files')}/4800 | {s1.get('blocked')} |",
        "",
        f"- `second_blocked` = `{payload.get('second_blocked')}`",
        f"- `ben_s1_blocked` = `{payload.get('ben_s1_blocked')}`",
        f"- Drive href = `{sec.get('drive_href')}`",
        f"- `test_45_present` = `{payload.get('test_45_present')}`",
        f"- `cdvqa_eval_overlap` = `{payload.get('cdvqa_eval_overlap')}`",
        f"- VRS zip bytes (Run 08 record / API) = `{vrs.get('zip_bytes')}` / `{vrs.get('api_size')}`",
        f"- VRS 20-id spot-check = `{vrs.get('spot_check')}`",
        f"- Modal profile at last write = `{payload.get('modal_profile')}`",
        "",
        "## Notes",
        "",
        payload.get("notes", ""),
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    lists = load_lists()
    print("LISTS", json.dumps({k: lists[k] for k in lists if k not in ("ben_indices", "second_basenames")}, indent=2), flush=True)
    vrs_meta = vrs_card()
    print("VRS_CARD", json.dumps(vrs_meta, indent=2), flush=True)
    levir = levir_copy(lists["levir_names"])
    print("LEVIR", json.dumps({k: v for k, v in levir.items() if k != "copied_sample"}, indent=2), flush=True)
    second = second_hunt(lists["second_basenames"], lists["cdvqa_eval_overlap_n"])
    print("SECOND", json.dumps({k: v for k, v in second.items() if k != "ieee"}, indent=2), flush=True)
    payload = {
        "task": "RASTER-HUNT-13",
        "partial": True,
        "modal_profile": os.environ.get("MODAL_PROFILE") or "not_run_this_cycle",
        "second_blocked": True,
        "ben_s1_blocked": True,
        "test_45_present": bool(levir.get("test_45_present")),
        "cdvqa_eval_overlap": lists["cdvqa_eval_overlap_n"],
        "drive_href": second.get("drive_href"),
        "vrs_zip_bytes": vrs_meta.get("run08_zip_bytes"),
        "vrs_spot_check": "pending_modal_spare_account",
        "samples_jsonl_untouched": True,
        "serve_ps1_untouched": True,
        "trained": False,
        "new_adapter": False,
        "new_gguf": False,
        "sources": {
            "vrs": {
                "url": vrs_meta["url"],
                "license": vrs_meta["license"],
                "size": vrs_meta.get("run08_zip_bytes"),
                "zip_bytes": vrs_meta.get("run08_zip_bytes"),
                "api_size": vrs_meta.get("api_size"),
                "where": "pending: satquery-vrsbench on spare Modal account (redownload Images_train.zip; do not put 8.4 GB on laptop)",
                "n_files": 0,
                "blocked": False,
                "spot_check": "pending_modal",
                **vrs_meta,
            },
            "levir": levir,
            "second": {**second, "where": "not downloaded", "n_files": 0},
            "ben_s2": {
                "url": "hackelle/BigEarthNetV2-LMDB (CDLA-Permissive) joined to BigEarthNet.txt; 06 volume satquery-ben-lmdb was deleted",
                "license": "CDLA-Permissive",
                "size": None,
                "where": "pending Modal rematerialize on spare account",
                "n_files": 0,
                "blocked": True,
                "blocked_reason": "waiting spare Modal profile; will not dump 155 GB on laptop",
            },
            "ben_s1": {
                "url": "same LMDB, VV band rendered gray-RGB (tools.vv_to_preview)",
                "license": "CDLA-Permissive",
                "size": None,
                "where": "pending",
                "n_files": 0,
                "blocked": True,
                "blocked_reason": "waiting spare Modal profile / LMDB; ben_s1_blocked until rematerialized",
            },
        },
        "lists": {
            "vrs_spot_basenames": lists["vrs_spot_basenames"],
            "levir_names": lists["levir_names"],
            "n_ben": lists["n_ben"],
            "n_second_basenames": lists["n_second_basenames"],
            "cdvqa_eval_overlap": lists["cdvqa_eval_overlap_n"],
        },
        "notes": (
            "Local cycle: LEVIR 40 copied if zip members matched; SECOND Drive href extracted; "
            "IEEE + HF CDVQA rechecked; VRS/BEN/SECOND rasters deferred until `modal profile current` is the spare account. "
            "User authorized redownload onto the new Modal account (not the laptop)."
        ),
    }
    write_report(payload)
    print("WROTE", REPORT_JSON, REPORT_MD, flush=True)
    return 0 if levir.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
