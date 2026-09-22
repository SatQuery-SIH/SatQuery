"""SECOND pair loader + binary change target for ChangeFormer-FT.

Prep spec A (integer 1..30 unequal) is kept as ``mask_rule_a``. CF-FT-LABEL-PROBE
found the hunt tree is captain-whu SCD RGB land-cover maps (6 classes + white
no-change), not 1..30 ids. The live loader uses ``CHOSEN_RULE`` (set from
``label_probe.json``). This is a **where-change** mask, not a built-up class
map. Demo ``built_up_direction`` stays ``not_determined``.

Do not rglob this tree into TRAIN-10. Do not rewrite samples.jsonl.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

SATQUERY = Path(__file__).resolve().parent.parent
CF_CACHE = SATQUERY / "gates" / "_cache" / "cf_ft"
IMPORTED_CKPT_ROOT = SATQUERY / "gates" / "_cache" / "changeformer"
CDVQA_EVAL = SATQUERY / "gates" / "cdvqa_eval_ids.json"
SECOND_MODAL_ROOT = Path("/png/png/second")
IMG_SIZE = 256
VALID_CLASS_MIN = 1
VALID_CLASS_MAX = 30
# Live mask rule from CF-FT-LABEL-PROBE (label_probe.json).
CHOSEN_RULE = "rgb6_decode_unequal"

# captain-whu SCD / SECOND visualization (rs15164095 Table 2). White = no-change.
# The "30 classes" in SECOND docs are 6 land-cover types × transitions, not pixel ids 1..30.
SECOND_RGB_TO_CLASS: dict[tuple[int, int, int], int] = {
    (255, 255, 255): 0,  # no-change / unlabeled
    (0, 0, 255): 1,  # water
    (128, 128, 128): 2,  # n.v.g. surface
    (0, 128, 0): 3,  # low vegetation
    (0, 255, 0): 4,  # tree
    (128, 0, 0): 5,  # buildings
    (255, 0, 0): 6,  # playgrounds
}
SECOND_CLASS_MAX = 6


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def imported_ckpt_path() -> Path:
    """Laptop path of the imported LEVIR ``best_ckpt.pt`` (not a live demo picker)."""
    hits = sorted(
        p
        for p in IMPORTED_CKPT_ROOT.rglob("best_ckpt.pt")
        if p.is_file() and p.stat().st_size > 1_000_000
    )
    if not hits:
        raise FileNotFoundError(f"No imported best_ckpt.pt under {IMPORTED_CKPT_ROOT}")
    return hits[0]


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def pair_is_readable(root: Path, name: str) -> tuple[bool, str | None]:
    """Open A/B/label1/label2. Returns (ok, error). Does not invent a mask."""
    root = Path(root)
    for split in ("A", "B", "label1", "label2"):
        p = root / split / name
        try:
            with Image.open(p) as im:
                im.load()
        except Exception as e:
            return False, f"{split}/{name}: {type(e).__name__}: {e}"
    return True, None


def inspect_label(path: Path) -> dict[str, Any]:
    """PIL mode / palette / unique values. Does not invent a remap."""
    with Image.open(path) as im:
        im.load()
        mode = im.mode
        size = list(im.size)
        has_palette = im.mode == "P" or im.palette is not None
        palette_len = None
        if im.mode == "P" and im.palette is not None:
            raw = im.palette.tobytes() if hasattr(im.palette, "tobytes") else bytes(im.palette.getdata()[1])
            palette_len = len(raw) // 3
        arr = np.asarray(im)
    rec: dict[str, Any] = {
        "path": str(path),
        "mode": mode,
        "size": size,
        "ndim": int(arr.ndim),
        "shape": list(arr.shape),
        "has_palette": bool(has_palette),
        "palette_len": palette_len,
        "dtype": str(arr.dtype),
    }
    if arr.ndim == 3:
        flat = arr.reshape(-1, arr.shape[-1])
        if arr.shape[-1] >= 3:
            rgb = flat[:, :3].astype(np.int32)
            uniq, counts = np.unique(rgb, axis=0, return_counts=True)
            rec["unique_rgb"] = [[int(x) for x in row] for row in uniq.tolist()]
            rec["unique_rgb_n"] = int(len(uniq))
            rec["unique_rgb_counts"] = [int(c) for c in counts.tolist()]
            rec["unique_ch0"] = [int(x) for x in np.unique(arr[..., 0]).tolist()]
        rec["scalar"] = arr[..., 0].astype(np.int32)
    else:
        rec["unique"] = [int(x) for x in np.unique(arr).tolist()]
        rec["unique_n"] = int(np.unique(arr).size)
        rec["scalar"] = arr.astype(np.int32)
        hist = {}
        vals, counts = np.unique(arr, return_counts=True)
        for v, c in zip(vals.tolist(), counts.tolist()):
            hist[str(int(v))] = int(c)
        rec["hist"] = hist
    return rec


def load_label(path: Path) -> np.ndarray:
    """Scalar HxW from a label PNG (P indices, L, or RGB channel 0). Prep-spec A/B/C path."""
    with Image.open(path) as im:
        if im.mode == "P":
            arr = np.asarray(im)
        else:
            arr = np.asarray(im)
            if arr.ndim == 3:
                arr = arr[..., 0]
    return arr.astype(np.int32, copy=False)


def load_label_rgb(path: Path) -> np.ndarray:
    """HxWx3 RGB (palette applied if mode P)."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def decode_second_rgb(rgb: np.ndarray) -> np.ndarray:
    """Map SECOND RGB colormap to class ids 0..6. Unmatched → 255 (ignored)."""
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"expected HxWx3 RGB, got {arr.shape}")
    h, w = arr.shape[:2]
    out = np.full((h, w), 255, dtype=np.int32)
    pix = arr[:, :, :3].astype(np.int32)
    for (r, g, b), cid in SECOND_RGB_TO_CLASS.items():
        hit = (pix[:, :, 0] == r) & (pix[:, :, 1] == g) & (pix[:, :, 2] == b)
        out[hit] = cid
    return out


def decode_second_label(path: Path) -> np.ndarray:
    """Class-id map. P-mode with small index set kept as ids; else RGB colormap."""
    with Image.open(path) as im:
        mode = im.mode
        arr = np.asarray(im)
    if mode == "P" or (arr.ndim == 2 and int(np.max(arr, initial=0)) <= SECOND_CLASS_MAX):
        ids = arr.astype(np.int32)
        if ids.ndim == 2 and set(int(x) for x in np.unique(ids).tolist()).issubset(
            set(range(SECOND_CLASS_MAX + 1)) | {255}
        ):
            return ids
    rgb = load_label_rgb(path)
    return decode_second_rgb(rgb)


def valid_class_mask(label: np.ndarray, lo: int = VALID_CLASS_MIN, hi: int = VALID_CLASS_MAX) -> np.ndarray:
    lab = np.asarray(label)
    return (lab >= lo) & (lab <= hi)


def mask_rule_a(label1: np.ndarray, label2: np.ndarray) -> np.ndarray:
    """Prep spec A: both labels in 1..30 and unequal. 0/255 ignored."""
    a = np.asarray(label1)
    b = np.asarray(label2)
    if a.shape != b.shape:
        raise ValueError(f"label shape mismatch {a.shape} vs {b.shape}")
    change = valid_class_mask(a, 1, 30) & valid_class_mask(b, 1, 30) & (a != b)
    return change.astype(np.uint8)


def mask_rule_b(label1: np.ndarray, label2: np.ndarray) -> np.ndarray:
    """Nonzero and not 255, unequal. 0/255 ignored. Value 128 counts as valid if present."""
    a = np.asarray(label1)
    b = np.asarray(label2)
    if a.shape != b.shape:
        raise ValueError(f"label shape mismatch {a.shape} vs {b.shape}")
    va = (a != 0) & (a != 255)
    vb = (b != 0) & (b != 255)
    return (va & vb & (a != b)).astype(np.uint8)


def mask_rule_c(label1: np.ndarray, label2: np.ndarray, positive: int = 255) -> np.ndarray:
    """Binary-map: change where the positive value differs across T1/T2.

    128 is never silently merged into 0 or 255. ``positive`` must be stated
    (default 255 = typical bright change / white-R collapse).
    """
    a = np.asarray(label1)
    b = np.asarray(label2)
    if a.shape != b.shape:
        raise ValueError(f"label shape mismatch {a.shape} vs {b.shape}")
    return ((a == int(positive)) != (b == int(positive))).astype(np.uint8)


def mask_rule_rgb6(label1_ids: np.ndarray, label2_ids: np.ndarray) -> np.ndarray:
    """Decoded SECOND 6-class ids: both in 1..6 and unequal. 0 (white/no-change) and 255 ignored."""
    a = np.asarray(label1_ids)
    b = np.asarray(label2_ids)
    if a.shape != b.shape:
        raise ValueError(f"label shape mismatch {a.shape} vs {b.shape}")
    change = valid_class_mask(a, 1, SECOND_CLASS_MAX) & valid_class_mask(b, 1, SECOND_CLASS_MAX) & (a != b)
    return change.astype(np.uint8)


def binary_change_mask(
    label1: np.ndarray,
    label2: np.ndarray,
    rule: str | None = None,
) -> np.ndarray:
    """Where-change mask under ``rule`` or ``CHOSEN_RULE``. Not built-up direction."""
    r = rule or CHOSEN_RULE
    if r in ("A", "a", "prep_1_30"):
        return mask_rule_a(label1, label2)
    if r in ("B", "b", "nonzero_unequal"):
        return mask_rule_b(label1, label2)
    if r in ("C", "c", "binary_255"):
        return mask_rule_c(label1, label2, positive=255)
    if r in ("C128", "binary_128"):
        return mask_rule_c(label1, label2, positive=128)
    if r in ("rgb6_decode_unequal", "rgb6", "D"):
        return mask_rule_rgb6(label1, label2)
    raise ValueError(f"unknown mask rule {r!r}")


def change_frac(mask: np.ndarray) -> float:
    m = np.asarray(mask).astype(bool)
    return float(m.mean()) if m.size else 0.0


def png_basenames(folder: Path) -> set[str]:
    if not folder.is_dir():
        return set()
    return {p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"}


def complete_pair_ids(root: Path) -> list[str]:
    a = png_basenames(root / "A")
    b = png_basenames(root / "B")
    l1 = png_basenames(root / "label1")
    l2 = png_basenames(root / "label2")
    return sorted(a & b & l1 & l2)


def eval_png_tokens(eval_obj: Any) -> set[str]:
    """PNG basenames + stems from cdvqa_eval_ids.json (no raw-substring false hits)."""
    tokens: set[str] = set()

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            if x.lower().endswith(".png"):
                tokens.add(Path(x).name)
                tokens.add(Path(x).stem)

    walk(eval_obj)
    if isinstance(eval_obj, dict):
        for key in ("pair_ids_test_union", "pair_ids", "ids", "file_names", "names"):
            vals = eval_obj.get(key)
            if isinstance(vals, list):
                for v in vals:
                    s = str(v)
                    tokens.add(s)
                    tokens.add(Path(s).name)
                    tokens.add(Path(s).stem)
    blob = json.dumps(eval_obj) if not isinstance(eval_obj, str) else eval_obj
    for name in re.findall(r"[0-9A-Za-z_.-]+\.png", blob, flags=re.I):
        tokens.add(name)
        tokens.add(Path(name).stem)
    return tokens


def overlap_cdvqa_eval(pair_ids: Iterable[str], eval_obj: Any) -> list[str]:
    tokens = eval_png_tokens(eval_obj)
    hit = []
    for name in pair_ids:
        stem = Path(name).stem
        if name in tokens or Path(name).name in tokens or stem in tokens:
            hit.append(name)
    return sorted(hit)


def freeze_val_ids(complete_ids: Sequence[str], seed: int = 42) -> list[str]:
    """Hold out 10% (round to int, at least 32) with seed=42. Output sorted."""
    ids = sorted(complete_ids)
    n = len(ids)
    if n == 0:
        return []
    n_val = max(32, int(round(0.10 * n)))
    n_val = min(n_val, n)
    rng = random.Random(seed)
    return sorted(rng.sample(ids, n_val))


def _resize_rgb(arr: np.ndarray, size: int) -> np.ndarray:
    im = Image.fromarray(arr).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(im)


def _resize_mask(arr: np.ndarray, size: int) -> np.ndarray:
    im = Image.fromarray(np.asarray(arr).astype(np.uint8), mode="L")
    if im.size != (size, size):
        im = im.resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(im)


def tensorize_rgb(arr: np.ndarray):
    """ImageNet-style ChangeFormer eval norm: to_tensor then mean=0.5 std=0.5."""
    import torch

    t = torch.from_numpy(np.asarray(arr).astype(np.float32) / 255.0)
    if t.ndim != 3:
        raise ValueError(f"RGB array must be HxWx3, got {t.shape}")
    t = t.permute(2, 0, 1)
    return (t - 0.5) / 0.5


class SecondPairDataset:
    """Pair loader from ``{A,B,label1,label2}/`` with the binary change mask above."""

    def __init__(
        self,
        root: str | Path,
        ids: Sequence[str],
        img_size: int = IMG_SIZE,
        rule: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.ids = list(ids)
        self.img_size = int(img_size)
        self.rule = rule or CHOSEN_RULE

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        name = self.ids[index]
        a = _resize_rgb(load_rgb(self.root / "A" / name), self.img_size)
        b = _resize_rgb(load_rgb(self.root / "B" / name), self.img_size)
        rule = self.rule
        if rule in ("rgb6_decode_unequal", "rgb6", "D"):
            l1 = _resize_mask(decode_second_label(self.root / "label1" / name), self.img_size)
            l2 = _resize_mask(decode_second_label(self.root / "label2" / name), self.img_size)
            mask = binary_change_mask(l1, l2, rule="rgb6_decode_unequal")
        else:
            l1 = _resize_mask(load_label(self.root / "label1" / name), self.img_size)
            l2 = _resize_mask(load_label(self.root / "label2" / name), self.img_size)
            mask = binary_change_mask(l1, l2, rule=rule)
        return {
            "A": tensorize_rgb(a),
            "B": tensorize_rgb(b),
            "L": torch.from_numpy(mask.astype(np.int64)),
            "name": name,
            "change_frac": change_frac(mask),
        }


def collate_pairs(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    return {
        "A": torch.stack([x["A"] for x in batch], dim=0),
        "B": torch.stack([x["B"] for x in batch], dim=0),
        "L": torch.stack([x["L"] for x in batch], dim=0),
        "name": [x["name"] for x in batch],
    }
