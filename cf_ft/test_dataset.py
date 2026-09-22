"""CPU tests for SECOND binary change target. No Modal. No GPU."""
from __future__ import annotations

import unittest

import numpy as np

from cf_ft.dataset import (
    CHOSEN_RULE,
    binary_change_mask,
    decode_second_rgb,
    freeze_val_ids,
    mask_rule_a,
    mask_rule_rgb6,
    overlap_cdvqa_eval,
    pair_is_readable,
    valid_class_mask,
)


class BinaryChangeMaskTests(unittest.TestCase):
    def test_rule_a_dummy_8x8_change_only_where_both_valid_and_unequal(self) -> None:
        label1 = np.zeros((8, 8), dtype=np.uint8)
        label2 = np.zeros((8, 8), dtype=np.uint8)

        label1[0, 0] = 1
        label2[0, 0] = 1  # valid, equal -> no change

        label1[0, 1] = 1
        label2[0, 1] = 2  # valid, unequal -> change

        label1[0, 2] = 0
        label2[0, 2] = 5  # 0 ignored

        label1[0, 3] = 7
        label2[0, 3] = 255  # 255 ignored

        label1[1, 0] = 31
        label2[1, 0] = 2  # 31 not in 1..30

        label1[1, 1] = 30
        label2[1, 1] = 1  # valid, unequal -> change

        label1[2, 2] = 255
        label2[2, 2] = 255  # both ignore

        change = mask_rule_a(label1, label2)
        self.assertEqual(change.shape, (8, 8))
        self.assertEqual(change.dtype, np.uint8)
        self.assertEqual(int(change.sum()), 2)
        self.assertEqual(int(change[0, 0]), 0)
        self.assertEqual(int(change[0, 1]), 1)
        self.assertEqual(int(change[0, 2]), 0)
        self.assertEqual(int(change[0, 3]), 0)
        self.assertEqual(int(change[1, 0]), 0)
        self.assertEqual(int(change[1, 1]), 1)
        self.assertEqual(int(change[2, 2]), 0)

    def test_valid_class_bounds(self) -> None:
        lab = np.array([[0, 1, 30, 31, 255]], dtype=np.int32)
        m = valid_class_mask(lab)
        self.assertEqual(m.tolist(), [[False, True, True, False, False]])

    def test_freeze_val_seed42_sorted_at_least_32(self) -> None:
        ids = [f"{i:05d}.png" for i in range(200)]
        val = freeze_val_ids(ids, seed=42)
        self.assertEqual(val, sorted(val))
        self.assertEqual(len(val), 32)  # max(32, round(20)) = 32
        val2 = freeze_val_ids(ids, seed=42)
        self.assertEqual(val, val2)

    def test_overlap_uses_png_tokens_not_qa_ids(self) -> None:
        eval_obj = {
            "pair_ids_test_union": ["00017.png"],
            "qa_ids_holdout_n100": ["test1:11448"],
        }
        hit = overlap_cdvqa_eval(["00017.png", "00003.png"], eval_obj)
        self.assertEqual(hit, ["00017.png"])

    def test_chosen_rule_is_rgb6(self) -> None:
        self.assertEqual(CHOSEN_RULE, "rgb6_decode_unequal")

    def test_rgb6_8x8_observed_encoding(self) -> None:
        """Fixture matching hunt RGB labels: 6-class colormap + white no-change.

        Channel-0 of this map is {0,128,255}; 128 is n.v.g. gray (128,128,128),
        not a silent ignore.
        """
        white = (255, 255, 255)
        gray = (128, 128, 128)  # n.v.g. class 2
        buildings = (128, 0, 0)  # class 5
        playgrounds = (255, 0, 0)  # class 6
        water = (0, 0, 255)  # class 1

        l1 = np.zeros((8, 8, 3), dtype=np.uint8)
        l2 = np.zeros((8, 8, 3), dtype=np.uint8)
        l1[:] = white
        l2[:] = white

        l1[0, 0] = gray
        l2[0, 0] = gray  # same class -> no change

        l1[0, 1] = gray
        l2[0, 1] = buildings  # 2 vs 5 -> change

        l1[0, 2] = white
        l2[0, 2] = buildings  # white (0) ignored -> no change

        l1[0, 3] = playgrounds
        l2[0, 3] = water  # 6 vs 1 -> change

        ids1 = decode_second_rgb(l1)
        ids2 = decode_second_rgb(l2)
        self.assertEqual(int(ids1[0, 0]), 2)
        self.assertEqual(int(ids2[0, 0]), 2)
        self.assertEqual(int(ids1[0, 1]), 2)
        self.assertEqual(int(ids2[0, 1]), 5)
        self.assertEqual(int(ids1[0, 2]), 0)
        self.assertEqual(int(ids2[0, 3]), 1)

        # Channel-0 unique set matches the prep probe {0,128,255}
        ch0 = set(int(x) for x in np.unique(l1[..., 0]).tolist()) | set(
            int(x) for x in np.unique(l2[..., 0]).tolist()
        )
        self.assertEqual(ch0, {0, 128, 255})

        change = mask_rule_rgb6(ids1, ids2)
        default = binary_change_mask(ids1, ids2)
        self.assertTrue(np.array_equal(change, default))
        self.assertEqual(int(change.sum()), 2)
        self.assertEqual(int(change[0, 0]), 0)
        self.assertEqual(int(change[0, 1]), 1)
        self.assertEqual(int(change[0, 2]), 0)
        self.assertEqual(int(change[0, 3]), 1)


class FrozenSplitTests(unittest.TestCase):
    def test_val_json_seed42_n128_train1152(self) -> None:
        from pathlib import Path
        import json

        val_path = Path(__file__).resolve().parent.parent / "gates" / "_cache" / "cf_ft" / "second_val_ids.json"
        rec = json.loads(val_path.read_text(encoding="utf-8"))
        self.assertEqual(rec.get("seed"), 42)
        self.assertEqual(int(rec.get("n")), 128)
        self.assertEqual(len(rec.get("ids") or []), 128)
        self.assertEqual(int(rec.get("n_train")), 1152)
        unique = Path(__file__).resolve().parent.parent / "gates" / "_cache" / "prod10" / "unique_images.json"
        blob = json.loads(unique.read_text(encoding="utf-8"))
        names = sorted(
            {
                Path(str(p).replace("\\", "/")).name
                for p in (blob.get("second") or [])
                if "/A/" in str(p).replace("\\", "/") and str(p).lower().endswith(".png")
            }
        )
        self.assertEqual(len(names), 1280)
        val_set = set(rec["ids"])
        train = [n for n in names if n not in val_set]
        self.assertEqual(len(train), 1152)
        self.assertFalse(val_set & set(train))


class ReadablePairTests(unittest.TestCase):
    def test_missing_split_is_unreadable(self) -> None:
        import tempfile
        from pathlib import Path
        from PIL import Image

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for split in ("A", "B", "label1"):
                (root / split).mkdir()
                Image.new("RGB", (8, 8), (255, 255, 255)).save(root / split / "x.png")
            (root / "label2").mkdir()
            (root / "label2" / "x.png").write_bytes(b"not a png")
            ok, err = pair_is_readable(root, "x.png")
            self.assertFalse(ok)
            self.assertIn("label2/x.png", err or "")


if __name__ == "__main__":
    unittest.main()
