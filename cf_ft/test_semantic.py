"""CPU tests for SEMANTIC-FROMTO compiler + leakage guards. No Modal. No GPU."""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cf_ft.registry import append_second_semantic_entry
from cf_ft.semantic import (
    BUILT_UP_CLASS_ID,
    CLASS_ID_TO_TOKEN,
    IGNORE_INDEX,
    NUM_CLASSES,
    RATIO_BINS,
    SEM_BATCH,
    SEM_EPOCHS,
    SEM_LR,
    SEM_SEED,
    bin_ratio,
    build_model,
    ce_ignore_loss,
    compile_answer,
    compile_from_features,
    config_hash,
    count_params,
    exact_token,
    extract_class_id,
    extract_time_side,
    features_from_maps,
    from_to_token,
    ids_to_train_target,
    project_semantic_cost,
    protocol_dict,
    refuse_label_path,
    semantic_config,
    train_ids_complement,
)


class ProtocolFrozenTests(unittest.TestCase):
    def test_built_up_and_bins_and_map(self) -> None:
        p = protocol_dict()
        self.assertEqual(p["built_up_class_id"], 5)
        self.assertEqual(p["built_up_token"], "buildings")
        self.assertEqual(BUILT_UP_CLASS_ID, 5)
        self.assertTrue(p["frozen_before_train"])
        self.assertTrue(p["labels_scorer_only"])
        self.assertFalse(p["approved_for_demo"])
        self.assertEqual(p["built_up_direction"], "not_determined")
        tokens = [b["token"] for b in RATIO_BINS]
        self.assertEqual(tokens[0], "0")
        self.assertIn("0_to_10", tokens)
        self.assertIn("90_to_100", tokens)
        self.assertEqual(from_to_token(2, 5), "buildings")
        self.assertEqual(CLASS_ID_TO_TOKEN[4], "trees")
        self.assertIn("1->5", p["from_to_token_map"])
        self.assertEqual(p["from_to_token_map"]["2->5"], "buildings")

    def test_ratio_bin_edges(self) -> None:
        self.assertEqual(bin_ratio(0.0), "0")
        self.assertEqual(bin_ratio(0.05), "0_to_10")
        self.assertEqual(bin_ratio(0.10), "0_to_10")
        self.assertEqual(bin_ratio(0.1000001), "10_to_20")
        self.assertEqual(bin_ratio(1.0), "90_to_100")

    def test_pre_registered_train_hparams(self) -> None:
        self.assertAlmostEqual(SEM_LR, 1e-4)
        self.assertEqual(SEM_BATCH, 16)
        self.assertEqual(SEM_EPOCHS, 12)
        self.assertEqual(SEM_SEED, 42)
        cfg = semantic_config()
        self.assertEqual(cfg["loss"], "CE-ignore")
        self.assertEqual(len(config_hash(cfg)), 64)


class CompilerMapTests(unittest.TestCase):
    def _maps(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.ones((4, 4), dtype=np.int32) * 2  # nvg
        b = np.ones((4, 4), dtype=np.int32) * 2
        a[:2, :] = 2
        b[:2, :] = 5  # nvg → buildings on top half
        a[2:, :] = 4
        b[2:, :] = 4
        change = (a != b).astype(np.uint8)
        return a, b, change

    def test_change_to_what_from_question_class(self) -> None:
        a, b, ch = self._maps()
        pred, err = compile_answer(
            "change_to_what",
            "What have the regions of non-vegetated ground surface in the first image mainly changed to?",
            cls_a=a,
            cls_b=b,
            change=ch,
        )
        self.assertIsNone(err)
        self.assertEqual(pred, "buildings")

    def test_largest_smallest_and_ratio_types(self) -> None:
        a, b, ch = self._maps()
        feat = features_from_maps(a, b, ch)
        largest, _ = compile_from_features(
            feat, "largest_change", "What is the largest change in the pre-change image?"
        )
        self.assertEqual(largest, "nvg_surface")
        smallest, _ = compile_from_features(
            feat, "smallest_change", "What is the smallest change in the second image?"
        )
        self.assertEqual(smallest, "buildings")
        ratio, _ = compile_from_features(
            feat,
            "change_ratio_types",
            "What is the change ratio of non-vegetated ground surface in the first image?",
        )
        self.assertEqual(ratio, "90_to_100")
        zero, _ = compile_from_features(
            feat,
            "change_ratio_types",
            "How much area of water has changed in the first image?",
        )
        self.assertEqual(zero, "0")

    def test_increase_decrease_whole_image_counts(self) -> None:
        a, b, ch = self._maps()
        inc, _ = compile_answer(
            "increase_or_not",
            "Did the areas of buildings increase?",
            cls_a=a,
            cls_b=b,
            change=ch,
        )
        dec, _ = compile_answer(
            "decrease_or_not",
            "Did the regions of non-vegetated ground surface decrease?",
            cls_a=a,
            cls_b=b,
            change=ch,
        )
        trees, _ = compile_answer(
            "increase_or_not",
            "Have the areas of trees increased?",
            cls_a=a,
            cls_b=b,
            change=ch,
        )
        self.assertEqual(inc, "yes")
        self.assertEqual(dec, "yes")
        self.assertEqual(trees, "no")

    def test_extractors(self) -> None:
        self.assertEqual(extract_class_id("Did the areas of trees decrease?"), 4)
        self.assertEqual(extract_time_side("in the post-event image"), "t2")
        self.assertEqual(extract_time_side("in the first image"), "t1")
        self.assertIsNone(extract_time_side("What is the largest change?"))


class LeakageGuardTests(unittest.TestCase):
    def test_refuse_label_paths(self) -> None:
        with self.assertRaises(RuntimeError):
            refuse_label_path(Path("png/png/second/label1/00003.png"))
        with self.assertRaises(RuntimeError):
            compile_answer(
                "change_to_what",
                "x",
                feat={"changed_pixels": 0, "from_to": [[0] * 6] * 6},
                label_path="foo/label2/bar.png",
            )
        ok = refuse_label_path(Path("png/second_unpack/train/SECOND_train_set/im1/00017.png"))
        self.assertTrue(str(ok).endswith("00017.png"))

    def test_train_disjoint_from_val_and_cdvqa_eval(self) -> None:
        train, meta = train_ids_complement()
        self.assertEqual(meta["n_train"], 1152)
        self.assertEqual(len(train), 1152)
        from cf_ft.semantic import cdvqa_test_ids, cdvqa_val_ids, frozen_val_ids

        val = set(frozen_val_ids())
        self.assertEqual(len(val), 128)
        self.assertEqual(set(train) & val, set())
        self.assertEqual(set(train) & set(cdvqa_test_ids()), set())
        self.assertEqual(set(train) & set(cdvqa_val_ids()), set())

    def test_white_ignored_in_ce(self) -> None:
        logits = torch.zeros(1, NUM_CLASSES, 2, 2)
        logits[0, 0, :, :] = 8.0
        target = torch.full((1, 2, 2), IGNORE_INDEX, dtype=torch.long)
        target[0, 0, 0] = 0  # one valid pixel; ignore-only tensors can be nan in CE
        loss = F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX)
        self.assertTrue(math.isfinite(float(loss)))
        t2 = ids_to_train_target(np.array([[0, 255], [1, 2]], dtype=np.int32))
        self.assertEqual(int(t2[0, 0]), IGNORE_INDEX)
        self.assertEqual(int(t2[0, 1]), IGNORE_INDEX)
        self.assertEqual(int(t2[1, 0]), 0)
        self.assertEqual(int(t2[1, 1]), 1)

    def test_ce_ignore_helper_finite(self) -> None:
        logits = torch.zeros(1, NUM_CLASSES, 2, 2)
        target = torch.zeros(1, 2, 2, dtype=torch.long)
        target[0, 0, 0] = IGNORE_INDEX
        loss = ce_ignore_loss({"out": logits, "aux": None}, target)
        self.assertTrue(math.isfinite(float(loss)))

    def test_model_under_50m_untrained(self) -> None:
        m = build_model(pretrained=False)
        n = count_params(m)
        self.assertLessEqual(n, 50_000_000)
        self.assertGreater(n, 1_000_000)

    def test_exact_token_casefold(self) -> None:
        self.assertEqual(exact_token("NVG_surface", "nvg_surface"), 1)
        self.assertEqual(exact_token("unknown", "buildings"), 0)

    def test_projection_fits_caps(self) -> None:
        rec = project_semantic_cost()
        self.assertTrue(rec["fits_cap"])
        self.assertLessEqual(float(rec["projected_new_usd"]), 4.0 + 1e-9)
        self.assertLessEqual(float(rec["projected_cumulative_usd"]), 8.0 + 1e-9)

    def test_registry_append_preserves_prior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            prior = {
                "approved_for_demo_any": True,
                "checkpoints": [
                    {"role": "imported_levir", "path": "imp.pt", "sha256": "aa", "approved_for_demo": False},
                    {
                        "role": "team_second",
                        "path": "gates/_cache/cf_ft/train/best_by_val.pt",
                        "sha256": "cbfefe6564613571e824cb2cee0f971b4793ca0dda6c1a5c5741cbcdf4c76fe6",
                        "approved_for_demo": True,
                        "route_scope": "second_like",
                    },
                    {
                        "role": "team_second_retention",
                        "path": None,
                        "sha256": None,
                        "retention_failed": True,
                        "approved_for_demo": False,
                    },
                ],
            }
            reg = d / "ckpt_registry.json"
            reg.write_text(json.dumps(prior, indent=2) + "\n", encoding="utf-8")
            ckpt = d / "best_semantic.pt"
            ckpt.write_bytes(b"semantic-ckpt")
            rec = append_second_semantic_entry(ckpt, 0.12, "deadbeef", n_params=11, path=reg)
            roles = [c["role"] for c in rec["checkpoints"]]
            self.assertEqual(
                roles,
                ["imported_levir", "team_second", "team_second_retention", "second_semantic"],
            )
            team = next(c for c in rec["checkpoints"] if c["role"] == "team_second")
            self.assertTrue(team["approved_for_demo"])
            self.assertEqual(team["route_scope"], "second_like")
            sem = rec["checkpoints"][-1]
            self.assertFalse(sem["approved_for_demo"])
            self.assertTrue(rec["approved_for_demo_any"])
            ret = next(c for c in rec["checkpoints"] if c["role"] == "team_second_retention")
            self.assertTrue(ret["retention_failed"])


class SemanticAttachHelpersTests(unittest.TestCase):
    def test_built_up_direction_rule(self) -> None:
        from cf_ft.semantic import built_up_direction_from_counts

        total = 10000
        thr = 50
        self.assertEqual(built_up_direction_from_counts(0, 51, total), "increase")
        self.assertEqual(built_up_direction_from_counts(51, 0, total), "decrease")
        self.assertEqual(built_up_direction_from_counts(10, 10 + thr, total), "no_change")
        self.assertEqual(built_up_direction_from_counts(10, 10 + thr + 1, total), "increase")

    def test_ratio_bins_from_deciles_keep_names(self) -> None:
        from cf_ft.semantic import RATIO_BIN_TOKENS, bin_ratio, ratio_bins_from_deciles

        deciles = {f"p{q}": q / 100.0 for q in range(10, 100, 10)}
        bins = ratio_bins_from_deciles(deciles)
        self.assertEqual([b["token"] for b in bins], list(RATIO_BIN_TOKENS))
        self.assertTrue(bins[0]["exact_zero"])
        self.assertEqual(bin_ratio(0.0, bins=bins), "0")
        self.assertEqual(bin_ratio(0.05, bins=bins), "0_to_10")
        self.assertEqual(bin_ratio(0.10, bins=bins), "0_to_10")
        self.assertEqual(bin_ratio(0.101, bins=bins), "10_to_20")
        self.assertEqual(bin_ratio(1.0, bins=bins), "90_to_100")

    def test_configured_max_combined_not_per_family(self) -> None:
        from cf_ft.semantic import configured_choice

        self.assertEqual(configured_choice(0.50, 0.51), "v2")
        self.assertEqual(configured_choice(0.51, 0.50), "v1")
        self.assertEqual(configured_choice(0.50, 0.50), "v1")

    def test_classify_type_family_not_generic_change(self) -> None:
        from cf_ft.semantic import classify_semantic_family

        self.assertIsNone(
            classify_semantic_family("What changed between these two dates, and where?")
        )
        self.assertEqual(
            classify_semantic_family("Did the areas of buildings increase?"),
            "increase_or_not",
        )
        self.assertEqual(
            classify_semantic_family(
                "Has built-up area increased, decreased, or remained unchanged?"
            ),
            "built_up_direction",
        )
        self.assertEqual(
            classify_semantic_family(
                "What is the change ratio of trees in the first image?"
            ),
            "change_ratio_types",
        )

    def test_fitlist_disjoint_on_disk(self) -> None:
        from cf_ft.semantic import assert_fitlist_disjoint, load_fitlist

        fit = load_fitlist()
        rec = assert_fitlist_disjoint(fit["ids"])
        self.assertTrue(rec["ok"])
        self.assertEqual(rec["n_fit"], 1152)
        self.assertTrue(rec["disjoint_cdvqa_test"])
        self.assertTrue(rec["disjoint_cdvqa_val"])
        self.assertTrue(rec["disjoint_our_val"])

    def test_freeze_mtime_before_rescore_tmp(self) -> None:
        import time

        from cf_ft.semantic import freeze_ratio_bins_v2, load_fitlist

        fit = load_fitlist()
        rows = []
        for i, name in enumerate(fit["ids"]):
            share = 0.05 + (i % 90) / 100.0
            rows.append(
                {
                    "id": name,
                    "changed_pixels": 100,
                    "total_pixels": 1000,
                    "dominant_share": share,
                    "error": None,
                }
            )
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "semantic_ratio_bins_v2.json"
            frozen = freeze_ratio_bins_v2({"rows": rows}, dest=dest, fitlist=fit)
            self.assertTrue(frozen["frozen_before_rescore"])
            mt = dest.stat().st_mtime
            time.sleep(0.2)
            later = Path(tmp) / "cdvqa_scores_sem_v2.json"
            later.write_text("{}", encoding="utf-8")
            self.assertGreaterEqual(later.stat().st_mtime + 1e-6, mt)
            self.assertEqual(len(frozen["ratio_bins"]), 11)

    def test_registry_mark_semantic_preserves_prior(self) -> None:
        from cf_ft.registry import mark_second_semantic_demo_route

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            prior = {
                "approved_for_demo_any": True,
                "checkpoints": [
                    {"role": "imported_levir", "path": "imp.pt", "sha256": "aa", "approved_for_demo": False},
                    {
                        "role": "team_second",
                        "path": "gates/_cache/cf_ft/train/best_by_val.pt",
                        "sha256": "cbfefe6564613571e824cb2cee0f971b4793ca0dda6c1a5c5741cbcdf4c76fe6",
                        "approved_for_demo": True,
                        "route_scope": "second_like",
                        "domains": ["second_like"],
                    },
                    {
                        "role": "team_second_retention",
                        "path": None,
                        "sha256": None,
                        "retention_failed": True,
                        "approved_for_demo": False,
                    },
                    {
                        "role": "second_semantic",
                        "path": "gates/_cache/cf_ft/semantic/best_semantic.pt",
                        "sha256": "438cac09be7c630254a12278550b64f86254ecc131ee0cdc723fd526210764d8",
                        "approved_for_demo": False,
                    },
                ],
            }
            reg = d / "ckpt_registry.json"
            reg.write_text(json.dumps(prior, indent=2) + "\n", encoding="utf-8")
            rec = mark_second_semantic_demo_route(path=reg, ratio_outcome="v2 combined 0.51 > v1 0.50")
            by = {c["role"]: c for c in rec["checkpoints"]}
            self.assertEqual(by["second_semantic"]["route_scope"], "second_like_type")
            self.assertEqual(by["second_semantic"]["domains"], ["second_like_type"])
            self.assertTrue(by["second_semantic"]["approved_for_demo"])
            self.assertTrue(by["team_second"]["approved_for_demo"])
            self.assertEqual(by["team_second"]["route_scope"], "second_like")
            self.assertTrue(by["team_second_retention"]["retention_failed"])
            self.assertFalse(by["imported_levir"].get("approved_for_demo"))


if __name__ == "__main__":
    unittest.main()
