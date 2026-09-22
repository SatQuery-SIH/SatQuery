"""SAR-AGREE CPU tests. Synthetic masks only — no files, no models, no server."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

DEMO = Path(__file__).resolve().parent
SAT = DEMO.parent
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))

from evidence_packet import build_packet, validate_packet  # noqa: E402
from planner import plan  # noqa: E402
from tools import (  # noqa: E402
    AGREE_IOU_T,
    AGREE_MISREG_PX,
    AGREE_MIN_FRAC,
    sar_agreement,
)


def _blob(h: int, w: int, cells: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Binary mask with 1s in the given (r0, r1, c0, c1) boxes."""
    m = np.zeros((h, w), dtype=np.uint8)
    for r0, r1, c0, c1 in cells:
        m[r0:r1, c0:c1] = 1
    return m


class SarAgreementVerdictTests(unittest.TestCase):
    def test_identical_blobs_both_support(self) -> None:
        m = _blob(10, 10, [(2, 8, 2, 8)])
        out = sar_agreement(m, m.copy())
        self.assertEqual(out["verdicts"]["water"], "both_support")
        self.assertEqual(out["verdicts"]["built_up"], "withheld_no_tool")
        self.assertEqual(out["iou"], 1.0)
        self.assertEqual(out["grid"], "same_grid")

    def test_disjoint_blobs_disagree(self) -> None:
        om = _blob(10, 10, [(1, 5, 1, 5)])
        sm = _blob(10, 10, [(6, 9, 6, 9)])
        out = sar_agreement(om, sm)
        self.assertEqual(out["verdicts"]["water"], "disagree")
        self.assertEqual(out["iou"], 0.0)

    def test_optical_only_and_sar_only(self) -> None:
        om = _blob(10, 10, [(1, 5, 1, 5)])
        sm = np.zeros((10, 10), dtype=np.uint8)
        out = sar_agreement(om, sm)
        self.assertEqual(out["verdicts"]["water"], "optical_only")
        out = sar_agreement(sm, om)
        self.assertEqual(out["verdicts"]["water"], "sar_only")

    def test_both_empty_is_absent(self) -> None:
        z = np.zeros((10, 10), dtype=np.uint8)
        out = sar_agreement(z, z.copy())
        self.assertEqual(out["verdicts"]["water"], "both_absent")
        self.assertIsNone(out["iou"])

    def test_uncalibrated_and_misregistered_withhold(self) -> None:
        m = _blob(10, 10, [(2, 8, 2, 8)])
        out = sar_agreement(m, m.copy(), sar_calibrated=False)
        self.assertEqual(out["verdicts"]["water"], "withheld_uncalibrated")
        out = sar_agreement(m, m.copy(), misreg_shift_px=8.0)
        self.assertEqual(out["verdicts"]["water"], "withheld_misregistered")
        out = sar_agreement(m, m.copy(), misreg_shift_px=AGREE_MISREG_PX)
        self.assertEqual(out["verdicts"]["water"], "both_support")

    def test_resample_discloses_grid(self) -> None:
        om = _blob(20, 20, [(4, 16, 4, 16)])
        sm = _blob(10, 10, [(2, 8, 2, 8)])
        out = sar_agreement(om, sm)
        self.assertEqual(out["grid"], "resampled_to_10x10_for_comparison")
        self.assertEqual(out["optical_mask"].shape, (10, 10))
        self.assertEqual(out["sar_mask"].shape, (10, 10))
        self.assertEqual(out["verdicts"]["water"], "both_support")

    def test_min_frac_boundary(self) -> None:
        # 10x10 = 100 px; AGREE_MIN_FRAC=0.005 -> need >= 1 px to count.
        om = _blob(10, 10, [(0, 1, 0, 1)])
        sm = np.zeros((10, 10), dtype=np.uint8)
        out = sar_agreement(om, sm)
        self.assertGreaterEqual(out["optical_water_fraction"], AGREE_MIN_FRAC)
        self.assertEqual(out["verdicts"]["water"], "optical_only")


class SarAgreementPacketTests(unittest.TestCase):
    def test_packet_claims_and_validation(self) -> None:
        om = _blob(10, 10, [(2, 8, 2, 8)])
        ag = sar_agreement(om, om.copy())
        tout = {
            "water_highlight": {"water_pixels": int(om.sum()), "method": "synthetic"},
            "sar_read": {"water_pixels": int(om.sum()), "water_calibrated": True},
            "sar_agreement": {
                k: v for k, v in ag.items() if k not in {"optical_mask", "sar_mask"}
            },
        }
        packet = build_packet(tout, "water_identification")
        preds = {c.get("predicate"): c for c in packet.get("claims") or []}
        self.assertIn("water_agreement", preds)
        self.assertEqual(preds["water_agreement"]["value"], "both_support")
        self.assertEqual(
            preds["water_agreement"]["provenance"]["tool"], "sar_agreement"
        )
        self.assertIn("built_up_agreement", preds)
        self.assertIn("agreement_iou", preds)
        self.assertIn("agreement_grid", preds)
        self.assertEqual(validate_packet(packet), [])
        self.assertTrue(
            any(
                "not pixel-level" in lim
                for lim in packet.get("limitations") or []
            )
        )

    def test_planner_emits_tool(self) -> None:
        p = plan("Identify water-covered regions.", "optical+sar")
        self.assertIn("sar_agreement", p["tools"])
        self.assertIn("sar_read", p["tools"])


if __name__ == "__main__":
    unittest.main()
