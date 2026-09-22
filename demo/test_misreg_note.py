"""MISREG-NOTE: phase-correlation ingest note. No realign, no tool number changes."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

DEMO = Path(__file__).resolve().parent
SAT = DEMO.parent
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))

STRESS = DEMO / "data" / "_stress"
SCENE2_BEFORE = DEMO / "data" / "scene2" / "before.png"
SCENE2_AFTER = DEMO / "data" / "scene2" / "after.png"
DX20_BEFORE = STRESS / "before_dx20.png"
DX20_AFTER = STRESS / "after_dx20.png"
EVIDENCE_DIR = SAT / "gates" / "_cache" / "sensor_stress" / "misreg_note"
EVIDENCE_JSON = EVIDENCE_DIR / "report.json"

NOTE_STEM = "estimated rigid shift ≈"
NOTE_TAIL = (
    "change mask may include registration artifacts; no automatic realignment performed"
)


def _write_evidence(rows: dict) -> Path:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    from ingest import (
        MISREG_DOWNSAMPLE_PX,
        MISREG_MIN_PEAK,
        MISREG_MIN_STD,
        MISREG_SHIFT_BAR_PX,
        MISREG_NOTE_TEMPLATE,
    )

    doc = {
        "task": "MISREG-NOTE",
        "method": {
            "downsample_px": MISREG_DOWNSAMPLE_PX,
            "shift_bar_px": MISREG_SHIFT_BAR_PX,
            "min_std": MISREG_MIN_STD,
            "min_peak": MISREG_MIN_PEAK,
            "peak_rule": (
                "normalized phase-correlation on 256×256 grayscale; "
                "inconclusive if either std < min_std or peak < min_peak; "
                "note if hypot(dx,dy) in original pixels > 5"
            ),
            "note_template": MISREG_NOTE_TEMPLATE,
            "warps_pixels": False,
        },
        "pairs": rows,
    }
    EVIDENCE_JSON.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    (EVIDENCE_DIR / "note_text.txt").write_text(
        json.dumps({k: (v or {}).get("misregistration_note") for k, v in rows.items()}, indent=2),
        encoding="utf-8",
    )
    return EVIDENCE_JSON


class MisregNoteTests(unittest.TestCase):
    def test_synthetic_shift_on_same_image(self) -> None:
        from ingest import pair_misreg_fields

        src = Image.open(SCENE2_AFTER).convert("RGB")
        arr = np.asarray(src)
        shifted = np.zeros_like(arr)
        shifted[:, 20:] = arr[:, :-20]
        tmp = Path(tempfile.mkdtemp())
        a = tmp / "a.png"
        b = tmp / "b.png"
        src.save(a)
        Image.fromarray(shifted).save(b)
        fields = pair_misreg_fields(a, b)
        self.assertEqual(fields["misreg_check"], "noted")
        self.assertIsNotNone(fields["misregistration_note"])
        self.assertGreater(fields["misreg_shift_px"], 5.0)
        self.assertAlmostEqual(fields["misreg_shift_px"], 20.0, delta=8.0)
        self.assertIn(NOTE_STEM, fields["misregistration_note"])
        self.assertIn(NOTE_TAIL, fields["misregistration_note"])

    def test_dx20_asset_notes(self) -> None:
        from pipeline import bind_inputs

        if not (DX20_BEFORE.is_file() and DX20_AFTER.is_file()):
            self.skipTest("dx20 stress assets missing")
        self.assertTrue(DX20_BEFORE.is_file() and DX20_AFTER.is_file(), "dx20 stress assets missing")
        bound = bind_inputs(
            "bi-temporal",
            scene=2,
            uploads={"before": DX20_BEFORE, "after": DX20_AFTER},
        )
        self.assertTrue(bound["ok"], bound.get("error"))
        self.assertEqual(bound.get("misreg_check"), "noted")
        note = bound.get("misregistration_note") or ""
        self.assertIn(NOTE_STEM, note)
        self.assertIn(NOTE_TAIL, note)
        self.assertIn(note, bound.get("ingest_note") or "")
        self.assertGreater(float(bound["misreg_shift_px"]), 5.0)
        self.assertAlmostEqual(float(bound["misreg_shift_px"]), 20.0, delta=8.0)

    def test_clean_control_silent(self) -> None:
        from pipeline import bind_inputs

        bound = bind_inputs(
            "bi-temporal",
            scene=2,
            uploads={"before": SCENE2_BEFORE, "after": SCENE2_BEFORE},
        )
        self.assertTrue(bound["ok"], bound.get("error"))
        self.assertEqual(bound.get("misreg_check"), "ok")
        self.assertIsNone(bound.get("misregistration_note"))
        self.assertNotIn(NOTE_STEM, bound.get("ingest_note") or "")
        self.assertLessEqual(float(bound["misreg_shift_px"] or 0.0), 5.0)

    def test_inconclusive_uniform_pair(self) -> None:
        from ingest import pair_misreg_fields

        tmp = Path(tempfile.mkdtemp())
        u1 = tmp / "u1.png"
        u2 = tmp / "u2.png"
        Image.new("RGB", (64, 64), (40, 40, 40)).save(u1)
        Image.new("RGB", (64, 64), (40, 40, 40)).save(u2)
        fields = pair_misreg_fields(u1, u2)
        self.assertEqual(fields["misreg_check"], "inconclusive")
        self.assertIsNone(fields["misregistration_note"])

    def test_scene2_measured_and_passthrough(self) -> None:
        from pipeline import bind_inputs, run_query

        bound = bind_inputs("bi-temporal", scene=2, uploads=None)
        self.assertTrue(bound["ok"])
        self.assertIn(bound.get("misreg_check"), {"ok", "noted", "inconclusive"})
        trace = run_query(
            "Show the change mask",
            "bi-temporal",
            scene=2,
            live=False,
        )
        self.assertEqual(trace.get("misreg_check"), bound.get("misreg_check"))
        self.assertEqual(trace.get("misregistration_note"), bound.get("misregistration_note"))
        if bound.get("misregistration_note"):
            self.assertIn(bound["misregistration_note"], trace.get("ingest_note") or "")
        # Record as-is; do not assert note vs silent.
        rows = {
            "dx20": bind_inputs(
                "bi-temporal",
                scene=2,
                uploads={"before": DX20_BEFORE, "after": DX20_AFTER},
            ),
            "control_before_vs_self": bind_inputs(
                "bi-temporal",
                scene=2,
                uploads={"before": SCENE2_BEFORE, "after": SCENE2_BEFORE},
            ),
            "scene2_prepared": bound,
        }
        slim = {}
        for name, b in rows.items():
            slim[name] = {
                "misreg_check": b.get("misreg_check"),
                "misregistration_note": b.get("misregistration_note"),
                "misreg_shift_px": b.get("misreg_shift_px"),
                "misreg_dx_px": b.get("misreg_dx_px"),
                "misreg_dy_px": b.get("misreg_dy_px"),
                "misreg_peak": b.get("misreg_peak"),
                "source": b.get("source"),
            }
        path = _write_evidence(slim)
        self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
