"""Eval-kit CPU tests. No llama-server. No Gradio launch. No Modal."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

DEMO = Path(__file__).resolve().parent
SAT = DEMO.parent
if str(SAT) not in sys.path:
    sys.path.insert(0, str(SAT))
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))

from judge_kit.adapters import cdvqa_like, generic, rsvqa_like, vrsbench_like  # noqa: E402
from judge_kit.drop_eval import (  # noqa: E402
    FROZEN_GGUF_SHA,
    FROZEN_MMPROJ_SHA,
    evaluate,
    exact_match,
    load_synonym_table,
    normalize_answer,
)
from judge_kit.schema import KIT_MODE_TO_INPUT_MODE, validate_row  # noqa: E402

PROTOCOL = SAT / "gates" / "_cache" / "scoreboard" / "protocol.json"
README = SAT / "judge_kit" / "README_JUDGE.txt"
SHARED = (
    DEMO / "tools.py",
    DEMO / "app.py",
    DEMO / "pipeline.py",
    DEMO / "report.py",
    DEMO / "planner.py",
    DEMO / "ingest.py",
    DEMO / "serve.ps1",
)
SHARED_SHA = {
    "tools.py": "8c56dc12fbe3d9633c9f8bdf328649c8b349d328a8d9e241ceba26036c1433d3",  # 2026-09-24 authorized C1-SENSOR-PROFILE (band identity via declared profile/metadata; unidentified bands withhold; evidence_class)
    "app.py": "be1f0ee9e58a2a3800674ac631f76a79cda271c9ee9f298bf7eb9b1089a87bf6",  # 2026-09-13 human-authorized Tab 2 copy edit (readable legend; needles unchanged)
    "pipeline.py": "c86ab8a1ea806b02aecae080c5cc46ab3476217c99b513027f4af4ccc1fe559b",  # 2026-09-24 authorized C1-SENSOR-PROFILE (sensor_profile threading; withheld water skips overlay/area/agreement artifacts)
    "report.py": "bbf5370bd67deacda91a817844797f166f793eeb79df03066eb54af52a0fdff6",  # 2026-09-24 authorized WEB-POLISH-V3
    "planner.py": "31c20c8f97056aacab425e9a0310491e1d00909236078809126db07b7fe1cee3",  # 2026-09-24 authorized ROUTE-PROBE (cdvqa_map→change_detect validator coupling; dead-tool fix)
    "ingest.py": "f6d0f8bf7c840d5ff2f7b4de375bcd2199771a2c89a91b572e75c36f35ab2e3e",  # 2026-09-24 authorized C1-SENSOR-PROFILE (SENSOR_PROFILES + resolve_band_map; SAR pol verify/refuse; profile-driven true-color)
    "serve.ps1": "19287c44083cd94fd4f91ff6cbae5e66b2c33b30978a0df2bc5b99a0f14dd172",
}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _png(path: Path, color=(10, 80, 30)) -> Path:
    Image.new("RGB", (32, 32), color).save(path)
    return path


def _fake_run(query, input_mode, uploads, vlm_url, live):
    return {
        "query": query,
        "input_mode": input_mode,
        "answer": "yes",
        "plan": {"supported": True, "tools": ["vqa"]},
        "tool_outputs": {"vqa": {"text": "yes"}},
        "evidence_packet": {"canonical_answer": "yes"},
        "vlm": {"url": vlm_url},
        "uploads": {k: str(v) for k, v in (uploads or {}).items()},
        "live": live,
    }


class SchemaTests(unittest.TestCase):
    def test_valid_single_and_pair(self) -> None:
        seen: set[str] = set()
        row, err = validate_row(
            {"id": "a", "mode": "single", "question": "Q?", "image": "x.png"},
            seen_ids=seen,
        )
        self.assertIsNone(err)
        self.assertEqual(row["mode"], "single")
        row, err = validate_row(
            {"id": "b", "mode": "change", "question": "Q?", "image_a": "a.png", "image_b": "b.png"},
            seen_ids=seen,
        )
        self.assertIsNone(err)
        row, err = validate_row(
            {"id": "c", "mode": "sar", "question": "Q?", "image_a": "o.png", "image_b": "s.npz"},
            seen_ids=seen,
        )
        self.assertIsNone(err)
        self.assertEqual(KIT_MODE_TO_INPUT_MODE["sar"], "optical+sar")
        self.assertEqual(KIT_MODE_TO_INPUT_MODE["change"], "bi-temporal")

    def test_violations_are_errors_not_repairs(self) -> None:
        seen: set[str] = set()
        row, err = validate_row({"id": "a", "mode": "single", "question": "Q?"}, seen_ids=seen)
        self.assertIsNone(row)
        self.assertIn("image", err or "")
        row, err = validate_row(
            {"id": "a", "mode": "change", "question": "Q?", "image": "x.png"},
            seen_ids=set(),
        )
        self.assertIsNone(row)
        self.assertIn("image_a", err or "")
        _, err = validate_row({"id": "z", "mode": "other", "question": "Q?", "image": "x.png"}, seen_ids=set())
        self.assertIsNotNone(err)
        seen2: set[str] = set()
        validate_row({"id": "dup", "mode": "single", "question": "Q?", "image": "x.png"}, seen_ids=seen2)
        _, err = validate_row(
            {"id": "dup", "mode": "single", "question": "Q?", "image": "y.png"},
            seen_ids=seen2,
        )
        self.assertIn("duplicate", err or "")


class AdapterTests(unittest.TestCase):
    def test_round_trips(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        img = _png(tmp / "05867_0000.png")
        vrs = tmp / "vrs.json"
        vrs.write_text(
            json.dumps(
                [
                    {
                        "example_id": "e1",
                        "question": "Is there a road?",
                        "image_id": img.name,
                        "gold": "Yes",
                    }
                ]
            ),
            encoding="utf-8",
        )
        vrows = vrsbench_like.convert(vrs, images_dir=tmp)
        self.assertEqual(vrows[0]["mode"], "single")
        self.assertEqual(vrows[0]["question"], "Is there a road?")
        self.assertNotIn("gold", vrows[0])
        self.assertTrue(Path(vrows[0]["image"]).is_file())

        rsrc = tmp / "rsvqa.json"
        rsrc.write_text(
            json.dumps(
                {
                    "questions": [{"id": 7, "question": "Is there a building?", "img_id": 3}],
                    "images": [{"id": 3, "filename": img.name}],
                }
            ),
            encoding="utf-8",
        )
        rrows = rsvqa_like.convert(rsrc, images_dir=tmp)
        self.assertEqual(rrows[0]["id"], "7")
        self.assertEqual(rrows[0]["mode"], "single")
        self.assertNotIn("gold", rrows[0])

        csrc = tmp / "cd.jsonl"
        csrc.write_text(
            json.dumps(
                {
                    "id": "p1",
                    "question": "What changed?",
                    "im1": "a.png",
                    "im2": "b.png",
                    "gold": "yes",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        crows = cdvqa_like.convert(csrc)
        self.assertEqual(crows[0]["mode"], "change")
        self.assertEqual(crows[0]["image_a"], "a.png")
        self.assertNotIn("gold", crows[0])

        schema = tmp / "schema.jsonl"
        schema.write_text(
            json.dumps({"id": "g1", "mode": "single", "question": "Q?", "image": str(img)}) + "\n",
            encoding="utf-8",
        )
        grows = generic.convert(schema)
        self.assertEqual(grows[0]["id"], "g1")


class EvalLoopTests(unittest.TestCase):
    def test_resume_idempotence_and_failure_row(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        img = _png(tmp / "ok.png")
        qpath = tmp / "questions.jsonl"
        rows = [
            {"id": "ok", "mode": "single", "question": "Is there a road?", "image": str(img)},
            {"id": "bad", "mode": "single", "question": "Is there a road?", "image": str(tmp / "missing.png")},
        ]
        qpath.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        out = tmp / "out"
        rec1 = evaluate(
            qpath,
            out,
            images_dir=tmp,
            vlm_url="http://127.0.0.1:8080",
            live=True,
            run_query_fn=_fake_run,
        )
        self.assertEqual(rec1["n"], 2)
        self.assertEqual(rec1["n_fail"], 1)
        self.assertEqual(rec1["n_resume_skip"], 0)
        preds = Path(rec1["preds"])
        blob1 = preds.read_bytes()
        lines = [json.loads(x) for x in preds.read_text(encoding="utf-8").splitlines() if x.strip()]
        self.assertEqual(len(lines), 2)
        by = {r["id"]: r for r in lines}
        self.assertEqual(by["ok"]["answer"], "yes")
        self.assertEqual(by["ok"]["prediction_raw"], "yes")
        self.assertIsNone(by["ok"]["error"])
        self.assertTrue(by["bad"]["error"])
        self.assertEqual(by["bad"]["answer"], "")
        man = (out / "MANIFEST.txt").read_text(encoding="utf-8")
        for key in (
            "gguf_sha256=",
            "mmproj_sha256=",
            "server=",
            "started_here=",
            "n=",
            "n_fail=",
            "wall_s=",
            "mode_single=",
        ):
            self.assertIn(key, man)
        rec2 = evaluate(
            qpath,
            out,
            images_dir=tmp,
            vlm_url="http://127.0.0.1:8080",
            live=True,
            run_query_fn=_fake_run,
        )
        self.assertEqual(rec2["n_resume_skip"], 2)
        self.assertEqual(preds.read_bytes(), blob1)

    def test_schema_error_stays_in_file(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        qpath = tmp / "q.jsonl"
        qpath.write_text(
            json.dumps({"id": "x", "mode": "single", "question": "Q?"}) + "\n",
            encoding="utf-8",
        )
        rec = evaluate(qpath, tmp / "out", run_query_fn=_fake_run)
        lines = [json.loads(x) for x in Path(rec["preds"]).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rec["n"], 1)
        self.assertEqual(rec["n_fail"], 1)
        self.assertIn("image", lines[0]["error"])

    def test_vlm_only_trace_is_not_an_error(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        img = _png(tmp / "ok.png")
        qpath = tmp / "q.jsonl"
        qpath.write_text(
            json.dumps({"id": "v1", "mode": "single", "question": "Q?", "image": str(img)}) + "\n",
            encoding="utf-8",
        )

        def vlm_only(query, input_mode, uploads, vlm_url, live):
            return {
                "answer": "north-south",
                "plan": {"supported": True, "tools": ["vqa"]},
                "tool_outputs": {},
                "vlm": {"text": "north-south", "complete_s": 0.7},
                "evidence_packet": {},
            }

        rec = evaluate(qpath, tmp / "out", run_query_fn=vlm_only)
        row = json.loads(Path(rec["preds"]).read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(rec["n_fail"], 0)
        self.assertIsNone(row["error"])
        self.assertEqual(row["answer"], "north-south")


class ScoreExactMatchTests(unittest.TestCase):
    def test_frozen_synonym_table_and_exact_match(self) -> None:
        if not PROTOCOL.is_file():
            self.skipTest("scoreboard protocol.json not present")
        table = load_synonym_table(PROTOCOL)
        proto = json.loads(PROTOCOL.read_text(encoding="utf-8"))["synonym_table"]
        self.assertEqual(table, {str(k): str(v) for k, v in proto.items()})
        self.assertTrue(exact_match("true", "yes", table))
        self.assertTrue(exact_match("Yes.", "true", table))
        self.assertTrue(exact_match("two", "2", table))
        self.assertFalse(exact_match("maybe", "yes", table))
        self.assertEqual(normalize_answer("Answer: No", table), "no")
        tmp = Path(tempfile.mkdtemp())
        img = _png(tmp / "a.png")
        qpath = tmp / "q.jsonl"
        qpath.write_text(
            json.dumps({"id": "s1", "mode": "single", "question": "Q?", "image": str(img)}) + "\n",
            encoding="utf-8",
        )
        gold = tmp / "gold.jsonl"
        gold.write_text(json.dumps({"id": "s1", "gold": "yes"}) + "\n", encoding="utf-8")
        rec = evaluate(
            qpath,
            tmp / "out",
            run_query_fn=_fake_run,
            score_gold_path=gold,
        )
        scores = rec["scores"]
        self.assertEqual(scores["n"], 1)
        self.assertEqual(scores["n_exact"], 1)
        self.assertEqual(scores["metric"], "exact-match")
        self.assertNotIn("secondary", scores)
        self.assertIn("synonym_table_sha256", scores)


class HygieneTests(unittest.TestCase):
    def test_readme_and_banned_phrases(self) -> None:
        lines = README.read_text(encoding="utf-8").splitlines()
        self.assertLessEqual(len(lines), 20)
        kit = SAT / "judge_kit"
        blob = ""
        for p in kit.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".py", ".ps1", ".txt", ".md"}:
                continue
            blob += p.read_text(encoding="utf-8", errors="replace")
        low = blob.lower()
        self.assertNotIn("local" + " judge", low)
        self.assertNotIn("local" + "-judge", low)
        self.assertNotIn("judge" + " model", low)
        self.assertNotIn("oll" + "ama", low)

    def test_shared_demo_files_untouched_and_no_kit_ids(self) -> None:
        for p in SHARED:
            self.assertEqual(_sha(p), SHARED_SHA[p.name], p.name)
            text = p.read_text(encoding="utf-8", errors="replace")
            self.assertNotIn("judge_kit", text)
            self.assertNotIn("drop_eval", text)
            self.assertNotIn("run_eval.ps1", text)


class FrozenWeightConstantsTests(unittest.TestCase):
    def test_sha_constants(self) -> None:
        self.assertEqual(
            FROZEN_GGUF_SHA,
            "67d1659bfe71b89d50b45a4ad1a9e5b997e5bb16ce5da66a6a6167abd569e9e2",
        )
        self.assertEqual(
            FROZEN_MMPROJ_SHA,
            "ca524100ebf825c9a870db1c580d03879e0da0ab2541697e2458e64891cf9d38",
        )
        ps1 = (SAT / "judge_kit" / "run_eval.ps1").read_text(encoding="utf-8")
        self.assertIn("8081", ps1)
        self.assertIn(FROZEN_GGUF_SHA.upper(), ps1)
        self.assertIn("--port 8081", ps1)
        self.assertIn("-ngl 99", ps1)


class RehearsalFixtureTests(unittest.TestCase):
    def test_rehearsal_questions_shape(self) -> None:
        path = SAT / "judge_kit" / "rehearsal" / "questions.jsonl"
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        self.assertEqual(len(rows), 25)
        ids = [r["id"] for r in rows]
        self.assertEqual(len(ids), len(set(ids)))
        modes = [r["mode"] for r in rows]
        self.assertEqual(modes.count("single"), 20)
        self.assertEqual(modes.count("change"), 5)
        for r in rows:
            self.assertNotIn("gold", r)
            self.assertNotIn("ground_truth", r)
            row, err = validate_row(r, seen_ids=set())
            self.assertIsNone(err, err)
            self.assertIsNotNone(row)


if __name__ == "__main__":
    unittest.main()
