"""WIRE-8091: canonical_vqa -> :8091 adapted RSVQA seat tests.

Mocked-HTTP tests must pass WITHOUT the server running; one live test hits
real :8091 and skips when it is down. The narrator seat :8080 is never used.
"""
from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import tools
from tools import canonical_vqa

DEMO = Path(__file__).resolve().parent
SERVED_ID = "canonical/Qwen3VL-8B-RSVQA-Q4_K_M.gguf"


def _fake_resp(payload: dict):
    class _R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    return _R(json.dumps(payload).encode("utf-8"))


def _make_urlopen(chat_text: str, captured: list | None = None):
    """Fake urlopen: GET /v1/models -> served id; POST chat -> answer text."""

    def fake(req, timeout=None, **kw):
        if isinstance(req, str):  # GET /v1/models
            assert req.endswith("/v1/models"), req
            return _fake_resp({"data": [{"id": SERVED_ID}]})
        url = getattr(req, "full_url", "")
        assert url.endswith("/v1/chat/completions"), url
        if captured is not None:
            captured.append(json.loads(req.data.decode("utf-8")))
        return _fake_resp(
            {"choices": [{"message": {"content": chat_text}}]}
        )

    return fake


def _tmp_png() -> str:
    from PIL import Image
    import numpy as np

    path = Path(tempfile.mkdtemp()) / "img.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)
    return str(path)


class CanonicalVqaMockedTests(unittest.TestCase):
    def test_happy_path_and_provenance(self) -> None:
        img = _tmp_png()
        with mock.patch.object(
            urllib.request, "urlopen", _make_urlopen("Yes.")
        ):
            out = canonical_vqa("Is there a road in the image?", [img])
        self.assertTrue(out["available"])
        self.assertEqual(out["answer"], "yes")
        self.assertEqual(out["text"], "Yes.")
        self.assertEqual(out["model"], SERVED_ID)
        self.assertEqual(out["seat"], "127.0.0.1:8091")
        self.assertEqual(out["gguf_sha256"], tools.CANONICAL_GGUF_SHA256)
        self.assertEqual(out["mmproj_sha256"], tools.CANONICAL_MMPROJ_SHA256)
        self.assertEqual(out["decode"]["temperature"], 0.0)
        self.assertEqual(out["decode"]["repeat_penalty"], 1.08)
        self.assertEqual(out["decode"]["max_tokens"], 16)
        self.assertIsNotNone(out["latency_s"])
        self.assertIn("8091", out["provenance"])
        self.assertIn("not the :8080 narrator", out["provenance"])

    def test_decode_contract_on_wire(self) -> None:
        img = _tmp_png()
        sent: list = []
        with mock.patch.object(
            urllib.request, "urlopen", _make_urlopen("3", sent)
        ):
            canonical_vqa("How many cars are visible?", [img])
        self.assertEqual(len(sent), 1)
        body = sent[0]
        self.assertEqual(body["model"], SERVED_ID)
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["repeat_penalty"], 1.08)
        self.assertEqual(body["max_tokens"], 16)
        self.assertFalse(body["stream"])
        content = body["messages"][0]["content"]
        self.assertTrue(any(p.get("type") == "image_url" for p in content))
        self.assertTrue(any(p.get("type") == "text" for p in content))

    def test_unavailable_is_structured_not_exception(self) -> None:
        def boom(req, timeout=None, **kw):
            raise urllib.error.URLError("connection refused")

        img = _tmp_png()
        with mock.patch.object(urllib.request, "urlopen", boom):
            out = canonical_vqa("Is there a road?", [img])
        self.assertFalse(out["available"])
        self.assertIsNone(out["answer"])
        self.assertIn("connection refused", out["error"])
        self.assertIn("no narrator substitution", out["provenance"])
        self.assertEqual(out["seat"], "127.0.0.1:8091")

    def test_empty_answer_stays_unavailable_for_claim(self) -> None:
        img = _tmp_png()
        with mock.patch.object(urllib.request, "urlopen", _make_urlopen("   ")):
            out = canonical_vqa("Is there a road?", [img])
        self.assertTrue(out["available"])
        self.assertIsNone(out["answer"])


class CanonicalVqaPacketTests(unittest.TestCase):
    def _cv_out(self, answer: str | None, available: bool = True) -> dict:
        return {
            "available": available,
            "answer": answer,
            "text": answer or "",
            "model": SERVED_ID if available else None,
            "gguf_sha256": tools.CANONICAL_GGUF_SHA256,
            "mmproj_sha256": tools.CANONICAL_MMPROJ_SHA256,
            "url": tools.CANONICAL_VLM_URL,
            "seat": "127.0.0.1:8091",
            "decode": dict(tools.CANONICAL_DECODE),
            "latency_s": 0.42,
            "error": None if available else "URLError: refused",
        }

    def test_packet_claim_and_canonical_answer(self) -> None:
        from evidence_packet import build_packet, validate_packet

        pkt = build_packet(
            {"canonical_vqa": self._cv_out("yes")}, "vqa"
        )
        claims = [c for c in pkt["claims"] if c["predicate"] == "canonical_answer"]
        self.assertEqual(len(claims), 1)
        c = claims[0]
        self.assertEqual(c["value"], "yes")
        self.assertEqual(c["confidence"]["level"], "measured")
        self.assertEqual(c["provenance"]["tool"], "canonical_vqa")
        self.assertEqual(c["provenance"]["model"], SERVED_ID)
        self.assertEqual(
            c["provenance"]["gguf_sha256"], tools.CANONICAL_GGUF_SHA256
        )
        self.assertEqual(c["provenance"]["seat"], "127.0.0.1:8091")
        self.assertEqual(pkt["canonical_answer"], "yes")
        self.assertEqual(validate_packet(pkt), [])

    def test_packet_unavailable_withholds(self) -> None:
        from evidence_packet import build_packet, validate_packet

        pkt = build_packet(
            {"canonical_vqa": self._cv_out(None, available=False)}, "vqa"
        )
        c = [
            c for c in pkt["claims"] if c["predicate"] == "canonical_answer"
        ][0]
        self.assertIsNone(c["value"])
        self.assertEqual(c["confidence"]["level"], "withheld")
        self.assertEqual(
            c["confidence"]["basis"], "canonical_vqa_unavailable"
        )
        self.assertIsNone(pkt["canonical_answer"])
        self.assertTrue(
            any("canonical_vqa unavailable" in lim for lim in pkt["limitations"])
        )
        self.assertEqual(validate_packet(pkt), [])

    def test_findings_header_and_narration_whitelist(self) -> None:
        from report import check_narration, findings_header

        tout = {"canonical_vqa": self._cv_out("617m2")}
        head = findings_header({"tool_outputs": tout})
        self.assertIn("canonical_vqa", head)
        self.assertIn("617m2", head)
        self.assertIn(SERVED_ID, head)
        # Narrator quoting the canonical answer's number is not "invented".
        check = check_narration(
            "The canonical seat reports 617 m2 of change.", tout
        )
        self.assertEqual(check["issues"], [], check)


class CanonicalVqaPlannerTests(unittest.TestCase):
    def test_routes_on_single_questions_only(self) -> None:
        from planner import plan

        p = plan("Is there a road in the image?", "single")
        self.assertIn("canonical_vqa", p["tools"])
        p = plan("Count the aircraft on the runway.", "single")
        self.assertIn("canonical_vqa", p["tools"])
        p = plan("Describe the land cover and major objects.", "single")
        self.assertNotIn("canonical_vqa", p["tools"])
        p = plan("What changed between these two dates?", "bi-temporal")
        self.assertNotIn("canonical_vqa", p["tools"])
        p = plan("Identify water-covered regions.", "optical+sar")
        self.assertNotIn("canonical_vqa", p["tools"])


def _server_up() -> bool:
    try:
        with urllib.request.urlopen(
            tools.CANONICAL_VLM_URL.rstrip("/") + "/v1/models", timeout=3
        ) as r:
            json.loads(r.read())
        return True
    except Exception:
        return False


class CanonicalVqaLiveTests(unittest.TestCase):
    @unittest.skipUnless(_server_up(), "canonical :8091 server is down")
    def test_live_8091_answer(self) -> None:
        img = DEMO / "data" / "scene1" / "05867_0000.png"
        if not img.is_file():
            self.skipTest("scene1 image missing")
        out = canonical_vqa("Is there a road in the image?", [img])
        self.assertTrue(out["available"], out.get("error"))
        self.assertIsNotNone(out["answer"])
        self.assertIn("Qwen3VL-8B-RSVQA", out["model"])
        self.assertLessEqual(len(out["answer"].split()), 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
