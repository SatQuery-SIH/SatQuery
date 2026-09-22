"""API-0080 contract tests — FastAPI TestClient.

Real ingest on synthetic GeoTIFFs (demo.ingest.write_test_geotiff); run_query
is MOCKED — no live llama seats needed. Seat probes point at a dead port.
Run:  python api/test_api.py   (from the SatQuery repo root)
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

API_DIR = Path(__file__).resolve().parent
SAT = API_DIR.parent
DEMO = SAT / "demo"
TMP = Path(tempfile.mkdtemp(prefix="satquery_api_test_"))

os.environ["SATQUERY_DB"] = str(TMP / "test.sqlite3")
os.environ["SATQUERY_DATA_DIR"] = str(TMP / "data")
os.environ["SATQUERY_NARRATOR_URL"] = "http://127.0.0.1:9"
os.environ["SATQUERY_CANONICAL_URL"] = "http://127.0.0.1:9"
for _p in (str(API_DIR), str(DEMO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

import main  # noqa: E402
import pipeline  # noqa: E402
from store import SqliteStore  # noqa: E402


def _fake_run(
    query,
    input_mode,
    scene=None,
    live=True,
    vlm_url=None,
    device_cd="cpu",
    uploads=None,
    cd_prefer="changeformer",
    on_event=None,
):
    if on_event:
        on_event({"event": "stage", "stage": "plan", "status": "done", "ts": 0.0})
    wd = TMP / f"wd_{input_mode}_{len(list(TMP.glob('wd_*')))}"
    wd.mkdir(parents=True, exist_ok=True)
    img = wd / "image.png"
    overlay = wd / "overlay_live.png"
    Image.new("RGB", (8, 8), (20, 60, 30)).save(img)
    Image.new("RGB", (8, 8), (200, 40, 40)).save(overlay)
    return {
        "query": query,
        "input_mode": input_mode,
        "scene": scene or 1,
        "live": live,
        "plan": {
            "supported": True,
            "tools": ["vqa"],
            "task": "vqa",
            "vlm_role": "narrate",
        },
        "tool_outputs": {"vqa": {"text": "yes"}},
        "answer": "yes",
        "visible_answer": "### Findings (from tools)\n\nyes",
        "first_token_s": 0.12,
        "complete_s": 0.5,
        "overlay_path": str(overlay),
        "images": [str(img)],
        "metrics_badge": None,
        "ingest_note": "test",
        "input_source": "upload" if uploads else "prepared",
        "gsd": {"gsd_m": None, "source": "none"},
        "misreg_check": None,
        "misregistration_note": None,
        "misreg_shift_px": None,
        "geo_exports": {},
        "evidence_packet": {
            "schema_version": "1.0",
            "task": "vqa",
            "canonical_answer": "yes",
            "claims": [],
            "limitations": ["test limitation"],
            "artifacts": [],
            "tool_outputs": {"vqa": {"text": "yes"}},
        },
        "uploads_seen": uploads,
    }


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = SqliteStore(TMP / f"{self._testMethodName}.sqlite3")
        self.client = TestClient(main.create_app(store=self.store))

    def tearDown(self) -> None:
        self.store.close()

    def _upload(self, mode: str, files: dict) -> "object":
        return self.client.post("/upload", data={"mode": mode}, files=files)

    def _tif(self, name: str = "s.tif", scale: float = 10.0) -> Path:
        from ingest import write_test_geotiff

        p = TMP / name
        arr = np.zeros((16, 20), dtype=np.uint8)
        return write_test_geotiff(p, arr, scale, scale, geographic=False)

    # --- seats / health -----------------------------------------------------

    def test_health_seats_down_still_200(self) -> None:
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIs(body["ok"], True)
        self.assertFalse(body["seats"]["narrator"]["up"])
        self.assertFalse(body["seats"]["canonical"]["up"])
        self.assertEqual(body["seats"]["narrator"]["url"], "http://127.0.0.1:9")
        self.assertIn("gguf_sha256", body["seats"]["canonical"])

    def test_seats_local_and_cloud_stub(self) -> None:
        r = self.client.get("/seats")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["profile"], "local")
        r = self.client.post("/seats", json={"profile": "cloud"})
        self.assertEqual(r.status_code, 501)
        self.assertEqual(r.json()["detail"], "cloud profile pending CLOUD-SEAT")
        r = self.client.post("/seats", json={"profile": "local"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["profile"], "local")

    # --- upload -------------------------------------------------------------

    def test_upload_geotiff_detected_block(self) -> None:
        tif = self._tif()
        with tif.open("rb") as fh:
            r = self._upload("single", {"image": ("s.tif", fh, "image/tiff")})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertTrue(body["upload_id"])
        self.assertTrue(Path(body["workdir"]).is_dir())
        det = body["detected"]
        self.assertAlmostEqual(det["gsd_m"], 10.0, places=5)
        self.assertEqual(det["gsd_source"], "geotransform")
        self.assertEqual(det["bands"], 1)
        self.assertEqual(det["dtype"], "uint8")
        self.assertIsInstance(body["warnings"], list)

    def test_upload_png_gsd_withheld(self) -> None:
        png = TMP / "plain.png"
        Image.new("RGB", (16, 16)).save(png)
        with png.open("rb") as fh:
            r = self._upload("single", {"image": ("plain.png", fh, "image/png")})
        self.assertEqual(r.status_code, 200, r.text)
        det = r.json()["detected"]
        self.assertIsNone(det["gsd_m"])
        self.assertEqual(det["gsd_source"], "none")
        self.assertTrue(r.json()["warnings"])

    def test_upload_incomplete_roles_422(self) -> None:
        png = TMP / "b.png"
        Image.new("RGB", (16, 16)).save(png)
        with png.open("rb") as fh:
            r = self._upload(
                "bi-temporal", {"before": ("b.png", fh, "image/png")}
            )
        self.assertEqual(r.status_code, 422)
        body = r.json()
        self.assertIn("error", body)
        self.assertIn("detail", body)

    def test_upload_files_list_order(self) -> None:
        png = TMP / "o.png"
        Image.new("RGB", (16, 16)).save(png)
        with png.open("rb") as fh:
            r = self.client.post(
                "/upload",
                data={"mode": "single"},
                files=[("files", ("o.png", fh, "image/png"))],
            )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()["upload_id"])

    # --- query / runs --------------------------------------------------------

    def test_query_mocked_bundle_and_runs(self) -> None:
        with mock.patch.object(pipeline, "run_query", _fake_run):
            r = self.client.post(
                "/query",
                json={
                    "query": "is there water?",
                    "input_mode": "single",
                    "live": False,
                },
            )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        for key in (
            "run_id",
            "answer",
            "visible_answer",
            "plan",
            "tool_outputs",
            "evidence_packet",
            "trace",
            "report",
            "artifacts",
        ):
            self.assertIn(key, body)
        self.assertEqual(body["answer"], "yes")
        self.assertEqual(body["trace"]["first_token_s"], 0.12)
        self.assertIn("limitations", body["report"])
        rid = body["run_id"]

        r = self.client.get("/runs")
        self.assertEqual(r.status_code, 200)
        rows = r.json()
        self.assertEqual(rows[0]["run_id"], rid)
        self.assertTrue(rows[0]["supported"])

        r = self.client.get(f"/runs/{rid}")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["run_id"], rid)
        self.assertEqual(r.json()["answer"], "yes")

    def test_query_with_upload_id_and_mode_mismatch(self) -> None:
        tif = self._tif("up.tif")
        with tif.open("rb") as fh:
            up = self._upload("single", {"image": ("up.tif", fh, "image/tiff")})
        uid = up.json()["upload_id"]

        with mock.patch.object(pipeline, "run_query", _fake_run):
            r = self.client.post(
                "/query",
                json={
                    "query": "describe",
                    "input_mode": "single",
                    "upload_id": uid,
                    "live": False,
                },
            )
            self.assertEqual(r.status_code, 200, r.text)
            self.assertTrue(r.json()["trace"]["uploads_seen"])

            bad = self.client.post(
                "/query",
                json={
                    "query": "describe",
                    "input_mode": "bi-temporal",
                    "upload_id": uid,
                    "live": False,
                },
            )
        self.assertEqual(bad.status_code, 422)
        self.assertEqual(bad.json()["error"], "mode_mismatch")

        r = self.client.post(
            "/query",
            json={
                "query": "q",
                "input_mode": "single",
                "upload_id": "nope",
                "live": False,
            },
        )
        self.assertEqual(r.status_code, 404)
        self.assertIn("error", r.json())
        self.assertIn("detail", r.json())

    # --- artifacts ------------------------------------------------------------

    def test_artifact_fetch_and_traversal_rejection(self) -> None:
        with mock.patch.object(pipeline, "run_query", _fake_run):
            body = self.client.post(
                "/query",
                json={"query": "q", "input_mode": "single", "live": False},
            ).json()
        rid = body["run_id"]
        names = {a["name"] for a in body["artifacts"]}
        self.assertIn("overlay_live.png", names)

        r = self.client.get(f"/artifacts/{rid}/overlay_live.png")
        self.assertEqual(r.status_code, 200)
        self.assertGreater(len(r.content), 50)

        for bad in ("..%2F..%2Fsecret.txt", "..", "not_listed.png"):
            r = self.client.get(f"/artifacts/{rid}/{bad}")
            self.assertIn(r.status_code, (400, 404), bad)

        r = self.client.get("/artifacts/no_such_run/x.png")
        self.assertEqual(r.status_code, 404)

    def test_query_validation_error_shape(self) -> None:
        r = self.client.post("/query", json={"input_mode": "single"})
        self.assertEqual(r.status_code, 422)
        self.assertIn("error", r.json())
        self.assertIn("detail", r.json())


def _sse_events(text: str) -> list[tuple[str, dict]]:
    out = []
    name = None
    for line in text.splitlines():
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: ") and name:
            import json as _json

            out.append((name, _json.loads(line[6:])))
            name = None
    return out


class StreamTests(unittest.TestCase):
    """POST /query/stream — SSE view of the same run store."""

    def setUp(self) -> None:
        self.store = SqliteStore(TMP / f"{self._testMethodName}.sqlite3")
        self.client = TestClient(main.create_app(store=self.store))

    def tearDown(self) -> None:
        self.store.close()

    def test_stream_real_refusal_sequence(self) -> None:
        # Real pipeline, no servers needed: counting refuses on bi-temporal.
        r = self.client.post(
            "/query/stream",
            json={
                "query": "how many buildings changed?",
                "input_mode": "bi-temporal",
                "scene": 2,
                "live": False,
            },
        )
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/event-stream", r.headers["content-type"])
        events = _sse_events(r.text)
        names = [e[0] for e in events]
        self.assertEqual(names[-1], "done")
        stages = [(d.get("stage"), d.get("status")) for n, d in events if n == "stage"]
        self.assertEqual(stages[0], ("plan", "start"))
        self.assertIn(("plan", "withheld"), stages)
        self.assertIn(("packet", "done"), stages)
        self.assertEqual(stages[-1], ("done", "done"))
        done = events[-1][1]
        self.assertTrue(done["run_id"])
        self.assertFalse(done["plan"]["supported"])
        # The streamed run landed in the store like a /query run.
        r2 = self.client.get(f"/runs/{done['run_id']}")
        self.assertEqual(r2.status_code, 200)
        self.assertFalse(r2.json()["plan"]["supported"])

    def test_stream_real_supported_run_order(self) -> None:
        # Real pipeline on prepared scene 1: water_highlight + area_calc run
        # on CPU; narration withheld (live=false). No llama seats needed.
        r = self.client.post(
            "/query/stream",
            json={
                "query": "highlight the water",
                "input_mode": "single",
                "scene": 1,
                "live": False,
            },
        )
        self.assertEqual(r.status_code, 200)
        events = _sse_events(r.text)
        seq = [(d.get("stage"), d.get("status"), d.get("tool")) for n, d in events if n == "stage"]

        def idx(stage, status, tool=None):
            for i, s in enumerate(seq):
                if s == (stage, status, tool):
                    return i
            return -1

        self.assertEqual(seq[0], ("plan", "start", None))
        self.assertEqual(seq[1], ("plan", "done", None))
        self.assertLess(idx("plan", "done"), idx("bind", "start"))
        self.assertLess(idx("bind", "done"), idx("tool", "start", "water_highlight"))
        self.assertLess(
            idx("tool", "done", "water_highlight"), idx("tool", "start", "area_calc")
        )
        self.assertLess(idx("tool", "done", "area_calc"), idx("packet", "start"))
        self.assertLess(idx("packet", "done"), idx("done", "done"))
        # done event carries the full bundle (run_id + report + artifacts)
        done = events[-1][1]
        for key in ("run_id", "answer", "plan", "tool_outputs", "report", "artifacts"):
            self.assertIn(key, done)
        # tool done events carry real latency_s
        wh_done = next(d for n, d in events if n == "stage" and d.get("tool") == "water_highlight" and d.get("status") == "done")
        self.assertIn("latency_s", wh_done["data"])
        self.assertIn("water_pixels", wh_done["data"])

    def test_stream_error_event_on_pipeline_failure(self) -> None:
        def _boom(*a, **kw):
            raise RuntimeError("kaboom")

        with mock.patch.object(pipeline, "run_query", _boom):
            r = self.client.post(
                "/query/stream",
                json={"query": "q", "input_mode": "single", "live": False},
            )
        self.assertEqual(r.status_code, 200)
        events = _sse_events(r.text)
        self.assertEqual(events[-1][0], "error")
        self.assertIn("kaboom", events[-1][1]["detail"])
        self.assertEqual(events[-1][1]["error"], "pipeline_error")

    def test_stream_unknown_upload_is_json_404(self) -> None:
        r = self.client.post(
            "/query/stream",
            json={
                "query": "q",
                "input_mode": "single",
                "upload_id": "nope",
                "live": False,
            },
        )
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.json()["error"], "unknown_upload")


class PipelineEventTests(unittest.TestCase):
    """demo.pipeline.run_query(on_event=...) — additive hook contract."""

    def test_ordering_supported_run(self) -> None:
        evs = []
        trace = pipeline.run_query(
            "highlight the water", "single", scene=1, live=False,
            on_event=evs.append,
        )
        self.assertIsInstance(trace, dict)
        seq = [(e["stage"], e["status"], e.get("tool")) for e in evs]
        self.assertEqual(seq[0], ("plan", "start", None))
        self.assertEqual(seq[1], ("plan", "done", None))
        self.assertEqual(seq[-1][0:2], ("done", "done"))
        stages = [s[0] for s in seq]
        self.assertLess(stages.index("bind"), stages.index("tool"))
        self.assertLess(stages.index("packet"), len(seq) - 1)
        tools = [s[2] for s in seq if s[0] == "tool" and s[1] == "done"]
        self.assertIn("water_highlight", tools)
        for e in evs:
            self.assertIn("ts", e)
            self.assertEqual(e["event"], "stage")

    def test_refusal_emits_withheld_then_done(self) -> None:
        evs = []
        trace = pipeline.run_query(
            "how many buildings changed?", "bi-temporal", scene=2,
            live=False, on_event=evs.append,
        )
        self.assertFalse(trace["plan"]["supported"])
        seq = [(e["stage"], e["status"]) for e in evs]
        self.assertIn(("plan", "withheld"), seq)
        self.assertIn(("packet", "done"), seq)
        self.assertEqual(seq[-1], ("done", "done"))
        w = next(e for e in evs if e["status"] == "withheld")
        self.assertIn("counting tool", w["data"]["refusal"])

    def test_bind_failure_emits_fail_then_done(self) -> None:
        # Partial upload = bind error (a missing file alone falls back to
        # prepared scenes by design).
        real = TMP / "real_before.png"
        Image.new("RGB", (16, 16)).save(real)
        evs = []
        trace = pipeline.run_query(
            "describe", "bi-temporal",
            uploads={"before": real, "after": TMP / "no_such_file.png"},
            live=False, on_event=evs.append,
        )
        seq = [(e["stage"], e["status"]) for e in evs]
        self.assertIn(("bind", "fail"), seq)
        self.assertEqual(seq[-1], ("done", "done"))

    def test_raising_callback_never_breaks_run(self) -> None:
        def _bad(_ev):
            raise RuntimeError("callback exploded")

        trace = pipeline.run_query(
            "highlight the water", "single", scene=1, live=False,
            on_event=_bad,
        )
        self.assertIsInstance(trace, dict)
        self.assertTrue(trace["plan"]["supported"])

    def test_no_callback_unchanged(self) -> None:
        trace = pipeline.run_query(
            "highlight the water", "single", scene=1, live=False
        )
        self.assertIsInstance(trace, dict)
        self.assertIn("tool_outputs", trace)


if __name__ == "__main__":
    unittest.main(verbosity=2)
