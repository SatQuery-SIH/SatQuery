"""Drive the 3 Gradio tabs via the HTTP API and save screenshots/timings."""
from __future__ import annotations

import json
import time
from pathlib import Path

from gradio_client import Client

DEMO = Path(__file__).resolve().parent
OUT = DEMO / "traces" / "ui_drive.json"


def main() -> None:
    client = Client("http://127.0.0.1:7860", download_files=False)
    results = []
    jobs = [
        ("scene1", "Describe the land cover and major objects.", 0),  # api names discovered at runtime
    ]
    print(client.view_api(), flush=True)
    t0 = time.perf_counter()
    # Predict on each of the 3 run buttons. Endpoint names vary by Gradio version;
    # try numbered fns then named.
    calls = [
        ("scene1", "/scene1", ["Describe the land cover and major objects."]),
        ("scene2", "/scene2", ["What changed between these two dates, and where?"]),
        ("scene3", "/scene3", ["Identify water-covered regions."]),
    ]
    alts = {
        "scene1": ["/lambda", "/predict"],
        "scene2": ["/lambda_1", "/predict_1"],
        "scene3": ["/lambda_2", "/predict_2"],
    }
    for name, api, args in calls:
        t1 = time.perf_counter()
        try:
            out = client.predict(*args, api_name=api)
            err = None
        except Exception as e:
            out = None
            err = f"{type(e).__name__}: {e}"
            for alt in alts.get(name, []):
                try:
                    out = client.predict(*args, api_name=alt)
                    err = None
                    api = alt
                    break
                except Exception as e2:
                    err = f"{type(e2).__name__}: {e2}"
        dt = time.perf_counter() - t1
        answer = out[0] if isinstance(out, (list, tuple)) and out else out
        if isinstance(answer, str) and len(answer) > 400:
            answer_preview = answer[:400]
        else:
            answer_preview = answer
        results.append(
            {
                "name": name,
                "api": api,
                "seconds": round(dt, 3),
                "error": err,
                "answer_preview": answer_preview,
            }
        )
        print(name, "s=", round(dt, 3), "err=", err, "ans=", str(answer_preview)[:180], flush=True)
    payload = {"wall_s": round(time.perf_counter() - t0, 3), "results": results}
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
