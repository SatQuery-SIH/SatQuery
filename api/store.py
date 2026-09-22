"""Run/upload persistence for the SatQuery API (API-0080).

Repository interface + a zero-dep SQLite backend (stdlib sqlite3). The DB
indexes ids -> record JSON; artifact files stay files on disk. The interface
carries no sqlite-only assumptions so a Postgres/PostGIS backend can replace
it for the deployed profile without handler changes.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Protocol

API_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = API_DIR / "_data" / "satquery.sqlite3"


class RunStore(Protocol):
    """Repository contract. Handlers call these methods only — never SQL."""

    def put_upload(self, record: dict[str, Any]) -> str:
        """Persist an upload record; returns its upload_id."""
        ...

    def get_upload(self, upload_id: str) -> dict[str, Any] | None:
        ...

    def put_run(self, record: dict[str, Any]) -> str:
        """Persist a run bundle record; returns its run_id."""
        ...

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        ...

    def list_runs(self, limit: int = 200) -> list[dict[str, Any]]:
        """Newest-first run summaries: {run_id, query, input_mode, ts, supported}."""
        ...

    def close(self) -> None:
        ...


class SqliteStore:
    """SQLite backend. One JSON column per record; files stay files."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self._path = str(db_path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS uploads ("
            "  upload_id TEXT PRIMARY KEY,"
            "  ts TEXT NOT NULL,"
            "  record TEXT NOT NULL)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS runs ("
            "  run_id TEXT PRIMARY KEY,"
            "  ts TEXT NOT NULL,"
            "  record TEXT NOT NULL)"
        )
        self._conn.commit()

    def put_upload(self, record: dict[str, Any]) -> str:
        upload_id = str(record["upload_id"])
        blob = json.dumps(record, default=str)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO uploads (upload_id, ts, record) VALUES (?,?,?)",
                (upload_id, str(record.get("ts") or ""), blob),
            )
            self._conn.commit()
        return upload_id

    def get_upload(self, upload_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT record FROM uploads WHERE upload_id = ?", (upload_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put_run(self, record: dict[str, Any]) -> str:
        run_id = str(record["run_id"])
        blob = json.dumps(record, default=str)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, ts, record) VALUES (?,?,?)",
                (run_id, str(record.get("ts") or ""), blob),
            )
            self._conn.commit()
        return run_id

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT record FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def list_runs(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT record FROM runs ORDER BY ts DESC LIMIT ?", (int(limit),)
            ).fetchall()
        out = []
        for (blob,) in rows:
            rec = json.loads(blob)
            out.append(
                {
                    "run_id": rec.get("run_id"),
                    "query": rec.get("query"),
                    "input_mode": rec.get("input_mode"),
                    "ts": rec.get("ts"),
                    "supported": rec.get("supported"),
                }
            )
        return out

    def close(self) -> None:
        with self._lock:
            self._conn.close()
