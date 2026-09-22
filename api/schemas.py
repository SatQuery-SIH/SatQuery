"""Frozen request/response shapes for the SatQuery API (API-0080).

This module is the contract WEB-FRONTEND mocks against. Field names and
nullability here are load-bearing — change them only with a spec bump.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

INPUT_MODES = ("single", "bi-temporal", "optical+sar")
InputMode = Literal["single", "bi-temporal", "optical+sar"]
SeatProfile = Literal["local", "cloud"]


class SeatState(BaseModel):
    """One inference seat. `up`/`model` are live-probe fields (GET /health);
    `gguf_sha256` pins the canonical seat's served weights (config view)."""

    url: str
    model: str | None = None
    up: bool | None = None
    gguf_sha256: str | None = None


class HealthResponse(BaseModel):
    ok: bool
    seats: dict[str, SeatState]


class SeatsResponse(BaseModel):
    profile: SeatProfile
    seats: dict[str, SeatState]


class SeatSwitchRequest(BaseModel):
    profile: SeatProfile


class DetectedBlock(BaseModel):
    """Judge-facing ingest facts, straight from demo.ingest on the real files.
    `calibrated` is set for the SAR input (optical+sar mode); null otherwise."""

    gsd_m: float | None
    gsd_source: str
    crs: str | None
    crs_note: str | None
    bands: int | None
    dtype: str | None
    calibrated: bool | None = None


class UploadResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    upload_id: str
    workdir: str | None
    detected: DetectedBlock
    warnings: list[str]


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    input_mode: InputMode
    scene: int | None = None
    upload_id: str | None = None
    live: bool = True


class ArtifactRef(BaseModel):
    type: str
    path: str
    name: str
    url: str


class RunBundle(BaseModel):
    """The run bundle. `trace`/`report` carry the full pipeline record;
    extra keys are preserved for forward compatibility."""

    model_config = ConfigDict(extra="allow")

    run_id: str
    answer: str | None
    visible_answer: str | None
    plan: dict[str, Any]
    tool_outputs: dict[str, Any]
    evidence_packet: dict[str, Any] | None
    trace: dict[str, Any]
    report: dict[str, Any]
    artifacts: list[ArtifactRef]


class RunListItem(BaseModel):
    run_id: str
    query: str | None
    input_mode: str | None
    ts: str | None
    supported: bool | None


class ErrorBody(BaseModel):
    error: str
    detail: str
