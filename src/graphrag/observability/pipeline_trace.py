"""Append one JSON line per pipeline stage for POST /query debugging."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_TRACE_FILE = Path(__file__).resolve().parent / "pipeline_trace.jsonl"
_FORMATTED_JSON = Path(__file__).resolve().parent / "pipeline_trace.formatted.json"
# Latest single-request pretty trace (intermediate stages only for that ``request_id``).
_SAMPLE_FORMATTED_JSON = Path(__file__).resolve().parent / "pipeline_trace_sample.formatted.json"


def append_pipeline_stage(
    *,
    request_id: str,
    stage: str,
    user_query: str,
    output: dict[str, Any] | list[Any] | str | float | int | None,
) -> None:
    """Append a single JSON object (one line) to ``pipeline_trace.jsonl``."""
    row: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "stage": stage,
        "user_query": user_query,
        "output": output,
    }
    try:
        line = json.dumps(row, ensure_ascii=False, default=str)
        with _LOCK:
            _TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with _TRACE_FILE.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            _sync_formatted_json()
    except Exception as exc:
        logger.warning("Could not append pipeline trace (%s): %s", stage, exc)


def _load_trace_rows_from_jsonl() -> list[dict[str, Any]]:
    """Parse every JSON object from ``pipeline_trace.jsonl`` (source of truth)."""
    rows: list[dict[str, Any]] = []
    if not _TRACE_FILE.exists():
        return rows
    for raw in _TRACE_FILE.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                rows.append(obj)
        except json.JSONDecodeError:
            continue
    return rows


def _sync_formatted_json() -> None:
    """Rewrite ``pipeline_trace.formatted.json`` from JSONL so it always matches on-disk trace."""
    rows = _load_trace_rows_from_jsonl()
    _FORMATTED_JSON.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def rebuild_formatted_from_jsonl() -> None:
    """Rewrite ``pipeline_trace.formatted.json`` from the full ``pipeline_trace.jsonl`` (repair / bootstrap)."""
    with _LOCK:
        rows = _load_trace_rows_from_jsonl()
        _FORMATTED_JSON.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )


def write_sample_formatted_for_request(request_id: str) -> None:
    """Write all trace rows for one ``request_id`` to ``pipeline_trace_sample.formatted.json`` (pretty JSON array)."""
    try:
        with _LOCK:
            rows = [r for r in _load_trace_rows_from_jsonl() if r.get("request_id") == request_id]
            _SAMPLE_FORMATTED_JSON.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
    except Exception as exc:
        logger.warning("Could not write sample formatted trace: %s", exc)
