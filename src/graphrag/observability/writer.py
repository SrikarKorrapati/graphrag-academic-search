"""Append-only JSONL store for each successful ``POST /query``."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DIR = Path(__file__).resolve().parent
_FILE = _DIR / "queries.jsonl"


def append_query_log(record: dict[str, Any]) -> None:
    """
    Append one JSON object as a single line to ``queries.jsonl``.

    Creates the directory if needed. Intended to run in a thread pool from FastAPI.
    """
    try:
        _DIR.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _FILE.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception as exc:
        logger.warning("Could not write query log: %s", exc)


def build_log_record(
    query: str,
    answer: str,
    latency_ms: float,
    parsed_dump: dict[str, Any],
    papers: list[dict[str, Any]],
) -> dict[str, Any]:
    """Shape a log row with UTC timestamp and retrieval summary."""
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "answer": answer,
        "latency_ms": latency_ms,
        "parsed_query": parsed_dump,
        "papers": papers,
    }
