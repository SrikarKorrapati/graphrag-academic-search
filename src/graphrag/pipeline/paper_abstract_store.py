"""Lazy in-memory map paper_id -> abstract from papers.jsonl for fusion / reranking."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock

from config import settings

logger = logging.getLogger(__name__)

_LOCK = Lock()
_STORE: dict[str, str] | None = None


def _resolved_jsonl_path() -> Path:
    p = Path(settings.papers_jsonl_path)
    if p.is_absolute():
        return p
    return Path(__file__).resolve().parent.parent / p


def ensure_loaded() -> None:
    """Scan JSONL once and populate the abstract map (thread-safe, idempotent)."""
    global _STORE
    with _LOCK:
        if _STORE is not None:
            return
        path = _resolved_jsonl_path()
        if not path.is_file():
            logger.warning("papers JSONL not found (%s); KG abstract enrichment disabled", path)
            _STORE = {}
            return
        store: dict[str, str] = {}
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = str(obj.get("id", "")).strip()
                if not pid:
                    continue
                store[pid] = str(obj.get("abstract") or "")
        _STORE = store
        logger.info("paper abstract store: loaded %s rows from %s", len(_STORE), path)


def abstract_for_paper(paper_id: str) -> str:
    """Return abstract text for ``paper_id``, or empty string if unknown."""
    ensure_loaded()
    assert _STORE is not None
    return _STORE.get(str(paper_id).strip(), "") or ""
