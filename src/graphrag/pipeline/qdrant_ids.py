"""Qdrant point ids: API accepts uint or UUID string; keep canonical ids in payload."""

from __future__ import annotations

import uuid


def paper_qdrant_point_id(paper_id: str) -> str:
    """Stable UUID string derived from paper_id (payload holds the real OpenAlex-style id)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"paper:{paper_id}"))
