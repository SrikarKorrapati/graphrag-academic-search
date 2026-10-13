"""Shared row formatting for KG traversal results."""

from __future__ import annotations

from typing import Any


def paper_meta(
    pid: str,
    title: str,
    year: Any,
    authors: list[str],
    venue: Any,
    topics: list[str],
    cited: str | None = None,
) -> dict[str, Any]:
    """Build metadata dict for a paper-centric KG row."""
    meta: dict[str, Any] = {
        "paper_id": pid,
        "title": title,
        "year": year,
        "authors": authors,
        "venue": venue,
        "topics": topics,
    }
    if cited:
        meta["cited_paper"] = cited
    return meta


def fmt_paper(title: str, year: Any, authors: list[str], venue: Any, topics: list[str]) -> str:
    """Human-readable one-line summary for fusion and context."""
    au = ", ".join(authors) if authors else "Unknown authors"
    top = ", ".join(topics) if topics else ""
    ven = venue or "unknown venue"
    return f"{title} ({year}) by {au}. Published in {ven}. Topics: {top}."


def clean(xs: list[Any]) -> list[str]:
    """Drop falsy Neo4j collect entries."""
    return [str(x) for x in xs if x]
