"""
Knowledge-graph (Neo4j) relevance scoring.

This module provides a single function, `compute_kg_score`, which converts a
paper's KG-derived metadata + query context into a normalized relevance score
in [0, 1]. The score is intended to be combined with dense-retrieval scores
in the GraphRAG pipeline.
"""

from __future__ import annotations

import math
from typing import Any


# --- Scoring constants (tunable) -------------------------------------------------

MAX_CITATIONS = 10_000
CURRENT_YEAR = 2026
MIN_YEAR = 1990

# Edge-type weights used for graph-connection relevance.
CONNECTION_WEIGHTS: dict[str, float] = {
    "CITES": 0.5,
    "CITED_BY": 0.5,
    "SAME_AUTHOR": 0.3,
    "SAME_TOPIC": 0.2,
    "SAME_VENUE": 0.1,
}

# Final score component weights.
FINAL_WEIGHTS: dict[str, float] = {
    "citation": 0.30,
    "connection": 0.20,
    "topic": 0.15,
    "author": 0.15,
    "recency": 0.10,
    "distance": 0.10,
}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _coerce_int(value: Any, default: int = 0) -> int:
    """Best-effort conversion to int; returns `default` on failure."""
    if value is None:
        return default
    try:
        # Handles ints, floats, numeric strings like "12".
        return int(value)
    except (TypeError, ValueError):
        return default


def _norm(s: Any) -> str:
    """Normalize a string for case-insensitive comparisons."""
    return str(s).strip().lower()


def _to_str_list(value: Any) -> list[str]:
    """Coerce metadata/query fields into a clean list[str]."""
    if value is None:
        return []
    if isinstance(value, str):
        xs = [value]
    else:
        xs = list(value) if isinstance(value, (list, tuple, set)) else [value]
    out: list[str] = []
    for x in xs:
        nx = _norm(x)
        if nx:
            out.append(nx)
    return out


def _author_match_score(metadata: dict[str, Any], query_context: dict[str, Any]) -> float:
    """Compute author match score in [0, 1].

    Rules:
    - if any query author exactly matches any metadata author -> 1.0
    - else partial overlap uses a heuristic match (substring or token overlap)
      counted per query author:
        (#matched_query_authors / #total_query_authors)
    """

    query_authors = list(dict.fromkeys(_to_str_list(query_context.get("authors"))))
    if not query_authors:
        return 0.0

    meta_authors_raw = metadata.get("authors") or []
    meta_authors = list(dict.fromkeys(_to_str_list(meta_authors_raw)))
    if not meta_authors:
        return 0.0

    meta_set = set(meta_authors)

    # Exact match check.
    if any(q in meta_set for q in query_authors):
        return 1.0

    # Partial match heuristic: per query author, count as "matched" if it
    # appears as a substring in any metadata author OR shares at least one token.
    meta_tokens = [set(a.split()) for a in meta_authors]

    matched = 0
    for q in query_authors:
        q_tokens = set(q.split())
        is_match = False
        for i, m in enumerate(meta_authors):
            if q and (q in m or m in q):
                is_match = True
                break
            if q_tokens and (q_tokens & meta_tokens[i]):
                is_match = True
                break
        if is_match:
            matched += 1

    return _clamp01(matched / max(1, len(query_authors)))


def compute_kg_score(metadata: dict, query_context: dict) -> float:
    """
    Compute a normalized KG relevance score in [0, 1].

    The function follows the component definitions and weights provided in the
    task description.
    """

    # --- 1) Citation Score (log-normalized importance) --------------------------
    citation_count = _coerce_int(metadata.get("citation_count"), default=0)
    citation_score = 0.0
    if MAX_CITATIONS > 0:
        citation_score = math.log(1.0 + max(0, citation_count)) / math.log(1.0 + MAX_CITATIONS)
    citation_score = _clamp01(citation_score)

    # --- 2) Recency Score (newer preferred) ------------------------------------
    year = metadata.get("year")
    year_i = _coerce_int(year, default=0)
    denom = max(1, CURRENT_YEAR - MIN_YEAR)
    # If year is missing/invalid, year_i becomes 0 which will clamp to 0.
    recency_score = (year_i - MIN_YEAR) / denom
    recency_score = _clamp01(recency_score)

    # --- 3) Author Match Score -------------------------------------------------
    author_score = _author_match_score(metadata, query_context)

    # --- 4) Topic Match Score --------------------------------------------------
    meta_topics = list(dict.fromkeys(_to_str_list(metadata.get("topics"))))
    query_topics = list(dict.fromkeys(_to_str_list(query_context.get("topics"))))

    if not query_topics:
        topic_score = 0.0
    else:
        common = set(meta_topics).intersection(query_topics)
        topic_score = len(common) / max(1, len(query_topics))
        topic_score = _clamp01(topic_score)

    # --- 5) Connection Score (weighted edge types) ------------------------------
    connection_types = _to_str_list(metadata.get("connection_types"))
    # Normalize to expected weight keys (case-sensitive in spec, but we accept
    # different casing in real-world payloads).
    connection_score = 0.0
    for t in connection_types:
        key = str(t).strip().upper()
        connection_score += float(CONNECTION_WEIGHTS.get(key, 0.0))
    connection_score = _clamp01(connection_score)

    # --- 6) Graph Distance Score -----------------------------------------------
    gd = metadata.get("graph_distance")
    distance_score = 0.0
    if gd is not None:
        gd_i = _coerce_int(gd, default=0)
        gd_i = max(0, gd_i)
        distance_score = 1.0 / (1.0 + gd_i)
        distance_score = _clamp01(distance_score)

    # --- Final weighted sum ----------------------------------------------------
    score = (
        FINAL_WEIGHTS["citation"] * citation_score
        + FINAL_WEIGHTS["connection"] * connection_score
        + FINAL_WEIGHTS["topic"] * topic_score
        + FINAL_WEIGHTS["author"] * author_score
        + FINAL_WEIGHTS["recency"] * recency_score
        + FINAL_WEIGHTS["distance"] * distance_score
    )
    score = _clamp01(float(score))
    return float(score)

