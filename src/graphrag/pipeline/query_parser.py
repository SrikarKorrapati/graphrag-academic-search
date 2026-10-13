"""Natural-language query parsing via configured LLM (single call, JSON-only)."""

from __future__ import annotations

import json
import logging
from typing import Any

from models.schemas import Entity, ParsedQuery, Relation, YearFilter
from pipeline.llm_backend import complete_chat

logger = logging.getLogger(__name__)

_SYSTEM = """You are a query parser for an academic paper knowledge graph.
Return ONE JSON object only (no markdown, no prose). Keys:
- entities: array of {text, type}. type must be one of:
  AUTHOR, PAPER, INSTITUTION, KEYWORD, VENUE, YEAR, UNKNOWN
- relations: array of {subject, predicate, object}.
  predicate one of: WROTE, HAS_TOPIC, AFFILIATED_WITH, PUBLISHED_IN, CITES
- intent: one of:
  find_papers, find_authors, find_by_topic, find_by_venue,
  find_by_citation, find_by_institution, comparative, exploratory
- keywords: research topic strings only (no person/venue/year names).
  Expand slightly if helpful (e.g. "attention mechanism" -> include related terms).
- sub_queries: if the query has multiple constraints, split into simpler strings; else []
- year_filter: {min: number|null, max: number|null}
  "recent papers" -> min=2022, max=null. "2023 papers" -> min=max=2023.
- cypher_hint: short Cypher MATCH template string for the main intent (use $params).

Examples:
"papers by Hinton on transformers" ->
  

"ICLR 2023 papers about diffusion" ->
  entities: ICLR 2023 VENUE, diffusion KEYWORD; intent find_by_venue;
  year_filter min=max=2023 if year tied to venue.

Map user wording to the closest intent. Output valid JSON only."""

_ALLOWED_ENTITY = frozenset(
    {"AUTHOR", "PAPER", "INSTITUTION", "KEYWORD", "VENUE", "YEAR", "UNKNOWN"}
)
_ALLOWED_INTENT = frozenset(
    {
        "find_papers",
        "find_authors",
        "find_by_topic",
        "find_by_venue",
        "find_by_citation",
        "find_by_institution",
        "comparative",
        "exploratory",
    }
)


def _coerce_payload(data: dict[str, Any], original: str) -> ParsedQuery:
    """Map arbitrary JSON dict fields into a strict ParsedQuery instance."""
    ents: list[Entity] = []
    for e in data.get("entities") or []:
        if not isinstance(e, dict):
            continue
        t = str(e.get("type", "UNKNOWN")).upper()
        if t not in _ALLOWED_ENTITY:
            t = "UNKNOWN"
        ents.append(Entity(text=str(e.get("text", "")).strip(), type=t))
    rels: list[Relation] = []
    for r in data.get("relations") or []:
        if not isinstance(r, dict):
            continue
        rels.append(
            Relation(
                subject=str(r.get("subject", "")),
                predicate=str(r.get("predicate", "")),
                object=str(r.get("object", "")),
            )
        )
    intent = str(data.get("intent", "exploratory")).lower()
    if intent not in _ALLOWED_INTENT:
        intent = "exploratory"
    yf = data.get("year_filter") or {}
    year_filter = YearFilter(
        min=yf.get("min") if yf.get("min") is not None else None,
        max=yf.get("max") if yf.get("max") is not None else None,
    )
    kws = [str(x).strip() for x in (data.get("keywords") or []) if str(x).strip()]
    subs = [str(x).strip() for x in (data.get("sub_queries") or []) if str(x).strip()]
    return ParsedQuery(
        original=original,
        entities=ents,
        relations=rels,
        intent=intent,
        keywords=kws,
        sub_queries=subs,
        year_filter=year_filter,
        cypher_hint=str(data.get("cypher_hint", "")),
    )


def _strip_json_fences(raw: str) -> str:
    """Remove optional ``` / ```json fences from model output."""
    s = raw.strip()
    if not s.startswith("```"):
        return s
    lines = s.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _fallback(original: str) -> ParsedQuery:
    """Return keyword-only structure after repeated parse failures."""
    return ParsedQuery(
        original=original,
        entities=[],
        relations=[],
        intent="exploratory",
        keywords=[original.strip()] if original.strip() else [],
        sub_queries=[],
        year_filter=YearFilter(),
        cypher_hint="",
    )


class QueryParser:
    """Parses user questions into structured ParsedQuery using the configured LLM."""

    def __init__(self) -> None:
        """Parser is stateless; backend reads ``settings`` per call."""

    def _call_model(self, user_text: str) -> str:
        """Run a single LLM generation and return raw text (JSON when possible)."""
        return complete_chat(_SYSTEM, f"Query:\n{user_text}", json_mode=True)

    def parse(self, query: str) -> ParsedQuery:
        """Parse natural language into ParsedQuery; retry once then fallback."""
        original = query.strip()
        for attempt in range(2):
            try:
                raw = _strip_json_fences(self._call_model(original))
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError("Top-level JSON must be an object")
                pq = _coerce_payload(data, original)
                pq.original = original
                return pq
            except Exception as exc:
                logger.warning("Query parse attempt %s failed: %s", attempt + 1, exc)
                if attempt == 1:
                    return _fallback(original)
        return _fallback(original)
