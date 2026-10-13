"""Neo4j graph traversal strategies keyed by parsed intent."""

from __future__ import annotations

import logging
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver

from config import settings
from models.schemas import KGResult, ParsedQuery
from pipeline import kg_cypher as KC
from pipeline.kg_explore import explore as explore_fallback
from pipeline.kg_explore import run_param_loop
from pipeline.kg_row_utils import clean, fmt_paper, paper_meta
from pipeline.kg_scoring import compute_kg_score

logger = logging.getLogger(__name__)


class KGTraversal:
    """Runs schema-specific Cypher based on ParsedQuery.intent."""

    def __init__(self, driver: AsyncDriver | None = None) -> None:
        """Optionally reuse an AsyncDriver owned by the application."""
        self._driver = driver
        self._owns_driver = driver is None

    async def _get_driver(self) -> AsyncDriver:
        """Lazily construct a driver when none was injected."""
        if self._driver is None:
            try:
                self._driver = AsyncGraphDatabase.driver(
                    settings.neo4j_uri,
                    auth=(settings.neo4j_user, settings.neo4j_password),
                )
            except Exception as exc:
                raise RuntimeError(f"Neo4j driver init failed: {exc}") from exc
        return self._driver

    async def close(self) -> None:
        """Close the underlying driver if this class created it."""
        if self._owns_driver and self._driver is not None:
            await self._driver.close()
            self._driver = None

    def _yp(self, pq: ParsedQuery) -> dict[str, Any]:
        """Neo4j parameters for optional year bounds on Paper nodes."""
        return {"min_year": pq.year_filter.min, "max_year": pq.year_filter.max}

    async def traverse(self, parsed_query: ParsedQuery) -> list[KGResult]:
        """Execute intent-specific traversals and return deduped KGResult rows."""
        try:
            driver = await self._get_driver()
            intent = parsed_query.intent
            if intent == "find_papers":
                rows = await self._by_author(driver, parsed_query)
            elif intent in ("find_by_topic", "comparative"):
                rows = await self._by_topic(driver, parsed_query)
            elif intent == "find_by_citation":
                rows = await self._by_citation(driver, parsed_query)
            elif intent == "find_by_venue":
                rows = await self._by_venue(driver, parsed_query)
            elif intent == "find_authors":
                rows = await self._find_authors(driver, parsed_query)
            elif intent == "find_by_institution":
                rows = await self._by_institution(driver, parsed_query)
            else:
                rows = await explore_fallback(driver, parsed_query)
            return self._apply_kg_scoring(self._dedupe_cap(rows), parsed_query)
        except Exception as exc:
            logger.exception("KG traversal failed: %s", exc)
            raise RuntimeError(f"Knowledge graph query failed: {exc}") from exc

    @staticmethod
    def _query_context(pq: ParsedQuery) -> dict[str, Any]:
        """Build ``query_context`` for ``compute_kg_score`` from the parsed query."""
        authors = [e.text for e in pq.entities if e.type == "AUTHOR"]
        topics = list(dict.fromkeys(pq.keywords + [e.text for e in pq.entities if e.type == "KEYWORD"]))
        return {"authors": authors, "topics": topics}

    def _apply_kg_scoring(self, rows: list[KGResult], pq: ParsedQuery) -> list[KGResult]:
        """Assign ``KGResult.score`` via ``kg_scoring.compute_kg_score``; sort best-first for RRF."""
        ctx = self._query_context(pq)
        scored = [
            r.model_copy(update={"score": round(compute_kg_score(r.metadata, ctx), 6)})
            for r in rows
        ]
        scored.sort(key=lambda r: float(r.score or 0.0), reverse=True)
        return scored

    def _dedupe_cap(self, rows: list[KGResult]) -> list[KGResult]:
        """Keep first occurrence per id and cap at 30 rows."""
        seen: dict[str, KGResult] = {}
        for r in rows:
            if r.id not in seen:
                seen[r.id] = r
        return list(seen.values())[:30]

    async def _by_author(self, driver: AsyncDriver, pq: ParsedQuery) -> list[KGResult]:
        """Papers written by extracted author entities."""
        names = [e.text for e in pq.entities if e.type == "AUTHOR"]
        if not names:
            return await explore_fallback(driver, pq)

        def b(rec: Any) -> KGResult:
            authors, topics = clean(rec["authors"]), clean(rec["topics"])
            pid = str(rec["id"])
            return KGResult(
                id=pid,
                text=fmt_paper(rec["title"], rec["year"], authors, rec["venue"], topics),
                metadata=paper_meta(pid, rec["title"], rec["year"], authors, rec["venue"], topics),
                score=None,
            )

        return await run_param_loop(driver, KC.AUTHOR_PAPERS, "name", names, self._yp(pq), b)

    async def _by_topic(self, driver: AsyncDriver, pq: ParsedQuery) -> list[KGResult]:
        """Papers linked to keyword entities or parser keywords."""
        kws = [e.text for e in pq.entities if e.type == "KEYWORD"] or pq.keywords
        if not kws:
            return await explore_fallback(driver, pq)

        def b(rec: Any) -> KGResult:
            authors, topics = clean(rec["authors"]), clean(rec["all_topics"])
            pid = str(rec["id"])
            return KGResult(
                id=pid,
                text=fmt_paper(rec["title"], rec["year"], authors, rec["venue"], topics),
                metadata=paper_meta(pid, rec["title"], rec["year"], authors, rec["venue"], topics),
                score=None,
            )

        return await run_param_loop(driver, KC.TOPIC_PAPERS, "keyword", kws, self._yp(pq), b)

    async def _by_citation(self, driver: AsyncDriver, pq: ParsedQuery) -> list[KGResult]:
        """Papers citing a paper matched by title fragment."""
        titles = [e.text for e in pq.entities if e.type == "PAPER"]
        if not titles:
            return await explore_fallback(driver, pq)

        def b(rec: Any) -> KGResult:
            authors = clean(rec["authors"])
            pid = str(rec["id"])
            meta = paper_meta(pid, rec["title"], rec["year"], authors, rec["venue"], [], str(rec["cites"]))
            return KGResult(
                id=pid,
                text=fmt_paper(rec["title"], rec["year"], authors, rec["venue"], []),
                metadata=meta,
                score=None,
            )

        return await run_param_loop(driver, KC.CITE_PAPERS, "title", titles, self._yp(pq), b)

    async def _by_venue(self, driver: AsyncDriver, pq: ParsedQuery) -> list[KGResult]:
        """Papers published in a venue matched by name fragment."""
        names = [e.text for e in pq.entities if e.type == "VENUE"] or (pq.keywords or [])
        if not names:
            return await explore_fallback(driver, pq)

        def b(rec: Any) -> KGResult:
            authors, topics = clean(rec["authors"]), clean(rec["topics"])
            pid = str(rec["id"])
            return KGResult(
                id=pid,
                text=fmt_paper(rec["title"], rec["year"], authors, rec["venue"], topics),
                metadata=paper_meta(pid, rec["title"], rec["year"], authors, rec["venue"], topics),
                score=None,
            )

        return await run_param_loop(driver, KC.VENUE_PAPERS, "venue_name", names, self._yp(pq), b)

    async def _find_authors(self, driver: AsyncDriver, pq: ParsedQuery) -> list[KGResult]:
        """Authors who wrote papers on keyword-matched topics."""
        kws = [e.text for e in pq.entities if e.type == "KEYWORD"] or pq.keywords
        if not kws:
            return []

        def b(rec: Any) -> KGResult:
            aid = str(rec["id"])
            papers = clean(rec["papers"])
            inst = rec["institution"]
            return KGResult(
                id=aid,
                text=(
                    f"{rec['name']} at {inst or 'unknown institution'} — "
                    f"{rec['paper_count']} papers on related topics."
                ),
                metadata={
                    "author_id": aid,
                    "name": rec["name"],
                    "papers": papers,
                    "institution": inst,
                    "paper_count": rec["paper_count"],
                    "kind": "author",
                },
                score=None,
            )

        return await run_param_loop(driver, KC.FIND_AUTHORS, "keyword", kws, {}, b)

    async def _by_institution(self, driver: AsyncDriver, pq: ParsedQuery) -> list[KGResult]:
        """Papers from authors affiliated with matched institutions."""
        insts = [e.text for e in pq.entities if e.type == "INSTITUTION"]
        if not insts:
            return await explore_fallback(driver, pq)

        def b(rec: Any) -> KGResult:
            authors, topics = clean(rec["authors"]), clean(rec["topics"])
            pid = str(rec["id"])
            return KGResult(
                id=pid,
                text=fmt_paper(rec["title"], rec["year"], authors, None, topics),
                metadata=paper_meta(pid, rec["title"], rec["year"], authors, None, topics),
                score=None,
            )

        return await run_param_loop(driver, KC.INST_PAPERS, "institution", insts, self._yp(pq), b)
