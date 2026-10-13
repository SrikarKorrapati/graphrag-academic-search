"""Fallback keyword scan and shared parameterized Cypher execution."""

from __future__ import annotations

from typing import Any, Callable

from neo4j import AsyncDriver

from models.schemas import KGResult, ParsedQuery
from pipeline import kg_cypher as KC


async def run_param_loop(
    driver: AsyncDriver,
    cypher: str,
    param: str,
    values: list[str],
    extra: dict[str, Any],
    build: Callable[[Any], KGResult],
) -> list[KGResult]:
    """Execute the same Cypher once per string parameter value."""
    out: list[KGResult] = []
    for v in values:
        async with driver.session() as session:
            res = await session.run(cypher, **{param: v}, **extra)
            async for rec in res:
                out.append(build(rec))
    return out


async def explore(driver: AsyncDriver, pq: ParsedQuery) -> list[KGResult]:
    """Run broad name/title CONTAINS search when intent is unknown or thin."""
    kws = pq.keywords or ([pq.original] if pq.original else [])
    if not kws:
        return []
    rows: list[KGResult] = []
    for kw in kws[:5]:
        async with driver.session() as session:
            res = await session.run(KC.EXPLORE, kw=kw)
            async for rec in res:
                n = rec["n"]
                labels, props = list(n.labels), dict(n)
                nid = str(props.get("id", n.element_id))
                title = props.get("title") or props.get("name") or "node"
                rows.append(
                    KGResult(
                        id=nid,
                        text=f"[{'/'.join(labels)}] {title}",
                        metadata={"labels": labels, **props},
                        score=None,
                    )
                )
    return rows
