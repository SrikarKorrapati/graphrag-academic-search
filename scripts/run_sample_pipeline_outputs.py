#!/usr/bin/env python3
"""Run fixtures from tests/fixtures/sample_test_queries.json through the full pipeline.

Writes one JSON per query under query_logs/sample_pipeline_runs/ with:
  query_parser, kg_traversal, dense_retrieval, hybrid_fusion, final (answer + papers + latency).

Run from the graphrag directory:
  python3 scripts/run_sample_pipeline_outputs.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Allow `python3 scripts/run_sample_pipeline_outputs.py` from repo graphrag/ root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from neo4j import AsyncGraphDatabase
from qdrant_client import AsyncQdrantClient

from config import settings
from models.schemas import DenseResult, FusedResult, KGResult, ParsedQuery
from pipeline.context_aggregator import ContextAggregator
from pipeline.dense_retrieval import DenseRetriever
from pipeline.hybrid_fusion import HybridFusion
from pipeline.kg_scoring import compute_kg_score
from pipeline.kg_traversal import KGTraversal
from pipeline.llm_generator import LLMGenerator
from pipeline.paper_abstract_store import abstract_for_paper
from pipeline.query_parser import QueryParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_sample_pipeline")


def _dump_kg(rows: list[KGResult], pq: ParsedQuery | None = None) -> list[dict[str, Any]]:
    ctx = KGTraversal._query_context(pq) if pq is not None else None
    out: list[dict[str, Any]] = []
    for r in rows:
        d = r.model_dump(mode="json")
        meta = d.get("metadata")
        if isinstance(meta, dict) and not str(meta.get("abstract", "")).strip():
            pid = str(meta.get("paper_id", d.get("id", ""))).strip()
            ab = abstract_for_paper(pid)
            if ab.strip():
                d["metadata"] = {**meta, "abstract": ab}
        if ctx is not None and d.get("score") is None:
            try:
                d["score"] = round(float(compute_kg_score(r.metadata, ctx)), 6)
            except Exception:
                d["score"] = 0.0
        out.append(d)
    return out


def _dump_dense(rows: list[DenseResult]) -> list[dict[str, Any]]:
    return [r.model_dump(mode="json") for r in rows]


def _dump_fused(rows: list[FusedResult]) -> list[dict[str, Any]]:
    return [r.model_dump(mode="json") for r in rows]


async def _safe_kg(kg: KGTraversal | None, parsed: ParsedQuery) -> tuple[list[KGResult], str | None]:
    if not kg:
        return [], None
    try:
        return await kg.traverse(parsed), None
    except Exception as exc:
        logger.warning("KG traversal failed: %s", exc)
        return [], str(exc)


async def _safe_dense(dense: DenseRetriever | None, parsed: ParsedQuery) -> tuple[list[DenseResult], str | None]:
    if not dense:
        return [], None
    try:
        return await dense.retrieve(parsed), None
    except Exception as exc:
        logger.warning("Dense retrieval failed: %s", exc)
        return [], str(exc)


async def run_all(
    fixture_path: Path,
    out_dir: Path,
    *,
    limit: int | None = None,
) -> None:
    neo4j_driver = None
    qdrant: AsyncQdrantClient | None = None
    try:
        try:
            neo4j_driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
        except Exception as exc:
            logger.error("Neo4j connect failed: %s", exc)
            neo4j_driver = None
        try:
            qdrant = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        except Exception as exc:
            logger.error("Qdrant connect failed: %s", exc)
            qdrant = None

        kg = KGTraversal(neo4j_driver) if neo4j_driver else None
        dense = DenseRetriever(qdrant) if qdrant else None
        try:
            fusion = HybridFusion()
        except Exception as exc:
            raise RuntimeError(f"HybridFusion init failed: {exc}") from exc
        parser = QueryParser()
        aggregator = ContextAggregator()
        generator = LLMGenerator()

        if not kg and not dense:
            raise RuntimeError("Neo4j and Qdrant both unavailable")

        raw = json.loads(fixture_path.read_text(encoding="utf-8"))
        cases: list[dict[str, Any]] = list(raw.get("queries") or [])
        if limit is not None:
            cases = cases[:limit]

        out_dir.mkdir(parents=True, exist_ok=True)

        for case in cases:
            qid = str(case.get("id", "unknown"))
            qtext = str(case.get("query", "")).strip()
            if not qtext:
                continue

            t0 = time.perf_counter()
            logger.info("Running %s …", qid)

            parsed = await asyncio.to_thread(parser.parse, qtext)

            kg_rows, kg_err = await _safe_kg(kg, parsed)
            dense_rows, dense_err = await _safe_dense(dense, parsed)

            fused = await fusion.fuse_and_rerank(
                qtext,
                kg_rows,
                dense_rows,
                top_k=settings.top_k_final,
            )
            context = aggregator.aggregate(fused)
            answer = await asyncio.to_thread(generator.generate, qtext, context, parsed.intent)
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)

            payload: dict[str, Any] = {
                "query_id": qid,
                "user_query": qtext,
                "grounded_in": case.get("grounded_in"),
                "query_parser": {"parsed_query": parsed.model_dump(mode="json")},
                "kg_traversal": {
                    "n_hits": len(kg_rows),
                    "error": kg_err,
                    "hits": _dump_kg(kg_rows, parsed),
                },
                "dense_retrieval": {
                    "n_hits": len(dense_rows),
                    "error": dense_err,
                    "hits": _dump_dense(dense_rows),
                },
                "hybrid_fusion": {
                    "top_k": settings.top_k_final,
                    "n_fused": len(fused),
                    "papers": _dump_fused(fused),
                },
                "final": {
                    "answer": answer,
                    "latency_ms": latency_ms,
                    "aggregated_context": context,
                    "papers": _dump_fused(fused),
                },
            }

            out_path = out_dir / f"{qid}.json"
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("Wrote %s", out_path)

    finally:
        if kg:
            await kg.close()
        if dense:
            await dense.close()
        if neo4j_driver:
            await neo4j_driver.close()
        if qdrant:
            await qdrant.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    root = _ROOT
    ap.add_argument(
        "--fixture",
        type=Path,
        default=root / "tests" / "fixtures" / "sample_test_queries.json",
        help="Path to sample_test_queries.json",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=root / "query_logs" / "sample_pipeline_runs",
        help="Directory for per-query JSON outputs",
    )
    ap.add_argument("--limit", type=int, default=None, help="Run only first N queries")
    args = ap.parse_args()
    fixture = args.fixture if args.fixture.is_absolute() else root / args.fixture
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    asyncio.run(run_all(fixture, out_dir, limit=args.limit))


if __name__ == "__main__":
    main()
