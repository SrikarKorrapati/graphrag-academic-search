#!/usr/bin/env python3
"""Run sample queries through the same pipeline as POST /query; one JSON file per query.

Stages captured (aligned with main.py):
  - parsed_query (query_parser)
  - kg_traversal (KGTraversal.traverse)
  - dense_retrieval (DenseRetriever.retrieve)
  - hybrid_fusion (HybridFusion.fuse_and_rerank)
  - final_output (ContextAggregator + LLMGenerator answer + fused papers)

Usage (from graphrag/):
  python3 scripts/run_sample_queries_dump.py
  python3 scripts/run_sample_queries_dump.py --fixture tests/fixtures/sample_test_queries.json --out-dir query_logs/sample_query_runs
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

# graphrag/ is cwd for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from models.schemas import DenseResult, FusedResult, KGResult, ParsedQuery
from neo4j import AsyncGraphDatabase
from pipeline.context_aggregator import ContextAggregator
from pipeline.dense_retrieval import DenseRetriever
from pipeline.hybrid_fusion import HybridFusion
from pipeline.kg_scoring import compute_kg_score
from pipeline.kg_traversal import KGTraversal
from pipeline.llm_generator import LLMGenerator
from pipeline.paper_abstract_store import abstract_for_paper
from pipeline.query_parser import QueryParser
from qdrant_client import AsyncQdrantClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
) -> None:
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    cases = raw.get("queries") or []
    if not cases:
        raise SystemExit("No queries in fixture")

    out_dir.mkdir(parents=True, exist_ok=True)

    neo4j_driver = None
    qdrant: AsyncQdrantClient | None = None
    try:
        neo4j_driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
    except Exception as exc:
        logger.error("Neo4j connect failed: %s", exc)
    try:
        qdrant = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    except Exception as exc:
        logger.error("Qdrant connect failed: %s", exc)

    kg = KGTraversal(neo4j_driver) if neo4j_driver else None
    dense = DenseRetriever(qdrant) if qdrant else None
    fusion = HybridFusion()
    parser = QueryParser()
    aggregator = ContextAggregator()
    generator = LLMGenerator()

    if not fusion:
        raise SystemExit("HybridFusion unavailable")

    for case in cases:
        qid = str(case.get("id", "unknown"))
        qtext = str(case.get("query", "")).strip()
        out_path = out_dir / f"{qid}.json"
        record: dict[str, Any] = {
            "query_id": qid,
            "user_query": qtext,
            "parsed_query": None,
            "kg_traversal": None,
            "dense_retrieval": None,
            "hybrid_fusion": None,
            "final_output": None,
            "pipeline_error": None,
        }
        t0 = time.perf_counter()
        try:
            parsed = await asyncio.to_thread(parser.parse, qtext)
            record["parsed_query"] = parsed.model_dump(mode="json")

            kg_results, kg_err = await _safe_kg(kg, parsed)
            dense_results, dense_err = await _safe_dense(dense, parsed)

            record["kg_traversal"] = {
                "module": "pipeline.kg_traversal",
                "n_hits": len(kg_results),
                "hits": _dump_kg(kg_results, parsed),
                "error": kg_err,
            }
            record["dense_retrieval"] = {
                "module": "pipeline.dense_retrieval",
                "n_hits": len(dense_results),
                "hits": _dump_dense(dense_results),
                "error": dense_err,
            }

            fused = await fusion.fuse_and_rerank(
                qtext,
                kg_results,
                dense_results,
                top_k=settings.top_k_final,
            )
            record["hybrid_fusion"] = {
                "module": "pipeline.hybrid_fusion",
                "top_k": settings.top_k_final,
                "n_fused": len(fused),
                "papers": _dump_fused(fused),
            }

            context = aggregator.aggregate(fused)
            answer = await asyncio.to_thread(
                generator.generate,
                qtext,
                context,
                parsed.intent,
            )
            ms = (time.perf_counter() - t0) * 1000
            record["final_output"] = {
                "module": "pipeline.llm_generator + pipeline.context_aggregator",
                "answer": answer,
                "context_aggregated": context,
                "papers_fused": _dump_fused(fused),
                "intent": parsed.intent,
                "latency_ms": round(ms, 2),
            }
        except Exception as exc:
            logger.exception("Pipeline failed for %s: %s", qid, exc)
            record["pipeline_error"] = str(exc)
            record["final_output"] = {
                "answer": None,
                "error": str(exc),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            }

        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Wrote %s", out_path)

    if neo4j_driver:
        await neo4j_driver.close()
    if qdrant:
        await qdrant.close()
    if dense:
        await dense.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    ap.add_argument(
        "--fixture",
        type=Path,
        default=root / "tests" / "fixtures" / "sample_test_queries.json",
        help="JSON fixture with a 'queries' array (id + query)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=root / "query_logs" / "sample_query_runs",
        help="Directory for one JSON file per query id",
    )
    args = ap.parse_args()
    fixture = args.fixture if args.fixture.is_absolute() else root / args.fixture
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    asyncio.run(run_all(fixture, out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
