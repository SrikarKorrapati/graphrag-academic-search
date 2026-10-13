#!/usr/bin/env python3
"""Run all sample_test_queries.json entries through the full pipeline and save one JSON per query.

Stages captured (aligned with query_logs/sample_query_runs/q01.json):
  - parsed_query (query_parser)
  - kg_traversal (KGTraversal.traverse)
  - dense_retrieval (DenseRetriever.retrieve)
  - hybrid_fusion (HybridFusion.fuse_and_rerank)
  - final_output (ContextAggregator + LLMGenerator)

Usage (from graphrag/):
  python3 scripts/run_sample_query_batch.py
  python3 scripts/run_sample_query_batch.py --out-dir query_logs/sample_query_runs
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

# graphrag/ as cwd for imports
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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
logger = logging.getLogger("run_sample_query_batch")


def _dump_kg(rows: list[KGResult], pq: ParsedQuery | None) -> list[dict[str, Any]]:
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


async def _run_one(
    user_query: str,
    query_id: str,
    parser: QueryParser,
    kg: KGTraversal | None,
    dense: DenseRetriever | None,
    fusion: HybridFusion,
    aggregator: ContextAggregator,
    generator: LLMGenerator,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    pipeline_error: str | None = None
    try:
        parsed = await asyncio.to_thread(parser.parse, user_query)
    except Exception as exc:
        logger.exception("Parse failed for %s", query_id)
        return {
            "query_id": query_id,
            "user_query": user_query,
            "parsed_query": None,
            "kg_traversal": {"module": "pipeline.kg_traversal", "n_hits": 0, "hits": [], "error": None},
            "dense_retrieval": {
                "module": "pipeline.dense_retrieval",
                "n_hits": 0,
                "hits": [],
                "error": None,
            },
            "hybrid_fusion": {"module": "pipeline.hybrid_fusion", "top_k": settings.top_k_final, "n_fused": 0, "papers": []},
            "final_output": {
                "module": "pipeline.llm_generator + pipeline.context_aggregator",
                "answer": "",
                "context_aggregated": "",
                "papers_fused": [],
                "intent": "exploratory",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            },
            "pipeline_error": f"query_parser: {exc}",
        }

    kg_err: str | None = None
    dense_err: str | None = None
    kg_results: list[KGResult] = []
    dense_results: list[DenseResult] = []

    async def _safe_kg() -> None:
        nonlocal kg_results, kg_err
        if not kg:
            return
        try:
            kg_results = await kg.traverse(parsed)
        except Exception as exc:
            kg_err = str(exc)
            logger.warning("KG failed %s: %s", query_id, exc)

    async def _safe_dense() -> None:
        nonlocal dense_results, dense_err
        if not dense:
            return
        try:
            dense_results = await dense.retrieve(parsed)
        except Exception as exc:
            dense_err = str(exc)
            logger.warning("Dense failed %s: %s", query_id, exc)

    await asyncio.gather(_safe_kg(), _safe_dense())

    fused: list[FusedResult] = []
    fusion_err: str | None = None
    try:
        fused = await fusion.fuse_and_rerank(
            user_query, kg_results, dense_results, top_k=settings.top_k_final
        )
    except Exception as exc:
        fusion_err = str(exc)
        logger.exception("Fusion failed for %s", query_id)
        pipeline_error = f"hybrid_fusion: {exc}"

    context = aggregator.aggregate(fused)
    answer = ""
    gen_err: str | None = None
    try:
        answer = await asyncio.to_thread(generator.generate, user_query, context, parsed.intent)
    except Exception as exc:
        gen_err = str(exc)
        if pipeline_error:
            pipeline_error += f"; llm_generator: {exc}"
        else:
            pipeline_error = f"llm_generator: {exc}"
        logger.exception("Generate failed for %s", query_id)

    ms = round((time.perf_counter() - t0) * 1000, 2)

    hybrid_block: dict[str, Any] = {
        "module": "pipeline.hybrid_fusion",
        "top_k": settings.top_k_final,
        "n_fused": len(fused),
        "papers": _dump_fused(fused),
    }
    if fusion_err:
        hybrid_block["error"] = fusion_err

    final_block: dict[str, Any] = {
        "module": "pipeline.llm_generator + pipeline.context_aggregator",
        "answer": answer,
        "context_aggregated": context,
        "papers_fused": _dump_fused(fused),
        "intent": parsed.intent,
        "latency_ms": ms,
    }
    if gen_err:
        final_block["error"] = gen_err

    return {
        "query_id": query_id,
        "user_query": user_query,
        "parsed_query": parsed.model_dump(mode="json"),
        "kg_traversal": {
            "module": "pipeline.kg_traversal",
            "n_hits": len(kg_results),
            "hits": _dump_kg(kg_results, parsed),
            "error": kg_err,
        },
        "dense_retrieval": {
            "module": "pipeline.dense_retrieval",
            "n_hits": len(dense_results),
            "hits": _dump_dense(dense_results),
            "error": dense_err,
        },
        "hybrid_fusion": hybrid_block,
        "final_output": final_block,
        "pipeline_error": pipeline_error,
    }


async def _async_main(args: argparse.Namespace) -> int:
    fixture = Path(args.fixture)
    if not fixture.is_file():
        logger.error("Fixture not found: %s", fixture)
        return 1

    data = json.loads(fixture.read_text(encoding="utf-8"))
    queries = data.get("queries") or []
    if not queries:
        logger.error("No queries in fixture")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    neo4j_driver = None
    qdrant: AsyncQdrantClient | None = None
    kg: KGTraversal | None = None
    dense: DenseRetriever | None = None

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

    try:
        fusion = HybridFusion()
    except Exception as exc:
        logger.error("HybridFusion init failed: %s", exc)
        return 1

    parser = QueryParser()
    aggregator = ContextAggregator()
    generator = LLMGenerator()

    n_ok = 0
    only_set: set[str] = set()
    if str(args.only).strip():
        only_set = {x.strip() for x in str(args.only).split(",") if x.strip()}

    try:
        for item in queries:
            qid = str(item.get("id", "")).strip() or "unknown"
            if only_set and qid not in only_set:
                continue
            qtext = str(item.get("query", "")).strip()
            if not qtext:
                continue
            logger.info("Running %s …", qid)
            record = await _run_one(qtext, qid, parser, kg, dense, fusion, aggregator, generator)
            out_path = out_dir / f"{qid}.json"
            out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("Wrote %s", out_path)
            n_ok += 1
            if args.sleep_seconds > 0:
                await asyncio.sleep(args.sleep_seconds)
    finally:
        if kg and neo4j_driver:
            await kg.close()
        if qdrant:
            await qdrant.close()
        if neo4j_driver:
            await neo4j_driver.close()

    logger.info("Finished %s queries", n_ok)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Run sample queries through GraphRAG pipeline")
    ap.add_argument(
        "--fixture",
        default=str(_ROOT / "tests" / "fixtures" / "sample_test_queries.json"),
        help="Path to sample_test_queries.json",
    )
    ap.add_argument(
        "--out-dir",
        default=str(_ROOT / "query_logs" / "sample_query_runs"),
        help="Directory for per-query JSON outputs",
    )
    ap.add_argument(
        "--sleep-seconds",
        type=float,
        default=2.5,
        help="Pause after each query (reduces LLM provider rate limits). Use 0 to disable.",
    )
    ap.add_argument(
        "--only",
        default="",
        help="Comma-separated query ids to run (e.g. q08,q12). Empty = all.",
    )
    args = ap.parse_args()
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
