"""FastAPI entrypoint wiring GraphRAG retrieval and configurable LLM generation."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from neo4j import AsyncGraphDatabase
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import settings
from models.schemas import (
    DenseResult,
    FusedResult,
    IngestRequest,
    IngestResponse,
    KGResult,
    ParsedQuery,
    QueryRequest,
    QueryResponse,
)
from pipeline.context_aggregator import ContextAggregator
from pipeline.dense_retrieval import DenseRetriever
from pipeline.qdrant_ids import paper_qdrant_point_id
from pipeline.hybrid_fusion import HybridFusion
from pipeline.kg_scoring import compute_kg_score
from pipeline.kg_traversal import KGTraversal
from pipeline.paper_abstract_store import abstract_for_paper
from pipeline.llm_generator import LLMGenerator
from pipeline.query_parser import QueryParser
from query_logs.pipeline_trace import append_pipeline_stage, write_sample_formatted_for_request
from query_logs.writer import append_query_log, build_log_record

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _qdrant_point_id(paper_id: str) -> uuid.UUID:
    """Stable Qdrant point id (API requires UUID or uint; keep real id in payload)."""
    return uuid.uuid5(uuid.NAMESPACE_URL, f"paper:{paper_id}")


def _vec_size() -> int:
    """Embedding dimension for collection creation."""
    p = settings.embedding_provider.lower()
    if p == "openai":
        return 1536 if "3-small" in settings.embedding_model else 3072
    if p == "gemini":
        return settings.embedding_dimensions
    if p in ("local", "sentence-transformers", "st"):
        return settings.embedding_dimensions
    return 384


class AppState:
    """Shared resources for the service lifetime."""

    neo4j_driver: Any = None
    qdrant: AsyncQdrantClient | None = None
    parser = QueryParser()
    fusion: HybridFusion | None = None
    aggregator = ContextAggregator()
    generator = LLMGenerator()
    kg: KGTraversal | None = None
    dense: DenseRetriever | None = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open Neo4j and Qdrant clients; build pipeline wrappers."""
    try:
        state.neo4j_driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
    except Exception as exc:
        logger.error("Neo4j connect failed: %s", exc)
        state.neo4j_driver = None
    try:
        state.qdrant = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
    except Exception as exc:
        logger.error("Qdrant connect failed: %s", exc)
        state.qdrant = None
    state.kg = KGTraversal(state.neo4j_driver) if state.neo4j_driver else None
    state.dense = DenseRetriever(state.qdrant) if state.qdrant else None
    try:
        state.fusion = HybridFusion()
    except Exception as exc:
        logger.error("HybridFusion init failed: %s", exc)
        state.fusion = None
    yield
    if state.qdrant:
        await state.qdrant.close()
    if state.neo4j_driver:
        await state.neo4j_driver.close()


app = FastAPI(title="GraphRAG Academic Search", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


def _trace_enabled() -> bool:
    return settings.query_pipeline_trace_enabled


def _dump_kg(rows: list[KGResult], pq: ParsedQuery | None = None) -> list[dict[str, Any]]:
    """Serialize KG hits; backfill ``abstract`` from JSONL and ``score`` from ``compute_kg_score`` for traces."""
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


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest) -> QueryResponse:
    """Run parse → KG + dense → fuse → aggregate → generate."""
    t0 = time.perf_counter()
    trace_on = _trace_enabled()
    request_id = str(uuid.uuid4())
    if not state.fusion:
        raise HTTPException(status_code=503, detail="Reranker stack unavailable")
    if not state.kg and not state.dense:
        raise HTTPException(
            status_code=503,
            detail="No retrieval backends (Neo4j and Qdrant both unavailable)",
        )
    try:
        parsed = await asyncio.to_thread(state.parser.parse, req.query)
        logger.info("Parsed intent=%s entities=%s", parsed.intent, len(parsed.entities))
        if trace_on:
            await asyncio.to_thread(
                append_pipeline_stage,
                request_id=request_id,
                stage="query_parser",
                user_query=req.query,
                output=parsed.model_dump(mode="json"),
            )

        async def _safe_kg() -> list[KGResult]:
            if not state.kg:
                return []
            try:
                return await state.kg.traverse(parsed)
            except Exception as exc:
                logger.warning("KG traversal failed (continuing without graph): %s", exc)
                return []

        async def _safe_dense() -> list[DenseResult]:
            if not state.dense:
                return []
            try:
                return await state.dense.retrieve(parsed)
            except Exception as exc:
                logger.warning("Dense retrieval failed (continuing without vectors): %s", exc)
                return []

        kg_results, dense_results = await asyncio.gather(_safe_kg(), _safe_dense())
        logger.info("KG hits=%s dense hits=%s", len(kg_results), len(dense_results))
        if trace_on:
            await asyncio.to_thread(
                append_pipeline_stage,
                request_id=request_id,
                stage="kg_traversal",
                user_query=req.query,
                output={"n_hits": len(kg_results), "hits": _dump_kg(kg_results, parsed)},
            )
            await asyncio.to_thread(
                append_pipeline_stage,
                request_id=request_id,
                stage="dense_retrieval",
                user_query=req.query,
                output={"n_hits": len(dense_results), "hits": _dump_dense(dense_results)},
            )

        fused = await state.fusion.fuse_and_rerank(
            req.query, kg_results, dense_results, top_k=settings.top_k_final
        )
        logger.info("Top rerank scores=%s", [f.rerank_score for f in fused[:5]])
        if trace_on:
            await asyncio.to_thread(
                append_pipeline_stage,
                request_id=request_id,
                stage="hybrid_fusion",
                user_query=req.query,
                output={
                    "top_k": settings.top_k_final,
                    "n_fused": len(fused),
                    "papers": _dump_fused(fused),
                },
            )

        context = state.aggregator.aggregate(fused)
        answer = await asyncio.to_thread(
            state.generator.generate, req.query, context, parsed.intent
        )
        ms = (time.perf_counter() - t0) * 1000
        latency = round(ms, 2)
        resp = QueryResponse(
            answer=answer,
            papers=fused,
            parsed_query=parsed,
            latency_ms=latency,
        )
        if trace_on:
            await asyncio.to_thread(
                append_pipeline_stage,
                request_id=request_id,
                stage="overall",
                user_query=req.query,
                output=resp.model_dump(mode="json"),
            )
            await asyncio.to_thread(write_sample_formatted_for_request, request_id)
        paper_rows = [
            {
                "id": p.id,
                "title": (p.metadata or {}).get("title"),
                "source": p.source,
            }
            for p in fused
        ]
        log_row = build_log_record(
            req.query,
            answer,
            float(latency),
            parsed.model_dump(mode="json"),
            paper_rows,
        )
        await asyncio.to_thread(append_query_log, log_row)
        return resp
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Query pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(body: IngestRequest) -> IngestResponse:
    """Embed papers and upsert into Qdrant."""
    if not state.qdrant or not state.dense:
        raise HTTPException(status_code=503, detail="Qdrant unavailable")
    try:
        cols = await state.qdrant.get_collections()
        if settings.qdrant_collection_name not in {c.name for c in cols.collections}:
            await state.qdrant.create_collection(
                collection_name=settings.qdrant_collection_name,
                vectors_config=VectorParams(size=_vec_size(), distance=Distance.COSINE),
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Qdrant setup failed: {exc}") from exc
    points: list[PointStruct] = []
    try:
        for p in body.papers:
            kw = ", ".join(p.keywords)
            # Embed content only; paper_id is not part of the model input.
            text = f"{p.title}. {p.abstract}. Keywords: {kw}"
            vec = await state.dense.embed_text(text)
            payload: dict[str, Any] = {
                "paper_id": p.paper_id,
                "title": p.title,
                "abstract": p.abstract,
                "authors": p.authors,
                "year": p.year,
                "venue": p.venue,
                "keywords": p.keywords,
                "text": text,
            }
            points.append(
                PointStruct(
                    id=paper_qdrant_point_id(p.paper_id),
                    vector=vec,
                    payload=payload,
                )
            )
        if points:
            await state.qdrant.upsert(
                collection_name=settings.qdrant_collection_name,
                points=points,
            )
        return IngestResponse(ingested=len(points))
    except Exception as exc:
        logger.exception("Ingest failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
