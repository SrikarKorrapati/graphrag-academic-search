#!/usr/bin/env python3
"""Ingest papers from JSONL into Qdrant using batched sentence-transformers encode."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import settings
from pipeline.dense_retrieval import DenseRetriever
from pipeline.qdrant_ids import paper_qdrant_point_id

logger = logging.getLogger(__name__)


def _text_for_embedding(title: str, abstract: str, keywords: list[str]) -> str:
    """Model input only; paper_id must not appear here."""
    kw = ", ".join(keywords)
    return f"{title}. {abstract}. Keywords: {kw}"


def _vec_size() -> int:
    p = settings.embedding_provider.lower()
    if p == "openai":
        return 1536 if "3-small" in settings.embedding_model else 3072
    if p == "gemini":
        return settings.embedding_dimensions
    if p in ("local", "sentence-transformers", "st"):
        return settings.embedding_dimensions
    return 384


def _parse_paper_row(obj: dict[str, Any]) -> dict[str, Any]:
    paper_id = str(obj["id"])
    title = str(obj.get("title") or "")
    abstract = str(obj.get("abstract") or "")
    authors = list(obj.get("authors") or [])
    venue = str(obj.get("venue") or "")
    keywords = list(obj.get("keywords") or [])
    year = int(obj["year"]) if obj.get("year") is not None else 0
    text = _text_for_embedding(title, abstract, keywords)
    return {
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "venue": venue,
        "keywords": keywords,
        "year": year,
        "text": text,
    }


async def _flush_upsert(
    client: AsyncQdrantClient,
    points: list[PointStruct],
) -> None:
    if not points:
        return
    await client.upsert(
        collection_name=settings.qdrant_collection_name,
        points=points,
    )


async def _run(
    jsonl_path: str,
    offset: int,
    limit: int,
    upsert_batch_size: int,
    st_device: str | None,
    qdrant_path: str | None,
    recreate_collection: bool,
) -> int:
    prov = settings.embedding_provider.lower()
    if prov not in ("local", "sentence-transformers", "st"):
        logger.error("This script requires EMBEDDING_PROVIDER=local (got %r)", settings.embedding_provider)
        return 1

    if offset < 0 or limit < 0:
        logger.error("offset must be non-negative; limit must be >= 0 (0 = read until EOF)")
        return 1

    if qdrant_path:
        os.makedirs(qdrant_path, exist_ok=True)
        client = AsyncQdrantClient(path=qdrant_path)
        logger.info("using embedded Qdrant storage at %s", qdrant_path)
    else:
        client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    dense = DenseRetriever(qdrant=client, st_device=st_device)
    emb_batch = max(1, settings.embedding_batch_size)
    try:
        cols = await client.get_collections()
        names = {c.name for c in cols.collections}
        if recreate_collection and settings.qdrant_collection_name in names:
            await client.delete_collection(collection_name=settings.qdrant_collection_name)
            names.discard(settings.qdrant_collection_name)
            logger.info("deleted collection %r for clean rebuild", settings.qdrant_collection_name)
        if settings.qdrant_collection_name not in names:
            await client.create_collection(
                collection_name=settings.qdrant_collection_name,
                vectors_config=VectorParams(size=_vec_size(), distance=Distance.COSINE),
            )

        pending: list[PointStruct] = []
        buf: list[dict[str, Any]] = []
        rows_read = 0
        points_upserted = 0

        with open(jsonl_path, encoding="utf-8") as f:
            for _ in range(offset):
                if f.readline() == "":
                    logger.error("EOF while skipping to offset %s", offset)
                    return 1

            read_all = limit == 0
            while read_all or rows_read < limit:
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                buf.append(_parse_paper_row(json.loads(line)))
                rows_read += 1

                if len(buf) < emb_batch and (read_all or rows_read < limit):
                    continue

                texts = [r["text"] for r in buf]
                vectors = await dense.embed_texts(texts)
                if len(vectors) != len(buf):
                    raise RuntimeError("embed_texts length mismatch")

                for r, vec in zip(buf, vectors, strict=True):
                    payload = {
                        "paper_id": r["paper_id"],
                        "title": r["title"],
                        "abstract": r["abstract"],
                        "authors": r["authors"],
                        "year": r["year"],
                        "venue": r["venue"],
                        "keywords": r["keywords"],
                        "text": r["text"],
                    }
                    pending.append(
                        PointStruct(
                            id=paper_qdrant_point_id(r["paper_id"]),
                            vector=vec,
                            payload=payload,
                        )
                    )
                buf.clear()

                while len(pending) >= upsert_batch_size:
                    chunk = pending[:upsert_batch_size]
                    del pending[:upsert_batch_size]
                    await _flush_upsert(client, chunk)
                    points_upserted += len(chunk)
                    logger.info(
                        "upserted %s points this run (rows read in segment: %s / %s)",
                        points_upserted,
                        rows_read,
                        limit,
                    )

        if buf:
            texts = [r["text"] for r in buf]
            vectors = await dense.embed_texts(texts)
            if len(vectors) != len(buf):
                raise RuntimeError("embed_texts length mismatch")
            for r, vec in zip(buf, vectors, strict=True):
                payload = {
                    "paper_id": r["paper_id"],
                    "title": r["title"],
                    "abstract": r["abstract"],
                    "authors": r["authors"],
                    "year": r["year"],
                    "venue": r["venue"],
                    "keywords": r["keywords"],
                    "text": r["text"],
                }
                pending.append(
                    PointStruct(
                        id=paper_qdrant_point_id(r["paper_id"]),
                        vector=vec,
                        payload=payload,
                    )
                )
            buf.clear()

        if pending:
            await _flush_upsert(client, pending)
            points_upserted += len(pending)
            pending.clear()

        logger.info(
            "done: upserted %s points (offset=%s, rows read=%s, limit=%s) -> %s",
            points_upserted,
            offset,
            rows_read,
            limit,
            settings.qdrant_collection_name,
        )
        return 0
    finally:
        await dense.close()
        await client.close()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_jsonl = os.path.join(root, "json", "papers.jsonl")
    parser = argparse.ArgumentParser(
        description="Batch-ingest JSONL papers into Qdrant (local sentence-transformers)",
    )
    parser.add_argument("--jsonl", default=default_jsonl, help="path to papers.jsonl")
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="number of JSONL lines to skip from the start (e.g. 5000 after first chunk)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="max non-empty lines after offset (0 = read JSONL until EOF)",
    )
    parser.add_argument(
        "--upsert-batch",
        "--batch",
        type=int,
        default=256,
        dest="upsert_batch",
        metavar="N",
        help="Qdrant upsert size (points per request); alias: --batch",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="force sentence-transformers to use CPU (device=cpu)",
    )
    parser.add_argument(
        "--qdrant-path",
        default="",
        help="optional directory for embedded Qdrant (bypasses Docker when host:port fails)",
    )
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="delete collection if it exists, then create empty (clean rebuild)",
    )
    args = parser.parse_args()
    st_device = "cpu" if args.cpu else None
    qpath = (args.qdrant_path or "").strip() or None
    return asyncio.run(
        _run(
            args.jsonl,
            args.offset,
            args.limit,
            args.upsert_batch,
            st_device,
            qpath,
            args.recreate_collection,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
