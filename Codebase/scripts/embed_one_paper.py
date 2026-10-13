#!/usr/bin/env python3
"""Embed the first line of papers.jsonl and upsert into Qdrant (same shape as POST /ingest)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import settings
from pipeline.dense_retrieval import DenseRetriever
from pipeline.qdrant_ids import paper_qdrant_point_id


def _text_for_embedding(
    title: str,
    abstract: str,
    keywords: list[str],
) -> str:
    """Text passed to the embedding model — excludes paper_id and other ids."""
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


async def _run(jsonl_path: str) -> int:
    with open(jsonl_path, encoding="utf-8") as f:
        paper: dict[str, Any] = json.loads(f.readline())

    paper_id = str(paper["id"])
    title = str(paper.get("title") or "")
    abstract = str(paper.get("abstract") or "")
    authors = list(paper.get("authors") or [])
    venue = str(paper.get("venue") or "")
    keywords = list(paper.get("keywords") or [])
    # JSONL has no year; ingest schema uses int — use 0 as unknown.
    year = int(paper["year"]) if paper.get("year") is not None else 0

    text = _text_for_embedding(title, abstract, keywords)

    client = AsyncQdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )
    dense = DenseRetriever(qdrant=client)
    try:
        cols = await client.get_collections()
        if settings.qdrant_collection_name not in {c.name for c in cols.collections}:
            await client.create_collection(
                collection_name=settings.qdrant_collection_name,
                vectors_config=VectorParams(size=_vec_size(), distance=Distance.COSINE),
            )

        vec = await dense.embed_text(text)
        payload: dict[str, Any] = {
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "year": year,
            "venue": venue,
            "keywords": keywords,
            "text": text,
        }
        qid = paper_qdrant_point_id(paper_id)
        await client.upsert(
            collection_name=settings.qdrant_collection_name,
            points=[
                PointStruct(id=qid, vector=vec, payload=payload),
            ],
        )

        res = await client.query_points(
            collection_name=settings.qdrant_collection_name,
            query=vec,
            limit=1,
            with_payload=True,
        )
        hit = res.points[0] if res.points else None
        print(f"OK: upserted paper_id={paper_id!r} qdrant_point_id={qid}")
        print(f"    collection={settings.qdrant_collection_name!r} dim={len(vec)}")
        if hit:
            print(f"    self-query score={float(hit.score):.4f}")
        return 0
    finally:
        await dense.close()


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_jsonl = os.path.join(root, "json", "papers.jsonl")
    parser = argparse.ArgumentParser(description="Embed first JSONL row into Qdrant")
    parser.add_argument(
        "--jsonl",
        default=default_jsonl,
        help="path to papers.jsonl (only first line is used)",
    )
    args = parser.parse_args()
    return asyncio.run(_run(args.jsonl))


if __name__ == "__main__":
    raise SystemExit(main())
