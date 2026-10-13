#!/usr/bin/env python3
"""Smoke test for dense retrieval (embeddings + Qdrant search)."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import settings
from models.schemas import ParsedQuery
from pipeline.dense_retrieval import DenseRetriever


def _build_query(text: str) -> ParsedQuery:
    """Build a minimal ParsedQuery for dense retrieval."""
    return ParsedQuery(original=text, keywords=[text], sub_queries=[], intent="exploratory")


async def _run(query_text: str, limit: int) -> int:
    """Run end-to-end dense retrieval and print top hits."""
    print("Dense retrieval smoke test")
    print(f"- provider: {settings.embedding_provider}")
    print(f"- embedding_model: {settings.embedding_model}")
    print(f"- qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    print(f"- collection: {settings.qdrant_collection_name}")
    print(f"- query: {query_text!r}")

    retriever = DenseRetriever()
    try:
        client = await retriever._client()  # noqa: SLF001 - test utility
        collections = await client.get_collections()
        names = {c.name for c in collections.collections}
        if settings.qdrant_collection_name not in names:
            print(
                "ERROR: collection not found. Available collections:",
                ", ".join(sorted(names)) if names else "<none>",
            )
            return 1

        results = await retriever.retrieve(_build_query(query_text))
        if not results:
            print("OK: retrieval call succeeded, but returned 0 hits.")
            return 0

        print(f"OK: retrieved {len(results)} hits (showing top {min(limit, len(results))})")
        for i, hit in enumerate(results[:limit], start=1):
            title = (hit.metadata or {}).get("title", "<no title>")
            year = (hit.metadata or {}).get("year")
            print(f"{i:02d}. score={hit.score:.4f} id={hit.id} year={year} title={title}")
        return 0
    except Exception as exc:
        print(f"ERROR: dense retrieval test failed: {exc}")
        return 1
    finally:
        await retriever.close()


def main() -> int:
    """Parse args and run async smoke test."""
    parser = argparse.ArgumentParser(description="Test DenseRetriever against Qdrant")
    parser.add_argument(
        "--query",
        default="papers about transformers",
        help="query text to embed and search",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="number of hits to print",
    )
    args = parser.parse_args()
    return asyncio.run(_run(args.query, args.top))


if __name__ == "__main__":
    sys.exit(main())
