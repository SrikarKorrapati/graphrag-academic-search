#!/usr/bin/env python3
"""Copy all points from one or more Qdrant sources into a single target (HTTP or embedded path).

Use this to unify e.g. Docker ``localhost:6333`` and embedded ``--path`` stores after a split ingest.

Examples:
  # Merge Docker + embedded tail into a new folder (app can use Qdrant path mode if you add it):
  python3 scripts/qdrant_merge_into.py \\
    --target path:./qdrant_unified \\
    --source path:/tmp/qdrant_papers_resume \\
    --source http:127.0.0.1:6333

  # Only embedded -> new path:
  python3 scripts/qdrant_merge_into.py --target path:./qdrant_unified --source path:/tmp/qdrant_papers_resume
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import settings


def _parse_location(spec: str) -> AsyncQdrantClient:
    """Build client from ``path:/dir`` or ``http:host:port`` (default port 6333)."""
    spec = spec.strip()
    if spec.startswith("path:"):
        path = spec[5:].strip()
        os.makedirs(path, exist_ok=True)
        return AsyncQdrantClient(path=path)
    if spec.startswith("http:"):
        rest = spec[5:].strip()
        if ":" in rest:
            host, port_s = rest.rsplit(":", 1)
            port = int(port_s)
        else:
            host, port = rest, 6333
        return AsyncQdrantClient(host=host, port=port)
    raise ValueError(f"Bad --source/--target (use path:... or http:host[:port]): {spec!r}")


async def _ensure_target_collection(
    target: AsyncQdrantClient,
    collection: str,
    vector_size: int,
) -> None:
    cols = await target.get_collections()
    names = {c.name for c in cols.collections}
    if collection in names:
        return
    await target.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


async def _copy_collection(
    src: AsyncQdrantClient,
    dst: AsyncQdrantClient,
    collection: str,
    upsert_batch: int,
) -> int:
    """Scroll source and upsert into target. Returns number of points copied."""
    n = 0
    offset: Any = None
    vec_size: int | None = None
    ensured = False
    while True:
        records, next_offset = await src.scroll(
            collection_name=collection,
            limit=upsert_batch,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )
        if not records:
            break
        if vec_size is None and records[0].vector is not None:
            v = records[0].vector
            if isinstance(v, dict):
                vec_size = len(next(iter(v.values())))
            else:
                vec_size = len(v)
        batch: list[PointStruct] = []
        for r in records:
            pl = r.payload if isinstance(r.payload, dict) else {}
            batch.append(
                PointStruct(
                    id=r.id,
                    vector=r.vector,
                    payload=pl,
                )
            )
        if vec_size is not None and not ensured:
            await _ensure_target_collection(dst, collection, vec_size)
            ensured = True
        await dst.upsert(collection_name=collection, points=batch)
        n += len(batch)
        if next_offset is None:
            break
        offset = next_offset
    return n


async def _run(
    target_spec: str,
    source_specs: list[str],
    collection: str,
    upsert_batch: int,
) -> int:
    target = _parse_location(target_spec)
    total = 0
    try:
        for i, spec in enumerate(source_specs):
            src = _parse_location(spec)
            try:
                copied = await _copy_collection(src, target, collection, upsert_batch)
                print(f"OK source[{i}] {spec!r} -> copied {copied} points")
                total += copied
            finally:
                await src.close()
        print(f"done: total upserted into {target_spec!r} collection={collection!r} -> {total}")
        return 0
    finally:
        await target.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge Qdrant collections (path: or http:) into one target",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="destination, e.g. path:./qdrant_unified or http:127.0.0.1:6333",
    )
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        required=True,
        help="repeat for each source, e.g. --source path:/tmp/qdrant_papers_resume --source http:127.0.0.1:6333",
    )
    parser.add_argument(
        "--collection",
        default=settings.qdrant_collection_name,
        help="collection name (default from settings)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=256,
        help="scroll/upsert batch size",
    )
    args = parser.parse_args()
    return asyncio.run(
        _run(args.target, args.sources, args.collection, args.batch),
    )


if __name__ == "__main__":
    raise SystemExit(main())
