"""Vector search over paper chunks in Qdrant."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, Range

from config import settings
from models.schemas import DenseResult, ParsedQuery

logger = logging.getLogger(__name__)


def _gemini_model_id() -> str:
    """Normalize embedding model name to Gemini API `models/...` id."""
    m = settings.embedding_model.strip()
    return m if m.startswith("models/") else f"models/{m}"


def _gemini_embed_sync(text: str, task_type: str) -> list[float]:
    """Blocking Gemini embedding call (runs in a thread pool)."""
    try:
        import google.generativeai as genai
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("google-generativeai is required for Gemini embeddings") from exc
    genai.configure(api_key=settings.google_api_key)
    try:
        res = genai.embed_content(
            model=_gemini_model_id(),
            content=text,
            task_type=task_type,
        )
    except Exception:
        res = genai.embed_content(model=_gemini_model_id(), content=text)
    emb = res.get("embedding") if isinstance(res, dict) else getattr(res, "embedding", None)
    if emb is None:
        raise RuntimeError("Gemini embed response missing embedding vector")
    return [float(x) for x in emb]


class DenseRetriever:
    """Embeds query text and retrieves top papers from Qdrant."""

    def __init__(
        self,
        qdrant: AsyncQdrantClient | None = None,
        *,
        st_device: str | None = None,
    ) -> None:
        """Prepare embedding backend and optional shared Qdrant client.

        st_device: optional override for sentence-transformers device (e.g. ``\"cpu\"``).
        """
        self._qdrant = qdrant
        self._owns_qdrant = qdrant is None
        self._openai: Any = None
        self._st: Any = None
        prov = settings.embedding_provider.lower()
        if prov == "openai":
            try:
                from openai import AsyncOpenAI

                self._openai = AsyncOpenAI(api_key=settings.openai_api_key or "")
            except Exception as exc:
                raise RuntimeError(f"OpenAI client init failed: {exc}") from exc
        elif prov == "gemini":
            if not (settings.google_api_key or "").strip():
                raise RuntimeError("GOOGLE_API_KEY is required for Gemini embeddings")
        elif prov in ("local", "sentence-transformers", "st"):
            try:
                from sentence_transformers import SentenceTransformer

                dev = (st_device or settings.embedding_device or "").strip()
                st_kw: dict[str, Any] = {}
                if dev:
                    st_kw["device"] = dev
                self._st = SentenceTransformer(settings.embedding_model, **st_kw)
            except Exception as exc:
                raise RuntimeError(f"SentenceTransformer init failed: {exc}") from exc
        else:
            raise RuntimeError(
                f"Unknown EMBEDDING_PROVIDER={prov!r} "
                "(use local, openai, or gemini)"
            )

    async def _client(self) -> AsyncQdrantClient:
        if self._qdrant is None:
            try:
                self._qdrant = AsyncQdrantClient(
                    host=settings.qdrant_host,
                    port=settings.qdrant_port,
                )
            except Exception as exc:
                raise RuntimeError(f"Qdrant client init failed: {exc}") from exc
        return self._qdrant

    async def close(self) -> None:
        """Close Qdrant client if owned by this retriever."""
        if self._owns_qdrant and self._qdrant is not None:
            await self._qdrant.close()
            self._qdrant = None

    def _search_text(self, pq: ParsedQuery, override: str | None = None) -> str:
        """Combine original query with concept keywords for embedding."""
        parts = [override or pq.original, *pq.keywords]
        return " ".join(p for p in parts if p).strip() or pq.original

    async def embed_text(self, text: str) -> list[float]:
        """Embed text for indexing (Gemini uses document task type)."""
        if settings.embedding_provider.lower() == "gemini":
            return await self._embed(text, gemini_task="RETRIEVAL_DOCUMENT")
        return await self._embed(text)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed strings for indexing (local sentence-transformers only; uses CPU/GPU of the model)."""
        if not texts:
            return []
        prov = settings.embedding_provider.lower()
        if prov not in ("local", "sentence-transformers", "st"):
            raise RuntimeError(
                "embed_texts only supports EMBEDDING_PROVIDER=local "
                f"(got {settings.embedding_provider!r})"
            )
        assert self._st is not None
        batch_size = max(1, settings.embedding_batch_size)

        def _enc_batch() -> list[list[float]]:
            arr = self._st.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return arr.tolist()

        return await asyncio.to_thread(_enc_batch)

    async def _embed(self, text: str, gemini_task: str | None = None) -> list[float]:
        """Produce one embedding vector for the given text."""
        try:
            prov = settings.embedding_provider.lower()
            if prov == "gemini":
                task = gemini_task or "RETRIEVAL_QUERY"
                return await asyncio.to_thread(_gemini_embed_sync, text, task)
            if self._openai:
                r = await self._openai.embeddings.create(
                    model=settings.embedding_model,
                    input=text,
                )
                return list(r.data[0].embedding)
            assert self._st is not None

            def _enc() -> list[float]:
                v = self._st.encode(text, normalize_embeddings=True)
                return v.tolist()

            return await asyncio.to_thread(_enc)
        except Exception as exc:
            raise RuntimeError(f"Embedding failed: {exc}") from exc

    def _year_filter(self, pq: ParsedQuery) -> Filter | None:
        """Build optional Qdrant filter for publication year."""
        lo, hi = pq.year_filter.min, pq.year_filter.max
        if lo is None and hi is None:
            return None
        return Filter(
            must=[
                FieldCondition(
                    key="year",
                    range=Range(gte=lo, lte=hi),
                )
            ]
        )

    async def _search_one(self, text: str, pq: ParsedQuery) -> list[DenseResult]:
        """Single vector search against the papers collection."""
        client = await self._client()
        vec = await self._embed(text)
        flt = self._year_filter(pq)
        try:
            if hasattr(client, "search"):
                hits = await client.search(
                    collection_name=settings.qdrant_collection_name,
                    query_vector=vec,
                    limit=20,
                    query_filter=flt,
                    with_payload=True,
                )
            else:
                # qdrant-client >= 1.17 uses query_points on async client.
                res = await client.query_points(
                    collection_name=settings.qdrant_collection_name,
                    query=vec,
                    limit=20,
                    query_filter=flt,
                    with_payload=True,
                )
                hits = list(getattr(res, "points", []) or [])
        except Exception as exc:
            raise RuntimeError(f"Qdrant search failed: {exc}") from exc
        out: list[DenseResult] = []
        for h in hits:
            pl = h.payload or {}
            pid = str(pl.get("paper_id", h.id))
            out.append(
                DenseResult(
                    id=pid,
                    text=str(pl.get("text", "")),
                    metadata=dict(pl),
                    score=float(h.score or 0.0),
                )
            )
        return out

    async def retrieve(self, parsed_query: ParsedQuery) -> list[DenseResult]:
        """Embed decomposed sub-queries (or one joint string) and merge hits."""
        try:
            texts = (
                [self._search_text(parsed_query, s) for s in parsed_query.sub_queries]
                if parsed_query.sub_queries
                else [self._search_text(parsed_query)]
            )
            seen: dict[str, DenseResult] = {}
            for t in texts:
                for r in await self._search_one(t, parsed_query):
                    if r.id not in seen or r.score > seen[r.id].score:
                        seen[r.id] = r
            return list(seen.values())
        except Exception as exc:
            logger.exception("Dense retrieval failed: %s", exc)
            raise RuntimeError(f"Dense retrieval failed: {exc}") from exc
