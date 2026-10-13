"""Reciprocal rank fusion plus cross-encoder reranking."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from config import settings
from models.schemas import DenseResult, FusedResult, KGResult
from pipeline.paper_abstract_store import abstract_for_paper

logger = logging.getLogger(__name__)


def _snippet(s: str, n: int = 400) -> str:
    """Trim long strings for the reranker input."""
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


def _rerank_text_from_metadata(meta: dict[str, Any]) -> str:
    """Build a dense paragraph for cross-encoder scoring."""
    title = str(meta.get("title", ""))
    abstract = _snippet(str(meta.get("abstract", "")), 400)
    authors = meta.get("authors") or []
    if isinstance(authors, str):
        authors = [authors]
    venue = meta.get("venue", "")
    topics = meta.get("topics") or meta.get("keywords") or []
    if isinstance(topics, str):
        topics = [topics]
    au = ", ".join(str(a) for a in authors if a)
    to = ", ".join(str(t) for t in topics if t)
    return f"{title}. {abstract} Authors: {au}. Venue: {venue}. Topics: {to}."


def _rerank_ce_text(meta: dict[str, Any], fallback: str) -> str:
    """Cross-encoder input: prefer abstract-rich dense text, always splice KG graph summary when present."""
    base = _rerank_text_from_metadata(meta).strip()
    kg_line = str(meta.get("kg_summary", "")).strip()
    if kg_line and kg_line not in base:
        merged = f"{base} | KG: {kg_line}" if base else kg_line
        return _snippet(merged, 500)
    if base:
        return _snippet(base, 500)
    return _snippet(kg_line or fallback, 500)


class HybridFusion:
    """Fuses KG and dense lists with RRF, then reranks with a cross-encoder."""

    def __init__(self) -> None:
        """Load the cross-encoder model from settings."""
        try:
            from sentence_transformers import CrossEncoder

            self._ce = CrossEncoder(settings.reranker_model)
        except Exception as exc:
            raise RuntimeError(f"CrossEncoder load failed: {exc}") from exc

    def _rrf(
        self,
        kg: list[KGResult],
        dense: list[DenseResult],
        k_const: int = 60,
        cap: int = 20,
    ) -> list[tuple[str, float, str, dict[str, Any], str]]:
        """Return tuples (id, rrf, source, metadata, text_for_rerank)."""
        scores: dict[str, float] = {}
        meta: dict[str, dict[str, Any]] = {}
        text: dict[str, str] = {}
        src: dict[str, set[str]] = {}
        for rank, r in enumerate(kg, start=1):
            rid = r.id
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k_const + rank)
            merged = {**r.metadata, **meta.get(rid, {}), "kg_summary": r.text}
            if not str(merged.get("abstract", "")).strip():
                ab = abstract_for_paper(merged.get("paper_id", rid))
                if ab.strip():
                    merged["abstract"] = ab
            meta[rid] = merged
            text[rid] = _rerank_ce_text(meta[rid], r.text)
            src.setdefault(rid, set()).add("kg")
        for rank, r in enumerate(dense, start=1):
            rid = r.id
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k_const + rank)
            prior = meta.get(rid, {})
            merged = {**prior, **r.metadata}
            if prior.get("kg_summary") and not str(r.metadata.get("kg_summary", "")).strip():
                merged["kg_summary"] = prior["kg_summary"]
            if str(prior.get("abstract", "")).strip() and not str(merged.get("abstract", "")).strip():
                merged["abstract"] = prior["abstract"]
            meta[rid] = merged
            text[rid] = _rerank_ce_text(merged, r.text)
            src.setdefault(rid, set()).add("dense")
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:cap]
        out: list[tuple[str, float, str, dict[str, Any], str]] = []
        for rid, sc in ranked:
            sset = src.get(rid, set())
            if sset == {"kg", "dense"}:
                lab = "both"
            elif "kg" in sset:
                lab = "kg"
            else:
                lab = "dense"
            out.append((rid, sc, lab, meta[rid], text[rid]))
        return out

    async def fuse_and_rerank(
        self,
        query: str,
        kg_results: list[KGResult],
        dense_results: list[DenseResult],
        top_k: int = 5,
    ) -> list[FusedResult]:
        """RRF-merge candidates, rerank with cross-encoder, return top_k."""
        try:
            cands = self._rrf(kg_results, dense_results)
            if not cands:
                return []
            pairs = [[query, row[-1]] for row in cands]

            def _pred() -> list[float]:
                return self._ce.predict(pairs, batch_size=16).tolist()

            rerank_scores = await asyncio.to_thread(_pred)
            order = sorted(
                range(len(cands)),
                key=lambda i: float(rerank_scores[i]),
                reverse=True,
            )[:top_k]
            fused: list[FusedResult] = []
            for i in order:
                rid, rrf_s, lab, m, _ = cands[i]
                fused.append(
                    FusedResult(
                        id=rid,
                        text=str(m.get("text") or _rerank_text_from_metadata(m)),
                        metadata=m,
                        rrf_score=float(rrf_s),
                        rerank_score=float(rerank_scores[i]),
                        source=lab,
                    )
                )
            return fused
        except Exception as exc:
            logger.exception("Hybrid fusion failed: %s", exc)
            raise RuntimeError(f"Hybrid fusion failed: {exc}") from exc
