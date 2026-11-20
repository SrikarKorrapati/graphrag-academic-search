"""Select final results directly with reciprocal rank fusion."""

from __future__ import annotations

from typing import Any

from models.schemas import DenseResult, FusedResult, KGResult
from pipeline.paper_abstract_store import abstract_for_paper


def _text_from_metadata(meta: dict[str, Any]) -> str:
    """Build fallback result text when the payload has no text field."""
    title = str(meta.get("title", ""))
    abstract = str(meta.get("abstract", "")).strip()
    if len(abstract) > 400:
        abstract = abstract[:400] + "…"
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


class HybridFusion:
    """Merge ranked KG/dense lists by ID and select top_k using RRF alone."""

    def fuse(
        self,
        kg_results: list[KGResult],
        dense_results: list[DenseResult],
        top_k: int = 5,
    ) -> list[FusedResult]:
        """Sum 1 / (60 + rank) per list, then return the highest-scoring results.

        Inputs must be ordered best-first and contain unique IDs within each
        list. Ties preserve first-seen order (KG, then dense). No intermediate
        candidate cap or model-based scoring is applied.
        """
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if top_k == 0:
            return []

        scores: dict[str, float] = {}
        metadata: dict[str, dict[str, Any]] = {}
        sources: dict[str, set[str]] = {}
        for rank, result in enumerate(kg_results, start=1):
            rid = result.id
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (60 + rank)
            metadata[rid] = {**result.metadata, "kg_summary": result.text}
            sources.setdefault(rid, set()).add("kg")

        for rank, result in enumerate(dense_results, start=1):
            rid = result.id
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (60 + rank)
            prior = metadata.get(rid, {})
            merged = {**prior, **result.metadata}
            if prior.get("kg_summary") and not str(result.metadata.get("kg_summary", "")).strip():
                merged["kg_summary"] = prior["kg_summary"]
            if str(prior.get("abstract", "")).strip() and not str(merged.get("abstract", "")).strip():
                merged["abstract"] = prior["abstract"]
            metadata[rid] = merged
            sources.setdefault(rid, set()).add("dense")

        ranked = sorted(scores, key=lambda rid: scores[rid], reverse=True)[:top_k]
        fused: list[FusedResult] = []
        for rid in ranked:
            meta = metadata[rid]
            source_set = sources[rid]
            # Abstracts enrich selected results for answer generation; they do
            # not affect ranking. Prefer abstracts already present in either store.
            if "kg" in source_set and not str(meta.get("abstract", "")).strip():
                abstract = abstract_for_paper(meta.get("paper_id", rid))
                if abstract.strip():
                    meta["abstract"] = abstract
            source = "both" if source_set == {"kg", "dense"} else next(iter(source_set))
            fused.append(
                FusedResult(
                    id=rid,
                    text=str(meta.get("text") or _text_from_metadata(meta)),
                    metadata=meta,
                    rrf_score=scores[rid],
                    source=source,
                )
            )
        return fused
