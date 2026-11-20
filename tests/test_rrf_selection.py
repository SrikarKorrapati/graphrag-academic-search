"""Ranking regressions that run without model downloads or remote services."""

import asyncio
import sys
from unittest.mock import AsyncMock, Mock

import pytest

from models.schemas import DenseResult, KGResult, ParsedQuery
from pipeline.dense_retrieval import DenseRetriever
from pipeline.hybrid_fusion import HybridFusion


def kg_hit(pid, **metadata):
    return KGResult(
        id=pid, text=f"Graph evidence for {pid}",
        metadata={"title": pid, "abstract": "Known abstract", **metadata},
    )


def dense_hit(pid, score=0.5, **metadata):
    return DenseResult(
        id=pid, text=f"Dense evidence for {pid}", score=score,
        metadata={"title": pid, "abstract": "Known abstract", **metadata},
    )


def test_rrf_selects_by_rank_and_combines_sources_without_model(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    graph = [kg_hit("A"), kg_hit("B"), kg_hit("C")]
    dense = [dense_hit("B"), dense_hit("D"), dense_hit("A")]

    results = HybridFusion().fuse(graph, dense)

    assert [r.id for r in results] == ["B", "A", "D", "C"]
    assert [r.rrf_score for r in results] == pytest.approx([
        1 / 62 + 1 / 61, 1 / 61 + 1 / 63, 1 / 62, 1 / 63,
    ])
    assert [r.source for r in results] == ["both", "both", "dense", "kg"]
    assert all("rerank_score" not in r.model_dump() for r in results)
    assert [r.id for r in HybridFusion().fuse(graph, dense, top_k=2)] == ["B", "A"]


@pytest.mark.parametrize("dense_abstract", ["", "Dense abstract"])
def test_metadata_survives_merge_without_mutating_inputs(dense_abstract):
    graph = kg_hit("A", abstract="Graph abstract", topics=["graphs"], cited_paper="B")
    dense = dense_hit("A", abstract=dense_abstract, text="Original payload text", keywords=["GNN"])
    before = (graph.model_dump(), dense.model_dump())

    result = HybridFusion().fuse([graph], [dense])[0]

    assert result.metadata["abstract"] == (dense_abstract or "Graph abstract")
    assert result.metadata["kg_summary"] == graph.text
    assert result.metadata["cited_paper"] == "B"
    assert result.metadata["topics"] == ["graphs"]
    assert result.metadata["keywords"] == ["GNN"]
    assert result.text == "Original payload text"
    assert before == (graph.model_dump(), dense.model_dump())


def test_abstract_lookup_only_enriches_selected_graph_results(monkeypatch):
    lookup = Mock(return_value="Corpus abstract")
    monkeypatch.setattr("pipeline.hybrid_fusion.abstract_for_paper", lookup)

    results = HybridFusion().fuse([kg_hit(str(i), abstract="") for i in range(25)], [], top_k=2)

    assert [r.id for r in results] == ["0", "1"]
    assert lookup.call_count == 2
    assert all(r.metadata["abstract"] == "Corpus abstract" for r in results)


@pytest.mark.parametrize("source", ["kg", "dense"])
def test_single_backend_honors_top_k_above_old_candidate_cap(source):
    graph = [kg_hit(str(i)) for i in range(30)] if source == "kg" else []
    dense = [dense_hit(str(i)) for i in range(30)] if source == "dense" else []

    results = HybridFusion().fuse(graph, dense, top_k=25)

    assert [r.id for r in results] == [str(i) for i in range(25)]
    assert results[-1].rrf_score == pytest.approx(1 / 85)
    assert all(r.source == source for r in results)


def test_empty_inputs_and_zero_limit():
    assert HybridFusion().fuse([], []) == []
    assert HybridFusion().fuse([kg_hit("A")], [], top_k=0) == []
    with pytest.raises(ValueError, match="non-negative"):
        HybridFusion().fuse([], [], top_k=-1)


def test_later_subquery_best_hits_receive_top_rrf_ranks():
    # Exercise the real merge logic without initializing an embedding backend.
    retriever = DenseRetriever.__new__(DenseRetriever)
    retriever._search_one = AsyncMock(side_effect=[
        [dense_hit(str(i), score=0.5 - i / 100) for i in range(20)],
        [dense_hit("later-best", score=0.99), dense_hit("19", score=0.95)],
    ])
    parsed = ParsedQuery(original="combined query", sub_queries=["first", "second"])

    results = asyncio.run(retriever.retrieve(parsed))
    fused = HybridFusion().fuse([], results)

    assert len(results) == 21
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
    assert [r.id for r in fused[:2]] == ["later-best", "19"]
    assert fused[0].rrf_score == pytest.approx(1 / 61)
    assert fused[1].rrf_score == pytest.approx(1 / 62)
