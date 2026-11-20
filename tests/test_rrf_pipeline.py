"""Verify API and sample runners use RRF with external services mocked."""

import asyncio
import importlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from config import settings
from models.schemas import DenseResult, KGResult, ParsedQuery


@pytest.fixture
def pipeline_mocks(monkeypatch):
    # Any leftover attempt to load a cross-encoder would fail in these checks.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.setattr(settings, "top_k_final", 2)
    monkeypatch.setattr(settings, "query_pipeline_trace_enabled", False)
    meta = lambda pid: {"title": f"Paper {pid}", "abstract": f"Abstract {pid}"}
    graph = [KGResult(id=pid, text=f"Graph {pid}", metadata=meta(pid)) for pid in ["A", "B"]]
    dense = [DenseResult(id=pid, text=f"Dense {pid}", metadata=meta(pid), score=score)
             for pid, score in [("B", 0.9), ("C", 0.8)]]
    parser = SimpleNamespace(parse=Mock(return_value=ParsedQuery(original="test query")))
    generator = SimpleNamespace(generate=Mock(return_value="Answer from selected papers"))
    kg = SimpleNamespace(traverse=AsyncMock(return_value=graph), close=AsyncMock())
    retriever = SimpleNamespace(retrieve=AsyncMock(return_value=dense), close=AsyncMock())
    driver, qdrant = SimpleNamespace(close=AsyncMock()), SimpleNamespace(close=AsyncMock())

    def configure(module):
        monkeypatch.setattr(module.AsyncGraphDatabase, "driver", Mock(return_value=driver))
        monkeypatch.setattr(module, "AsyncQdrantClient", Mock(return_value=qdrant))
        monkeypatch.setattr(module, "KGTraversal", Mock(return_value=kg))
        monkeypatch.setattr(module, "DenseRetriever", Mock(return_value=retriever))
        monkeypatch.setattr(module, "QueryParser", Mock(return_value=parser))
        monkeypatch.setattr(module, "LLMGenerator", Mock(return_value=generator))
        # The runner trace helpers ask KGTraversal for a query context.
        module.KGTraversal._query_context.return_value = {"authors": [], "topics": []}
        return parser, generator

    return configure, generator, driver, qdrant


def assert_selection(papers):
    assert [p["id"] for p in papers] == ["B", "A"]
    assert papers[0]["rrf_score"] == pytest.approx(1 / 62 + 1 / 61)
    assert papers[1]["rrf_score"] == pytest.approx(1 / 61)
    assert all("rerank_score" not in p for p in papers)


def test_api_startup_and_query_need_no_ranking_model(monkeypatch, pipeline_mocks):
    module = importlib.import_module("main")
    configure, generator, driver, qdrant = pipeline_mocks
    parser, _ = configure(module)
    state = module.AppState()
    state.parser, state.generator = parser, generator
    monkeypatch.setattr(module, "state", state)
    monkeypatch.setattr(module, "append_query_log", Mock())

    with TestClient(module.app) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/query", json={"query": "test query"})

    assert response.status_code == 200, response.text
    assert_selection(response.json()["papers"])
    assert response.json()["answer"] == "Answer from selected papers"
    parser.parse.assert_called_once_with("test query")
    generator.generate.assert_called_once()
    context = generator.generate.call_args.args[1]
    assert context.index("Paper B") < context.index("Paper A")
    driver.close.assert_awaited_once()
    qdrant.close.assert_awaited_once()


@pytest.mark.parametrize("name", [
    "run_sample_queries_dump", "run_sample_pipeline_outputs", "run_sample_query_batch",
])
def test_sample_runners_select_with_rrf(name, tmp_path, pipeline_mocks):
    module = importlib.import_module(name)
    configure, generator, _, _ = pipeline_mocks
    configure(module)
    fixture = tmp_path / "queries.json"
    fixture.write_text(json.dumps({"queries": [{"id": "q01", "query": "test query"}]}))
    out_dir = tmp_path / "runs"

    if name == "run_sample_query_batch":
        args = SimpleNamespace(fixture=str(fixture), out_dir=str(out_dir), only="", sleep_seconds=0)
        assert asyncio.run(module._async_main(args)) == 0
    else:
        asyncio.run(module.run_all(fixture, out_dir))

    record = json.loads((out_dir / "q01.json").read_text())
    assert not record.get("pipeline_error")
    assert_selection(record["hybrid_fusion"]["papers"])
    generator.generate.assert_called_once()
