"""Pydantic v2 schemas for GraphRAG academic search."""

from __future__ import annotations

from pydantic import BaseModel, Field


class YearFilter(BaseModel):
    """Inclusive year bounds extracted from the user query."""

    min: int | None = None
    max: int | None = None


class Entity(BaseModel):
    """A named span classified against the knowledge-graph schema."""

    text: str
    type: str


class Relation(BaseModel):
    """A subject–predicate–object triple aligned to graph relations."""

    subject: str
    predicate: str
    object: str


class ParsedQuery(BaseModel):
    """Structured output from the query parser LLM."""

    original: str = ""
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    intent: str = "exploratory"
    keywords: list[str] = Field(default_factory=list)
    sub_queries: list[str] = Field(default_factory=list)
    year_filter: YearFilter = Field(default_factory=YearFilter)
    cypher_hint: str = ""


class KGResult(BaseModel):
    """One row from knowledge-graph traversal, paper- or author-centric."""

    id: str
    text: str
    metadata: dict
    score: float | None = None


class DenseResult(BaseModel):
    """Semantic hit from the vector index."""

    id: str
    text: str
    metadata: dict
    score: float


class FusedResult(BaseModel):
    """Hybrid retrieval output after RRF and cross-encoder reranking."""

    id: str
    text: str
    metadata: dict
    rrf_score: float
    rerank_score: float
    source: str


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    papers: list[FusedResult]
    parsed_query: ParsedQuery
    latency_ms: float


class PaperIngestItem(BaseModel):
    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    year: int
    venue: str
    keywords: list[str]


class IngestRequest(BaseModel):
    papers: list[PaperIngestItem]


class IngestResponse(BaseModel):
    ingested: int
