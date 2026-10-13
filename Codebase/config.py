"""Application settings loaded from environment (.env)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    google_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    llm_provider: str = "groq"
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2"

    openai_api_key: str | None = None

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "your_password"

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "academic_papers"

    # Used to enrich KG rows with abstracts before cross-encoder (path relative to graphrag/).
    papers_jsonl_path: str = "json/papers.jsonl"

    embedding_provider: str = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    embedding_batch_size: int = 64
    # Empty = auto (e.g. MPS/CUDA if available). Set to "cpu" to force CPU for local ST.
    embedding_device: str = ""
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_final: int = 5

    # Append per-stage JSON lines to query_logs/pipeline_trace.jsonl on each POST /query.
    query_pipeline_trace_enabled: bool = True


settings = Settings()
