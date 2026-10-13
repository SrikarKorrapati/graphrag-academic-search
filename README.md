# GraphRAG Academic Search

Hybrid GraphRAG API that combines:
- **Knowledge Graph retrieval** from Neo4j
- **Dense vector retrieval** from Qdrant
- **Hybrid fusion + reranking**
- **LLM answer generation** via Groq / Gemini / Ollama

The service exposes a FastAPI endpoint (`POST /query`) for academic paper search-style questions.

## Tech Stack

- **Backend API:** FastAPI, Uvicorn
- **Knowledge Graph:** Neo4j
- **Vector DB:** Qdrant
- **Embeddings:** sentence-transformers (default), or OpenAI/Gemini
- **Reranker:** cross-encoder (`ms-marco-MiniLM-L-6-v2`)
- **LLMs:** Groq (default), Gemini, or Ollama
- **Validation / config:** Pydantic + pydantic-settings

## Directory Structure

```text
graphrag/
├── main.py                     # FastAPI entrypoint (/health, /query, /ingest)
├── config.py                   # Environment-based settings
├── requirements.txt
├── json/
│   └── papers.jsonl            # Source papers dataset
├── models/
│   └── schemas.py              # API and pipeline schemas
├── pipeline/
│   ├── query_parser.py
│   ├── kg_traversal.py         # Neo4j retrieval
│   ├── dense_retrieval.py      # Qdrant retrieval + embedding
│   ├── hybrid_fusion.py        # Merge + rerank KG + dense hits
│   └── llm_generator.py        # Final answer generation
├── scripts/
│   ├── seed_neo4j.py           # Seed sample graph into Neo4j
│   ├── test_neo4j_connection.py
│   ├── ingest_jsonl_batch.py   # Batch ingest JSONL into Qdrant
│   └── eval_query_cases.py
├── tests/
│   └── fixtures/
│       └── query_eval_cases.json
└── query_logs/
    └── ...                     # query logs + pipeline traces + sample runs
```

## Prerequisites

- Python **3.10+**
- Neo4j running and reachable from your machine
- Qdrant running and reachable from your machine (or use embedded mode for ingestion script)

Optional (depends on provider choices):
- Groq API key, Gemini API key, OpenAI API key
- Ollama running locally if using `LLM_PROVIDER=ollama`

## Setup

### 1) Install dependencies

```bash
cd graphrag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure environment (`.env`)

Create/update `graphrag/.env`:

```env
# LLM provider
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.1-8b-instant

# Optional if using ollama
# OLLAMA_BASE_URL=http://127.0.0.1:11434
# OLLAMA_MODEL=llama3.2

# Optional if using gemini/openai paths
GOOGLE_API_KEY=
OPENAI_API_KEY=

# Neo4j
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=academic_papers

# Embedding + rerank
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSIONS=384
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
TOP_K_FINAL=5

# Optional
GEMINI_MODEL=gemini-2.0-flash
```

### 3) Start Neo4j and Qdrant

If you use Docker:

```bash
# Neo4j
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_neo4j_password \
  neo4j:5

# Qdrant
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  qdrant/qdrant:latest
```

Then verify Neo4j connection:

```bash
python3 scripts/test_neo4j_connection.py
```

### 4) Seed KG (Neo4j)

```bash
python3 scripts/seed_neo4j.py
```

### 5) Ingest vectors into Qdrant

```bash
python3 scripts/ingest_jsonl_batch.py --jsonl json/papers.jsonl --limit 5000 --upsert-batch 256
```

Notes:
- Use `--limit 0` to ingest until EOF.
- Use `--recreate-collection` for a clean rebuild.
- Use `--qdrant-path ./qdrant_unified` to ingest into embedded Qdrant storage.

## How to connect to KG and Qdrant (from this app)

The app reads these settings from `.env` via `config.py`:
- **KG / Neo4j:** `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- **Qdrant:** `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_COLLECTION_NAME`

At startup (`main.py` lifespan):
- Opens Neo4j async driver (`AsyncGraphDatabase.driver(...)`)
- Opens Qdrant async client (`AsyncQdrantClient(host=..., port=...)`)
- Builds:
  - `KGTraversal` wrapper over Neo4j
  - `DenseRetriever` wrapper over Qdrant

If one backend is unavailable, pipeline still runs with the other backend. If both are unavailable, `/query` returns `503`.

## Run the API

```bash
python3 -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## 2 Sample Queries

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"papers on graph neural networks for citation graphs"}'
```

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"transformer attention mechanisms in natural language processing"}'
```

## Optional evaluation

Run fixture-based query checks:

```bash
python3 scripts/eval_query_cases.py --base-url http://127.0.0.1:8000
```

Run LLM-as-judge scoring on sample query outputs:

```bash
python3 scripts/eval_sample_queries_llm_judge.py
```
# GraphRAG Academic Search

FastAPI-based GraphRAG service that combines:
- **Knowledge Graph retrieval** from **Neo4j**
- **Dense vector retrieval** from **Qdrant**
- **Hybrid fusion + reranking** for final paper ranking
- **LLM answer generation** from fused context

## Tech Stack

- **Backend/API**: Python, FastAPI, Uvicorn
- **Knowledge Graph**: Neo4j
- **Vector DB**: Qdrant
- **Embeddings**:
  - Local `sentence-transformers` (default)
  - Optional OpenAI / Gemini embeddings
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM providers**:
  - Groq (default in `.env`)
  - Ollama
  - Gemini
- **Config**: `.env` via `pydantic-settings`

## Directory Structure

```text
graphrag/
├── main.py                     # FastAPI app (/health, /query, /ingest)
├── config.py                   # Environment-backed settings
├── requirements.txt
├── json/
│   └── papers.jsonl            # Paper corpus for vector ingestion
├── models/
│   └── schemas.py              # Pydantic request/response models
├── pipeline/
│   ├── query_parser.py
│   ├── kg_traversal.py         # Neo4j retrieval
│   ├── dense_retrieval.py      # Qdrant retrieval + embedding
│   ├── hybrid_fusion.py
│   ├── context_aggregator.py
│   └── llm_generator.py
├── scripts/
│   ├── seed_neo4j.py
│   ├── test_neo4j_connection.py
│   ├── ingest_jsonl_batch.py
│   ├── test_dense_retrieval.py
│   └── run_sample_query_batch.py
├── tests/
│   └── fixtures/
│       └── sample_test_queries.json
└── query_logs/                 # Pipeline traces + sample outputs
```

## Setup

### 1) Create Python environment and install dependencies

```bash
cd graphrag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure environment

Create/update `.env` in `graphrag/`:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.1-8b-instant

NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=academic_papers

EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSIONS=384
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
TOP_K_FINAL=5
```

### 3) Start Neo4j and Qdrant

Use Docker (recommended):

```bash
# Neo4j
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:5

# Qdrant
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  qdrant/qdrant
```

### 4) Connect and seed Knowledge Graph (Neo4j)

```bash
python3 scripts/test_neo4j_connection.py
python3 scripts/seed_neo4j.py
```

### 5) Ingest vectors into Qdrant

```bash
python3 scripts/ingest_jsonl_batch.py \
  --jsonl json/papers.jsonl \
  --offset 0 \
  --limit 5000 \
  --upsert-batch 256
```

Sanity check dense retrieval:

```bash
python3 scripts/test_dense_retrieval.py --query "papers about transformers" --top 5
```

### 6) Run the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

## How It Connects to KG + Qdrant

- At startup (`main.py` lifespan), the app initializes:
  - Neo4j driver via `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
  - Qdrant async client via `QDRANT_HOST`, `QDRANT_PORT`
- Query flow (`POST /query`):
  1. Parse query (`query_parser`)
  2. Retrieve from KG (`kg_traversal`) and dense vectors (`dense_retrieval`) in parallel
  3. Fuse + rerank (`hybrid_fusion`)
  4. Aggregate context and generate final answer (`llm_generator`)

If one backend is unavailable, the pipeline can continue with the other backend (unless both are down).

## Sample Queries (2)

Run these against `POST /query`:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"papers on federated learning and privacy-preserving deep learning for clinical data"}'
```

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"who cites MobileNetV2 Inverted Residuals and Linear Bottlenecks"}'
```

## Optional: Run all sample fixture queries

```bash
python3 scripts/run_sample_query_batch.py
```

Outputs are written under `query_logs/sample_query_runs/`.

