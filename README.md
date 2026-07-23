# CRAG — Agentic RAG over Product Reviews

An end-to-end agentic retrieval-augmented generation system that answers natural language
questions about e-commerce products by routing each query to a SQL agent, a hybrid vector
search agent, or both — then synthesizing a grounded answer with citations.

## Table of contents

- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Running the pipeline](#running-the-pipeline)
- [Running the API](#running-the-api)
- [Running the frontend](#running-the-frontend)
- [Evaluation](#evaluation)
- [Known limitations](#known-limitations)
- [Possible next steps](#possible-next-steps)

## Architecture

![Agentic RAG architecture](architecture-diagram.svg)

```
User query
    │
    ▼
┌─────────┐
│ router  │  Groq LLM classifies query into: sql | vector | hybrid
└────┬────┘
     │
     ├── sql ──────► sql_agent ──────────────────────┐
     │               (Groq generates SQL,             │
     │                validated, executed              │
     │                against Postgres)                │
     │                                                  │
     ├── vector ───► vector_agent ────────────────────►│
     │               (intent extraction → query         │
     │                expansion → hybrid dense+sparse    │
     │                search → RRF fusion → rerank)      │
     │                                                  │
     └── hybrid ───► sql_agent + vector_agent ──────────┤
                                                          ▼
                                                    ┌───────────┐
                                                    │ generator │  Groq synthesizes
                                                    └─────┬─────┘  grounded answer
                                                          │
                                                          ▼
                                                    ┌───────────┐
                                                    │  logger   │  writes to query_logs
                                                    └───────────┘
```

The graph itself (routing, conditional branching, node sequencing) is built with **LangGraph**.
Every node's internal logic (SQL generation, embedding, reranking, etc.) is plain Python —
LangGraph only owns orchestration, not the actual retrieval/generation logic.

### Hybrid retrieval pipeline

![Hybrid retrieval pipeline](retrieval-pipeline-diagram.svg)

The vector route runs each query through multi-query expansion (3 Groq-generated paraphrases
+ the original), searches all 4 variants against both dense and sparse Qdrant vector spaces,
fuses the ranked lists with Reciprocal Rank Fusion, then narrows the merged candidate pool
down with a cross-encoder reranker before the top results reach the generator.

## Tech stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (`StateGraph`) |
| LLM | Groq (`llama-3.1-8b-instant` for routing/agents, `llama-3.3-70b-versatile` for RAGAS judging) |
| Structured data | PostgreSQL + SQLAlchemy |
| Vector search | Qdrant (local, file-based) — hybrid dense + BM25 sparse vectors, fused with Reciprocal Rank Fusion |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Evaluation | RAGAS (faithfulness, answer relevancy, context precision, context recall) |
| API | FastAPI |
| Frontend | Static HTML/CSS/JS (no framework, no build step) |

## Project structure

```
CRAG_project/
├── agents/
│   ├── router.py           # Classifies query into sql / vector / hybrid
│   ├── sql_agent.py         # Text-to-SQL generation + safe execution
│   ├── vector_agent.py       # Intent extraction + calls into pipeline/
│   └── orchestrator.py      # LangGraph StateGraph wiring all agents together
│
├── pipeline/
│   ├── retriever.py          # Hybrid dense + sparse search, RRF fusion
│   ├── reranker.py            # Cross-encoder reranking
│   └── query_rewriter.py     # Multi-query expansion
│
├── vector_store/
│   └── vector_manager.py     # Builds the Qdrant index from Postgres reviews
│
├── database/
│   └── schemas.py             # SQLAlchemy models (Product, Review, QueryLog, ConversationalCache)
│
├── evaluation/
│   ├── generate_test_set.py  # Builds grounded Q&A test set from real reviews
│   ├── evaluate_pipeline.py  # RAGAS evaluation harness
│   └── debug_eval_rows.py     # Manual per-row inspection tool
│
├── main.py                    # FastAPI app exposing /query and /health
├── index.html                 # Standalone frontend (query console UI)
├── bm25_vocab.json             # BM25 vocabulary/IDF, generated at indexing time
└── .env                        # DATABASE_URL, GROQ_API_KEY, QDRANT_PATH, COLLECTION_NAME
```

## Setup

```bash
uv add langgraph langchain-core langchain-groq groq qdrant-client sentence-transformers \
       sqlalchemy python-dotenv psycopg2-binary fastapi uvicorn ragas datasets
```

`.env` file:
```
DATABASE_URL=postgresql://user:pass@localhost:5432/yourdb
GROQ_API_KEY=gsk_your_key_here
QDRANT_PATH=qdrant_storage
COLLECTION_NAME=product_reviews
BM25_VOCAB_PATH=bm25_vocab.json
```

Postgres also needs the `pgcrypto` extension (used by the query logger):
```sql
CREATE EXTENSION IF NOT EXISTS pgcrypto;
```

## Running the pipeline

Build the vector index (run once, or after any change to reviews/vocab):
```bash
python -m vector_store.vector_manager --reset
```

Test each agent standalone before wiring them together:
```bash
python -m agents.router
python -m agents.sql_agent
python -m agents.vector_agent
```

Run a query through the full pipeline:
```bash
python -m agents.orchestrator
```

## Running the API

```bash
uvicorn main:app --reload
```

Interactive docs: `http://127.0.0.1:8000/docs`

**`POST /query`**
```json
// request
{ "query": "best earphone for battery life under 2000" }

// response
{
  "query": "best earphone for battery life under 2000",
  "answer": "...",
  "route": "hybrid",
  "sources": [{ "product_name": "...", "review_summary": "...", "rating": 5.0 }],
  "latency_ms": 1234
}
```

**`GET /health`** — liveness check.

## Running the frontend

Open `index.html` directly in a browser — no build step. It calls the API at
`http://127.0.0.1:8000` by default (configurable in the UI). Make sure `main.py` is
running first.

## Evaluation

```bash
python -m evaluation.generate_test_set     # builds evaluation/test_set.json
python -m evaluation.evaluate_pipeline     # scores the live pipeline with RAGAS
```

By default this evaluates the full `hybrid_rerank` config (query expansion + hybrid
search + cross-encoder reranking). To compare against simpler retrieval configs:
```bash
python -m evaluation.evaluate_pipeline --config all
```

### Latest documented scores

*(fill in after your next clean run — n=50 questions, `hybrid_rerank` config)*

| Metric | Score |
|---|---|
| faithfulness | — |
| answer_relevancy | — |
| context_precision | — |
| context_recall | — |

## Known limitations

- **Router precision on ambiguous questions.** Fixed a major bug where fact-style questions
  ("does X have Y") were being misrouted to `sql` even when the fact only existed in review
  text — see commit history / this doc's changelog for the fix. Some edge cases may remain.
- **Duplicate reviews across product variants.** Deduplication logic exists in
  `vector_manager.py` (`deduplicate_reviews`) but requires a full `--reset` re-index to take
  effect on an existing collection.
- **Retrieval recall.** Even after tuning `top_k`/pool size, some ground-truth reviews aren't
  retrieved for narrowly-specific questions — a known gap between semantic similarity and exact
  factual matching.
- **Local Qdrant (file-based), not a server.** Fine for a single-machine demo; would need
  Qdrant Cloud or a self-hosted server for concurrent multi-user access.
- **No auth on the API.** `main.py` has no authentication layer — add one before any public deployment.

## Possible next steps

- Add a dense-only "naive" baseline search path (currently `retrieve_naive` in
  `evaluate_pipeline.py` falls back to the same hybrid search) for a true 3-way comparison.
- Cache repeated query embeddings to cut hybrid-route latency.
- Add conversational memory / follow-up question support.
- Deploy the API + frontend somewhere reachable (Render, Railway, Fly.io) instead of localhost-only.