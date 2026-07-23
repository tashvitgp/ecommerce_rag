import logging
import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.orchestrator import run_pipeline

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CRAG Project API",
    description="Agentic RAG system over product reviews — routes queries to SQL, vector search, or both.",
    version="1.0.0",
)

# Allow a local frontend (e.g. served on a different port) to call this API.
# Tighten allow_origins to your actual frontend URL before deploying anywhere public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Natural language product question")


class SourceItem(BaseModel):
    product_name: str | None = None
    review_summary: str | None = None
    rating: float | None = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    route: str
    sources: list[SourceItem]
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    version: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Simple liveness check — does not verify DB/Qdrant/Groq connectivity."""
    return HealthResponse(status="ok", version=app.version)


def _run_query(text: str) -> QueryResponse:
    start = time.time()

    if not text.strip():
        raise HTTPException(status_code=400, detail="Query text is required")

    try:
        result = run_pipeline(text)
    except Exception as e:
        logger.error(f"Pipeline failed for query '{text}': {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    latency_ms = int((time.time() - start) * 1000)

    return QueryResponse(
        query=text,
        answer=result["answer"],
        route=result["route"],
        sources=[SourceItem(**s) for s in (result.get("sources") or [])],
        latency_ms=latency_ms,
    )


@app.get("/query", response_model=QueryResponse)
def query_get(query_text: str = Query(..., max_length=500)):
    return _run_query(query_text)


@app.post("/query", response_model=QueryResponse)
def query_post(request: QueryRequest):
    return _run_query(request.query)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)