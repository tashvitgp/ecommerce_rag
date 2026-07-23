# import json
# import logging
# import os
# from typing import TypedDict, Optional

# from dotenv import load_dotenv
# from groq import Groq
# from langgraph.graph import StateGraph, END
# from sqlalchemy import create_engine, text as sql_text

# from agents.router import classify_query
# from agents.sql_agent import run_sql_agent
# from agents.vector_agent import run_vector_agent

# load_dotenv()

# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# logger = logging.getLogger(__name__)

# GROQ_MODEL = "llama-3.1-8b-instant"
# client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# DATABASE_URL = os.getenv("DATABASE_URL")
# engine = create_engine(DATABASE_URL)


# # ---------------------------------------------------------------------------
# # State schema
# # ---------------------------------------------------------------------------

# class PipelineState(TypedDict):
#     query: str
#     route: Optional[str]
#     sql_result: Optional[dict]
#     vector_result: Optional[dict]
#     answer: Optional[str]
#     sources: Optional[list]


# # ---------------------------------------------------------------------------
# # Nodes
# # ---------------------------------------------------------------------------

# def router_node(state: PipelineState) -> PipelineState:
#     result = classify_query(state["query"])
#     state["route"] = result["route"]
#     return state


# def sql_node(state: PipelineState) -> PipelineState:
#     state["sql_result"] = run_sql_agent(state["query"])
#     return state


# def vector_node(state: PipelineState) -> PipelineState:
#     state["vector_result"] = run_vector_agent(state["query"])
#     return state


# def generator_node(state: PipelineState) -> PipelineState:
#     """Synthesize final answer from whichever results are present."""
#     context_parts = []
#     sources = []

#     if state.get("sql_result") and state["sql_result"]["success"]:
#         context_parts.append(f"SQL results:\n{json.dumps(state['sql_result']['results'], default=str)}")

#     if state.get("vector_result"):
#         reviews = state["vector_result"]["results"]
#         context_parts.append(
#             "Review excerpts:\n" + "\n".join(
#                 f"- [{r['product_name']}] ({r['rating']}★, {r['sentiment']}): {r['review_summary']}"
#                 for r in reviews
#             )
#         )
#         sources = [
#             {"product_name": r["product_name"], "review_summary": r["review_summary"], "rating": r["rating"]}
#             for r in reviews
#         ]

#     context = "\n\n".join(context_parts) if context_parts else "No data retrieved."

#     prompt = f"""Answer the user's question using ONLY the context below. Cite specific products/ratings where relevant. Be concise.

# Question: {state['query']}

# Context:
# {context}
# """
#     response = client.chat.completions.create(
#         model=GROQ_MODEL,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.3,
#         max_tokens=500,
#     )
#     state["answer"] = response.choices[0].message.content
#     state["sources"] = sources
#     return state


# def logger_node(state: PipelineState) -> PipelineState:
#     """Write the query + route + retrieved chunk ids to query_logs."""
#     try:
#         chunk_ids = [r.get("review_id") for r in state.get("vector_result", {}).get("results", [])] \
#             if state.get("vector_result") else []

#         with engine.connect() as conn:
#             conn.execute(
#                 sql_text("""
#                     INSERT INTO query_logs (log_id, query_text, agent_route, retrieved_chunk_ids)
#                     VALUES (gen_random_uuid()::text, :query_text, :agent_route, :retrieved_chunk_ids)
#                 """),
#                 {
#                     "query_text": state["query"],
#                     "agent_route": state["route"],
#                     "retrieved_chunk_ids": json.dumps(chunk_ids),
#                 },
#             )
#             conn.commit()
#     except Exception as e:
#         logger.error(f"Failed to write query log: {e}")
#     return state


# # ---------------------------------------------------------------------------
# # Routing logic (conditional edge)
# # ---------------------------------------------------------------------------

# def route_decision(state: PipelineState) -> str:
#     return state["route"]  # "sql" | "vector" | "hybrid"


# # ---------------------------------------------------------------------------
# # Build the graph
# # ---------------------------------------------------------------------------

# def build_graph():
#     graph = StateGraph(PipelineState)

#     graph.add_node("router", router_node)
#     graph.add_node("sql_agent", sql_node)
#     graph.add_node("vector_agent", vector_node)
#     graph.add_node("generator", generator_node)
#     graph.add_node("logger", logger_node)

#     graph.set_entry_point("router")

#     # Conditional branching based on router's decision
#     graph.add_conditional_edges(
#         "router",
#         route_decision,
#         {
#             "sql": "sql_agent",
#             "vector": "vector_agent",
#             "hybrid": "sql_agent",  # hybrid runs sql_agent first, then vector_agent
#         },
#     )

#     graph.add_conditional_edges(
#         "sql_agent",
#         lambda state: "vector_agent" if state["route"] == "hybrid" else "generator",
#         {"vector_agent": "vector_agent", "generator": "generator"},
#     )

#     graph.add_edge("vector_agent", "generator")
#     graph.add_edge("generator", "logger")
#     graph.add_edge("logger", END)

#     return graph.compile()


# _app = None


# def get_app():
#     global _app
#     if _app is None:
#         _app = build_graph()
#     return _app


# def run_pipeline(query: str) -> dict:
#     """Entry point: run a query through the full agentic pipeline."""
#     app = get_app()
#     final_state = app.invoke({
#         "query": query,
#         "route": None,
#         "sql_result": None,
#         "vector_result": None,
#         "answer": None,
#         "sources": None,
#     })
#     return {
#         "answer": final_state["answer"],
#         "sources": final_state["sources"],
#         "route": final_state["route"],
#     }


# if __name__ == "__main__":
#     result = run_pipeline("best earphone for battery life under 2000")
#     print(json.dumps(result, indent=2))


import json
import logging
import os
from typing import TypedDict, Optional

from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import StateGraph, END
from sqlalchemy import create_engine, text as sql_text

from agents.router import classify_query
from agents.sql_agent import run_sql_agent
from agents.vector_agent import run_vector_agent

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.1-8b-instant"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    query: str
    route: Optional[str]
    sql_result: Optional[dict]
    vector_result: Optional[dict]
    answer: Optional[str]
    sources: Optional[list]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def router_node(state: PipelineState) -> PipelineState:
    result = classify_query(state["query"])
    state["route"] = result["route"]
    return state


def sql_node(state: PipelineState) -> PipelineState:
    state["sql_result"] = run_sql_agent(state["query"])
    return state


def vector_node(state: PipelineState) -> PipelineState:
    state["vector_result"] = run_vector_agent(state["query"])
    return state


def generator_node(state: PipelineState) -> PipelineState:
    """Synthesize final answer from whichever results are present."""
    context_parts = []
    sources = []

    if state.get("sql_result") and state["sql_result"]["success"]:
        context_parts.append(f"SQL results:\n{json.dumps(state['sql_result']['results'], default=str)}")

    if state.get("vector_result"):
        reviews = state["vector_result"]["results"]
        context_parts.append(
            "Review excerpts:\n" + "\n".join(
                f"- [{r['product_name']}] ({r['rating']}★, {r['sentiment']}): {r['review_summary']}"
                for r in reviews
            )
        )
        sources = [
            {"product_name": r["product_name"], "review_summary": r["review_summary"], "rating": r["rating"]}
            for r in reviews
        ]

    context = "\n\n".join(context_parts) if context_parts else "No data retrieved."

    prompt = f"""Answer the user's question using ONLY the context below.

STRICT RULES:
- If the context does not explicitly mention the specific fact/feature/spec asked about, say
  clearly that the reviews don't mention it — do NOT guess, infer, or assume an answer based on
  what similar products typically have.
- Only state something as fact if it is explicitly written in the context below. Do not fill
  gaps with outside knowledge, even if it seems like a reasonable assumption.
- Cite specific products/ratings where relevant.
- Be concise.

Question: {state['query']}

Context:
{context}
"""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500,
    )
    state["answer"] = response.choices[0].message.content
    state["sources"] = sources
    return state


def logger_node(state: PipelineState) -> PipelineState:
    """Write the query + route + retrieved chunk ids to query_logs."""
    try:
        chunk_ids = [r.get("review_id") for r in state.get("vector_result", {}).get("results", [])] \
            if state.get("vector_result") else []

        with engine.connect() as conn:
            conn.execute(
                sql_text("""
                    INSERT INTO query_logs (log_id, query_text, agent_route, retrieved_chunk_ids)
                    VALUES (gen_random_uuid()::text, :query_text, :agent_route, :retrieved_chunk_ids)
                """),
                {
                    "query_text": state["query"],
                    "agent_route": state["route"],
                    "retrieved_chunk_ids": json.dumps(chunk_ids),
                },
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to write query log: {e}")
    return state


# ---------------------------------------------------------------------------
# Routing logic (conditional edge)
# ---------------------------------------------------------------------------

def route_decision(state: PipelineState) -> str:
    return state["route"]  # "sql" | "vector" | "hybrid"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("router", router_node)
    graph.add_node("sql_agent", sql_node)
    graph.add_node("vector_agent", vector_node)
    graph.add_node("generator", generator_node)
    graph.add_node("logger", logger_node)

    graph.set_entry_point("router")

    # Conditional branching based on router's decision
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "sql": "sql_agent",
            "vector": "vector_agent",
            "hybrid": "sql_agent",  # hybrid runs sql_agent first, then vector_agent
        },
    )

    graph.add_conditional_edges(
        "sql_agent",
        lambda state: "vector_agent" if state["route"] == "hybrid" else "generator",
        {"vector_agent": "vector_agent", "generator": "generator"},
    )

    graph.add_edge("vector_agent", "generator")
    graph.add_edge("generator", "logger")
    graph.add_edge("logger", END)

    return graph.compile()


_app = None


def get_app():
    global _app
    if _app is None:
        _app = build_graph()
    return _app


def run_pipeline(query: str) -> dict:
    """Entry point: run a query through the full agentic pipeline."""
    app = get_app()
    final_state = app.invoke({
        "query": query,
        "route": None,
        "sql_result": None,
        "vector_result": None,
        "answer": None,
        "sources": None,
    })
    return {
        "answer": final_state["answer"],
        "sources": final_state["sources"],
        "route": final_state["route"],
    }


if __name__ == "__main__":
    result = run_pipeline("best earphone for battery life under 2000")
    print(json.dumps(result, indent=2))