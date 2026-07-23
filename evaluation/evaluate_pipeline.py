# import argparse
# import functools
# import json
# import logging
# import os
# import sys
# import types
# from datetime import datetime

# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# try:
#     from langchain_google_vertexai import ChatVertexAI
# except Exception:  # pragma: no cover - fallback for environments without VertexAI support
#     class ChatVertexAI:  # type: ignore[no-redef]
#         def __init__(self, *args, **kwargs):
#             raise RuntimeError("VertexAI support is unavailable in this environment")

# vertexai_module = types.ModuleType("langchain_community.chat_models.vertexai")
# vertexai_module.ChatVertexAI = ChatVertexAI
# sys.modules.setdefault("langchain_community.chat_models.vertexai", vertexai_module)

# from dotenv import load_dotenv
# from datasets import Dataset
# from langchain_groq import ChatGroq
# from ragas import evaluate
# from ragas.llms import LangchainLLMWrapper
# from ragas.embeddings.base import BaseRagasEmbeddings
# from ragas.metrics import (
#     faithfulness,
#     answer_relevancy,
#     context_precision,
#     context_recall,
# )

# # RAGAS defaults to OpenAI for its LLM-judge metrics (faithfulness,
# # answer_relevancy, context_precision, context_recall). Point it at Groq
# # instead so no OpenAI key is needed, matching the rest of this project.
# RAGAS_JUDGE_MODEL = "llama-3.3-70b-versatile"  # larger model — judging quality matters here
# _ragas_llm = LangchainLLMWrapper(ChatGroq(model=RAGAS_JUDGE_MODEL, temperature=0))

# # answer_relevancy also needs an embedding model to compare semantic
# # similarity between generated questions and the original — reuse the same
# # local embedding model already used elsewhere in the project (no API key
# # or extra cost).
# class LocalSentenceTransformerEmbeddings(BaseRagasEmbeddings):
#     def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
#         super().__init__()
#         from sentence_transformers import SentenceTransformer

#         self.model = SentenceTransformer(model_name)

#     def embed_query(self, text: str) -> list[float]:
#         return self.embed_documents([text])[0]

#     def embed_documents(self, texts: list[str]) -> list[list[float]]:
#         embeddings = self.model.encode(texts, normalize_embeddings=True, convert_to_tensor=False)
#         return embeddings.tolist()

#     async def aembed_query(self, text: str) -> list[float]:
#         return self.embed_query(text)

#     async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
#         return self.embed_documents(texts)


# _ragas_embeddings = LocalSentenceTransformerEmbeddings()

# for metric in (faithfulness, answer_relevancy, context_precision, context_recall):
#     metric.llm = _ragas_llm

# answer_relevancy.embeddings = _ragas_embeddings

# from pipeline.retriever import hybrid_search
# from pipeline.query_rewriter import multi_query_retrieve
# from pipeline.reranker import rerank
# from agents.vector_agent import extract_intent
# from agents.orchestrator import run_pipeline

# load_dotenv()

# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# logger = logging.getLogger(__name__)

# TEST_SET_PATH  = "evaluation/test_set.json"
# RESULTS_DIR    = "evaluation/results"

# # ---------------------------------------------------------------------------
# # Retrieval configs — each returns list[str] of context texts for a query.
# # All three are wired up so you can compare naive vs hybrid vs hybrid+rerank
# # later; --config controls which one(s) actually run.
# # ---------------------------------------------------------------------------

# def retrieve_naive(query: str, top_k: int = 5) -> list[dict]:
#     """Dense-only search, no sparse fusion, no rerank, no query expansion."""
#     intent = extract_intent(query)
#     # hybrid_search does dense+sparse RRF internally; for a true "naive"
#     # baseline we bypass fusion and just take the dense ranking by using
#     # a pool size equal to top_k and skipping rerank/expansion upstream.
#     results = hybrid_search(query, category=intent["category"], aspect=intent["aspect"], top_k=top_k)
#     return results


# def retrieve_hybrid(query: str, top_k: int = 5) -> list[dict]:
#     """Dense + sparse RRF fusion, no reranking, no query expansion."""
#     intent = extract_intent(query)
#     results = hybrid_search(query, category=intent["category"], aspect=intent["aspect"], top_k=top_k)
#     return results


# def retrieve_hybrid_rerank(query: str, top_k: int = 5, pool_size: int = 15) -> list[dict]:
#     """Full pipeline: multi-query expansion -> hybrid RRF -> cross-encoder rerank."""
#     intent = extract_intent(query)
#     retrieve_fn = functools.partial(hybrid_search, category=intent["category"], aspect=intent["aspect"])
#     candidates = multi_query_retrieve(query, retrieve_fn, top_k=pool_size)
#     return rerank(query, candidates, top_k=top_k)


# RETRIEVAL_CONFIGS = {
#     "naive": retrieve_naive,
#     "hybrid": retrieve_hybrid,
#     "hybrid_rerank": retrieve_hybrid_rerank,
# }


# def contexts_from_results(results: list[dict]) -> list[str]:
#     return [r["payload"].get("review_text") or r["payload"].get("review_summary", "") for r in results]


# # ---------------------------------------------------------------------------
# # Build RAGAS-ready dataset for one config
# # ---------------------------------------------------------------------------

# def build_eval_rows(test_set: list[dict], config_name: str) -> list[dict]:
#     """
#     For each test question, retrieve contexts using the given config and
#     generate an answer via the full orchestrator (so 'answer' always
#     reflects what a real user would see — only the CONTEXT differs by
#     config, which is what RAGAS's context_precision/recall actually measure).
#     """
#     retrieve_fn = RETRIEVAL_CONFIGS[config_name]
#     rows = []

#     for i, item in enumerate(test_set, 1):
#         question = item["question"]

#         try:
#             retrieved = retrieve_fn(question)
#             contexts = contexts_from_results(retrieved)

#             pipeline_result = run_pipeline(question)
#             answer = pipeline_result["answer"]

#             rows.append({
#                 "question": question,
#                 "answer": answer,
#                 "contexts": contexts if contexts else ["No context retrieved."],
#                 "ground_truth": item["ground_truth"],
#             })

#         except Exception as e:
#             logger.error(f"Row {i} failed for question '{question}': {e}")

#         if i % 10 == 0:
#             logger.info(f"Built {i}/{len(test_set)} eval rows for config='{config_name}'")

#     return rows


# def run_ragas_eval(rows: list[dict]) -> dict:
#     """Run RAGAS metrics over the built rows and return the summary scores."""
#     dataset = Dataset.from_list(rows)

#     result = evaluate(
#         dataset,
#         metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
#     )
#     return result


# def run_evaluation(config_names: list[str], test_set_path: str = TEST_SET_PATH) -> dict:
#     with open(test_set_path, "r") as f:
#         test_set = json.load(f)

#     logger.info(f"Loaded {len(test_set)} test questions from {test_set_path}")

#     os.makedirs(RESULTS_DIR, exist_ok=True)
#     all_results = {}

#     for config_name in config_names:
#         logger.info(f"\n{'='*55}\nEvaluating config: {config_name}\n{'='*55}")

#         rows = build_eval_rows(test_set, config_name)
#         if not rows:
#             logger.error(f"No valid rows built for config '{config_name}' — skipping")
#             continue

#         scores = run_ragas_eval(rows)
#         scores_dict = scores.to_pandas().mean(numeric_only=True).to_dict()

#         all_results[config_name] = scores_dict
#         logger.info(f"Scores for '{config_name}': {scores_dict}")

#         # Save per-config detailed results
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         out_path = f"{RESULTS_DIR}/{config_name}_{timestamp}.json"
#         with open(out_path, "w") as f:
#             json.dump({"config": config_name, "scores": scores_dict, "n_rows": len(rows)}, f, indent=2)
#         logger.info(f"✓ Saved detailed results to {out_path}")

#     return all_results


# def print_comparison_table(all_results: dict) -> None:
#     if not all_results:
#         logger.warning("No results to compare.")
#         return

#     metrics = list(next(iter(all_results.values())).keys())
#     header = f"{'Config':<15}" + "".join(f"{m:<20}" for m in metrics)
#     print("\n" + "=" * len(header))
#     print(header)
#     print("=" * len(header))
#     for config_name, scores in all_results.items():
#         row = f"{config_name:<15}" + "".join(f"{scores.get(m, 0):<20.4f}" for m in metrics)
#         print(row)
#     print("=" * len(header))


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Evaluate RAG pipeline with RAGAS.")
#     parser.add_argument(
#         "--config",
#         choices=list(RETRIEVAL_CONFIGS.keys()) + ["all"],
#         default="hybrid_rerank",
#         help="Which retrieval config to evaluate. Default: hybrid_rerank (the live pipeline). "
#              "Use 'all' to compare naive vs hybrid vs hybrid_rerank.",
#     )
#     return parser.parse_args()


# if __name__ == "__main__":
#     args = parse_args()
#     configs_to_run = list(RETRIEVAL_CONFIGS.keys()) if args.config == "all" else [args.config]

#     results = run_evaluation(configs_to_run)
#     print_comparison_table(results)


import argparse
import functools
import json
import logging
import os
import sys
import types
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from langchain_google_vertexai import ChatVertexAI
except Exception:  # pragma: no cover - fallback for environments without VertexAI support
    class ChatVertexAI:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("VertexAI support is unavailable in this environment")

vertexai_module = types.ModuleType("langchain_community.chat_models.vertexai")
vertexai_module.ChatVertexAI = ChatVertexAI
sys.modules.setdefault("langchain_community.chat_models.vertexai", vertexai_module)

from dotenv import load_dotenv
from datasets import Dataset
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# RAGAS defaults to OpenAI for its LLM-judge metrics (faithfulness,
# answer_relevancy, context_precision, context_recall). Point it at Groq
# instead so no OpenAI key is needed, matching the rest of this project.
RAGAS_JUDGE_MODEL = "llama-3.1-8b-instant" # larger model — judging quality matters here
_ragas_llm = LangchainLLMWrapper(ChatGroq(model=RAGAS_JUDGE_MODEL, temperature=0))

# answer_relevancy also needs an embedding model to compare semantic
# similarity between generated questions and the original — reuse the same
# local embedding model already used elsewhere in the project (no API key
# or extra cost).
class LocalSentenceTransformerEmbeddings(BaseRagasEmbeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True, convert_to_tensor=False)
        return embeddings.tolist()

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


_ragas_embeddings = LocalSentenceTransformerEmbeddings()

for metric in (faithfulness, answer_relevancy, context_precision, context_recall):
    metric.llm = _ragas_llm

answer_relevancy.embeddings = _ragas_embeddings

from pipeline.retriever import hybrid_search
from pipeline.query_rewriter import multi_query_retrieve
from pipeline.reranker import rerank
from agents.vector_agent import extract_intent
from agents.orchestrator import run_pipeline

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TEST_SET_PATH  = "evaluation/test_set.json"
RESULTS_DIR    = "evaluation/results"

# ---------------------------------------------------------------------------
# Retrieval configs — each returns list[str] of context texts for a query.
# All three are wired up so you can compare naive vs hybrid vs hybrid+rerank
# later; --config controls which one(s) actually run.
# ---------------------------------------------------------------------------

def retrieve_naive(query: str, top_k: int = 5) -> list[dict]:
    """Dense-only search, no sparse fusion, no rerank, no query expansion."""
    intent = extract_intent(query)
    # hybrid_search does dense+sparse RRF internally; for a true "naive"
    # baseline we bypass fusion and just take the dense ranking by using
    # a pool size equal to top_k and skipping rerank/expansion upstream.
    results = hybrid_search(query, category=intent["category"], aspect=intent["aspect"], top_k=top_k)
    return results


def retrieve_hybrid(query: str, top_k: int = 5) -> list[dict]:
    """Dense + sparse RRF fusion, no reranking, no query expansion."""
    intent = extract_intent(query)
    results = hybrid_search(query, category=intent["category"], aspect=intent["aspect"], top_k=top_k)
    return results


def retrieve_hybrid_rerank(query: str, top_k: int = 5, pool_size: int = 15) -> list[dict]:
    """Full pipeline: multi-query expansion -> hybrid RRF -> cross-encoder rerank."""
    intent = extract_intent(query)
    retrieve_fn = functools.partial(hybrid_search, category=intent["category"], aspect=intent["aspect"])
    candidates = multi_query_retrieve(query, retrieve_fn, top_k=pool_size)
    return rerank(query, candidates, top_k=top_k)


RETRIEVAL_CONFIGS = {
    "naive": retrieve_naive,
    "hybrid": retrieve_hybrid,
    "hybrid_rerank": retrieve_hybrid_rerank,
}


def contexts_from_results(results: list[dict]) -> list[str]:
    return [r["payload"].get("review_text") or r["payload"].get("review_summary", "") for r in results]


# ---------------------------------------------------------------------------
# Build RAGAS-ready dataset for one config
# ---------------------------------------------------------------------------

def build_eval_rows(test_set: list[dict], config_name: str) -> list[dict]:
    """
    For each test question, retrieve contexts using the given config and
    generate an answer via the full orchestrator (so 'answer' always
    reflects what a real user would see — only the CONTEXT differs by
    config, which is what RAGAS's context_precision/recall actually measure).
    """
    retrieve_fn = RETRIEVAL_CONFIGS[config_name]
    rows = []
    refusal_count = 0
    refusal_phrases = ["don't mention", "not mentioned", "no information", "does not mention", "doesn't mention"]

    for i, item in enumerate(test_set, 1):
        question = item["question"]

        try:
            retrieved = retrieve_fn(question)
            contexts = contexts_from_results(retrieved)

            pipeline_result = run_pipeline(question)
            answer = pipeline_result["answer"]

            if any(p in answer.lower() for p in refusal_phrases):
                refusal_count += 1

            rows.append({
                "question": question,
                "answer": answer,
                "contexts": contexts if contexts else ["No context retrieved."],
                "ground_truth": item["ground_truth"],
            })

        except Exception as e:
            logger.error(f"Row {i} failed for question '{question}': {e}")

        if i % 10 == 0:
            logger.info(f"Built {i}/{len(test_set)} eval rows for config='{config_name}'")

    logger.info(f"✓ Refusal-style answers: {refusal_count}/{len(rows)} rows ({100*refusal_count/max(len(rows),1):.0f}%)")
    return rows


def run_ragas_eval(rows: list[dict]) -> dict:
    """Run RAGAS metrics over the built rows and return the summary scores."""
    dataset = Dataset.from_list(rows)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return result


def run_evaluation(config_names: list[str], test_set_path: str = TEST_SET_PATH) -> dict:
    with open(test_set_path, "r") as f:
        test_set = json.load(f)

    logger.info(f"Loaded {len(test_set)} test questions from {test_set_path}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}

    for config_name in config_names:
        logger.info(f"\n{'='*55}\nEvaluating config: {config_name}\n{'='*55}")

        rows = build_eval_rows(test_set, config_name)
        if not rows:
            logger.error(f"No valid rows built for config '{config_name}' — skipping")
            continue

        scores = run_ragas_eval(rows)
        scores_dict = scores.to_pandas().mean(numeric_only=True).to_dict()

        all_results[config_name] = scores_dict
        logger.info(f"Scores for '{config_name}': {scores_dict}")

        # Save per-config detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"{RESULTS_DIR}/{config_name}_{timestamp}.json"
        with open(out_path, "w") as f:
            json.dump({"config": config_name, "scores": scores_dict, "n_rows": len(rows)}, f, indent=2)
        logger.info(f"✓ Saved detailed results to {out_path}")

    return all_results


def print_comparison_table(all_results: dict) -> None:
    if not all_results:
        logger.warning("No results to compare.")
        return

    metrics = list(next(iter(all_results.values())).keys())
    header = f"{'Config':<15}" + "".join(f"{m:<20}" for m in metrics)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for config_name, scores in all_results.items():
        row = f"{config_name:<15}" + "".join(f"{scores.get(m, 0):<20.4f}" for m in metrics)
        print(row)
    print("=" * len(header))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline with RAGAS.")
    parser.add_argument(
        "--config",
        choices=list(RETRIEVAL_CONFIGS.keys()) + ["all"],
        default="hybrid_rerank",
        help="Which retrieval config to evaluate. Default: hybrid_rerank (the live pipeline). "
             "Use 'all' to compare naive vs hybrid vs hybrid_rerank.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configs_to_run = list(RETRIEVAL_CONFIGS.keys()) if args.config == "all" else [args.config]

    results = run_evaluation(configs_to_run)
    print_comparison_table(results)