"""
RAGAS evaluation harness.

Runs the actual RAG pipeline (retrieve -> generate, no cache) against the
hand-authored Q&A ground-truth set and scores it with RAGAS: faithfulness,
answer relevancy, context precision, context recall.

Free-tier resilience (Groq/OpenRouter only, no paid APIs):
  - The judge LLM runs on OpenRouter (settings.llm_fallback_model), not
    Groq: the RAG pipeline's own generation already spends Groq's tight
    daily token budget (200K/day on this account), and a 15-question x
    4-metric run needs many more judge calls than that budget survives.
    Splitting judge calls onto a separate provider/quota means pipeline
    generation and judging no longer compete for the same cap.
  - Judge LLM calls are disk-cached (data/eval/.ragas_cache) keyed on the
    actual prompt sent, so re-running against unchanged questions never
    re-spends quota.
  - RunConfig retries with exponential backoff on rate-limit/transient
    errors before giving up on a single metric call.
  - ragas.evaluate(..., raise_exceptions=False) means a failing
    question/metric becomes NaN in the results instead of crashing the
    run -- and the summary reports how many of the N rows actually
    succeeded per metric, so a partial sample is never mistaken for a
    full-dataset score.

Faithfulness and Context Precision are the primary decision signals (does
the answer stay grounded, does retrieval surface the right chunks);
Answer Relevancy and Context Recall are secondary diagnostics.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import openai
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.cache import DiskCacheBackend
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

from rag_app.config.settings import settings
from rag_app.core.interfaces import EmbeddingProvider, LLMProvider, VectorStore
from rag_app.embeddings.fastembed_provider import FastEmbedProvider
from rag_app.evaluation.datasets import QAItem, load_qa_dataset
from rag_app.llm.groq_provider import GroqProvider
from rag_app.llm.openrouter_provider import OpenRouterProvider
from rag_app.llm.router import LLMRouter
from rag_app.observability.provider import ObservabilityProvider
from rag_app.rag.graph import build_rag_graph
from rag_app.reranking.cross_encoder_reranker import CrossEncoderReranker
from rag_app.retrieval.bm25_index import BM25Index
from rag_app.vectorstore.qdrant_store import QdrantStore

PRIMARY_METRICS = {"faithfulness", "context_precision"}
SECONDARY_METRICS = {"answer_relevancy", "context_recall"}
ALL_METRIC_NAMES = ("faithfulness", "context_precision", "answer_relevancy", "context_recall")

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = _PROJECT_ROOT / "data" / "eval" / "results"
JUDGE_CACHE_DIR = _PROJECT_ROOT / "data" / "eval" / ".ragas_cache"


@dataclass
class MetricScore:
    """Aggregate score for one metric, honest about partial samples."""

    mean: float
    succeeded: int  # rows with a non-NaN score for this metric
    total: int  # rows attempted


@dataclass
class EvalRunResult:
    """Summary of a single RAGAS evaluation run."""

    scores: dict[str, MetricScore]  # metric -> mean + how many rows it covers
    per_question: list[dict]  # raw per-row results, including NaN failures
    results_path: Path


async def _run_pipeline(graph, question: str, top_k: int) -> tuple[str, list[str]]:
    """Invoke the RAG graph for one question, returning (answer, contexts)."""
    result = await graph.ainvoke({"query": question, "top_k": top_k, "filters": None})
    answer = result.get("generated_answer", "")
    chunks = result.get("reranked_chunks") or result.get("retrieved_chunks", [])
    contexts = [c.content for c in chunks]
    return answer, contexts


async def build_eval_dataset(
    qa_items: list[QAItem], graph, top_k: int = 5
) -> EvaluationDataset:
    """Run each question through the live RAG pipeline and assemble a RAGAS dataset."""
    rows = []
    for item in qa_items:
        answer, contexts = await _run_pipeline(graph, item["question"], top_k)
        rows.append(
            {
                "user_input": item["question"],
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": item["ground_truth"],
            }
        )
    return EvaluationDataset.from_list(rows)


def _build_judge_llm() -> LangchainLLMWrapper:
    """RAGAS judge LLM: OpenRouter, with retry/backoff and disk-cached calls.

    Deliberately NOT Groq -- the RAG pipeline's own generation already
    spends from Groq's tight daily token budget, and judge calls (several
    per question per metric) would compete for the same cap.
    """
    JUDGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    chat = ChatOpenAI(
        model=settings.llm_fallback_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0,
    )
    run_config = RunConfig(
        max_retries=6,
        max_wait=90,
        exception_types=(
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
        ),
    )
    cache = DiskCacheBackend(cache_dir=str(JUDGE_CACHE_DIR))
    return LangchainLLMWrapper(chat, run_config=run_config, cache=cache)


def _build_judge_embeddings() -> LangchainEmbeddingsWrapper:
    """RAGAS judge embeddings: the same FastEmbed model used in production."""
    embeddings = FastEmbedEmbeddings(model_name=settings.embedding_model)
    return LangchainEmbeddingsWrapper(embeddings)


async def run_ragas_eval(
    qa_items: list[QAItem] | None = None,
    top_k: int = 5,
    obs: ObservabilityProvider | None = None,
    use_hyde: bool = False,
    use_reranker: bool = False,
    use_bm25: bool = False,
) -> EvalRunResult:
    """Run the full RAGAS evaluation: live pipeline -> RAGAS scoring -> persist.

    Builds its own provider instances rather than reusing FastAPI's DI
    container (matches scripts/seed_vectorstore.py). Runs the RAG graph
    WITHOUT the Redis cache so every question hits real retrieval and
    generation, uncontaminated by cached answers from a prior eval run.

    Args:
        use_hyde: Insert the hyde_expand node before retrieve, so the
            query embedding comes from a generated hypothetical answer
            passage instead of the raw question. Costs one extra LLM call
            per question. Used to measure before/after when comparing
            against a baseline run with use_hyde=False.
        use_reranker: Retrieve a wider candidate pool and re-score it with
            a local cross-encoder before generation, trimming back down to
            top_k. Purely local compute, no extra API/quota cost.
        use_bm25: Also run BM25 keyword search and merge it into the
            candidate pool before reranking. Only has an effect when
            use_reranker is also True -- otherwise nothing re-scores the
            merged set.
    """
    obs = obs or ObservabilityProvider()
    qa_items = qa_items if qa_items is not None else load_qa_dataset()

    embedder: EmbeddingProvider = FastEmbedProvider()
    vector_store: VectorStore = QdrantStore(obs=obs, embedding_dimension=embedder.dimension())
    llm_provider: LLMProvider = LLMRouter(
        primary=GroqProvider(), fallback=OpenRouterProvider(), obs=obs
    )
    reranker = CrossEncoderReranker() if use_reranker else None
    keyword_search = (
        BM25Index(qdrant_url=settings.qdrant_url, collection_name=settings.qdrant_collection_name)
        if use_bm25
        else None
    )
    graph = build_rag_graph(
        embedder,
        vector_store,
        llm_provider,
        cache=None,
        use_hyde=use_hyde,
        reranker=reranker,
        keyword_search=keyword_search,
    )

    with obs.span("evaluation.run", {"question_count": len(qa_items)}):
        start = time.perf_counter()
        dataset = await build_eval_dataset(qa_items, graph, top_k=top_k)

        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=_build_judge_llm(),
            embeddings=_build_judge_embeddings(),
            raise_exceptions=False,
            show_progress=True,
        )

        df = result.to_pandas()
        total = len(df)
        scores: dict[str, MetricScore] = {}
        for metric in ALL_METRIC_NAMES:
            if metric not in df.columns:
                continue
            non_nan = df[metric].dropna()
            if len(non_nan) == 0:
                continue
            scores[metric] = MetricScore(
                mean=float(non_nan.mean()), succeeded=len(non_nan), total=total
            )

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        obs.log_event(
            "evaluation.run.completed",
            {
                "scores": {k: vars(v) for k, v in scores.items()},
                "question_count": len(qa_items),
                "duration_ms": duration_ms,
            },
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_path = RESULTS_DIR / f"ragas_run_{timestamp}.json"
    per_question = df.to_dict(orient="records")
    results_path.write_text(
        json.dumps(
            {"scores": {k: vars(v) for k, v in scores.items()}, "per_question": per_question},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    return EvalRunResult(scores=scores, per_question=per_question, results_path=results_path)


def print_summary(result: EvalRunResult) -> None:
    """Print a human-readable summary table, primary metrics first.

    Always shows how many rows a score actually covers -- a mean over 1
    surviving row and a mean over 15 are not comparable, so both are
    printed rather than just the number.
    """
    print(f"\nRAGAS evaluation results ({len(result.per_question)} questions)")
    print(f"Saved to: {result.results_path}\n")
    print(f"{'Metric':<20}{'Score':<10}{'Coverage':<12}{'Signal'}")
    print("-" * 60)
    for metric in ALL_METRIC_NAMES:
        signal = "primary" if metric in PRIMARY_METRICS else "secondary"
        if metric not in result.scores:
            print(f"{metric:<20}{'N/A':<10}{'0/?':<12}{signal} -- all rows failed")
            continue
        score = result.scores[metric]
        coverage = f"{score.succeeded}/{score.total}"
        note = "" if score.succeeded == score.total else " -- PARTIAL, interpret with caution"
        print(f"{metric:<20}{score.mean:<10.4f}{coverage:<12}{signal}{note}")
