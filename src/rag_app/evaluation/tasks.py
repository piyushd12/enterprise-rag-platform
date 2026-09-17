"""
Celery task for running the RAGAS evaluation harness asynchronously.

A full evaluation run makes many LLM calls (generation + RAGAS judge calls
per metric) and can take several minutes, so POST /evaluate enqueues it
here rather than blocking the request, mirroring ingestion/tasks.py.
"""

from __future__ import annotations

import asyncio

from rag_app.evaluation.datasets import load_qa_dataset
from rag_app.evaluation.harness import run_ragas_eval
from rag_app.observability.provider import ObservabilityProvider
from rag_app.queue.celery_app import celery_app


@celery_app.task(bind=True, name="rag_app.evaluation.tasks.run_evaluation")
def run_evaluation(
    self,
    sample_size: int | None = None,
    use_hyde: bool = False,
    use_reranker: bool = False,
    use_bm25: bool = False,
):
    """Run the RAGAS evaluation harness and return the scores.

    Args:
        sample_size: Evaluate only the first N questions (default: all 15).
        use_hyde: Run the pipeline with the HyDE query-expansion node enabled,
            to compare against a baseline run with use_hyde=False.
        use_reranker: Run the pipeline with the cross-encoder reranker
            enabled, to compare against a baseline run with use_reranker=False.
        use_bm25: Also merge BM25 keyword search into the candidate pool
            before reranking (only has an effect when use_reranker is True).
    """
    obs = ObservabilityProvider()
    task_id = self.request.id

    obs.log_event(
        "evaluation.task.started",
        {
            "task_id": task_id,
            "sample_size": sample_size,
            "use_hyde": use_hyde,
            "use_reranker": use_reranker,
            "use_bm25": use_bm25,
        },
    )

    qa_items = load_qa_dataset()
    if sample_size is not None:
        qa_items = qa_items[:sample_size]

    self.update_state(
        state="STARTED",
        meta={"message": f"Evaluating {len(qa_items)} question(s)", "total_chunks": len(qa_items)},
    )

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            run_ragas_eval(
                qa_items=qa_items,
                obs=obs,
                use_hyde=use_hyde,
                use_reranker=use_reranker,
                use_bm25=use_bm25,
            )
        )
    finally:
        loop.close()
        asyncio.set_event_loop(None)

    scores = {k: vars(v) for k, v in result.scores.items()}

    obs.log_event(
        "evaluation.task.succeeded",
        {"task_id": task_id, "scores": scores, "question_count": len(qa_items)},
    )

    return {
        "scores": scores,
        "question_count": len(qa_items),
        "results_path": str(result.results_path),
    }
