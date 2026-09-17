"""
Evaluation endpoint — triggers the RAGAS evaluation harness asynchronously.

A full run makes many LLM calls (generation + judge calls per metric) and
can take several minutes, so this enqueues a Celery task and returns
immediately, mirroring /ingest's async pattern. Poll GET /evaluate/status/{task_id}
for progress and final scores.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from rag_app.api.dependencies import get_obs, get_queue
from rag_app.api.schemas import EvalRequest, EvalResponse, TaskStatusResponse
from rag_app.core.interfaces import TaskQueue
from rag_app.observability.provider import ObservabilityProvider

router = APIRouter(tags=["evaluation"])


@router.post("/evaluate", response_model=EvalResponse)
async def evaluate(
    body: EvalRequest,
    obs: ObservabilityProvider = Depends(get_obs),
    queue: TaskQueue = Depends(get_queue),
) -> EvalResponse:
    """Run the RAGAS evaluation harness (faithfulness, answer relevancy,
    context precision, context recall) against the ground-truth Q&A set.

    Runs asynchronously via Celery — poll GET /evaluate/status/{task_id}.
    """
    try:
        task_id = queue.enqueue(
            "rag_app.evaluation.tasks.run_evaluation",
            sample_size=body.sample_size,
            use_hyde=body.use_hyde,
            use_reranker=body.use_reranker,
            use_bm25=body.use_bm25,
        )
    except Exception as e:
        obs.log_error("evaluate.enqueue.failed", {"error": str(e)})
        raise HTTPException(status_code=503, detail=f"Task queue unavailable: {e}")

    return EvalResponse(task_id=task_id, status="queued", message="Evaluation task enqueued")


@router.get("/evaluate/status/{task_id}", response_model=TaskStatusResponse)
async def get_evaluate_status(
    task_id: str,
    queue: TaskQueue = Depends(get_queue),
) -> TaskStatusResponse:
    """Poll the status of an evaluation task: queued/started/succeeded/failed."""
    try:
        status = queue.get_status(task_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query task status: {e}")

    return TaskStatusResponse(
        task_id=status["task_id"],
        status=status["status"],
        result=status.get("result"),
        error=status.get("error"),
        progress=status.get("progress"),
    )
