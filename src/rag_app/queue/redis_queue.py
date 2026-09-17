"""
RedisQueue — TaskQueue implementation using Celery with Redis broker.

Wraps Celery's task dispatch and result query behind the TaskQueue interface
so that route handlers depend on an ABC, never on Celery directly.
"""

from __future__ import annotations

from typing import Any

from celery.result import AsyncResult

from rag_app.core.interfaces import TaskQueue
from rag_app.observability.provider import ObservabilityProvider
from rag_app.queue.celery_app import celery_app


class RedisQueue(TaskQueue):
    """Celery-backed task queue with Logfire lifecycle logging."""

    def __init__(self, obs: ObservabilityProvider | None = None):
        self._obs = obs

    def enqueue(self, task_name: str, *args: Any, **kwargs: Any) -> str:
        """Dispatch a Celery task and return its task ID immediately.

        Args:
            task_name: Fully qualified task name (e.g. 'rag_app.ingestion.tasks.ingest_document').
            *args: Positional arguments to pass to the task.
            **kwargs: Keyword arguments to pass to the task.

        Returns:
            The Celery task ID string.
        """
        result: AsyncResult = celery_app.send_task(task_name, args=args, kwargs=kwargs)

        if self._obs:
            self._obs.log_event(
                "queue.task.enqueued",
                {"task_id": result.id, "task_name": task_name},
            )

        return result.id

    def get_status(self, task_id: str) -> dict:
        """Query the current state and metadata of a Celery task.

        Returns:
            Dict with keys: task_id, status, result, error, progress.
        """
        result = AsyncResult(task_id, app=celery_app)
        state = result.state  # PENDING, STARTED, RETRY, SUCCESS, FAILURE, REVOKED

        status_map = {
            "PENDING": "queued",
            "STARTED": "started",
            "RETRY": "retried",
            "SUCCESS": "succeeded",
            "FAILURE": "failed",
        }
        status = status_map.get(state, state.lower())

        response: dict[str, Any] = {
            "task_id": task_id,
            "status": status,
            "result": None,
            "error": None,
            "progress": None,
        }

        # Extract progress metadata from task state (set via update_state)
        if state in ("STARTED", "RETRY") and result.info:
            info = result.info if isinstance(result.info, dict) else {}
            response["progress"] = {
                "chunks_processed": info.get("chunks_processed", 0),
                "total_chunks": info.get("total_chunks", 0),
            }

        if state == "SUCCESS":
            response["result"] = result.result

        if state == "FAILURE":
            response["error"] = str(result.info)

        if state == "RETRY" and result.info:
            info = result.info if isinstance(result.info, dict) else {}
            response["error"] = info.get("message", str(result.info))

        return response

    def ping(self) -> bool:
        """Check if the Celery broker is reachable."""
        try:
            insp = celery_app.control.inspect(timeout=3)
            active = insp.active_queues()
            return active is not None
        except Exception:
            return False
