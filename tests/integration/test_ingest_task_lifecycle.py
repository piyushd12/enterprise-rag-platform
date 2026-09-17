"""Integration test for the /ingest endpoint's full async task lifecycle.

Exercises the real FastAPI route, real Celery broker, and a real worker
process, so it requires the full local stack to be up (`make infra` +
`make worker`); skips itself if the broker isn't reachable. Cleans up the
document it ingests from the real vector store afterward so it never
pollutes the app's actual corpus.
"""

from __future__ import annotations

import hashlib
import time

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

from rag_app.config.settings import settings
from rag_app.main import app
from rag_app.queue.celery_app import celery_app


def _broker_reachable() -> bool:
    try:
        insp = celery_app.control.inspect(timeout=2)
        return insp.active_queues() is not None
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _broker_reachable(), reason="Celery broker/worker is not reachable"
)


def test_ingest_task_reaches_success_via_status_polling():
    client = TestClient(app)
    content = b"Integration test fixture document for /ingest lifecycle polling."
    source_id = hashlib.sha256(content).hexdigest()[:16]

    try:
        response = client.post(
            "/ingest", files={"file": ("lifecycle_test.txt", content, "text/plain")}
        )
        assert response.status_code == 200
        task_id = response.json()["task_id"]

        seen_statuses = set()
        final_status = None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            status_response = client.get(f"/ingest/status/{task_id}")
            assert status_response.status_code == 200
            body = status_response.json()
            seen_statuses.add(body["status"])
            if body["status"] in ("succeeded", "failed"):
                final_status = body
                break
            time.sleep(1)

        assert final_status is not None, f"task did not finish in time; saw {seen_statuses}"
        assert final_status["status"] == "succeeded", final_status
        assert final_status["result"]["chunk_count"] >= 1
    finally:
        # Never leave the test document in the real corpus.
        QdrantClient(url=settings.qdrant_url).delete(
            collection_name=settings.qdrant_collection_name,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))])
            ),
        )
