"""
Celery application factory.

Creates a Celery instance configured with Redis as both broker and result
backend. This module is the entry point for Celery worker discovery:

    celery -A rag_app.queue.celery_app worker

The worker uses environment variables (set via .env / pydantic-settings)
for Redis URL configuration.
"""

from celery import Celery

from rag_app.config.settings import settings

celery_app = Celery(
    "rag_app",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

# Celery configuration
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Task execution
    task_acks_late=True,                 # Re-deliver if worker crashes mid-task
    task_reject_on_worker_lost=True,     # Ensure task gets re-queued on crash
    task_track_started=True,             # Track PENDING → STARTED transition

    # Retry defaults (overridable per-task)
    task_default_retry_delay=5,          # 5 seconds initial
    task_max_retries=3,

    # Task discovery: scan for task modules
    include=["rag_app.ingestion.tasks", "rag_app.evaluation.tasks"],

    # Result backend settings
    result_expires=3600,                 # Results expire after 1 hour

    # Worker settings
    worker_prefetch_multiplier=1,        # One task at a time for accuracy
    # Default concurrency (one forked process per CPU core) spawns a
    # separate FastEmbed model instance per process; on memory-constrained
    # hosts, running several large ingestion/evaluation tasks in parallel
    # this way exhausts RAM and gets processes SIGKILLed by the OOM killer.
    # Ingestion and evaluation are CPU/memory-heavy, not I/O-bound, so high
    # concurrency buys little anyway -- keep it modest.
    worker_concurrency=2,
)
