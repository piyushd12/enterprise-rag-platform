"""
ObservabilityProvider — thin wrapper so call sites never import
LangSmith or Logfire SDKs directly.

This concrete implementation uses Logfire for infrastructure events
and spans. LangSmith is handled natively by LangChain/LangGraph via
env-var-based auto-tracing and is NOT called through this wrapper.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Generator

import logfire


class ObservabilityProvider:
    """Thin observability facade over Logfire.

    All infrastructure logging (cache, queue, fallback, HTTP) goes through
    this class. LLM pipeline tracing goes through LangSmith natively.
    """

    def log_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Log a discrete event (e.g. cache hit, fallback triggered)."""
        logfire.info(name, **(attributes or {}))

    def log_warning(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Log a warning-level event."""
        logfire.warn(name, **(attributes or {}))

    def log_error(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Log an error-level event."""
        logfire.error(name, **(attributes or {}))

    @contextmanager
    def span(
        self, name: str, attributes: dict[str, Any] | None = None
    ) -> Generator[dict[str, Any], None, None]:
        """Create a timed span for a block of work.

        Yields a mutable dict that callers can enrich with additional
        attributes before the span closes.

        Usage:
            with obs.span("ingestion.embed", {"chunk_count": 10}) as span_data:
                # do work
                span_data["embedding_time_ms"] = 42.0
        """
        span_data: dict[str, Any] = dict(attributes or {})
        start = time.perf_counter()

        with logfire.span(name, **span_data):
            try:
                yield span_data
            finally:
                span_data["duration_ms"] = round(
                    (time.perf_counter() - start) * 1000, 2
                )
