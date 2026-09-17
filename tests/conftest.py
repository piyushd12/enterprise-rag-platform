"""Shared pytest fixtures."""

from __future__ import annotations

import logfire
import pytest


@pytest.fixture(autouse=True, scope="session")
def _configure_logfire_for_tests():
    """Configure Logfire in local-only mode so tests never try to reach
    the real backend and never emit LogfireNotConfiguredWarning noise."""
    logfire.configure(send_to_logfire=False)
