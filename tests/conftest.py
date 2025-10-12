"""Pytest configuration file for test suite setup.

This module configures pytest behavior and ensures the project root is added
to sys.path for proper module imports during testing.
"""
from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_root_on_path() -> None:
    """Add the project root directory to sys.path if not already present."""
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def pytest_sessionstart(session):  # type: ignore[override]
    """Pytest hook called at the start of the test session.

    Ensures the project root is on sys.path before tests run.
    """
    ensure_project_root_on_path()
