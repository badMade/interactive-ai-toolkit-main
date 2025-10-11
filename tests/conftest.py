from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def pytest_sessionstart(session):  # type: ignore[override]
    ensure_project_root_on_path()
