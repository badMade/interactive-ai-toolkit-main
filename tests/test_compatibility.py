"""Unit tests for runtime compatibility helpers."""
from __future__ import annotations

import types

import pytest

from compatibility import (
    NumpyCompatibilityError,
    ensure_numpy_compatible,
    validate_numpy_version,
)


def test_validate_numpy_version_accepts_major_one() -> None:
    """Versions from the 1.x series should pass validation."""

    assert validate_numpy_version("1.26.4") == "1.26.4"


def test_validate_numpy_version_rejects_missing() -> None:
    """An empty version string should trigger a compatibility error."""

    with pytest.raises(NumpyCompatibilityError):
        validate_numpy_version("")


def test_validate_numpy_version_rejects_major_two() -> None:
    """Major version two must be rejected to keep binary compatibility."""

    with pytest.raises(NumpyCompatibilityError):
        validate_numpy_version("2.3.3")


def test_ensure_numpy_compatible_uses_provided_module() -> None:
    """A provided module object should be returned unchanged."""

    module = types.SimpleNamespace(__version__="1.24.0")
    assert ensure_numpy_compatible(module) is module
