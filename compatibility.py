"""Runtime compatibility helpers shared across the toolkit."""
from __future__ import annotations

import importlib
import re
from types import ModuleType

NUMPY_PINNED_SPEC = "numpy<2"
NUMPY_REINSTALL_COMMAND = 'python -m pip install "numpy<2"'


class NumpyCompatibilityError(RuntimeError):
    """Raised when the installed NumPy version is unsupported."""


def validate_numpy_version(version: str) -> str:
    """Ensure *version* satisfies the toolkit's NumPy constraints.

    Args:
        version: Raw version string retrieved from the NumPy module.

    Returns:
        str: Normalized version string when it meets the compatibility policy.

    Raises:
        NumpyCompatibilityError: If the version cannot be parsed or violates
        the toolkit's compatibility requirements.
    """

    normalized = str(version or "").strip()
    if not normalized:
        raise NumpyCompatibilityError(
            "Unable to determine the installed NumPy version. "
            f"Reinstall NumPy with `{NUMPY_REINSTALL_COMMAND}` and retry."
        )

    match = re.match(r"(\d+)\.(\d+)", normalized)
    if match is None:
        raise NumpyCompatibilityError(
            "Could not parse the NumPy version string "
            f"'{version}'. Reinstall NumPy with `{NUMPY_REINSTALL_COMMAND}`."
        )

    major = int(match.group(1))
    if major >= 2:
        raise NumpyCompatibilityError(
            "Detected NumPy version "
            f"{normalized}. The toolkit requires {NUMPY_PINNED_SPEC} "
            "for compatibility. "
            f"Run `{NUMPY_REINSTALL_COMMAND}` to install a supported build."
        )

    return normalized


def ensure_numpy_compatible(module: ModuleType | None = None) -> ModuleType:
    """Import NumPy and validate that it satisfies project requirements.

    Args:
        module: Optional pre-imported NumPy module. When ``None``, NumPy will
            be imported lazily.

    Returns:
        ModuleType: The validated NumPy module instance.

    Raises:
        NumpyCompatibilityError: If NumPy is missing or incompatible.
    """

    try:
        numpy_module = module or importlib.import_module("numpy")
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive guard
        raise NumpyCompatibilityError(
            "NumPy is not installed. "
            f"Install it with `{NUMPY_REINSTALL_COMMAND}`."
        ) from exc

    validate_numpy_version(getattr(numpy_module, "__version__", ""))
    return numpy_module
