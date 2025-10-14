"""Environment diagnostics for Inclusive AI Toolkit development workflows."""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Callable, Iterable, Sequence

import transcribe


class DiagnosticError(RuntimeError):
    """Raised when an actionable diagnostic cannot be completed."""


ExecutableLocator = Callable[[str], str | None]


@dataclass(frozen=True)
class DiagnosticResult:
    """Structured representation of a diagnostic outcome."""

    name: str
    status: str
    details: str
    recommendation: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize the result into a JSON-friendly dictionary."""

        payload: dict[str, object] = {
            "name": self.name,
            "status": self.status,
            "details": self.details,
        }
        if self.recommendation:
            payload["recommendation"] = self.recommendation
        return payload


def locate_executable(
    candidates: Iterable[str],
    *,
    which: ExecutableLocator | None = None,
) -> Path | None:
    """Return the first executable found in *candidates* or ``None``.

    The helper searches the system ``PATH`` using :func:`shutil.which` by
    default. A custom *which* callable can be injected to aid unit testing.
    """

    finder = which or shutil.which
    for candidate in candidates:
        resolved = finder(candidate)
        if resolved:
            return Path(resolved)
    return None


def diagnose_python_version(
    *, expected: tuple[int, int] = (3, 12), info: tuple[int, int, int] | None = None
) -> DiagnosticResult:
    """Describe whether the active interpreter matches the required version."""

    current_info = info or sys.version_info[:3]
    current_major, current_minor, current_micro = current_info
    expected_major, expected_minor = expected
    version_text = f"{current_major}.{current_minor}.{current_micro}"
    if (current_major, current_minor) == expected:
        return DiagnosticResult(
            name="python",
            status="available",
            details=f"Interpreter version {version_text} matches Python {expected_major}.{expected_minor}.",
        )
    recommendation = (
        "Activate the project's Python 3.12 virtual environment before running diagnostics."
    )
    return DiagnosticResult(
        name="python",
        status="unavailable",
        details=(
            "Interpreter version "
            f"{version_text} detected; Python {expected_major}.{expected_minor} is required for the toolkit."
        ),
        recommendation=recommendation,
    )


def determine_markitdown_launcher(
    *,
    which: ExecutableLocator | None = None,
    find_spec: Callable[[str], object | None] | None = None,
) -> list[str]:
    """Return a portable command that launches the MarkItDown server.

    The lookup prefers the dedicated ``uvx`` front-end. When ``uvx`` is not
    available it falls back to ``uv tool run`` or ``python -m uv`` provided the
    ``uv`` module is importable. If none of the strategies are viable a
    :class:`DiagnosticError` is raised with remediation guidance.
    """

    finder = which or shutil.which
    spec_finder = find_spec or importlib.util.find_spec

    uvx = locate_executable(("uvx", "uvx.cmd"), which=finder)
    if uvx:
        return [str(uvx), "markitdown"]

    uv = locate_executable(("uv", "uv.exe"), which=finder)
    if uv:
        return [str(uv), "tool", "run", "markitdown"]

    if spec_finder("uv") is not None:
        return [sys.executable, "-m", "uv", "tool", "run", "markitdown"]

    raise DiagnosticError(
        "Unable to locate a command capable of launching MarkItDown. "
        "Install the `uv` package (https://github.com/astral-sh/uv) and ensure "
        "its executables are on PATH. For example: `pip install uv`."
    )


def diagnose_markitdown(
    *,
    which: ExecutableLocator | None = None,
    find_spec: Callable[[str], object | None] | None = None,
) -> DiagnosticResult:
    """Produce a structured diagnostic describing MarkItDown availability."""

    try:
        command = determine_markitdown_launcher(which=which, find_spec=find_spec)
    except DiagnosticError as exc:
        return DiagnosticResult(
            name="markitdown",
            status="unavailable",
            details=str(exc),
            recommendation=(
                "Install `uv` and re-run this diagnostic. On macOS/Linux run "
                "`pip install uv` or install the standalone binary from the uv "
                "documentation. On Windows install uv via winget or pip and "
                "restart your terminal so PATH updates take effect."
            ),
        )

    return DiagnosticResult(
        name="markitdown",
        status="available",
        details=" ".join(command),
        recommendation=(
            "Use the displayed command to start MarkItDown. The helper will "
            "prefer `uvx` when present and falls back to `uv tool run`."
        ),
    )


def diagnose_whisper(
    *, importer: Callable[[str], object] | None = None
) -> DiagnosticResult:
    """Report whether the OpenAI Whisper dependency can be imported."""

    load_module = importer or import_module
    try:
        load_module("whisper")
    except ModuleNotFoundError as exc:
        return DiagnosticResult(
            name="whisper",
            status="unavailable",
            details=f"Import failed: {exc}",
            recommendation=transcribe.MISSING_WHISPER_MESSAGE,
        )

    return DiagnosticResult(
        name="whisper",
        status="available",
        details="Module 'whisper' successfully imported.",
    )


def render_results(results: Sequence[DiagnosticResult]) -> str:
    """Return a human-readable summary of diagnostic *results*."""

    lines = []
    for result in results:
        prefix = "[OK]" if result.status == "available" else "[ERROR]"
        lines.append(f"{prefix} {result.name}: {result.details}")
        if result.recommendation:
            lines.append(f"    → {result.recommendation}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point that prints JSON or text diagnostics based on *argv*."""

    args = list(argv or sys.argv[1:])
    as_json = "--json" in args
    if as_json:
        args.remove("--json")

    if args:
        print(
            "Unexpected arguments supplied. Only the optional '--json' flag is supported.",
            file=sys.stderr,
        )
        return 2

    results = [
        diagnose_python_version(),
        diagnose_markitdown(),
        diagnose_whisper(),
    ]
    if as_json:
        print(json.dumps([result.to_dict() for result in results], indent=2))
    else:
        print(render_results(results))

    return 0 if all(result.status == "available" for result in results) else 1


if __name__ == "__main__":  # pragma: no cover - CLI execution
    raise SystemExit(main())
