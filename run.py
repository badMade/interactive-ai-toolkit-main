#!/usr/bin/env python3
"""Convenience wrapper to launch the Inclusive AI Toolkit CLI."""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Mapping, Sequence

import ffmpeg_support

PROJECT_ROOT = Path(__file__).resolve().parent
SETUP_LOG_RELATIVE_PATH = Path("logs") / "setup_state.json"


def get_project_root() -> Path:
    """Return the directory that contains the repository's entry points."""

    return PROJECT_ROOT


def get_virtualenv_path() -> Path:
    """Return the path where the project's virtual environment resides."""

    return get_project_root() / ".venv"


def get_virtualenv_python_path(venv_path: Path) -> Path:
    """Return the Python interpreter inside the virtual environment."""

    if os.name == "nt":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def get_requirements_script_path() -> Path:
    """Return the helper script responsible for synchronizing dependencies."""

    return get_project_root() / "scripts" / "ensure_requirements.py"


def get_setup_log_path() -> Path:
    """Return the path to the setup log file."""

    return get_project_root() / SETUP_LOG_RELATIVE_PATH


def get_setup_env_script_path() -> Path:
    """Return the path to the setup_env.py script."""

    return get_project_root() / "setup_env.py"


def get_requirements_path() -> Path:
    """Return the path to the requirements.txt file."""

    return get_project_root() / "requirements.txt"


PACKAGE_NAME_ALIASES: Mapping[str, str] = {
    "openai_whisper": "whisper",
}

EXTRA_VALIDATION_MODULES: Mapping[tuple[str, str], tuple[str, ...]] = {
    ("imageio", "ffmpeg"): ("imageio_ffmpeg",),
}


def _canonical_package_name(name: str) -> str:
    """Return a normalized representation of *name* for comparisons."""

    normalized = name.replace("-", "_").strip().lower()
    return PACKAGE_NAME_ALIASES.get(normalized, normalized)


def read_required_packages(requirements_path: Path) -> dict[str, str]:
    """Parse *requirements_path* and return canonical requirement names."""

    if not requirements_path.is_file():
        raise RuntimeError(
            "Could not find requirements.txt. "
            "Please add it before running the CLI."
        )

    cleaned: dict[str, str] = {}
    with open(requirements_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            candidate = stripped.split("#", 1)[0].strip()
            candidate = candidate.split(";", 1)[0].strip()
            if not candidate:
                continue
            extras: tuple[str, ...] = ()
            base = candidate
            for delimiter in ("==", ">=", "<=", "~=", "!=", ">", "<"):
                if delimiter in base:
                    base = base.split(delimiter, 1)[0].strip()
                    break
            if "[" in base and "]" in base:
                package_name, extra_section = base.split("[", 1)
                extra_tokens, _, _ = extra_section.partition("]")
                extras = tuple(
                    token.strip().lower()
                    for token in extra_tokens.split(",")
                    if token.strip()
                )
                base = package_name.strip()
            else:
                base = base.strip()
            if not base:
                continue
            canonical = _canonical_package_name(base)
            cleaned[canonical] = base
            for extra in extras:
                for module in EXTRA_VALIDATION_MODULES.get((canonical, extra),
                                                           ()):
                    cleaned[module] = module
    return cleaned


@dataclass(frozen=True)
class SetupLogStatus:
    """Describe the state of the setup log and unmet requirements."""

    is_valid: bool
    reason: str | None
    missing_requirements: tuple[str, ...]
    data: Mapping[str, object] | None

    @property
    def has_missing_requirements(self) -> bool:
        """
        Return True when there are recorded missing requirements.

        Returns:
            bool: True if the `missing_requirements` collection
            contains any items, otherwise False.
        """
        return bool(self.missing_requirements)


def validate_setup_log(
    log_path: Path, requirements_path: Path
) -> SetupLogStatus:
    """
    Inspect *log_path* and confirm that it satisfies project requirements.
    Args:
        log_path (Path): The path to the setup log file.
        requirements_path (Path): The path to the requirements.txt file.
    Returns:
        SetupLogStatus: An object describing the state of the setup log."""

    required_packages = read_required_packages(requirements_path)
    required_keys = tuple(required_packages.keys())

    if not log_path.is_file():
        return SetupLogStatus(
            is_valid=False,
            reason="setup log not found",
            missing_requirements=tuple(),
            data=None,
        )

    try:
        with open(log_path, "r", encoding="utf-8") as handle:
            log_data = json.load(handle)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        return SetupLogStatus(
            is_valid=False,
            reason=f"malformed setup log: {exc}",
            missing_requirements=tuple(),
            data=None,
        )
    except OSError as exc:
        return SetupLogStatus(
            is_valid=False,
            reason=f"unable to read setup log: {exc}",
            missing_requirements=tuple(),
            data=None,
        )

    if not isinstance(log_data, dict):
        return SetupLogStatus(
            is_valid=False,
            reason="setup log has unexpected structure",
            missing_requirements=tuple(),
            data=None,
        )

    if not log_data.get("setup_completed"):
        return SetupLogStatus(
            is_valid=False,
            reason="setup has not been marked complete",
            missing_requirements=tuple(),
            data=log_data,
        )

    installed_packages = log_data.get("installed_packages")
    if not isinstance(installed_packages, Mapping):
        return SetupLogStatus(
            is_valid=False,
            reason="setup log is missing installed package metadata",
            missing_requirements=tuple(),
            data=log_data,
        )

    canonical_installed = {
        _canonical_package_name(str(package_name))
        for package_name in installed_packages.keys()
    }

    missing_keys = [
        key
        for key in required_keys
        if key and key not in canonical_installed
    ]

    if missing_keys:
        missing_display = tuple(required_packages[key] for key in missing_keys)
        return SetupLogStatus(
            is_valid=False,
            reason="setup log is missing required packages",
            missing_requirements=missing_display,
            data=log_data,
        )

    return SetupLogStatus(
        is_valid=True,
        reason=None,
        missing_requirements=tuple(),
        data=log_data,
    )


def verify_setup_log() -> bool:
    """Return ``True`` when the recorded setup log satisfies requirements."""

    status = validate_setup_log(
        get_setup_log_path(),
        get_requirements_path(),
    )
    return status.is_valid


def run_setup_env(
    *,
    reason: str | None = None,
    missing_requirements: Sequence[str] | None = None,
) -> None:
    """Run setup_env.py to configure the environment."""
    setup_script = get_setup_env_script_path()

    if not setup_script.exists():
        raise RuntimeError(
            f"Setup script not found at {setup_script}. "
            "Cannot configure the environment."
        )

    details: list[str] = []
    if reason:
        details.append(reason)
    if missing_requirements:
        missing = ", ".join(sorted(set(missing_requirements)))
        details.append(f"missing requirements: {missing}")

    if details:
        print(f"Running environment setup ({'; '.join(details)})...")
    else:
        print("Running environment setup...")
    try:
        subprocess.run(
            [sys.executable, str(setup_script)],
            check=True,
            text=True,
            capture_output=False,
        )
    except subprocess.CalledProcessError as exc:
        context = "; ".join(details)
        if context:
            context = f" ({context})"
        raise RuntimeError(
            "Failed to complete environment setup"
            f"{context}. "
            "Please run setup_env.py manually and fix any errors."
        ) from exc


def is_running_inside_virtualenv(venv_path: Path) -> bool:
    """Return ``True`` when the current interpreter belongs to *venv_path*."""

    resolved_venv = venv_path.resolve()
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env and Path(virtual_env).resolve() == resolved_venv:
        return True

    executable_path = Path(sys.executable).resolve()
    try:
        executable_path.relative_to(resolved_venv)
        return True
    except ValueError:
        pass

    prefix_path = Path(sys.prefix).resolve()
    try:
        prefix_path.relative_to(resolved_venv)
        return True
    except ValueError:
        return False


def ensure_virtual_environment() -> None:
    """Guarantee that the CLI executes from the managed virtual environment."""

    venv_path = get_virtualenv_path()
    if is_running_inside_virtualenv(venv_path):
        return

    requirements_path = get_requirements_path()
    setup_log_path = get_setup_log_path()
    status = validate_setup_log(setup_log_path, requirements_path)
    if not status.is_valid:
        run_setup_env(
            reason=status.reason,
            missing_requirements=status.missing_requirements
        )
        status = validate_setup_log(setup_log_path, requirements_path)
        if not status.is_valid:
            details: list[str] = []
            if status.reason:
                details.append(status.reason)
            if status.missing_requirements:
                missing = ", ".join(status.missing_requirements)
                details.append(f"missing requirements: {missing}")
            detail_text = "; ".join(details)
            if detail_text:
                detail_text = f" ({detail_text})"
            raise RuntimeError(
                "Environment setup is incomplete "
                "even after running setup_env.py"
                f"{detail_text}."
            )

    ensure_script = get_requirements_script_path()
    if not ensure_script.is_file():
        raise RuntimeError(
            "Missing helper script to synchronize project dependencies: "
            f"{ensure_script}"
        )

    try:
        subprocess.run(
            [sys.executable, str(ensure_script)],
            check=True,
            text=True,
            capture_output=False,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to prepare the project's virtual environment."
        ) from exc

    venv_python = get_virtualenv_python_path(venv_path)
    if not venv_python.exists():
        raise RuntimeError(
            "The virtual environment does not expose a Python interpreter. "
            "Re-run scripts/ensure_requirements.py and try again."
        )

    command = [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]]
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv_path)
    bin_dir = venv_python.parent
    path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join([str(bin_dir),
                                   path]) if path else str(bin_dir)

    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=False,
            env=env,
        )
    except FileNotFoundError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(
            "Failed to launch the project's virtual environment interpreter."
        ) from exc

    raise SystemExit(result.returncode)


def _missing_whisper_message(exc: ModuleNotFoundError) -> str | None:
    """Return a user-facing message when Whisper is not installed."""

    message = str(exc).strip()
    current: BaseException | None = exc
    while current is not None:
        is_missing_whisper = (
            isinstance(current, ModuleNotFoundError)
            and getattr(current, "name", None) == "whisper"
        )
        if is_missing_whisper:
            return message or str(current).strip()
        current = current.__cause__
    return None


def load_transcribe_module() -> ModuleType:
    """Safely load the ``transcribe`` module that powers the CLI."""
    try:
        return importlib.import_module("transcribe")
    except ModuleNotFoundError as exc:
        whisper_message = _missing_whisper_message(exc)
        if whisper_message is not None:
            print(whisper_message, file=sys.stderr)
            raise SystemExit(1) from None
        if getattr(exc, "name", None) == "transcribe":
            raise RuntimeError(
                "Could not import the 'transcribe' module required to "
                "launch the program."
            ) from exc
        raise


def main() -> None:
    """Import and execute the project's primary command-line entry point."""

    ensure_virtual_environment()
    try:
        ffmpeg_support.ensure_ffmpeg_available()
    except ffmpeg_support.FFmpegInstallationError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
    module = load_transcribe_module()
    try:
        launch = module.main
    except AttributeError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(
            "The 'transcribe' module does not expose a callable 'main'."
        ) from exc
    launch()


if __name__ == "__main__":
    main()
