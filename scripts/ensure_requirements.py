#!/usr/bin/env python3
"""Ensure the project's virtual environment contains every dependency."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Protocol, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VENV_PATH = PROJECT_ROOT / ".venv"
DEFAULT_REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"

NUMPY_PINNED_SPEC = "numpy<2"

REQUIRED_PYTHON_VERSION = (3, 12)


class EnvironmentProvisioningError(RuntimeError):
    """Raised when the virtual environment cannot be synchronized."""


class CommandRunner(Protocol):
    """Protocol that matches :func:`subprocess.run` for dependency injection."""

    def __call__(
        self,
        args: Sequence[str],
        *,
        check: bool,
        text: bool,
        capture_output: bool,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        ...


def run_command(
    args: Sequence[str],
    *,
    capture_output: bool = False,
    env: dict[str, str] | None = None,
    runner: CommandRunner | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute *args* and raise :class:`EnvironmentProvisioningError` on failure."""

    active_runner = runner or subprocess.run
    try:
        return active_runner(
            list(args),
            check=True,
            text=True,
            capture_output=capture_output,
            env=env,
        )
    except FileNotFoundError as error:  # pragma: no cover - defensive guard
        raise EnvironmentProvisioningError(
            f"Command not found: {args[0]}"
        ) from error
    except subprocess.CalledProcessError as error:
        message = f"Command failed with exit code {error.returncode}: {' '.join(args)}"
        if capture_output:
            stderr = (error.stderr or "").strip()
            if stderr:
                message = f"{message}\n{stderr}"
        raise EnvironmentProvisioningError(message) from error


def ensure_requirements_file(path: Path) -> Path:
    """Ensure that a requirements file exists before continuing."""

    if not path.is_file():
        raise EnvironmentProvisioningError(
            f"Could not find a requirements file at {path}."
        )
    return path


_PYTHON_VERSION_DETECTION_CODE = "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"


def _python_command_matches_requirement(
    command: Sequence[str], *, runner: CommandRunner | None = None
) -> bool:
    """Return ``True`` when *command* launches Python 3.12."""

    try:
        result = run_command(
            [*command, "-c", _PYTHON_VERSION_DETECTION_CODE],
            capture_output=True,
            runner=runner,
        )
    except EnvironmentProvisioningError:
        return False
    version = result.stdout.strip()
    return version == f"{REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]}"


def resolve_python_command(
    *, runner: CommandRunner | None = None
) -> Sequence[str]:
    """Return a command that launches the required Python interpreter."""

    if (sys.version_info.major, sys.version_info.minor) == REQUIRED_PYTHON_VERSION:
        return (sys.executable,)

    candidate_commands: list[Sequence[str]] = []
    if os.name == "nt":
        candidate_commands.append(
            ("py", f"-{REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]}")
        )
    candidate_commands.extend(
        (
            ("python3.12",),
            ("python3",),
            ("python",),
        )
    )

    for command in candidate_commands:
        if _python_command_matches_requirement(command, runner=runner):
            return command

    raise EnvironmentProvisioningError(
        "Python 3.12 is required but was not found on the system PATH. "
        "Install Python 3.12 and retry."
    )


def _read_pyvenv_version(venv_path: Path) -> tuple[int, int] | None:
    """Return the interpreter version recorded in ``pyvenv.cfg`` if present."""

    config_path = venv_path / "pyvenv.cfg"
    if not config_path.exists():
        return None
    try:
        for line in config_path.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("version"):
                _, _, value = line.partition("=")
                cleaned = value.strip()
                parts = cleaned.split(".")
                if len(parts) < 2:
                    return None
                return int(parts[0]), int(parts[1])
    except (OSError, ValueError):
        return None
    return None


def _virtualenv_uses_required_python(
    venv_path: Path, *, runner: CommandRunner | None = None
) -> bool:
    """Return ``True`` when the virtual environment uses Python 3.12."""

    recorded_version = _read_pyvenv_version(venv_path)
    if recorded_version == REQUIRED_PYTHON_VERSION:
        return True
    if recorded_version and recorded_version != REQUIRED_PYTHON_VERSION:
        return False
    try:
        venv_python = get_virtualenv_python_path(venv_path)
    except EnvironmentProvisioningError:
        return False
    try:
        result = run_command(
            [str(venv_python), "-c", _PYTHON_VERSION_DETECTION_CODE],
            capture_output=True,
            runner=runner,
        )
    except EnvironmentProvisioningError:
        return False
    version = result.stdout.strip()
    return version == f"{REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]}"


def ensure_virtualenv_exists(
    venv_path: Path,
    *,
    python_command: Sequence[str] | None = None,
    python_executable: str | None = None,
    runner: CommandRunner | None = None,
) -> None:
    """Create a virtual environment when one is not present or outdated."""

    if python_command and python_executable:
        raise EnvironmentProvisioningError(
            "Specify only one of 'python_command' or 'python_executable'."
        )
    if python_executable:
        python_command = (python_executable,)
    python_command = python_command or resolve_python_command(runner=runner)
    pyvenv_cfg = venv_path / "pyvenv.cfg"
    if pyvenv_cfg.exists() and _virtualenv_uses_required_python(
        venv_path, runner=runner
    ):
        return
    if pyvenv_cfg.exists():
        shutil.rmtree(venv_path)
    venv_path.parent.mkdir(parents=True, exist_ok=True)
    run_command([
        *python_command,
        "-m",
        "venv",
        str(venv_path),
    ], runner=runner)


def get_virtualenv_python_path(venv_path: Path) -> Path:
    """Return the Python interpreter located inside *venv_path*."""

    if os.name == "nt":
        python_path = venv_path / "Scripts" / "python.exe"
    else:
        python_path = venv_path / "bin" / "python"
    if not python_path.exists():
        raise EnvironmentProvisioningError(
            "The virtual environment is missing its Python executable."
        )
    return python_path


def collect_requirements(requirements_path: Path) -> list[str]:
    """Return a clean list of requirements from *requirements_path*."""

    requirements: list[str] = []
    for line in requirements_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        requirements.append(stripped)
    return requirements


def ensure_numpy_requirement(requirements: Sequence[str]) -> None:
    """Ensure that *requirements* pins NumPy below major version two."""

    normalized_requirements = [
        requirement.lower().replace(" ", "") for requirement in requirements
    ]
    numpy_entries = [
        requirement for requirement in normalized_requirements if requirement.startswith("numpy")
    ]
    if not numpy_entries:
        raise EnvironmentProvisioningError(
            "The requirements file must include the dependency "
            f"'{NUMPY_PINNED_SPEC}' to keep compatibility with PyTorch and Whisper."
        )
    for requirement in numpy_entries:
        if "<2" not in requirement:
            raise EnvironmentProvisioningError(
                "The NumPy dependency must be pinned below version 2.0. "
                "Update requirements.txt to include 'numpy<2' and rerun the setup."
            )


_CHECK_REQUIREMENT_CODE = """
import sys
try:
    from pkg_resources import DistributionNotFound, VersionConflict, require
except Exception:  # pragma: no cover - handled by caller
    sys.exit(2)
requirement = sys.argv[1]
try:
    require(requirement)
except (DistributionNotFound, VersionConflict):
    sys.exit(1)
sys.exit(0)
"""


def _run_pkg_resources_check(
    venv_python: Path,
    *,
    runner: CommandRunner | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute a lightweight probe for :mod:`pkg_resources`."""

    active_runner = runner or subprocess.run
    try:
        return active_runner(
            [str(venv_python), "-c", "import pkg_resources"],
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as error:  # pragma: no cover - defensive guard
        raise EnvironmentProvisioningError(
            f"The Python interpreter at {venv_python} could not be executed."
        ) from error


def ensure_setuptools(
    venv_python: Path,
    *,
    venv_path: Path,
    runner: CommandRunner | None = None,
) -> None:
    """Install :mod:`setuptools` when it is not present in the virtualenv."""

    result = _run_pkg_resources_check(venv_python, runner=runner)
    if result.returncode == 0:
        return

    env = build_venv_environment(venv_path)
    run_command(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
        ],
        runner=runner,
        env=env,
    )

    result = _run_pkg_resources_check(venv_python, runner=runner)
    if result.returncode != 0:
        raise EnvironmentProvisioningError(
            "Failed to install the 'setuptools' package in the virtual environment."
        )


def is_requirement_satisfied(
    venv_python: Path,
    requirement: str,
    *,
    runner: CommandRunner | None = None,
) -> bool:
    """Return ``True`` when *requirement* is present in the environment."""

    active_runner = runner or subprocess.run
    try:
        result = active_runner(
            [str(venv_python), "-c", _CHECK_REQUIREMENT_CODE, requirement],
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as error:  # pragma: no cover - defensive guard
        raise EnvironmentProvisioningError(
            f"The Python interpreter at {venv_python} could not be executed."
        ) from error
    if result.returncode == 0:
        return True
    if result.returncode == 2:
        raise EnvironmentProvisioningError(
            "The virtual environment is missing the 'setuptools' package."
        )
    return False


def missing_requirements(
    venv_python: Path,
    requirements: Iterable[str],
    *,
    runner: CommandRunner | None = None,
) -> list[str]:
    """Return requirements that are not currently satisfied."""

    missing: list[str] = []
    for requirement in requirements:
        if not is_requirement_satisfied(venv_python, requirement, runner=runner):
            missing.append(requirement)
    return missing


def build_venv_environment(venv_path: Path) -> dict[str, str]:
    """Construct environment variables for executing inside the venv."""

    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv_path)
    bin_dir = venv_path / ("Scripts" if os.name == "nt" else "bin")
    path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join([str(bin_dir), path]) if path else str(bin_dir)
    return env


def install_requirements(
    venv_python: Path,
    requirements_path: Path,
    *,
    venv_path: Path,
    runner: CommandRunner | None = None,
) -> None:
    """Install requirements into the virtual environment."""

    env = build_venv_environment(venv_path)
    run_command(
        [str(venv_python), "-m", "pip", "install", "-r", str(requirements_path)],
        runner=runner,
        env=env,
    )


def synchronize_environment(
    requirements_path: Path = DEFAULT_REQUIREMENTS_FILE,
    venv_path: Path = DEFAULT_VENV_PATH,
    *,
    runner: CommandRunner | None = None,
) -> list[str]:
    """Ensure *requirements_path* are installed in *venv_path*.

    Returns a list of dependencies that were missing prior to synchronization.
    """

    ensure_requirements_file(requirements_path)
    ensure_virtualenv_exists(venv_path, runner=runner)
    venv_python = get_virtualenv_python_path(venv_path)
    ensure_setuptools(venv_python, venv_path=venv_path, runner=runner)
    requirements = collect_requirements(requirements_path)
    ensure_numpy_requirement(requirements)
    if not requirements:
        return []
    missing_before = missing_requirements(
        venv_python,
        requirements,
        runner=runner,
    )
    if not missing_before:
        return []
    install_requirements(
        venv_python,
        requirements_path,
        venv_path=venv_path,
        runner=runner,
    )
    missing_after = missing_requirements(
        venv_python,
        requirements,
        runner=runner,
    )
    if missing_after:
        raise EnvironmentProvisioningError(
            "Unable to install the following dependencies: "
            + ", ".join(sorted(missing_after))
        )
    return missing_before


def main() -> int:
    """CLI entry point for synchronizing the virtual environment."""

    try:
        missing = synchronize_environment()
    except EnvironmentProvisioningError as error:
        print(f"Failed to ensure project requirements: {error}", file=sys.stderr)
        return 1
    if missing:
        print("Installed missing packages: " + ", ".join(missing))
    else:
        print("All requirements already satisfied in the virtual environment.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI behavior
    sys.exit(main())
