"""
Utility script to provision a dedicated Python environment for the project.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Protocol

from logging.handlers import RotatingFileHandler

from run import SETUP_LOG_RELATIVE_PATH, read_required_packages
from shared_messages import MISSING_WHISPER_MESSAGE


class CommandRunner(Protocol):
    """Protocol to allow dependency injection for subprocess execution."""

    def __call__(
        self,
        command: list[str],
        *,
        check: bool,
        text: bool,
        capture_output: bool,
    ) -> subprocess.CompletedProcess[str]:
        ...


class SetupError(RuntimeError):
    """Raised when the environment setup cannot be completed."""


REQUIRED_PYTHON_VERSION = (3, 12)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "setup_env.log"
LOG_PATH_ENV_VAR = "SETUP_ENV_LOG_PATH"

_PYTHON_VERSION_DETECTION_CODE = (
    "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
)


def _read_pyvenv_version(venv_path: Path) -> tuple[int, int] | None:
    """Return the Python version recorded in ``pyvenv.cfg`` if available."""

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


NUMPY_PINNED_SPEC = "numpy==1.26.4"


FIX_ENV_GUIDANCE = (
    "On macOS x86_64 hosts you can run './fix_env.sh' to recreate the environment automatically."
)


DEFAULT_VALIDATION_PACKAGES: tuple[str, ...] = (
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "soundfile",
    "whisper",
    "sentencepiece",
    "numpy",
    "pytest",
)


def ensure_supported_python() -> None:
    """Fail fast when running on an unsupported Python interpreter."""

    current = sys.version_info
    required_major, required_minor = REQUIRED_PYTHON_VERSION
    if (current.major, current.minor) < (required_major, required_minor):
        raise SetupError(
            "Python 3.12 or newer is required. "
            f"Detected {current.major}.{current.minor}.{current.micro}. "
            "Please install Python 3.12 and re-run the setup. "
            f"{FIX_ENV_GUIDANCE}"
        )


ANSI_COLORS = {
    "green": "\033[92m",
    "red": "\033[91m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "reset": "\033[0m",
}


def configure_logging(log_path: Path | None = None) -> Path:
    """Configure application logging with both console and file handlers."""

    if log_path is None:
        env_override = os.getenv(LOG_PATH_ENV_VAR)
        if env_override:
            candidate = Path(env_override).expanduser()
            if not candidate.is_absolute():
                candidate = (PROJECT_ROOT / candidate).resolve()
        else:
            candidate = DEFAULT_LOG_PATH
    else:
        candidate = Path(log_path).expanduser()
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()

    candidate.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)

    file_handler = RotatingFileHandler(
        candidate,
        maxBytes=1_048_576,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return candidate


def colorize(message: str, color: str) -> str:
    """Wrap *message* in ANSI color codes when running in a POSIX terminal."""
    if color not in ANSI_COLORS:
        return message
    return f"{ANSI_COLORS[color]}{message}{ANSI_COLORS['reset']}"


def run_command(
    command: list[str],
    *,
    capture_output: bool = False,
    runner: CommandRunner | None = None,
) -> str:
    """Execute *command* and return captured output when requested."""
    active_runner = runner or subprocess.run
    try:
        result = active_runner(
            command,
            check=True,
            text=True,
            capture_output=capture_output,
        )
    except FileNotFoundError as exc:
        missing = command[0] if command else "command"
        raise SetupError(f"Required command not found: {missing}") from exc
    except subprocess.CalledProcessError as exc:
        details = []
        if capture_output:
            stdout = getattr(exc, "stdout", "") or getattr(exc, "output", "")
            stderr = getattr(exc, "stderr", "")
            combined = f"{stdout}{stderr}".strip()
            if combined:
                details.append(combined)
        message = f"Command failed: {' '.join(command)}"
        if details:
            message = f"{message}\n{details[0]}"
        raise SetupError(message) from exc
    if capture_output:
        output = (result.stdout or "") + (result.stderr or "")
        return output.strip()
    return ""


def _python_command_matches_requirement(
    command: Iterable[str], *, runner: CommandRunner | None = None
) -> bool:
    """Return ``True`` when *command* launches Python 3.12."""

    try:
        version = run_command(
            [*command, "-c", _PYTHON_VERSION_DETECTION_CODE],
            capture_output=True,
            runner=runner,
        )
    except SetupError:
        return False
    return version.strip() == f"{REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]}"


def resolve_python_command(
    *, runner: CommandRunner | None = None
) -> tuple[str, ...]:
    """Return a command that launches the required Python interpreter."""

    if (sys.version_info.major, sys.version_info.minor) == REQUIRED_PYTHON_VERSION:
        return (sys.executable,)

    candidates: list[tuple[str, ...]] = []
    if os.name == "nt":
        candidates.append(
            ("py", f"-{REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]}")
        )
    candidates.extend((("python3.12",), ("python3",), ("python",)))

    for command in candidates:
        if _python_command_matches_requirement(command, runner=runner):
            return command

    raise SetupError(
        "Python 3.12 is required but could not be located. "
        "Install Python 3.12 and ensure it is on PATH."
    )


def _virtualenv_uses_required_python(
    venv_path: Path, *, runner: CommandRunner | None = None
) -> bool:
    """Return ``True`` when the virtual environment already targets Python 3.12."""

    recorded_version = _read_pyvenv_version(venv_path)
    if recorded_version == REQUIRED_PYTHON_VERSION:
        return True
    if recorded_version and recorded_version != REQUIRED_PYTHON_VERSION:
        return False
    venv_python = get_venv_python_path(venv_path)
    if not venv_python.exists():
        return False
    try:
        version = run_command(
            [str(venv_python), "-c", _PYTHON_VERSION_DETECTION_CODE],
            capture_output=True,
            runner=runner,
        )
    except SetupError:
        return False
    return version.strip() == f"{REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]}"


def ensure_ffmpeg_available(*, runner: CommandRunner | None = None) -> None:
    """Verify that FFmpeg is available on the host system."""

    def _probe() -> str:
        return run_command(
            ["ffmpeg", "-version"],
            capture_output=True,
            runner=runner,
        )

    try:
        version_output = _probe()
    except SetupError as error:
        platform = sys.platform.lower()
        is_windows = os.name == "nt" or platform.startswith("win")
        if is_windows:
            guidance = (
                "Install FFmpeg from https://ffmpeg.org/download.html or "
                "run `winget install Gyan.FFmpeg`."
            )
            raise SetupError(
                "FFmpeg is required but was not detected. "
                f"{guidance}"
            ) from error

        if os.name != "posix":
            raise SetupError(
                "FFmpeg is required but automatic installation is only supported "
                "on POSIX-compliant systems. Install FFmpeg manually from "
                "https://ffmpeg.org/download.html."
            ) from error

        install_script = PROJECT_ROOT / "scripts" / "install_ffmpeg.sh"
        try:
            run_command([str(install_script)], capture_output=True, runner=runner)
        except SetupError as install_error:
            raise SetupError(
                "FFmpeg is required but automatic installation failed.\n"
                f"{install_error}"
            ) from install_error

        version_output = _probe()

    if version_output:
        first_line = version_output.splitlines()[0]
        logging.info("FFmpeg detected: %s", first_line)


def ensure_requirements_file(requirements_path: Path) -> Path:
    """Ensure that *requirements_path* exists before continuing."""
    if not requirements_path.is_file():
        raise SetupError(
            "Could not find requirements.txt. "
            "Please add it before running setup. "
            f"{FIX_ENV_GUIDANCE}"
        )
    logging.info("Using requirements file at %s", requirements_path)
    return requirements_path


def create_virtualenv(
    venv_path: Path,
    *,
    runner: CommandRunner | None = None,
) -> None:
    """Create a virtual environment unless one already exists."""
    python_command = resolve_python_command(runner=runner)
    pyvenv_cfg = venv_path / "pyvenv.cfg"
    if pyvenv_cfg.exists():
        if _virtualenv_uses_required_python(venv_path, runner=runner):
            logging.info(
                "Virtual environment already exists at %s with Python %s.%s",
                venv_path,
                *REQUIRED_PYTHON_VERSION,
            )
            return
        logging.info(
            "Removing virtual environment at %s because it is not using Python %s.%s",
            venv_path,
            *REQUIRED_PYTHON_VERSION,
        )
        shutil.rmtree(venv_path)
    logging.info("Creating virtual environment at %s", venv_path)
    venv_path.parent.mkdir(parents=True, exist_ok=True)
    run_command([*python_command, "-m", "venv", str(venv_path)], runner=runner)


def get_venv_python_path(venv_path: Path) -> Path:
    """Return the Python executable inside the virtual environment."""
    if os.name == "nt":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def install_requirements(
    venv_python: Path,
    requirements_path: Path,
    *,
    runner: CommandRunner | None = None,
) -> None:
    """Install project dependencies into the virtual environment."""
    logging.info("Upgrading pip inside the virtual environment")
    run_command([str(venv_python),
                 "-m",
                 "pip",
                 "install",
                 "--upgrade",
                 "pip"], runner=runner)
    logging.info("Installing dependencies from requirements.txt")
    run_command(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "-r",
            str(requirements_path),
        ],
        runner=runner,
    )


def verify_installation(
    venv_python: Path,
    packages: Iterable[str],
    *,
    runner: CommandRunner | None = None,
) -> dict[str, str]:
    """
    Validate the new environment by reporting toolchain and package versions.

    Returns:
        dict: Dictionary mapping package names to their installed versions.
    """
    python_version = run_command(
        [str(venv_python), "--version"], capture_output=True, runner=runner
    )
    logging.info("Python interpreter: %s", python_version)

    pip_version = run_command(
        [str(venv_python), "-m", "pip", "--version"],
        capture_output=True,
        runner=runner,
    )
    logging.info("pip version: %s", pip_version)

    installed_packages = {}
    for package in packages:
        code = (
            "import importlib; "
            f"module = importlib.import_module('{package}'); "
            "version = getattr(module, '__version__', 'installed'); "
            "print(version)"
        )
        try:
            version = run_command(
                [str(venv_python), "-c", code],
                capture_output=True,
                runner=runner,
            )
        except (SetupError, ModuleNotFoundError) as exc:
            if package.lower() == "whisper":
                guidance = MISSING_WHISPER_MESSAGE
                details = str(exc)
                if details and guidance not in details:
                    guidance = f"{guidance}\nOriginal error: {details}"
                raise SetupError(guidance) from exc
            raise
        if package.lower() == "numpy":
            _ensure_supported_numpy(version)
        logging.info("Verified %s %s", package, version or "installed")
        installed_packages[package] = version or "installed"

    summary = {
        "python": python_version,
        "pip": pip_version,
        "packages": installed_packages,
    }
    logging.getLogger(__name__).info(
        "INSTALLATION_SUMMARY %s", json.dumps(summary, sort_keys=True)
    )

    return installed_packages


def _ensure_supported_numpy(version: str) -> None:
    """Validate that the NumPy version satisfies the project's constraints."""

    normalized = version.strip()
    if not normalized:
        raise SetupError(
            "Unable to determine the installed NumPy version. "
            "Reinstall NumPy with `python -m pip install numpy==1.26.4` and retry."
        )
    match = re.match(r"(\d+)\.(\d+)", normalized)
    if not match:
        raise SetupError(
            "Could not parse the NumPy version string "
            f"'{version}'. Reinstall NumPy with `python -m pip install numpy==1.26.4`."
        )
    major = int(match.group(1))
    if major >= 2:
        raise SetupError(
            "Detected NumPy version "
            f"{normalized}. The toolkit requires {NUMPY_PINNED_SPEC} for compatibility. "
            "Run `python -m pip install numpy==1.26.4` inside the virtual environment."
        )


def write_setup_log(
    log_path: Path,
    installed_packages: dict[str, str],
    requirements_path: Path,
) -> None:
    """Write a log file documenting the successful setup.

    Args:
        log_path: Path where the setup log should be written.
        installed_packages: Dictionary of package names to versions.
        requirements_path: Path to the requirements.txt file used.
    """
    log_data = {
        "setup_completed": True,
        "timestamp": datetime.now().isoformat(),
        "requirements_file": str(requirements_path),
        "installed_packages": installed_packages,
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

    logging.info("Setup log written to %s", log_path)


def main() -> int:
    """Entry point used when executing the setup script from the CLI."""
    ensure_supported_python()
    log_path = configure_logging()
    logging.info("Logging setup output to %s", log_path)
    project_root = PROJECT_ROOT
    venv_path = project_root / ".venv"
    requirements_path = project_root / "requirements.txt"
    setup_log_path = project_root / SETUP_LOG_RELATIVE_PATH

    try:
        ensure_requirements_file(requirements_path)
        ensure_ffmpeg_available()
        create_virtualenv(venv_path)
        venv_python = get_venv_python_path(venv_path)
        install_requirements(venv_python, requirements_path)
        required_packages = read_required_packages(requirements_path)
        packages_to_verify: tuple[str, ...]
        if required_packages:
            packages_to_verify = tuple(required_packages.keys())
        else:
            packages_to_verify = DEFAULT_VALIDATION_PACKAGES

        installed_packages = verify_installation(
            venv_python,
            packages=packages_to_verify,
        )
        write_setup_log(setup_log_path, installed_packages, requirements_path)
    except SetupError as error:
        logging.error(colorize(f"Environment setup failed: {error}", "red"))
        print(colorize("Environment setup failed. "
                       "See log output above for details.", "red"))
        if sys.platform == "darwin":
            print(
                colorize(
                    "Hint: run './fix_env.sh' to recreate the virtual environment on macOS x86_64.",
                    "yellow",
                )
            )
        return 1

    print(colorize("Environment setup completed successfully!", "green"))
    print("Activate the virtual environment with:")
    print(colorize(f"  macOS/Linux: source {venv_path}/bin/activate", "blue"))
    print(colorize("  Windows PowerShell: "
                   f"{venv_path}\\Scripts\\Activate.ps1", "yellow"))
    print(colorize("  Windows Command Prompt: "
                   f"{venv_path}\\Scripts\\activate.bat", "yellow"))
    print("To deactivate, run 'deactivate'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
