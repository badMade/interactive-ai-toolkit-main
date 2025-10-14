"""
Utility script to provision a dedicated Python environment for the project.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Protocol

from logging.handlers import RotatingFileHandler

from run import SETUP_LOG_RELATIVE_PATH, read_required_packages


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


MINIMUM_SUPPORTED_PYTHON = (3, 10)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "setup_env.log"
LOG_PATH_ENV_VAR = "SETUP_ENV_LOG_PATH"


DEFAULT_VALIDATION_PACKAGES: tuple[str, ...] = (
    "torch",
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
    required_major, required_minor = MINIMUM_SUPPORTED_PYTHON
    if (current.major, current.minor) < (required_major, required_minor):
        raise SetupError(
            "Python 3.10 or newer is required. "
            f"Detected {current.major}.{current.minor}.{current.micro}. "
            "Please install or launch Python 3 before continuing."
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
        combined_output = ""
        if capture_output:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            combined_output = (stdout + stderr).strip()
        message = f"Command failed: {' '.join(command)}"
        if combined_output:
            message = f"{message}\n{combined_output}"
        raise SetupError(message) from exc
    if capture_output:
        output = (result.stdout or "") + (result.stderr or "")
        return output.strip()
    return ""


def ensure_ffmpeg_available(*, runner: CommandRunner | None = None) -> None:
    """Verify that FFmpeg is available on the host system."""

    def probe() -> str:
        return run_command(
            ["ffmpeg", "-version"],
            capture_output=True,
            runner=runner,
        )

    try:
        version_output = probe()
    except SetupError:
        installer = PROJECT_ROOT / "scripts" / "install_ffmpeg.sh"
        if not installer.is_file():
            raise SetupError(
                "FFmpeg is required but automatic installation failed because the "
                "installer script is missing."
            )

        logging.info("FFmpeg not found; invoking %s", installer)
        try:
            installation_output = run_command(
                [str(installer)],
                capture_output=True,
                runner=runner,
            )
            if installation_output:
                logging.info("%s", installation_output.splitlines()[-1])
        except SetupError:
            raise

        version_output = probe()

    if version_output:
        first_line = version_output.splitlines()[0]
        logging.info("FFmpeg detected: %s", first_line)


def ensure_requirements_file(requirements_path: Path) -> Path:
    """Ensure that *requirements_path* exists before continuing."""
    if not requirements_path.is_file():
        raise SetupError(
            "Could not find requirements.txt. "
            "Please add it before running setup."
        )
    logging.info("Using requirements file at %s", requirements_path)
    return requirements_path


def create_virtualenv(
    venv_path: Path,
    *,
    runner: CommandRunner | None = None,
) -> None:
    """Create a virtual environment unless one already exists."""
    pyvenv_cfg = venv_path / "pyvenv.cfg"
    if pyvenv_cfg.exists():
        logging.info("Virtual environment already exists at %s", venv_path)
        return
    logging.info("Creating virtual environment at %s", venv_path)
    run_command([sys.executable, "-m", "venv", str(venv_path)], runner=runner)


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
        [str(venv_python),
         "-m",
         "pip",
         "install",
         "-r",
         str(requirements_path)],
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
        version = run_command(
            [str(venv_python), "-c", code],
            capture_output=True,
            runner=runner,
        )
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
