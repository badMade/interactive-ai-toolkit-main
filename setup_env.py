"""
Utility script to provision a dedicated Python environment for the project.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Protocol


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


def configure_logging() -> None:
    """Configure application logging to show concise, informative messages."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


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
        raise SetupError(f"Command failed: {' '.join(command)}") from exc
    if capture_output:
        output = (result.stdout or "") + (result.stderr or "")
        return output.strip()
    return ""


def ensure_ffmpeg_available(*, runner: CommandRunner | None = None) -> None:
    """Verify that FFmpeg is available on the host system."""

    try:
        version_output = run_command(
            ["ffmpeg", "-version"],
            capture_output=True,
            runner=runner,
        )
    except SetupError as error:
        if os.name == "nt":
            guidance = (
                "Install FFmpeg from https://ffmpeg.org/download.html or "
                "run `winget install Gyan.FFmpeg`."
            )
        elif sys.platform == "darwin":
            guidance = "Install FFmpeg via Homebrew using `brew install ffmpeg`."
        else:
            guidance = "Install FFmpeg with your package manager, e.g. `sudo apt install ffmpeg`."
        raise SetupError(
            "FFmpeg is required but was not detected. "
            f"{guidance}"
        ) from error

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
) -> None:
    """
    Validate the new environment by reporting toolchain and package versions.
    """
    python_version = run_command([str(venv_python), "--version"],
                                 capture_output=True, runner=runner)
    logging.info("Python interpreter: %s", python_version)

    pip_version = run_command(
        [str(venv_python), "-m", "pip", "--version"],
        capture_output=True,
        runner=runner,
    )
    logging.info("pip version: %s", pip_version)

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


def main() -> int:
    """Entry point used when executing the setup script from the CLI."""
    ensure_supported_python()
    configure_logging()
    project_root = Path(__file__).resolve().parent
    venv_path = project_root / ".venv"
    requirements_path = project_root / "requirements.txt"

    try:
        ensure_requirements_file(requirements_path)
        ensure_ffmpeg_available()
        create_virtualenv(venv_path)
        venv_python = get_venv_python_path(venv_path)
        install_requirements(venv_python, requirements_path)
        verify_installation(
            venv_python,
            packages=("torch", "transformers", "soundfile", "whisper"),
        )
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
