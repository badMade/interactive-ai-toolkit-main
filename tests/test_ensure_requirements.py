from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import ensure_requirements as ensure


class StubRunner:
    """Simple stub that mimics :func:`subprocess.run` for testing."""

    def __init__(self, responses: list[tuple[int, str, str]] | None = None) -> None:
        self.responses = list(responses or [])
        self.commands: list[dict[str, object]] = []

    def __call__(
        self,
        command: list[str] | tuple[str, ...],
        *,
        check: bool,
        text: bool,
        capture_output: bool,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        self.commands.append(
            {
                "command": list(command),
                "check": check,
                "capture_output": capture_output,
                "env": env,
            }
        )
        if self.responses:
            returncode, stdout, stderr = self.responses.pop(0)
        else:
            returncode, stdout, stderr = 0, "", ""
        if check and returncode != 0:
            raise subprocess.CalledProcessError(returncode, command, stdout, stderr)
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def test_collect_requirements_ignores_comments(tmp_path: Path) -> None:
    requirements_file = tmp_path / "requirements.txt"
    requirements_file.write_text("""# comment\npackage-one\n\npackage-two==1.0\n""")

    result = ensure.collect_requirements(requirements_file)

    assert result == ["package-one", "package-two==1.0"]


def test_missing_requirements_identifies_unsatisfied(tmp_path: Path) -> None:
    venv_python = tmp_path / "python"
    runner = StubRunner([(1, "", ""), (0, "", "")])

    missing = ensure.missing_requirements(
        venv_python,
        ["pkgA", "pkgB"],
        runner=runner,
    )

    assert missing == ["pkgA"]
    assert len(runner.commands) == 2


def test_synchronize_environment_installs_and_verifies(tmp_path: Path) -> None:
    project_root = tmp_path
    requirements_path = project_root / "requirements.txt"
    requirements_path.write_text("alpha\nbeta\n")

    venv_path = project_root / ".venv"
    python_dir = venv_path / "bin"
    python_dir.mkdir(parents=True)
    (venv_path / "pyvenv.cfg").write_text("home = /usr/bin/python\n")
    (python_dir / "python").write_text("#!/usr/bin/env python3\n")

    runner = StubRunner(
        [
            (1, "", ""),  # alpha missing
            (1, "", ""),  # beta missing
            (0, "", ""),  # pip install
            (0, "", ""),  # alpha satisfied after install
            (0, "", ""),  # beta satisfied after install
        ]
    )

    missing = ensure.synchronize_environment(
        requirements_path=requirements_path,
        venv_path=venv_path,
        runner=runner,
    )

    assert missing == ["alpha", "beta"]
    pip_command = runner.commands[2]["command"]
    assert pip_command[:4] == [
        str(python_dir / "python"),
        "-m",
        "pip",
        "install",
    ]


def test_synchronize_environment_skips_install_when_satisfied(tmp_path: Path) -> None:
    project_root = tmp_path
    requirements_path = project_root / "requirements.txt"
    requirements_path.write_text("alpha\nbeta\n")

    venv_path = project_root / ".venv"
    python_dir = venv_path / "bin"
    python_dir.mkdir(parents=True)
    (venv_path / "pyvenv.cfg").write_text("home = /usr/bin/python\n")
    (python_dir / "python").write_text("#!/usr/bin/env python3\n")

    runner = StubRunner([(0, "", ""), (0, "", "")])

    missing = ensure.synchronize_environment(
        requirements_path=requirements_path,
        venv_path=venv_path,
        runner=runner,
    )

    assert missing == []
    assert len(runner.commands) == 2


def test_ensure_virtualenv_exists_creates_environment(tmp_path: Path) -> None:
    venv_path = tmp_path / "env"
    runner = StubRunner([(0, "", "")])

    ensure.ensure_virtualenv_exists(
        venv_path,
        python_executable="python",
        runner=runner,
    )

    assert runner.commands[0]["command"] == ["python", "-m", "venv", str(venv_path)]
