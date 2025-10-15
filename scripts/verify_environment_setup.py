#!/usr/bin/env python3
"""Analyze environment setup consistency across project assets."""
from __future__ import annotations

import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_FILES = (
    REPO_ROOT / "setup_env.py",
    REPO_ROOT / "run.py",
    REPO_ROOT / "scripts" / "self_debug.py",
)
ALL_PYTHON_FILES = tuple(sorted(REPO_ROOT.rglob("*.py")))
DOCUMENTATION_FILES = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "notes.txt",
)
COMMON_ENV_VARS = {
    "PATH",
    "VIRTUAL_ENV",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "SSL_CERT_FILE",
}


@dataclass
class SyntaxFinding:
    """Describe the result of parsing a Python source file."""

    file: str
    status: str
    message: str | None = None
    line: int | None = None
    column: int | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"file": self.file, "status": self.status}
        if self.message:
            payload["message"] = self.message
        if self.line is not None:
            payload["line"] = self.line
        if self.column is not None:
            payload["column"] = self.column
        return payload


class ImportCollector(ast.NodeVisitor):
    """Collect imported module names from the provided AST."""

    def __init__(self) -> None:
        self.modules: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802 - stdlib API
        for alias in node.names:
            self.modules.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802 - stdlib API
        if node.module is None:
            return
        self.modules.add(node.module)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - stdlib API
        func = node.func
        module_name: str | None = None
        if isinstance(func, ast.Name) and func.id == "__import__":
            if node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    module_name = arg.value
        elif isinstance(func, ast.Name) and func.id == "import_module":
            if node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    module_name = arg.value
        elif isinstance(func, ast.Attribute):
            if (
                func.attr == "import_module"
                and isinstance(func.value, ast.Name)
                and func.value.id == "importlib"
            ):
                if node.args:
                    arg = node.args[0]
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        module_name = arg.value
        if module_name:
            self.modules.add(module_name)
        self.generic_visit(node)


class EnvironmentVariableCollector(ast.NodeVisitor):
    """Extract environment variable usage from AST."""

    def __init__(self) -> None:
        self.env_like_names: set[str] = {"os.environ"}
        self.variables: set[str] = set()
        self.constant_strings: dict[str, str] = {}

    def _resolve_constant(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802 - stdlib API
        value = node.value
        if (
            isinstance(value, ast.Constant)
            and isinstance(value.value, str)
        ):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    self.constant_strings[target.id] = value.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
            func = value.func
            if (
                func.attr == "copy"
                and isinstance(func.value, ast.Attribute)
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "os"
                and func.value.attr == "environ"
            ):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.env_like_names.add(target.id)
        elif isinstance(value, ast.Name):
            # Track aliasing of environment-like dicts.
            if value.id in self.env_like_names:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.env_like_names.add(target.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - stdlib API
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id == "os" and func.attr in {"getenv"}:
                if node.args:
                    raw = node.args[0]
                    name = self._resolve_constant(raw)
                    if name is None and isinstance(raw, ast.Name):
                        name = self.constant_strings.get(raw.id)
                    if name:
                        self.variables.add(name)
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute):
            if (
                isinstance(func.value.value, ast.Name)
                and func.value.value.id == "os"
                and func.value.attr == "environ"
                and func.attr in {"get", "pop"}
                and node.args
            ):
                raw = node.args[0]
                name = self._resolve_constant(raw)
                if name is None and isinstance(raw, ast.Name):
                    name = self.constant_strings.get(raw.id)
                if name:
                    self.variables.add(name)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: N802 - stdlib API
        target = node.value
        raw_slice = getattr(node, "slice", None)
        key = self._resolve_constant(raw_slice)
        if key is None and isinstance(node.slice, ast.Index):  # pragma: no cover
            key = self._resolve_constant(node.slice.value)
        if key is None and isinstance(raw_slice, ast.Name):
            key = self.constant_strings.get(raw_slice.id)
        if key is None:
            self.generic_visit(node)
            return
        if isinstance(target, ast.Attribute):
            if (
                isinstance(target.value, ast.Name)
                and target.value.id == "os"
                and target.attr == "environ"
            ):
                self.variables.add(key)
        elif isinstance(target, ast.Name) and target.id in self.env_like_names:
            self.variables.add(key)
        self.generic_visit(node)


def check_syntax(paths: Sequence[Path]) -> list[SyntaxFinding]:
    findings: list[SyntaxFinding] = []
    for path in paths:
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            findings.append(
                SyntaxFinding(
                    file=str(path.relative_to(REPO_ROOT)),
                    status="error",
                    message=exc.msg,
                    line=exc.lineno,
                    column=exc.offset,
                )
            )
        else:
            findings.append(
                SyntaxFinding(
                    file=str(path.relative_to(REPO_ROOT)),
                    status="ok",
                )
            )
    return findings


def discover_local_modules(root: Path) -> set[str]:
    modules: set[str] = set()
    for path in root.rglob("*.py"):
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        parts = relative.parts
        if not parts:
            continue
        top_level = parts[0]
        if top_level == "__pycache__":
            continue
        modules.add(top_level.replace(".py", ""))
    return modules


def collect_imports(paths: Sequence[Path]) -> dict[str, list[str]]:
    imports: dict[str, list[str]] = {}
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        visitor = ImportCollector()
        visitor.visit(tree)
        imports[str(path.relative_to(REPO_ROOT))] = sorted(visitor.modules)
    return imports


def top_level_module(name: str) -> str:
    return name.split(".", 1)[0]


def canonicalize(name: str) -> str:
    normalized = name.replace("-", "_").lower()
    aliases = {"openai_whisper": "whisper"}
    return aliases.get(normalized, normalized)


def load_requirements() -> Mapping[str, str]:
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from run import get_requirements_path, read_required_packages  # type: ignore
    finally:
        sys.path.pop(0)
    return read_required_packages(get_requirements_path())


def analyze_dependencies(
    import_map: Mapping[str, list[str]],
    local_modules: Iterable[str],
    required_packages: Mapping[str, str],
) -> dict[str, object]:
    local = set(local_modules)
    stdlib_modules = set(sys.stdlib_module_names)
    external_usage: dict[str, set[str]] = {}
    for file, modules in import_map.items():
        for module in modules:
            candidate = top_level_module(module)
            canonical = canonicalize(candidate)
            if canonical in local:
                continue
            if candidate in stdlib_modules:
                continue
            external_usage.setdefault(canonical, set()).add(file)
    required_keys = set(required_packages.keys())
    missing_in_requirements = sorted(external_usage.keys() - required_keys)

    unused_requirements = [
        required_packages[key]
        for key in sorted(required_keys - set(external_usage.keys()))
    ]

    usage_details = {
        key: sorted(files) for key, files in sorted(external_usage.items())
    }

    return {
        "usage": usage_details,
        "missing_in_requirements": missing_in_requirements,
        "unused_requirements": unused_requirements,
    }


def analyze_environment_variables(paths: Sequence[Path]) -> dict[str, object]:
    variables: dict[str, set[str]] = {}
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        visitor = EnvironmentVariableCollector()
        visitor.visit(tree)
        if visitor.variables:
            variables[str(path.relative_to(REPO_ROOT))] = visitor.variables
    all_variables = sorted({var for vars_ in variables.values() for var in vars_})
    documentation_text = "\n".join(
        doc.read_text(encoding="utf-8", errors="ignore") for doc in DOCUMENTATION_FILES
    )
    undocumented = [
        var
        for var in all_variables
        if var not in COMMON_ENV_VARS and var not in documentation_text
    ]
    return {
        "files": {key: sorted(values) for key, values in sorted(variables.items())},
        "undocumented": sorted(set(undocumented)),
    }


def check_python_version_alignment(required_version: tuple[int, int]) -> dict[str, object]:
    pattern = re.compile(r"python\s*3\.12", re.IGNORECASE)
    documentation_mentions: dict[str, bool] = {}
    for doc in DOCUMENTATION_FILES:
        text = doc.read_text(encoding="utf-8", errors="ignore")
        documentation_mentions[str(doc.relative_to(REPO_ROOT))] = bool(pattern.search(text))
    all_documented = all(documentation_mentions.values())
    return {
        "required_version": f"{required_version[0]}.{required_version[1]}",
        "documentation_mentions": documentation_mentions,
        "is_consistent": all_documented,
    }


def build_report() -> dict[str, object]:
    syntax_results = check_syntax(TARGET_FILES)
    import_map = collect_imports(TARGET_FILES)
    repo_import_map = collect_imports(ALL_PYTHON_FILES)
    local_modules = discover_local_modules(REPO_ROOT)
    required_packages = load_requirements()
    dependency_analysis = analyze_dependencies(
        repo_import_map, local_modules, required_packages
    )
    env_analysis = analyze_environment_variables(TARGET_FILES)

    sys.path.insert(0, str(REPO_ROOT))
    try:
        import setup_env  # type: ignore
    finally:
        sys.path.pop(0)
    required_version = getattr(setup_env, "REQUIRED_PYTHON_VERSION", (3, 12))
    python_alignment = check_python_version_alignment(tuple(required_version[:2]))

    recommendations: list[str] = []
    if dependency_analysis["missing_in_requirements"]:
        recommendations.append(
            "Add the missing dependencies to requirements.txt so setup_env.py installs them automatically."
        )
    if python_alignment["is_consistent"] is False:
        recommendations.append(
            "Update README.md and notes.txt to mention the required Python version."
        )
    if env_analysis["undocumented"]:
        recommendations.append(
            "Document the environment variables used by the setup tooling to aid advanced configuration."
        )
    if not recommendations:
        recommendations.append("Environment setup artifacts are consistent. Keep dependencies in sync.")

    return {
        "syntax": [finding.to_dict() for finding in syntax_results],
        "imports": import_map,
        "dependencies": dependency_analysis,
        "environment_variables": env_analysis,
        "python_version": python_alignment,
        "recommendations": recommendations,
    }


def main() -> None:
    report = build_report()
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
