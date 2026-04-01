"""
In-memory repository store.

Loads the mock codebase source files at construction time and exposes
clean query methods (open_file, search, inspect_function, run_tests)
that the environment's action router delegates to.
"""
from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from typing import Dict, List, Tuple

from env.task_config import load_task


def _load_files(task) -> Dict[str, str]:
    """Read all accessible files from a task pack into memory."""
    base = task.task_root
    contents: Dict[str, str] = {}
    for rel_path in task.accessible_files:
        abs_path = os.path.normpath(os.path.join(base, rel_path))
        try:
            with open(abs_path, "r", encoding="utf-8") as fh:
                contents[rel_path] = fh.read()
        except FileNotFoundError:
            contents[rel_path] = f"# File not found: {rel_path}\n"
    return contents


class Repository:
    """Read-only view of the mock codebase for the RL environment.

    Methods mirror the agent's action vocabulary so the action router
    can delegate directly.
    """

    def __init__(self, task=None) -> None:
        self.task = task or load_task()
        self.sandbox_mode = "ephemeral-tempdir-copy"
        self._files: Dict[str, str] = _load_files(self.task)

    # ── public query API ─────────────────────────────────────────────────

    def list_files(self) -> List[str]:
        """Return the names of all files in the repository."""
        return list(self._files.keys())

    def open_file(self, filename: str) -> Tuple[bool, str]:
        """Return the full source of *filename*.

        Args:
            filename: Relative path as returned by list_files().

        Returns:
            (success, content) — success is False when the file is not found.
        """
        # Normalise separators so "repo\\discount.py" also works on Windows
        key = filename.replace("\\", "/").strip()
        if key in self._files:
            return True, self._files[key]
        # Fuzzy match: try appending common prefixes
        for candidate in self._files:
            if candidate.endswith("/" + key) or candidate == key:
                return True, self._files[candidate]
        available = "\n".join(f"  {f}" for f in self._files)
        return False, (
            f"File '{filename}' not found in the repository.\n"
            f"Available files:\n{available}"
        )

    def search(self, keyword: str) -> Tuple[bool, str]:
        """Search all files for lines containing *keyword* (case-insensitive).

        Args:
            keyword: Plain-text string to search for.

        Returns:
            (found, formatted_results)
        """
        keyword_lower = keyword.lower()
        hits: List[str] = []
        for filename, source in self._files.items():
            for lineno, line in enumerate(source.splitlines(), start=1):
                if keyword_lower in line.lower():
                    hits.append(f"{filename}:{lineno}:  {line.rstrip()}")
        if hits:
            return True, "\n".join(hits)
        return False, f"No matches found for '{keyword}'."

    def inspect_function(self, filename: str, function_name: str) -> Tuple[bool, str]:
        """Return the source of a single named function inside *filename*.

        Uses the AST to find the exact line range so the result is precise
        without requiring the agent to parse the full file.

        Args:
            filename:      Relative path of the file.
            function_name: Name of the function to extract.

        Returns:
            (success, source_snippet)
        """
        ok, content = self.open_file(filename)
        if not ok:
            return False, content

        try:
            tree = ast.parse(content)
        except SyntaxError as exc:
            return False, f"Could not parse '{filename}': {exc}"

        lines = content.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == function_name:
                    start = node.lineno - 1  # 0-indexed
                    # `end_lineno` is not available on Python < 3.8.
                    if hasattr(node, "end_lineno") and node.end_lineno is not None:
                        end = node.end_lineno  # exclusive for Python slicing
                    else:
                        end = _infer_block_end(lines, start)
                    snippet = "\n".join(lines[start:end])
                    return True, (
                        f"# {filename} — {function_name}()\n"
                        + textwrap.dedent(snippet)
                    )

        return False, (
            f"Function '{function_name}' not found in '{filename}'.\n"
            f"Tip: use open_file to see all function definitions."
        )

    def run_tests(self) -> Tuple[bool, str]:
        """Execute the test suite and return the output.

        Runs the benchmark failing test in a subprocess so the real Python
        interpreter catches the actual rounding assertion failure.

        Returns:
            (tests_passed, output_text)
        """
        with tempfile.TemporaryDirectory(prefix="bug-env-") as sandbox_root:
            self._populate_sandbox(sandbox_root)
            timeout_seconds = max(1.0, float(self.task.timeout_seconds))
            env = os.environ.copy()
            env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["PYTHONNOUSERSITE"] = "1"
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "-m",
                        "pytest",
                        self.task.benchmark_test_target,
                        "-v",
                        "--tb=short",
                    ],
                    cwd=sandbox_root,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    env=env,
                )
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                partial_output = (stdout + stderr).strip()
                message = (
                    "Pytest timed out after "
                    f"{timeout_seconds:.1f}s while running {self.task.benchmark_test_target}."
                )
                if partial_output:
                    message += "\n\nPartial output:\n" + partial_output
                return False, message
            output = result.stdout + result.stderr
            passed = result.returncode == 0
            return passed, output.strip()

    def _populate_sandbox(self, sandbox_root: str) -> None:
        repo_dst = os.path.join(sandbox_root, self.task.repo_subdir)
        tests_dst = os.path.join(sandbox_root, self.task.tests_subdir)
        ignore = shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            ".pytest_cache",
            ".mypy_cache",
        )
        shutil.copytree(self.task.repo_dir, repo_dst, ignore=ignore)
        shutil.copytree(self.task.tests_dir, tests_dst, ignore=ignore)


def _infer_block_end(lines: List[str], start: int) -> int:
    """Infer a function block end when AST end positions are unavailable.

    Args:
        lines: Full file source split by lines.
        start: 0-indexed line where the `def` starts.

    Returns:
        0-indexed exclusive end line suitable for slicing: `lines[start:end]`.
    """
    if start < 0 or start >= len(lines):
        return len(lines)

    def_line = lines[start]
    def_indent = len(def_line) - len(def_line.lstrip())

    for idx in range(start + 1, len(lines)):
        line = lines[idx]
        stripped = line.strip()

        if not stripped:
            continue
        if line.lstrip().startswith("#"):
            continue

        indent = len(line) - len(line.lstrip())
        if indent <= def_indent:
            return idx

    return len(lines)
