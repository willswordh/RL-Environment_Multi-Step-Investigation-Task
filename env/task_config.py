"""Task-pack loading for BugInvestigationEnv."""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _project_root() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def _default_tasks_root() -> str:
    return os.path.join(_project_root(), "tasks")


@dataclass
class TaskSpec:
    """Configuration for a single investigation task."""

    task_id: str
    title: str
    task_root: str
    repo_subdir: str
    tests_subdir: str
    accessible_files: List[str]
    benchmark_test_target: str
    max_steps: int
    max_invalid_actions: int
    timeout_seconds: float
    bug_file: str
    bug_function: str
    mechanism_keywords: List[str]
    fix_keywords: List[str]
    evidence_functions: List[str]

    @property
    def instruction_path(self) -> str:
        return os.path.join(self.task_root, "instruction.md")

    @property
    def repo_dir(self) -> str:
        return os.path.join(self.task_root, self.repo_subdir)

    @property
    def tests_dir(self) -> str:
        return os.path.join(self.task_root, self.tests_subdir)

    def read_instruction(self) -> str:
        with open(self.instruction_path, "r", encoding="utf-8") as fh:
            return fh.read().strip()


def load_task(
    task_name: Optional[str] = None,
    tasks_root: Optional[str] = None,
) -> TaskSpec:
    """Load a task pack from ``tasks/<task_name>/task.toml``."""
    root = os.path.normpath(tasks_root or _default_tasks_root())
    resolved_name = task_name or _discover_default_task(root)
    task_root = os.path.join(root, resolved_name)
    task_toml = os.path.join(task_root, "task.toml")

    if not os.path.exists(task_toml):
        raise FileNotFoundError("Task config not found: {}".format(task_toml))

    raw = _parse_simple_toml(task_toml)
    metadata = raw.get("metadata", {})
    environment = raw.get("environment", {})
    ground_truth = raw.get("ground_truth", {})

    task_id = metadata.get("task_id", resolved_name)
    title = metadata.get("title", task_id)
    accessible_files = list(environment.get("accessible_files", []))

    if not accessible_files:
        raise ValueError("Task '{}' does not define accessible_files.".format(task_id))

    return TaskSpec(
        task_id=task_id,
        title=title,
        task_root=task_root,
        repo_subdir=environment.get("repo_subdir", "repo"),
        tests_subdir=environment.get("tests_subdir", "tests"),
        accessible_files=accessible_files,
        benchmark_test_target=environment["benchmark_test_target"],
        max_steps=int(environment.get("max_steps", 15)),
        max_invalid_actions=int(environment.get("max_invalid_actions", 5)),
        timeout_seconds=float(environment.get("timeout_seconds", 120.0)),
        bug_file=ground_truth["bug_file"],
        bug_function=ground_truth["bug_function"],
        mechanism_keywords=list(ground_truth.get("mechanism_keywords", [])),
        fix_keywords=list(ground_truth.get("fix_keywords", [])),
        evidence_functions=list(ground_truth.get("evidence_functions", [])),
    )


def _discover_default_task(tasks_root: str) -> str:
    if not os.path.isdir(tasks_root):
        raise FileNotFoundError("Tasks directory not found: {}".format(tasks_root))

    candidates = []
    for name in sorted(os.listdir(tasks_root)):
        if os.path.exists(os.path.join(tasks_root, name, "task.toml")):
            candidates.append(name)

    if not candidates:
        raise FileNotFoundError("No task packs found in {}".format(tasks_root))

    return candidates[0]


def _parse_simple_toml(path: str) -> Dict[str, Dict[str, Any]]:
    """Parse the small TOML subset used by this repository."""
    data: Dict[str, Dict[str, Any]] = {}
    current_section: Optional[str] = None

    with open(path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1].strip()
                data.setdefault(current_section, {})
                continue

            key, value = [part.strip() for part in line.split("=", 1)]
            parsed = _parse_value(value)
            if current_section is None:
                data[key] = parsed  # type: ignore[assignment]
            else:
                data[current_section][key] = parsed

    return data


def _parse_value(value: str) -> Any:
    normalised = value.strip()
    if normalised.lower() == "true":
        return True
    if normalised.lower() == "false":
        return False
    return ast.literal_eval(normalised)
