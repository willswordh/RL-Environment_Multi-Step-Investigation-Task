"""
Internal episode state for the BugInvestigationEnv.
"""
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Set, List

@dataclass
class EpisodeState:
    """All mutable state tracked across one episode.

    This object is private to the environment and is never sent to
    the agent directly — the agent can only infer what it has done
    from the observations it receives.
    """

    # ── step budget ──────────────────────────────────────────────────
    step_count: int = 0
    max_steps: int = 15

    # ── exploration tracking ─────────────────────────────────────────
    files_opened: Set[str] = field(default_factory=set)
    functions_inspected: Set[str] = field(default_factory=set)
    keywords_searched: List[str] = field(default_factory=list)
    tests_run: bool = False
    invalid_action_count: int = 0
    repeated_action_count: int = 0
    parse_error_count: int = 0
    trajectory: List[Dict[str, Any]] = field(default_factory=list)

    # ── milestone flags (for partial rewards) ────────────────────────
    correct_file_opened: bool = False
    correct_function_inspected: bool = False

    # ── terminal state ───────────────────────────────────────────────
    submitted: bool = False
    submission_root_cause: str = ""
    submission_fix: str = ""
    submission_bug_file: str = ""
    submission_bug_function: str = ""
    submission_mechanism: str = ""
    submission_fix_summary: str = ""
    stop_reason: str = ""

    # ── runtime tracking ──────────────────────────────────────────────
    start_time: float = field(default_factory=time.monotonic)
    elapsed_seconds: float = 0.0
    timed_out: bool = False
    max_invalid_reached: bool = False
    last_action_signature: str = ""
    consecutive_repeated_actions: int = 0

    # ── ground truth (immutable reference) ───────────────────────────
    task_id: str = ""
    bug_file: str = ""
    bug_function: str = ""
    mechanism_keywords: List[str] = field(default_factory=list)
    fix_keywords: List[str] = field(default_factory=list)
    evidence_function_entries: List[str] = field(default_factory=list)

    # ── helpers ──────────────────────────────────────────────────────
    @classmethod
    def for_task(cls, task, max_steps: int = None) -> "EpisodeState":
        return cls(
            max_steps=max_steps if max_steps is not None else task.max_steps,
            task_id=task.task_id,
            bug_file=task.bug_file,
            bug_function=task.bug_function,
            mechanism_keywords=list(task.mechanism_keywords),
            fix_keywords=list(task.fix_keywords),
            evidence_function_entries=list(task.evidence_functions),
        )

    @property
    def steps_remaining(self) -> int:
        return max(0, self.max_steps - self.step_count)

    @property
    def budget_exhausted(self) -> bool:
        return self.step_count >= self.max_steps

    def refresh_elapsed(self) -> None:
        self.elapsed_seconds = max(0.0, time.monotonic() - self.start_time)

    def record_file_opened(self, filename: str) -> None:
        self.files_opened.add(filename)
        if filename == self.bug_file:
            self.correct_file_opened = True

    def record_function_inspected(self, filename: str, function: str) -> None:
        key = f"{filename}::{function}"
        self.functions_inspected.add(key)
        self.files_opened.add(filename)
        if filename == self.bug_file:
            self.correct_file_opened = True
        if filename == self.bug_file and function == self.bug_function:
            self.correct_function_inspected = True

    def record_action_signature(self, signature: str) -> bool:
        """Track repeated actions. Returns True when repetition penalty should apply."""
        if signature == self.last_action_signature and signature:
            self.consecutive_repeated_actions += 1
        else:
            self.last_action_signature = signature
            self.consecutive_repeated_actions = 1

        # Penalize once the same action is repeated back-to-back.
        if self.consecutive_repeated_actions >= 2:
            self.repeated_action_count += 1
            return True
        return False

    def record_transition(
        self,
        *,
        action_taken: str,
        reward: float,
        done: bool,
        result_preview: str,
    ) -> None:
        self.trajectory.append(
            {
                "step": self.step_count,
                "action": action_taken,
                "reward": reward,
                "done": done,
                "stop_reason": self.stop_reason or "(running)",
                "elapsed_seconds": self.elapsed_seconds,
                "result_preview": result_preview,
            }
        )
