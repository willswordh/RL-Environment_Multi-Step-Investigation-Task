"""
BugInvestigationEnv — the main RL environment class.

Interface:
    env = BugInvestigationEnv()
    observation = env.reset()
    observation, reward, done, info = env.step(action)

where `action` is a plain dict, e.g.:
    {"type": "run_tests"}
    {"type": "open_file", "filename": "repo/discount.py"}
    {"type": "submit_answer", "root_cause": "...", "fix": "..."}
"""
from __future__ import annotations
import os
from typing import Any, Dict, Tuple

from env.tools.actions import (
    Action,
    LIST_FILES,
    OPEN_FILE,
    SEARCH,
    RUN_TESTS,
    INSPECT_FUNCTION,
    SUBMIT_ANSWER,
    action_help,
    parse_action,
)
from env.episode_state import EpisodeState
from env.task_config import load_task
from env.tools.repository import Repository
from env.reward import compute_reward, compute_step_reward

# Type aliases
Observation = Dict[str, Any]
Info        = Dict[str, Any]


_DEFAULT_TASK_NAME = "discount-rounding"
_DEFAULT_TASKS_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "tasks")
)


class BugInvestigationEnv:
    """Gymnasium-style RL environment for multi-step bug investigation.

    The agent interacts with an in-memory mock codebase by calling
    tool-like actions.  Each call to step() consumes one step of the
    fixed step budget.  The episode ends when the agent submits an
    answer or the budget is exhausted.

    Attributes:
        max_steps (int): Maximum steps per episode (default 15).
        repo      (Repository): Read-only view of the mock codebase.
        state     (EpisodeState): Current internal episode state.
    """

    def __init__(
        self,
        max_steps: int = None,
        max_invalid_actions: int = None,
        episode_timeout_seconds: float = None,
        task_name: str = _DEFAULT_TASK_NAME,
        tasks_root: str = _DEFAULT_TASKS_ROOT,
    ) -> None:
        self.task_name = task_name
        self.tasks_root = tasks_root
        self.task = load_task(task_name=self.task_name, tasks_root=self.tasks_root)
        self.max_steps = max_steps if max_steps is not None else self.task.max_steps
        self.max_invalid_actions = (
            max_invalid_actions
            if max_invalid_actions is not None
            else self.task.max_invalid_actions
        )
        self.episode_timeout_seconds = (
            episode_timeout_seconds
            if episode_timeout_seconds is not None
            else self.task.timeout_seconds
        )
        self.repo      = Repository(task=self.task)
        self.state     = EpisodeState.for_task(self.task, max_steps=self.max_steps)

    # ── Gymnasium-style interface ────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset the environment and return the initial observation.

        Returns:
            The initial observation dict containing the bug report and
            instructions.
        """
        self.task = load_task(task_name=self.task_name, tasks_root=self.tasks_root)
        self.repo = Repository(task=self.task)
        self.state = EpisodeState.for_task(self.task, max_steps=self.max_steps)
        return self._make_observation(
            action_taken="[episode start]",
            result=self._build_initial_prompt(),
        )

    def step(self, action: Dict[str, Any]) -> Tuple[Observation, float, bool, Info]:
        """Execute one action and return (observation, reward, done, info).

        Args:
            action: A dict with at minimum a "type" key.

        Returns:
            observation : dict — information returned to the agent.
            reward      : float — immediate reward for this step.
            done        : bool  — True when the episode has ended.
            info        : dict  — diagnostic information (not for training).
        """
        # ── guard: already done ─────────────────────────────────────────
        if (
            self.state.submitted
            or self.state.budget_exhausted
            or self.state.max_invalid_reached
            or self.state.timed_out
        ):
            obs = self._make_observation(
                action_taken="[attempted step after episode end]",
                result="Episode has already ended. Call reset() to start a new episode.",
            )
            self.state.record_transition(
                action_taken=obs["action_taken"],
                reward=0.0,
                done=True,
                result_preview=self._preview_result(obs["result"]),
            )
            return obs, 0.0, True, self._info()

        self.state.refresh_elapsed()
        if (
            self.episode_timeout_seconds is not None
            and self.state.elapsed_seconds >= self.episode_timeout_seconds
        ):
            self.state.timed_out = True
            self.state.stop_reason = "timeout"
            breakdown = compute_reward(self.state)
            obs = self._make_observation(
                action_taken=str(action),
                result="[Episode Ended] Wall-clock timeout reached before action execution.",
            )
            self.state.record_transition(
                action_taken=obs["action_taken"],
                reward=breakdown.total,
                done=True,
                result_preview=self._preview_result(obs["result"]),
            )
            return obs, breakdown.total, True, self._info(breakdown=breakdown)

        # ── parse the action ────────────────────────────────────────────
        parsed_action, error = parse_action(action)
        if error:
            obs = self._make_observation(
                action_taken=str(action),
                result=f"[Action Error] {error}",
            )
            # Invalid actions still cost a step
            self.state.step_count += 1
            self.state.invalid_action_count += 1
            self.state.parse_error_count += 1
            self.state.refresh_elapsed()
            step_reward = compute_step_reward(self.state, invalid_action=True)
            done = self.state.budget_exhausted

            if self.state.invalid_action_count >= self.max_invalid_actions:
                self.state.max_invalid_reached = True
                self.state.stop_reason = "max_invalid_actions"
                done = True
            elif done:
                self.state.stop_reason = "budget_exhausted"

            if done:
                final_breakdown = compute_reward(self.state)
                step_reward = final_breakdown.total
                self.state.record_transition(
                    action_taken=obs["action_taken"],
                    reward=step_reward,
                    done=done,
                    result_preview=self._preview_result(obs["result"]),
                )
                return obs, step_reward, done, self._info(breakdown=final_breakdown)
            self.state.record_transition(
                action_taken=obs["action_taken"],
                reward=step_reward,
                done=done,
                result_preview=self._preview_result(obs["result"]),
            )
            return obs, step_reward, done, self._info()

        # ── execute the action ──────────────────────────────────────────
        repeated_action = self.state.record_action_signature(str(parsed_action))
        result_text = self._execute(parsed_action)

        # ── increment step counter ──────────────────────────────────────
        self.state.step_count += 1
        self.state.refresh_elapsed()

        # ── compute reward ──────────────────────────────────────────────
        timeout_reached = (
            self.episode_timeout_seconds is not None
            and self.state.elapsed_seconds >= self.episode_timeout_seconds
        )

        if parsed_action.type == SUBMIT_ANSWER:
            self.state.stop_reason = "submitted"
        elif self.state.budget_exhausted:
            self.state.stop_reason = "budget_exhausted"
        elif timeout_reached:
            self.state.timed_out = True
            self.state.stop_reason = "timeout"

        if (
            parsed_action.type == SUBMIT_ANSWER
            or self.state.budget_exhausted
            or timeout_reached
        ):
            # Terminal step — compute full episode reward
            breakdown = compute_reward(self.state)
            step_reward = breakdown.total
            done = True
        else:
            # Non-terminal step — only the step penalty
            step_reward = compute_step_reward(
                self.state,
                repeated_action=repeated_action,
            )
            done = False
            breakdown = None

        obs = self._make_observation(
            action_taken=str(parsed_action),
            result=result_text,
        )
        self.state.record_transition(
            action_taken=obs["action_taken"],
            reward=step_reward,
            done=done,
            result_preview=self._preview_result(obs["result"]),
        )
        info = self._info(breakdown=breakdown)
        return obs, step_reward, done, info

    # ── Action execution ────────────────────────────────────────────────

    def _execute(self, action: Action) -> str:
        """Route a validated action to the appropriate repository method."""

        if action.type == LIST_FILES:
            files = self.repo.list_files()
            return "Repository files:\n" + "\n".join(f"  {f}" for f in files)

        elif action.type == OPEN_FILE:
            ok, content = self.repo.open_file(action.filename)
            if ok:
                self.state.record_file_opened(action.filename)
            return content

        elif action.type == SEARCH:
            self.state.keywords_searched.append(action.keyword)
            ok, content = self.repo.search(action.keyword)
            return content

        elif action.type == RUN_TESTS:
            self.state.tests_run = True
            passed, output = self.repo.run_tests()
            header = "✅ All tests passed.\n\n" if passed else "❌ Test suite FAILED.\n\n"
            return header + output

        elif action.type == INSPECT_FUNCTION:
            ok, content = self.repo.inspect_function(
                action.filename, action.function_name
            )
            if ok:
                self.state.record_function_inspected(
                    action.filename, action.function_name
                )
            return content

        elif action.type == SUBMIT_ANSWER:
            self.state.submitted           = True
            self.state.submission_root_cause = action.root_cause
            self.state.submission_fix       = action.fix
            self.state.submission_bug_file = action.bug_file
            self.state.submission_bug_function = action.bug_function
            self.state.submission_mechanism = action.mechanism
            self.state.submission_fix_summary = action.proposed_fix
            return (
                "Answer submitted. Episode ending.\n\n"
                f"Root cause: {action.root_cause}\n\n"
                f"Proposed fix: {action.fix if action.fix else '(none provided)'}\n"
                f"Structured bug file: {action.bug_file or '(not provided)'}\n"
                f"Structured bug function: {action.bug_function or '(not provided)'}\n"
                f"Structured mechanism: {action.mechanism or '(not provided)'}"
            )

        return f"[Internal Error] Unhandled action type: {action.type}"

    # ── Observation / Info builders ──────────────────────────────────────

    def _make_observation(self, action_taken: str, result: str) -> Observation:
        return {
            "step":          self.state.step_count,
            "max_steps":     self.state.max_steps,
            "steps_remaining": self.state.steps_remaining,
            "action_taken":  action_taken,
            "result":        result,
        }

    def _info(self, breakdown=None) -> Info:
        solved = False
        info: Info = {
            "files_opened":            list(self.state.files_opened),
            "functions_inspected":     list(self.state.functions_inspected),
            "keywords_searched":       list(self.state.keywords_searched),
            "tests_run":               self.state.tests_run,
            "invalid_action_count":    self.state.invalid_action_count,
            "parse_error_count":       self.state.parse_error_count,
            "repeated_action_count":   self.state.repeated_action_count,
            "correct_file_opened":     self.state.correct_file_opened,
            "correct_function_found":  self.state.correct_function_inspected,
            "submitted":               self.state.submitted,
            "elapsed_seconds":         self.state.elapsed_seconds,
            "stop_reason":             self.state.stop_reason or "(running)",
            "timed_out":               self.state.timed_out,
            "max_invalid_reached":     self.state.max_invalid_reached,
            "trajectory_length":       len(self.state.trajectory),
            "last_transition":         self.state.trajectory[-1] if self.state.trajectory else None,
            "task_id":                 self.task.task_id,
            "sandbox_mode":            self.repo.sandbox_mode,
            "solved":                  solved,
        }
        if breakdown is not None:
            reward_breakdown = breakdown.as_dict()
            solved = (
                self.state.submitted
                and reward_breakdown["correctness"] >= 0.5
                and reward_breakdown["wrong_penalty"] == 0.0
                and reward_breakdown["evidence_penalty"] == 0.0
                and reward_breakdown["contradiction_penalty"] == 0.0
            )
            info["reward_breakdown"] = reward_breakdown
            info["solved"] = solved
        return info

    def _build_initial_prompt(self) -> str:
        return (
            "Bug Investigation Task\n"
            "======================\n\n"
            + self.task.read_instruction()
            + "\n\n"
            + action_help()
            + "\n"
            + "NOTE: The list of repository files is not shown here.\n"
            + "Use list_files() to discover the codebase.\n"
            + "Tests are executed in a fresh ephemeral sandbox copy of the task files.\n"
        )

    @staticmethod
    def _preview_result(result: str, limit: int = 220) -> str:
        one_line = " ".join(result.split())
        if len(one_line) <= limit:
            return one_line
        return one_line[:limit] + "..."

    # ── Convenience ─────────────────────────────────────────────────────

    def render(self, obs: Observation) -> None:
        """Pretty-print an observation to stdout."""
        sep = "─" * 64
        print(sep)
        print(f"  Step {obs['step']} / {obs['max_steps']}  "
              f"(steps remaining: {obs['steps_remaining']})")
        print(f"  Action: {obs['action_taken']}")
        print(sep)
        print(obs["result"])
        print()
