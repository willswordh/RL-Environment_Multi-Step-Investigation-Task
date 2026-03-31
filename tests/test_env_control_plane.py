"""Tests for env-level safeguards and stop conditions."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env import BugInvestigationEnv


def test_max_invalid_actions_stops_episode():
    env = BugInvestigationEnv(max_steps=10, max_invalid_actions=2, episode_timeout_seconds=60.0)
    env.reset()

    _, reward_1, done_1, info_1 = env.step({"type": "open_file"})  # missing filename
    assert done_1 is False
    assert info_1["invalid_action_count"] == 1
    assert reward_1 < -0.02

    _, reward_2, done_2, info_2 = env.step({"type": "inspect_function", "filename": "repo/discount.py"})
    assert done_2 is True
    assert info_2["stop_reason"] == "max_invalid_actions"
    assert reward_2 < 0.0
    assert info_2["reward_breakdown"]["max_invalid_penalty"] < 0.0


def test_repeated_action_gets_extra_penalty():
    env = BugInvestigationEnv(max_steps=10, max_invalid_actions=5, episode_timeout_seconds=60.0)
    env.reset()

    _, reward_1, done_1, _ = env.step({"type": "list_files"})
    _, reward_2, done_2, info_2 = env.step({"type": "list_files"})

    assert done_1 is False
    assert done_2 is False
    assert reward_2 < reward_1
    assert info_2["repeated_action_count"] >= 1


def test_timeout_sets_stop_reason():
    env = BugInvestigationEnv(max_steps=10, max_invalid_actions=5, episode_timeout_seconds=0.0)
    env.reset()

    _, reward, done, info = env.step({"type": "list_files"})
    assert done is True
    assert info["stop_reason"] == "timeout"
    assert reward < 0.0


def test_terminal_solved_false_when_submission_contradicts_structured_fields():
    env = BugInvestigationEnv(max_steps=10, max_invalid_actions=5, episode_timeout_seconds=60.0)
    env.reset()

    env.step({"type": "search", "keyword": "discount"})
    env.step({
        "type": "inspect_function",
        "filename": "repo/discount.py",
        "function": "apply_discount",
    })
    env.step({
        "type": "inspect_function",
        "filename": "repo/math_utils.py",
        "function": "round_currency",
    })
    _, _, done, info = env.step({
        "type": "submit_answer",
        "root_cause": (
            "math_utils.py round_currency uses banker half-to-even rounding "
            "which rounds 20.825 to 20.82."
        ),
        "fix": "Use Decimal quantize with ROUND_HALF_UP.",
        "bug_file": "repo/config.py",
        "bug_function": "round_currency",
        "mechanism": "banker rounding",
    })

    assert done is True
    assert info["reward_breakdown"]["contradiction_penalty"] < 0.0
    assert info["solved"] is False


def test_terminal_info_exposes_solved_and_trajectory():
    env = BugInvestigationEnv(max_steps=10, max_invalid_actions=5, episode_timeout_seconds=60.0)
    env.reset()

    env.step({"type": "search", "keyword": "discount"})
    env.step({
        "type": "inspect_function",
        "filename": "repo/discount.py",
        "function": "apply_discount",
    })
    env.step({
        "type": "inspect_function",
        "filename": "repo/math_utils.py",
        "function": "round_currency",
    })
    _, reward, done, info = env.step({
        "type": "submit_answer",
        "root_cause": (
            "math_utils.py round_currency uses banker half-to-even rounding "
            "which rounds 20.825 to 20.82."
        ),
        "fix": "Use Decimal quantize with ROUND_HALF_UP.",
        "bug_file": "repo/math_utils.py",
        "bug_function": "round_currency",
        "mechanism": "banker rounding",
    })

    assert done is True
    assert reward > 0.0
    assert info["solved"] is True
    assert info["sandbox_mode"] == "ephemeral-tempdir-copy"
    assert info["trajectory_length"] == 4
    assert info["last_transition"] is not None
    assert info["last_transition"]["done"] is True
