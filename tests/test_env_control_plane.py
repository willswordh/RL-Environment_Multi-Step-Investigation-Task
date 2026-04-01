"""Tests for env-level safeguards, solved gating, and reward accounting."""
import math
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
    assert reward_1 < -0.01

    _, reward_2, done_2, info_2 = env.step({"type": "inspect_function", "filename": "repo/discount.py"})
    assert done_2 is True
    assert info_2["stop_reason"] == "max_invalid_actions"
    assert reward_2 < 0.0
    assert info_2["reward_breakdown"]["max_invalid_penalty"] < 0.0


def test_invalid_action_observation_reflects_consumed_step():
    env = BugInvestigationEnv(max_steps=10, max_invalid_actions=5, episode_timeout_seconds=60.0)
    env.reset()

    obs, reward, done, info = env.step({"type": "open_file"})

    assert obs["step"] == 1
    assert obs["steps_remaining"] == 9
    assert reward < 0.0
    assert done is False
    assert info["invalid_action_count"] == 1


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

    env.step({"type": "run_tests"})
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


def test_structured_fields_do_not_unlock_solved_episode():
    env = BugInvestigationEnv(max_steps=10, max_invalid_actions=5, episode_timeout_seconds=60.0)
    env.reset()

    env.step({"type": "run_tests"})
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
        "root_cause": "I am not fully sure, but the structured fields below are my guess.",
        "fix": "Maybe change the rounding logic somehow.",
        "bug_file": "repo/math_utils.py",
        "bug_function": "round_currency",
        "mechanism": "banker rounding",
        "proposed_fix": "Use Decimal quantize with ROUND_HALF_UP.",
    })

    assert done is True
    assert info["reward_breakdown"]["reasoning_score"] == 0.0
    assert info["reward_breakdown"]["fix_bonus"] == 0.0
    assert info["reward_breakdown"]["wrong_penalty"] < 0.0
    assert info["solved"] is False


def test_terminal_solved_false_when_free_text_contradicts_correct_structured_fields():
    env = BugInvestigationEnv(max_steps=10, max_invalid_actions=5, episode_timeout_seconds=60.0)
    env.reset()

    env.step({"type": "run_tests"})
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
        "root_cause": "The bug is in config.py, not in the rounding helper.",
        "fix": "Use Decimal quantize with ROUND_HALF_UP.",
        "bug_file": "repo/math_utils.py",
        "bug_function": "round_currency",
        "mechanism": "banker rounding",
    })

    assert done is True
    assert info["reward_breakdown"]["contradiction_penalty"] < 0.0
    assert info["reward_breakdown"]["correctness"] <= 0.5
    assert info["solved"] is False


def test_terminal_solved_false_when_fix_is_self_negating():
    env = BugInvestigationEnv(max_steps=10, max_invalid_actions=5, episode_timeout_seconds=60.0)
    env.reset()

    env.step({"type": "run_tests"})
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
        "fix": "Do not change the implementation; just document Decimal quantize and ROUND_HALF_UP.",
        "bug_file": "repo/math_utils.py",
        "bug_function": "round_currency",
        "mechanism": "banker rounding",
    })

    assert done is True
    assert info["reward_breakdown"]["fix_bonus"] == 0.0
    assert info["solved"] is False


def test_terminal_solved_false_without_direct_bug_evidence():
    env = BugInvestigationEnv(max_steps=10, max_invalid_actions=5, episode_timeout_seconds=60.0)
    env.reset()

    env.step({"type": "run_tests"})
    env.step({"type": "search", "keyword": "discount"})
    env.step({
        "type": "inspect_function",
        "filename": "repo/discount.py",
        "function": "apply_discount",
    })
    _, _, done, info = env.step({
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
    assert info["reward_breakdown"]["direct_bug_evidence"] is False
    assert info["reward_breakdown"]["correctness"] <= 0.5
    assert info["reward_breakdown"]["evidence_penalty"] < 0.0
    assert info["solved"] is False


def test_opening_bug_file_without_inspecting_bug_function_is_not_direct_evidence():
    env = BugInvestigationEnv(max_steps=10, max_invalid_actions=5, episode_timeout_seconds=60.0)
    env.reset()

    env.step({"type": "run_tests"})
    env.step({"type": "search", "keyword": "discount"})
    env.step({"type": "open_file", "filename": "repo/math_utils.py"})
    env.step({
        "type": "inspect_function",
        "filename": "repo/discount.py",
        "function": "apply_discount",
    })
    _, _, done, info = env.step({
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
    assert info["reward_breakdown"]["direct_bug_evidence"] is False
    assert info["reward_breakdown"]["evidence_penalty"] < 0.0
    assert info["solved"] is False


def test_keyword_only_submission_does_not_unlock_solved_episode():
    env = BugInvestigationEnv(max_steps=10, max_invalid_actions=5, episode_timeout_seconds=60.0)
    env.reset()

    env.step({"type": "run_tests"})
    env.step({"type": "search", "keyword": "discount"})
    env.step({"type": "open_file", "filename": "repo/math_utils.py"})
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
        "root_cause": "repo/math_utils.py round_currency banker",
        "fix": "Change to Decimal quantize with ROUND_HALF_UP.",
        "bug_file": "repo/math_utils.py",
        "bug_function": "round_currency",
        "mechanism": "banker rounding",
    })

    assert done is True
    assert info["reward_breakdown"]["reasoning_score"] < 0.5
    assert info["reward_breakdown"]["wrong_penalty"] < 0.0
    assert info["solved"] is False


def test_terminal_solved_false_without_required_path_evidence():
    env = BugInvestigationEnv(max_steps=10, max_invalid_actions=5, episode_timeout_seconds=60.0)
    env.reset()

    env.step({"type": "run_tests"})
    env.step({"type": "search", "keyword": "round"})
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
    assert reward <= 0.0
    assert info["reward_breakdown"]["required_path_evidence"] is False
    assert info["reward_breakdown"]["correctness"] == 0.0
    assert info["reward_breakdown"]["fix_bonus"] == 0.0
    assert info["solved"] is False


def test_immediate_correct_guess_without_investigation_is_not_rewarded():
    env = BugInvestigationEnv(max_steps=10, max_invalid_actions=5, episode_timeout_seconds=60.0)
    env.reset()

    _, reward, done, info = env.step({
        "type": "submit_answer",
        "root_cause": (
            "Root cause is in repo/math_utils.py round_currency because it uses "
            "banker half-to-even rounding, which rounds 20.825 down to 20.82 "
            "and causes the one-cent drift."
        ),
        "fix": "Use Decimal quantize with ROUND_HALF_UP.",
    })

    assert done is True
    assert reward <= 0.0
    assert info["reward_breakdown"]["correctness"] == 0.0
    assert info["reward_breakdown"]["evidence_penalty"] < 0.0
    assert info["solved"] is False


def test_step_rewards_sum_to_terminal_episode_total():
    env = BugInvestigationEnv(max_steps=10, max_invalid_actions=5, episode_timeout_seconds=60.0)
    env.reset()

    total_reward = 0.0
    _, reward_1, _, _ = env.step({"type": "search", "keyword": "discount"})
    total_reward += reward_1
    _, reward_2, done, info = env.step({
        "type": "submit_answer",
        "root_cause": "Bug is in config defaults.",
        "fix": "Change the minimum order threshold.",
    })
    total_reward += reward_2

    assert done is True
    assert total_reward == info["reward_breakdown"]["total"]
    assert math.isclose(
        info["reward_emitted_so_far"],
        info["reward_breakdown"]["total"],
        abs_tol=1e-12,
    )


def test_terminal_info_exposes_solved_and_trajectory():
    env = BugInvestigationEnv(max_steps=10, max_invalid_actions=5, episode_timeout_seconds=60.0)
    env.reset()

    env.step({"type": "run_tests"})
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
    assert info["correct_function_inspected"] is True
    assert math.isclose(
        info["reward_emitted_so_far"],
        info["reward_breakdown"]["total"],
        abs_tol=1e-12,
    )
    assert info["trajectory_length"] == 5
    assert info["last_transition"] is not None
    assert info["last_transition"]["done"] is True
