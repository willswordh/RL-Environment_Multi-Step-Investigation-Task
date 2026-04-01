"""Tests for config-backed reward loading and aggregation."""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.episode_state import EpisodeState
from env.reward import (
    award_shaping_reward,
    compute_reward,
    compute_step_reward,
    compute_terminal_reward_delta,
    load_reward_config,
)
from env.task_config import load_task

DEFAULT_TASK = load_task()


def make_state(max_steps=15):
    return EpisodeState.for_task(DEFAULT_TASK, max_steps=max_steps)


def test_reward_config_loads_expected_sections():
    config = load_reward_config()

    assert config.terminal_rewards["correct_root_cause"] == 1.0
    assert config.terminal_rewards["correct_mechanism"] == 0.25
    assert config.terminal_rewards["correct_fix"] == 0.20
    assert config.step_penalties["per_step"] == -0.01
    assert config.shaping_rewards["first_test_run"] == 0.05
    assert config.anti_hacking_penalties["wrong_submission"] == -0.40
    assert config.thresholds["solved_correctness"] == 0.80
    assert config.caps["shaping_max_total"] == 0.20


def test_step_reward_uses_config_values():
    config = load_reward_config()
    state = make_state()

    reward = compute_step_reward(
        state,
        invalid_action=True,
        repeated_action=True,
        shaping_reward=config.shaping_rewards["first_test_run"],
        config=config,
    )

    expected = (
        config.step_penalties["per_step"]
        + config.step_penalties["invalid_action"]
        + config.step_penalties["repeated_action"]
        + config.shaping_rewards["first_test_run"]
    )
    assert reward == expected


def test_success_breakdown_uses_configured_component_values():
    config = load_reward_config()
    state = make_state()
    state.step_count = 5
    state.tests_run = True
    state.record_search("discount", useful=True)
    state.record_function_inspected("repo/discount.py", "apply_discount")
    state.record_function_inspected("repo/math_utils.py", "round_currency")
    state.mark_shaping_event_once(
        "first_test_run",
        reward_value=config.shaping_rewards["first_test_run"],
        shaping_max_total=config.caps["shaping_max_total"],
    )
    state.mark_shaping_event_once(
        "useful_search",
        reward_value=config.shaping_rewards["useful_search"],
        shaping_max_total=config.caps["shaping_max_total"],
    )
    state.mark_shaping_event_once(
        "correct_file_opened",
        reward_value=config.shaping_rewards["correct_file_opened"],
        shaping_max_total=config.caps["shaping_max_total"],
    )
    state.mark_shaping_event_once(
        "correct_function_inspected",
        reward_value=config.shaping_rewards["correct_function_inspected"],
        shaping_max_total=config.caps["shaping_max_total"],
    )
    state.submitted = True
    state.submission_root_cause = (
        "Root cause is in repo/math_utils.py round_currency: banker half-to-even "
        "rounding causes 0.01 drift."
    )
    state.submission_fix = "Use Decimal quantize with ROUND_HALF_UP."

    breakdown = compute_reward(state, config=config)

    assert breakdown.correctness_score == config.terminal_rewards["correct_root_cause"]
    assert breakdown.mechanism_bonus == config.terminal_rewards["correct_mechanism"]
    assert breakdown.fix_bonus == config.terminal_rewards["correct_fix"]
    assert breakdown.file_bonus == config.terminal_rewards["correct_file"]
    assert breakdown.function_bonus == config.terminal_rewards["correct_function"]
    assert breakdown.shaping_reward_total == pytest.approx(
        sum(config.shaping_rewards.values()),
        abs=1e-12,
    )
    assert breakdown.step_penalty == config.step_penalties["per_step"] * state.step_count


def test_penalty_components_use_configured_values():
    config = load_reward_config()

    evidence_state = make_state()
    evidence_state.step_count = 3
    evidence_state.record_function_inspected("repo/math_utils.py", "round_currency")
    evidence_state.submitted = True
    evidence_state.submission_root_cause = (
        "Root cause is in repo/math_utils.py round_currency: banker half-to-even rounding."
    )
    evidence_state.submission_fix = "Use Decimal quantize with ROUND_HALF_UP."
    evidence_breakdown = compute_reward(evidence_state, config=config)
    assert evidence_breakdown.evidence_penalty == (
        config.anti_hacking_penalties["insufficient_evidence"]
    )

    wrong_state = make_state()
    wrong_state.step_count = 2
    wrong_state.submitted = True
    wrong_state.submission_root_cause = "Bug is in config defaults."
    wrong_breakdown = compute_reward(wrong_state, config=config)
    assert wrong_breakdown.wrong_penalty == config.anti_hacking_penalties["wrong_submission"]

    contradiction_state = make_state()
    contradiction_state.step_count = 4
    contradiction_state.tests_run = True
    contradiction_state.record_search("discount", useful=True)
    contradiction_state.record_function_inspected("repo/discount.py", "apply_discount")
    contradiction_state.record_function_inspected("repo/math_utils.py", "round_currency")
    contradiction_state.submitted = True
    contradiction_state.submission_root_cause = (
        "The bug is in config.py, not in math_utils round_currency."
    )
    contradiction_state.submission_bug_file = "repo/math_utils.py"
    contradiction_state.submission_bug_function = "round_currency"
    contradiction_state.submission_mechanism = "banker rounding"
    contradiction_breakdown = compute_reward(contradiction_state, config=config)
    assert contradiction_breakdown.contradiction_penalty == (
        config.anti_hacking_penalties["contradiction"]
    )

    terminal_state = make_state(max_steps=3)
    terminal_state.step_count = 3
    terminal_state.invalid_action_count = 2
    terminal_state.repeated_action_count = 1
    terminal_state.timed_out = True
    terminal_state.max_invalid_reached = True
    terminal_state.stop_reason = "max_invalid_actions"
    terminal_breakdown = compute_reward(terminal_state, config=config)
    assert terminal_breakdown.budget_penalty == config.anti_hacking_penalties["budget_exhausted"]
    assert terminal_breakdown.timeout_penalty == config.anti_hacking_penalties["timeout"]
    assert terminal_breakdown.max_invalid_penalty == (
        config.anti_hacking_penalties["max_invalid_reached"]
    )
    assert terminal_breakdown.invalid_action_penalty == (
        config.step_penalties["invalid_action"] * terminal_state.invalid_action_count
    )
    assert terminal_breakdown.repeated_action_penalty == (
        config.step_penalties["repeated_action"] * terminal_state.repeated_action_count
    )


def test_shaping_rewards_are_capped_and_one_shot():
    config = load_reward_config()
    state = make_state()
    state.tests_run = True
    state.useful_search_observed = True
    state.correct_file_opened = True
    state.correct_function_inspected = True

    first_award = award_shaping_reward(state, config=config)
    second_award = award_shaping_reward(state, config=config)

    assert math.isclose(first_award, sum(config.shaping_rewards.values()), abs_tol=1e-12)
    assert second_award == 0.0
    assert state.shaping_reward_total <= config.caps["shaping_max_total"]


def test_total_reward_aggregation_is_deterministic():
    config = load_reward_config()
    state = make_state()
    state.step_count = 5
    state.tests_run = True
    state.record_search("discount", useful=True)
    state.record_function_inspected("repo/discount.py", "apply_discount")
    state.record_function_inspected("repo/math_utils.py", "round_currency")
    state.useful_search_observed = True
    state.shaping_reward_total = sum(config.shaping_rewards.values())
    state.submitted = True
    state.submission_root_cause = (
        "Root cause is in repo/math_utils.py round_currency: banker half-to-even "
        "rounding causes 0.01 drift."
    )
    state.submission_fix = "Use Decimal quantize with ROUND_HALF_UP."

    first = compute_reward(state, config=config)
    second = compute_reward(state, config=config)

    assert math.isclose(first.total, second.total, abs_tol=1e-12)
    assert first.component_values() == second.component_values()

    emitted = compute_step_reward(
        state,
        shaping_reward=config.shaping_rewards["first_test_run"],
        config=config,
    )
    terminal_delta = compute_terminal_reward_delta(first, reward_emitted_so_far=emitted)
    assert math.isclose(emitted + terminal_delta, first.total, abs_tol=1e-12)
