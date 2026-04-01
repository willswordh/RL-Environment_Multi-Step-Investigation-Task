"""Tests for simplified reward gating and anti-hacking behavior."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.episode_state import EpisodeState
from env.reward import award_shaping_reward, compute_reward, load_reward_config
from env.task_config import load_task

DEFAULT_TASK = load_task()


def make_state(max_steps=15):
    return EpisodeState.for_task(DEFAULT_TASK, max_steps=max_steps)


def _mark_full_evidence(state, config):
    state.tests_run = True
    state.record_search("discount", useful=True)
    state.record_function_inspected("repo/discount.py", "apply_discount")
    state.record_function_inspected("repo/math_utils.py", "round_currency")
    award_shaping_reward(state, config=config)


def test_correct_root_cause_but_low_evidence_does_not_get_full_reward():
    config = load_reward_config()
    state = make_state()
    state.step_count = 3
    state.record_function_inspected("repo/math_utils.py", "round_currency")
    state.submitted = True
    state.submission_root_cause = (
        "Root cause is in repo/math_utils.py round_currency: banker half-to-even "
        "rounding causes the cent drift."
    )
    state.submission_fix = "Use Decimal quantize with ROUND_HALF_UP."

    breakdown = compute_reward(state, config=config)

    assert breakdown.reasoning_score == 1.0
    assert breakdown.evidence_score < config.thresholds["evidence_required_for_full_credit"]
    assert breakdown.correctness_score == 0.0
    assert breakdown.evidence_penalty == config.anti_hacking_penalties["insufficient_evidence"]
    assert breakdown.fix_bonus == 0.0
    assert breakdown.file_bonus == 0.0
    assert breakdown.function_bonus == 0.0


def test_wrong_root_cause_does_not_unlock_terminal_bonuses():
    config = load_reward_config()
    state = make_state()
    state.step_count = 4
    _mark_full_evidence(state, config)
    state.submitted = True
    state.submission_root_cause = "Bug is in config defaults."
    state.submission_fix = "Change the minimum order threshold."

    breakdown = compute_reward(state, config=config)

    assert breakdown.reasoning_score == 0.0
    assert breakdown.correctness_score == 0.0
    assert breakdown.mechanism_bonus == 0.0
    assert breakdown.fix_bonus == 0.0
    assert breakdown.file_bonus == 0.0
    assert breakdown.function_bonus == 0.0
    assert breakdown.wrong_penalty == config.anti_hacking_penalties["wrong_submission"]


def test_shaping_reward_cap_works():
    config = load_reward_config()
    state = make_state()
    state.tests_run = True
    state.useful_search_observed = True
    state.correct_file_opened = True
    state.correct_function_inspected = True

    awarded = award_shaping_reward(state, config=config)

    assert awarded == pytest.approx(sum(config.shaping_rewards.values()), abs=1e-12)
    assert state.shaping_reward_total <= config.caps["shaping_max_total"]


def test_repeated_event_farming_does_not_increase_shaping_reward():
    config = load_reward_config()
    state = make_state()
    state.tests_run = True

    first = award_shaping_reward(state, config=config)
    second = award_shaping_reward(state, config=config)

    assert first == config.shaping_rewards["first_test_run"]
    assert second == 0.0
    assert state.shaping_reward_total == config.shaping_rewards["first_test_run"]


def test_correct_solve_with_adequate_evidence_gets_full_terminal_reward():
    config = load_reward_config()
    state = make_state()
    state.step_count = 5
    _mark_full_evidence(state, config)
    state.submitted = True
    state.submission_root_cause = (
        "Root cause is in repo/math_utils.py round_currency: banker half-to-even "
        "rounding causes the cent drift."
    )
    state.submission_fix = "Use Decimal quantize with ROUND_HALF_UP."

    breakdown = compute_reward(state, config=config)

    assert breakdown.reasoning_score == 1.0
    assert breakdown.evidence_score == 1.0
    assert breakdown.correctness_score == config.terminal_rewards["correct_root_cause"]
    assert breakdown.mechanism_bonus == config.terminal_rewards["correct_mechanism"]
    assert breakdown.fix_bonus == config.terminal_rewards["correct_fix"]
    assert breakdown.file_bonus == config.terminal_rewards["correct_file"]
    assert breakdown.function_bonus == config.terminal_rewards["correct_function"]
    assert breakdown.evidence_penalty == 0.0
    assert breakdown.wrong_penalty == 0.0


def test_low_quality_submission_gets_wrong_submission_penalty():
    config = load_reward_config()
    state = make_state()
    state.step_count = 2
    state.submitted = True
    state.submission_root_cause = "Something is wrong somewhere."
    state.submission_fix = "Try a different constant."

    breakdown = compute_reward(state, config=config)

    assert breakdown.reasoning_score < config.thresholds["reasonable_answer"]
    assert breakdown.wrong_penalty == config.anti_hacking_penalties["wrong_submission"]


def test_structured_fields_do_not_add_positive_reasoning_or_fix_credit():
    config = load_reward_config()
    state = make_state()
    state.step_count = 5
    _mark_full_evidence(state, config)
    state.submitted = True
    state.submission_root_cause = "I am not fully sure, but the structured fields below are my guess."
    state.submission_fix = "Maybe change the rounding logic somehow."
    state.submission_bug_file = "repo/math_utils.py"
    state.submission_bug_function = "round_currency"
    state.submission_mechanism = "banker half-to-even rounding"
    state.submission_fix_summary = "Use Decimal quantize with ROUND_HALF_UP."

    breakdown = compute_reward(state, config=config)

    assert breakdown.reasoning_score == 0.0
    assert breakdown.correctness_score == 0.0
    assert breakdown.fix_bonus == 0.0
    assert breakdown.wrong_penalty == config.anti_hacking_penalties["wrong_submission"]


def test_keyword_bag_submission_does_not_count_as_reasoned_answer():
    config = load_reward_config()
    state = make_state()
    state.step_count = 5
    _mark_full_evidence(state, config)
    state.submitted = True
    state.submission_root_cause = "repo/math_utils.py round_currency banker"
    state.submission_fix = "Change to Decimal quantize with ROUND_HALF_UP."

    breakdown = compute_reward(state, config=config)

    assert breakdown.reasoning_score < config.thresholds["reasonable_answer"]
    assert breakdown.correctness_score == 0.0
    assert breakdown.wrong_penalty == config.anti_hacking_penalties["wrong_submission"]
    assert breakdown.fix_bonus == 0.0


def test_explicit_wrong_free_text_triggers_contradiction_penalty():
    config = load_reward_config()
    state = make_state()
    state.step_count = 5
    _mark_full_evidence(state, config)
    state.submitted = True
    state.submission_root_cause = "The bug is in config.py, not in the rounding helper."
    state.submission_bug_file = "repo/math_utils.py"
    state.submission_bug_function = "round_currency"
    state.submission_mechanism = "banker half-to-even rounding"
    state.submission_fix_summary = "Use Decimal quantize with ROUND_HALF_UP."

    breakdown = compute_reward(state, config=config)

    assert breakdown.contradiction_penalty == config.anti_hacking_penalties["contradiction"]
    assert breakdown.correctness_score <= config.thresholds["reasonable_answer"]
