"""Tests for reward-quality safeguards (anti-shortcut behavior)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.episode_state import EpisodeState
from env.task_config import load_task
from env.reward import compute_reward

DEFAULT_TASK = load_task()


def make_state(max_steps=15):
    return EpisodeState.for_task(DEFAULT_TASK, max_steps=max_steps)


def test_keyword_rich_submission_without_evidence_is_penalized():
    state = make_state(max_steps=15)
    state.step_count = 1
    state.submitted = True
    state.submission_root_cause = (
        "math_utils round_currency uses banker half-to-even rounding"
    )
    state.submission_fix = "Use Decimal quantize with round half up."

    breakdown = compute_reward(state)

    assert breakdown.reasoning_score >= 0.8
    assert breakdown.evidence_score == 0.0
    assert breakdown.correctness_score <= 0.2
    assert breakdown.evidence_penalty < 0.0
    assert breakdown.fix_bonus == 0.0


def test_well_investigated_submission_scores_high():
    state = make_state(max_steps=15)
    state.step_count = 5
    state.tests_run = True
    state.keywords_searched.append("discount")
    state.record_function_inspected("repo/discount.py", "apply_discount")
    state.record_function_inspected("repo/math_utils.py", "round_currency")
    state.submitted = True
    state.submission_root_cause = (
        "Root cause is in math_utils round_currency: banker half-to-even "
        "rounding causes 0.01 drift."
    )
    state.submission_fix = "Use Decimal quantize with ROUND_HALF_UP."

    breakdown = compute_reward(state)

    assert breakdown.reasoning_score == 1.0
    assert breakdown.evidence_score == pytest.approx(1.0, abs=1e-9)
    assert breakdown.correctness_score == 1.0
    assert breakdown.fix_bonus == 0.5
    assert breakdown.evidence_penalty == 0.0
    assert breakdown.wrong_penalty == 0.0


def test_wrong_submission_receives_wrong_penalty():
    state = make_state(max_steps=15)
    state.step_count = 2
    state.submitted = True
    state.submission_root_cause = "Bug is in config defaults."
    state.submission_fix = "Change the minimum order threshold."

    breakdown = compute_reward(state)

    assert breakdown.reasoning_score == 0.0
    assert breakdown.wrong_penalty == -0.5


def test_structured_submission_can_drive_reasoning_score():
    state = make_state(max_steps=15)
    state.step_count = 4
    state.tests_run = True
    state.record_function_inspected("repo/discount.py", "apply_discount")
    state.record_function_inspected("repo/math_utils.py", "round_currency")
    state.submitted = True
    state.submission_root_cause = "Issue identified."
    state.submission_bug_file = "repo/math_utils.py"
    state.submission_bug_function = "round_currency"
    state.submission_mechanism = "banker half-to-even rounding causes cent-level error"
    state.submission_fix_summary = "Use Decimal quantize with ROUND_HALF_UP."

    breakdown = compute_reward(state)

    assert breakdown.reasoning_score == 1.0
    assert breakdown.correctness_score == pytest.approx(1.0, abs=1e-9)
    assert breakdown.fix_bonus == 0.5


def test_structured_contradiction_penalty_applies():
    state = make_state(max_steps=15)
    state.step_count = 4
    state.tests_run = True
    state.record_function_inspected("repo/discount.py", "apply_discount")
    state.record_function_inspected("repo/math_utils.py", "round_currency")
    state.keywords_searched.append("round")
    state.submitted = True
    state.submission_root_cause = (
        "The bug is in math_utils round_currency and uses banker half-to-even rounding."
    )
    state.submission_bug_file = "repo/config.py"  # contradictory structured field
    state.submission_bug_function = "round_currency"
    state.submission_mechanism = "banker rounding"

    breakdown = compute_reward(state)

    assert breakdown.reasoning_score == 1.0
    assert breakdown.contradiction_penalty < 0.0
