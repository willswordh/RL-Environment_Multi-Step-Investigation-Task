"""
Reward computation for BugInvestigationEnv.

Reward structure (three tiers):
  1. Correctness  — dense partial credit based on submission quality.
  2. Efficiency   — per-step penalty to encourage concise investigation.
  3. Commitment   — terminal penalty for exhausting the budget without submitting.

Score range: approximately [-1.65, +1.92]
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from env.episode_state import EpisodeState


# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------

# Correctness (submission)
R_CORRECT_ROOT_CAUSE   = +1.00   # agent correctly explains the root cause
R_CORRECT_FIX          = +0.50   # agent proposes a correct fix
R_CORRECT_FILE         = +0.20   # agent names the correct file in submission
R_CORRECT_FUNCTION     = +0.20   # agent names the correct function in submission
R_WRONG_SUBMISSION     = -0.50   # confident wrong answer

# Efficiency
R_STEP_PENALTY         = -0.02   # charged per step taken
R_INVALID_ACTION_STEP  = -0.03   # charged for invalid action format/args
R_REPEATED_ACTION_STEP = -0.01   # charged for immediate repeated actions

# Commitment
R_BUDGET_EXHAUSTED     = -0.30   # episode ended by max_steps, no submission
R_INSUFFICIENT_EVIDENCE = -0.25  # strong claim without enough investigation trace
R_MAX_INVALID_REACHED  = -0.20   # stop due to too many invalid actions
R_TIMEOUT              = -0.20   # stop due to wall-clock timeout
R_CONTRADICTION        = -0.15   # structured submission fields contradict each other

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class RewardBreakdown:
    """Detailed record of how the episode reward was computed."""
    reasoning_score: float    = 0.0   # [0.0, 1.0] from submission text only
    evidence_score: float     = 0.0   # [0.0, 1.0] from investigation actions
    correctness_score: float = 0.0   # [0.0, 1.0] — partial credit
    fix_bonus: float         = 0.0   # 0 or R_CORRECT_FIX
    file_bonus: float        = 0.0   # 0 or R_CORRECT_FILE
    function_bonus: float    = 0.0   # 0 or R_CORRECT_FUNCTION
    wrong_penalty: float     = 0.0   # 0 or R_WRONG_SUBMISSION
    evidence_penalty: float  = 0.0   # 0 or R_INSUFFICIENT_EVIDENCE
    contradiction_penalty: float = 0.0  # 0 or R_CONTRADICTION
    step_penalty: float      = 0.0   # R_STEP_PENALTY * steps_taken
    invalid_action_penalty: float = 0.0  # R_INVALID_ACTION_STEP * invalid actions
    repeated_action_penalty: float = 0.0  # R_REPEATED_ACTION_STEP * repeated actions
    budget_penalty: float    = 0.0   # 0 or R_BUDGET_EXHAUSTED
    timeout_penalty: float   = 0.0   # 0 or R_TIMEOUT
    max_invalid_penalty: float = 0.0  # 0 or R_MAX_INVALID_REACHED

    @property
    def total(self) -> float:
        return (
            self.correctness_score
            + self.fix_bonus
            + self.file_bonus
            + self.function_bonus
            + self.wrong_penalty
            + self.evidence_penalty
            + self.contradiction_penalty
            + self.step_penalty
            + self.invalid_action_penalty
            + self.repeated_action_penalty
            + self.budget_penalty
            + self.timeout_penalty
            + self.max_invalid_penalty
        )

    def as_dict(self) -> dict:
        return {
            "reasoning_score":  self.reasoning_score,
            "evidence_score":   self.evidence_score,
            "correctness":      self.correctness_score,
            "fix_bonus":        self.fix_bonus,
            "file_bonus":       self.file_bonus,
            "function_bonus":   self.function_bonus,
            "wrong_penalty":    self.wrong_penalty,
            "evidence_penalty": self.evidence_penalty,
            "contradiction_penalty": self.contradiction_penalty,
            "step_penalty":     self.step_penalty,
            "invalid_action_penalty": self.invalid_action_penalty,
            "repeated_action_penalty": self.repeated_action_penalty,
            "budget_penalty":   self.budget_penalty,
            "timeout_penalty":  self.timeout_penalty,
            "max_invalid_penalty": self.max_invalid_penalty,
            "total":            self.total,
        }


def compute_reward(state: EpisodeState) -> RewardBreakdown:
    """Compute the episode reward given the terminal EpisodeState.

    This function should be called once, at the end of the episode
    (either after submit_answer or after budget exhaustion).

    Args:
        state: The final episode state.

    Returns:
        A RewardBreakdown with the total reward and a full breakdown.
    """
    breakdown = RewardBreakdown()

    # ── step penalty (always applied) ───────────────────────────────────
    breakdown.step_penalty = R_STEP_PENALTY * state.step_count
    breakdown.invalid_action_penalty = R_INVALID_ACTION_STEP * state.invalid_action_count
    breakdown.repeated_action_penalty = R_REPEATED_ACTION_STEP * state.repeated_action_count

    if state.timed_out or state.stop_reason == "timeout":
        breakdown.timeout_penalty = R_TIMEOUT

    if state.max_invalid_reached or state.stop_reason == "max_invalid_actions":
        breakdown.max_invalid_penalty = R_MAX_INVALID_REACHED

    # ── budget exhausted without submission ─────────────────────────────
    if not state.submitted:
        if state.budget_exhausted:
            breakdown.budget_penalty = R_BUDGET_EXHAUSTED
        return breakdown

    # ── submission evaluation ────────────────────────────────────────────
    root_cause_text = state.submission_root_cause.lower()
    fix_text        = state.submission_fix.lower()

    text_reasoning = _evaluate_root_cause(root_cause_text, state)
    structured_reasoning = _evaluate_structured_submission(state)
    reasoning = max(text_reasoning, structured_reasoning)
    evidence = _evaluate_investigation_evidence(state)
    corrected_score = _apply_evidence_gate(reasoning, evidence)
    contradiction = _evaluate_submission_contradiction(state)

    breakdown.reasoning_score = reasoning
    breakdown.evidence_score = evidence
    breakdown.correctness_score = corrected_score
    breakdown.contradiction_penalty = contradiction

    if corrected_score >= 0.5:
        # Reasonable root cause identified — check for correct fix
        structured_fix_text = state.submission_fix_summary.lower()
        if _evaluate_fix(f"{fix_text}\n{structured_fix_text}", state):
            breakdown.fix_bonus = R_CORRECT_FIX

        # Partial credit: correct file mentioned in submission
        if (
            any(signal in root_cause_text for signal in _bug_file_text_signals(state.bug_file))
            or _is_correct_bug_file(state.submission_bug_file, state.bug_file)
        ):
            breakdown.file_bonus = R_CORRECT_FILE

        # Partial credit: correct function mentioned in submission
        if (
            state.bug_function.lower() in root_cause_text
            or _is_correct_bug_function(state.submission_bug_function, state.bug_function)
        ):
            breakdown.function_bonus = R_CORRECT_FUNCTION
    else:
        if reasoning >= 0.5 and evidence < 0.4:
            # Submission text appears correct but lacks investigation trace.
            breakdown.evidence_penalty = R_INSUFFICIENT_EVIDENCE
        else:
            # Wrong answer — apply penalty
            breakdown.wrong_penalty = R_WRONG_SUBMISSION

    return breakdown


def compute_step_reward(
    state: EpisodeState,
    *,
    invalid_action: bool = False,
    repeated_action: bool = False,
) -> float:
    """Return per-step shaping reward.

    Called after each non-terminal step so the agent gets immediate
    feedback about efficiency, rather than waiting until episode end.
    """
    reward = R_STEP_PENALTY
    if invalid_action:
        reward += R_INVALID_ACTION_STEP
    if repeated_action:
        reward += R_REPEATED_ACTION_STEP
    return reward


# ---------------------------------------------------------------------------
# Private evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate_root_cause(text: str, state: EpisodeState) -> float:
    """Score a root-cause explanation in [0, 1].

    Scoring breakdown:
      0.50 — correct file mentioned
      0.30 — correct function mentioned
      0.20 — correct mechanism mentioned

    Returns a float in [0.0, 1.0].  A score >= 0.5 is considered a
    "reasonable" answer that avoids the wrong-answer penalty.
    """
    score = 0.0

    if any(signal in text for signal in _bug_file_text_signals(state.bug_file)):
        score += 0.50

    if state.bug_function and state.bug_function.lower() in text:
        score += 0.30

    if _has_mechanism_keyword(text, state.mechanism_keywords):
        score += 0.20

    return min(score, 1.0)


def _evaluate_investigation_evidence(state: EpisodeState) -> float:
    """Score how much evidence the agent gathered before submitting."""
    score = 0.0

    if state.tests_run:
        score += 0.25

    if state.keywords_searched:
        score += 0.15

    if state.evidence_function_entries:
        if any(entry in state.functions_inspected for entry in state.evidence_function_entries):
            score += 0.20
    elif state.functions_inspected:
        score += 0.20

    if state.correct_file_opened:
        score += 0.20

    if state.correct_function_inspected:
        score += 0.20

    return min(score, 1.0)


def _evaluate_structured_submission(state: EpisodeState) -> float:
    """Score correctness from structured submission fields."""
    score = 0.0
    if _is_correct_bug_file(state.submission_bug_file, state.bug_file):
        score += 0.50
    if _is_correct_bug_function(state.submission_bug_function, state.bug_function):
        score += 0.30
    if _has_mechanism_keyword(state.submission_mechanism.lower(), state.mechanism_keywords):
        score += 0.20
    return min(score, 1.0)


def _evaluate_submission_contradiction(state: EpisodeState) -> float:
    """Penalty for explicit structured contradictions."""
    file_present = bool(state.submission_bug_file)
    function_present = bool(state.submission_bug_function)

    if file_present and not _is_correct_bug_file(state.submission_bug_file, state.bug_file):
        return R_CONTRADICTION
    if function_present and not _is_correct_bug_function(
        state.submission_bug_function, state.bug_function
    ):
        return R_CONTRADICTION
    return 0.0


def _is_correct_bug_file(candidate: str, bug_file: str) -> bool:
    c = (candidate or "").replace("\\", "/").strip().lower()
    b = bug_file.replace("\\", "/").strip().lower()
    if not c:
        return False
    return c == b or c.endswith("/" + b) or b.endswith("/" + c)


def _is_correct_bug_function(candidate: str, bug_function: str) -> bool:
    c = (candidate or "").strip().lower()
    b = bug_function.strip().lower()
    if not c:
        return False
    return c == b


def _has_mechanism_keyword(text: str, keywords) -> bool:
    return any(kw.lower() in text for kw in keywords)


def _bug_file_text_signals(bug_file: str):
    filename = os.path.basename(bug_file).lower()
    stem = os.path.splitext(filename)[0]
    candidates = [bug_file.lower(), filename, stem]
    return [item for idx, item in enumerate(candidates) if item and item not in candidates[:idx]]


def _apply_evidence_gate(reasoning_score: float, evidence_score: float) -> float:
    """Cap text-only correctness when investigation evidence is weak."""
    # With zero evidence, max correctness is 0.2; with full evidence, full score is possible.
    evidence_cap = min(1.0, evidence_score + 0.2)
    return min(reasoning_score, evidence_cap)


def _evaluate_fix(text: str, state: EpisodeState) -> bool:
    """Return True if the proposed fix mentions the correct solution.

    A correct fix must reference at least one task-specific fix signal.
    """
    return any(kw.lower() in text for kw in state.fix_keywords)
