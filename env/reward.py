"""
Reward computation for BugInvestigationEnv.

Policy values live in ``reward_config.yaml``. This module keeps the
execution logic in Python so the environment can express task-specific
matching, evidence validation, and anti-double-counting behavior without
introducing a schema engine.
"""
from __future__ import annotations

import ast
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional

from env.episode_state import EpisodeState


_DEFAULT_REWARD_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "reward_config.yaml",
)


@dataclass(frozen=True)
class RewardConfig:
    """Loaded reward policy values."""

    terminal_rewards: Dict[str, float]
    step_penalties: Dict[str, float]
    shaping_rewards: Dict[str, float]
    anti_hacking_penalties: Dict[str, float]
    thresholds: Dict[str, float]
    caps: Dict[str, float]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RewardConfig":
        terminal_rewards = _require_float_mapping(raw, "terminal_rewards")
        step_penalties = _require_float_mapping(raw, "step_penalties")
        shaping_rewards = _require_float_mapping(raw, "shaping_rewards")
        anti_hacking_penalties = _require_float_mapping(raw, "anti_hacking_penalties")
        thresholds = _require_float_mapping(raw, "thresholds")
        caps = _require_float_mapping(raw, "caps")

        return cls(
            terminal_rewards=dict(terminal_rewards),
            step_penalties=dict(step_penalties),
            shaping_rewards=dict(shaping_rewards),
            anti_hacking_penalties=dict(anti_hacking_penalties),
            thresholds=dict(thresholds),
            caps=dict(caps),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "terminal_rewards": dict(self.terminal_rewards),
            "step_penalties": dict(self.step_penalties),
            "shaping_rewards": dict(self.shaping_rewards),
            "anti_hacking_penalties": dict(self.anti_hacking_penalties),
            "thresholds": dict(self.thresholds),
            "caps": dict(self.caps),
        }


@dataclass
class RewardBreakdown:
    """Detailed record of how the episode reward was computed."""

    reasoning_score: float = 0.0
    evidence_score: float = 0.0
    direct_bug_evidence: bool = False
    required_path_evidence: bool = False
    correctness_score: float = 0.0
    mechanism_bonus: float = 0.0
    fix_bonus: float = 0.0
    file_bonus: float = 0.0
    function_bonus: float = 0.0
    shaping_reward_total: float = 0.0
    wrong_penalty: float = 0.0
    evidence_penalty: float = 0.0
    contradiction_penalty: float = 0.0
    step_penalty: float = 0.0
    invalid_action_penalty: float = 0.0
    repeated_action_penalty: float = 0.0
    budget_penalty: float = 0.0
    timeout_penalty: float = 0.0
    max_invalid_penalty: float = 0.0

    @property
    def total(self) -> float:
        return sum(self.component_values().values())

    def component_values(self) -> Dict[str, float]:
        return {
            "correctness": self.correctness_score,
            "mechanism_bonus": self.mechanism_bonus,
            "fix_bonus": self.fix_bonus,
            "file_bonus": self.file_bonus,
            "function_bonus": self.function_bonus,
            "shaping_reward_total": self.shaping_reward_total,
            "wrong_penalty": self.wrong_penalty,
            "evidence_penalty": self.evidence_penalty,
            "contradiction_penalty": self.contradiction_penalty,
            "step_penalty": self.step_penalty,
            "invalid_action_penalty": self.invalid_action_penalty,
            "repeated_action_penalty": self.repeated_action_penalty,
            "budget_penalty": self.budget_penalty,
            "timeout_penalty": self.timeout_penalty,
            "max_invalid_penalty": self.max_invalid_penalty,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "reasoning_score": self.reasoning_score,
            "evidence_score": self.evidence_score,
            "direct_bug_evidence": self.direct_bug_evidence,
            "required_path_evidence": self.required_path_evidence,
            **self.component_values(),
            "total": self.total,
        }


def load_reward_config(path: Optional[str] = None) -> RewardConfig:
    """Load reward policy from ``reward_config.yaml``."""
    resolved = os.path.abspath(path or _DEFAULT_REWARD_CONFIG_PATH)
    return _load_reward_config_cached(resolved)


@lru_cache(maxsize=None)
def _load_reward_config_cached(path: str) -> RewardConfig:
    data = _parse_simple_yaml(path)
    return RewardConfig.from_dict(data)


def compute_reward(
    state: EpisodeState,
    *,
    config: Optional[RewardConfig] = None,
) -> RewardBreakdown:
    """Compute the full episode reward for a terminal state."""
    cfg = config or load_reward_config()
    breakdown = RewardBreakdown()

    breakdown.step_penalty = cfg.step_penalties["per_step"] * state.step_count
    breakdown.shaping_reward_total = state.shaping_reward_total
    breakdown.invalid_action_penalty = (
        cfg.step_penalties["invalid_action"] * state.invalid_action_count
    )
    breakdown.repeated_action_penalty = (
        cfg.step_penalties["repeated_action"] * state.repeated_action_count
    )

    if state.timed_out or state.stop_reason == "timeout":
        breakdown.timeout_penalty = cfg.anti_hacking_penalties["timeout"]

    if state.max_invalid_reached or state.stop_reason == "max_invalid_actions":
        breakdown.max_invalid_penalty = (
            cfg.anti_hacking_penalties["max_invalid_reached"]
        )

    if not state.submitted:
        if state.budget_exhausted:
            breakdown.budget_penalty = cfg.anti_hacking_penalties["budget_exhausted"]
        return breakdown

    reasoning = score_reasoning(state, cfg)
    evidence = score_evidence(state, cfg)
    direct_bug_evidence = _has_direct_bug_evidence(state)
    required_path_evidence = _has_required_path_evidence(state)
    contradiction = _evaluate_submission_contradiction(state, cfg)

    breakdown.reasoning_score = reasoning
    breakdown.evidence_score = evidence
    breakdown.direct_bug_evidence = direct_bug_evidence
    breakdown.required_path_evidence = required_path_evidence
    breakdown.contradiction_penalty = contradiction

    apply_terminal_reward_with_gates(state, breakdown, cfg)

    return breakdown


def compute_step_reward(
    state: EpisodeState,
    *,
    invalid_action: bool = False,
    repeated_action: bool = False,
    shaping_reward: float = 0.0,
    config: Optional[RewardConfig] = None,
) -> float:
    """Return per-step shaping reward."""
    cfg = config or load_reward_config()
    reward = cfg.step_penalties["per_step"]
    if invalid_action:
        reward += cfg.step_penalties["invalid_action"]
    if repeated_action:
        reward += cfg.step_penalties["repeated_action"]
    return reward + shaping_reward


def award_shaping_reward(
    state: EpisodeState,
    *,
    config: Optional[RewardConfig] = None,
) -> float:
    """Award one-shot shaping rewards for newly observed useful events."""
    cfg = config or load_reward_config()
    return cap_shaping_rewards(state, cfg)


def compute_terminal_reward_delta(
    breakdown: RewardBreakdown,
    *,
    reward_emitted_so_far: float,
) -> float:
    """Return the terminal delta needed to avoid double-counting."""
    return breakdown.total - reward_emitted_so_far


def is_solved_episode(
    state: EpisodeState,
    breakdown: RewardBreakdown,
    *,
    config: Optional[RewardConfig] = None,
) -> bool:
    """Return whether a terminal state counts as solved."""
    cfg = config or load_reward_config()
    return (
        state.submitted
        and breakdown.direct_bug_evidence is True
        and breakdown.required_path_evidence is True
        and breakdown.evidence_score >= cfg.thresholds["evidence_required_for_full_credit"]
        and breakdown.correctness_score >= cfg.thresholds["solved_correctness"]
        and breakdown.fix_bonus > 0.0
        and breakdown.wrong_penalty == 0.0
        and breakdown.evidence_penalty == 0.0
        and breakdown.contradiction_penalty == 0.0
    )


def score_reasoning(
    state: EpisodeState,
    cfg: RewardConfig,
) -> float:
    """Return a normalized root-cause score in [0, 1]."""
    root_cause_text = state.submission_root_cause.lower()
    signals = _collect_reasoning_signals(state)
    weights = {
        "file": cfg.terminal_rewards["correct_file"],
        "function": cfg.terminal_rewards["correct_function"],
        "mechanism": cfg.terminal_rewards["correct_mechanism"],
    }
    total_weight = sum(weights.values())
    if total_weight <= 0.0:
        return 0.0

    matched_weight = sum(
        weights[name] for name, matched in signals.items() if matched
    )
    score = matched_weight / total_weight
    has_explanation = _has_explanatory_root_cause(root_cause_text, state)

    if not has_explanation:
        return min(score, max(0.0, cfg.thresholds["reasonable_answer"] - 1e-9))
    if not (signals["file"] or signals["function"]):
        return min(score, max(0.0, cfg.thresholds["reasonable_answer"] - 1e-9))
    return min(score, 1.0)


def score_evidence(
    state: EpisodeState,
    cfg: RewardConfig,
) -> float:
    """Return a normalized evidence score in [0, 1]."""
    weights = cfg.shaping_rewards
    total_available = sum(max(0.0, value) for value in weights.values())
    if total_available <= 0.0:
        return 0.0

    observed = 0.0
    if state.tests_run:
        observed += weights["first_test_run"]
    if state.correct_file_opened:
        observed += weights["correct_file_opened"]
    if state.correct_function_inspected:
        observed += weights["correct_function_inspected"]
    if state.useful_search_observed:
        observed += weights["useful_search"]

    return min(1.0, observed / total_available)


def apply_terminal_reward_with_gates(
    state: EpisodeState,
    breakdown: RewardBreakdown,
    cfg: RewardConfig,
) -> None:
    """Apply terminal rewards and penalties using a small set of explicit gates."""
    thresholds = cfg.thresholds
    if breakdown.reasoning_score < thresholds["reasonable_answer"]:
        breakdown.wrong_penalty = cfg.anti_hacking_penalties["wrong_submission"]
        return

    correctness_reward = (
        cfg.terminal_rewards["correct_root_cause"] * breakdown.reasoning_score
    )
    evidence_sufficient = (
        breakdown.evidence_score >= thresholds["evidence_required_for_full_credit"]
        and breakdown.direct_bug_evidence
        and breakdown.required_path_evidence
    )

    if not evidence_sufficient:
        breakdown.evidence_penalty = cfg.anti_hacking_penalties["insufficient_evidence"]
        # Do not award root-cause credit unless the agent gathered enough
        # evidence to justify the answer.
        correctness_reward = 0.0

    if breakdown.contradiction_penalty != 0.0:
        correctness_reward = min(
            correctness_reward,
            cfg.terminal_rewards["correct_root_cause"] * thresholds["reasonable_answer"],
        )

    breakdown.correctness_score = correctness_reward

    bonuses_unlocked = (
        breakdown.reasoning_score >= thresholds["solved_correctness"]
        and evidence_sufficient
        and breakdown.contradiction_penalty == 0.0
    )
    if not bonuses_unlocked:
        return

    signals = _collect_reasoning_signals(state)
    if signals["mechanism"]:
        breakdown.mechanism_bonus = cfg.terminal_rewards["correct_mechanism"]
    if signals["file"]:
        breakdown.file_bonus = cfg.terminal_rewards["correct_file"]
    if signals["function"]:
        breakdown.function_bonus = cfg.terminal_rewards["correct_function"]

    if _evaluate_fix(state.submission_fix.lower(), state):
        breakdown.fix_bonus = cfg.terminal_rewards["correct_fix"]


def cap_shaping_rewards(state: EpisodeState, cfg: RewardConfig) -> float:
    """Award each shaping event once and enforce the global shaping cap."""
    cap = cfg.caps["shaping_max_total"]
    rewards = cfg.shaping_rewards
    awarded = 0.0

    if state.tests_run:
        awarded += state.mark_shaping_event_once(
            "first_test_run",
            reward_value=rewards["first_test_run"],
            shaping_max_total=cap,
        )
    if state.correct_file_opened:
        awarded += state.mark_shaping_event_once(
            "correct_file_opened",
            reward_value=rewards["correct_file_opened"],
            shaping_max_total=cap,
        )
    if state.correct_function_inspected:
        awarded += state.mark_shaping_event_once(
            "correct_function_inspected",
            reward_value=rewards["correct_function_inspected"],
            shaping_max_total=cap,
        )
    if state.useful_search_observed:
        awarded += state.mark_shaping_event_once(
            "useful_search",
            reward_value=rewards["useful_search"],
            shaping_max_total=cap,
        )

    return awarded


def _collect_reasoning_signals(state: EpisodeState) -> Dict[str, bool]:
    root_cause_text = state.submission_root_cause.lower()
    return {
        "file": (
            any(
                signal in root_cause_text
                for signal in _bug_file_text_signals(state.bug_file)
            )
        ),
        "function": state.bug_function.lower() in root_cause_text,
        "mechanism": _has_mechanism_keyword(root_cause_text, state.mechanism_keywords),
    }


def _evaluate_submission_contradiction(
    state: EpisodeState,
    cfg: RewardConfig,
) -> float:
    """Penalty for explicit structured contradictions."""
    file_present = bool(state.submission_bug_file)
    function_present = bool(state.submission_bug_function)

    if file_present and not _is_correct_bug_file(state.submission_bug_file, state.bug_file):
        return cfg.anti_hacking_penalties["contradiction"]
    if function_present and not _is_correct_bug_function(
        state.submission_bug_function,
        state.bug_function,
    ):
        return cfg.anti_hacking_penalties["contradiction"]
    if _mentions_wrong_file_as_root_cause(state.submission_root_cause.lower(), state):
        return cfg.anti_hacking_penalties["contradiction"]
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


def _has_direct_bug_evidence(state: EpisodeState) -> bool:
    return state.correct_function_inspected


def _has_explanatory_root_cause(text: str, state: EpisodeState) -> bool:
    """Require minimal explanatory content beyond a bag of keywords."""
    normalized = " ".join((text or "").lower().split())
    if not normalized:
        return False

    word_tokens = re.findall(r"[a-zA-Z_]{3,}", normalized)
    if len(word_tokens) < 6 or len(set(word_tokens)) < 5:
        return False

    explanation_markers = (
        "because",
        "caus",
        "due to",
        "uses",
        "using",
        "instead of",
        "rounds",
        "rounding",
        "returns",
        "leading to",
        "which",
    )
    if not any(marker in normalized for marker in explanation_markers):
        return False

    # The explanation should connect the buggy helper or mechanism to behavior.
    file_or_function_signals = tuple(_bug_file_text_signals(state.bug_file)) + (
        state.bug_function.lower(),
    )
    return any(signal and signal in normalized for signal in file_or_function_signals)


def _has_required_path_evidence(state: EpisodeState) -> bool:
    if not state.evidence_function_entries:
        return True
    return any(entry in state.functions_inspected for entry in state.evidence_function_entries)


def _evaluate_fix(text: str, state: EpisodeState) -> bool:
    """Return True if the proposed fix mentions the correct solution."""
    if not text.strip():
        return False

    if not any(kw.lower() in text for kw in state.fix_keywords):
        return False

    negated_change_patterns = (
        r"\bdo not change\b",
        r"\bdon't change\b",
        r"\bno change\b",
        r"\bunchanged\b",
        r"\bleave (?:the )?(?:behavior|implementation) unchanged\b",
    )
    if any(re.search(pattern, text) for pattern in negated_change_patterns):
        return False

    positive_change_markers = (
        "use",
        "replace",
        "switch",
        "change",
        "update",
        "convert",
        "set",
    )
    documentation_only_markers = ("document", "mention", "comment")
    has_positive_change = any(marker in text for marker in positive_change_markers)
    has_documentation_only = (
        any(marker in text for marker in documentation_only_markers)
        and not has_positive_change
    )
    if has_documentation_only:
        return False

    return has_positive_change or "should use" in text or "should replace" in text


def _mentions_wrong_file_as_root_cause(text: str, state: EpisodeState) -> bool:
    if not text.strip():
        return False

    cause_patterns = (
        r"\bbug\b[^.\n]{0,80}?\b(?:in|inside|within)\s+__SIGNAL__\b",
        r"\broot cause\b[^.\n]{0,80}?\b(?:in|inside|within)\s+__SIGNAL__\b",
        r"\bissue\b[^.\n]{0,80}?\b(?:in|inside|within)\s+__SIGNAL__\b",
        r"\bproblem\b[^.\n]{0,80}?\b(?:in|inside|within)\s+__SIGNAL__\b",
    )
    for candidate in state.accessible_files:
        if _is_correct_bug_file(candidate, state.bug_file):
            continue
        for signal in _candidate_file_text_signals(candidate):
            escaped = re.escape(signal.lower())
            for pattern in cause_patterns:
                if re.search(pattern.replace("__SIGNAL__", escaped), text):
                    return True
    return False


def _candidate_file_text_signals(path: str):
    filename = os.path.basename(path).lower()
    stem = os.path.splitext(filename)[0]
    candidates = [path.lower(), filename, stem]
    return [item for idx, item in enumerate(candidates) if item and item not in candidates[:idx]]


def _require_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ValueError("Reward config section '{}' must be a mapping.".format(key))
    return value


def _require_float_mapping(raw: Mapping[str, Any], key: str) -> Dict[str, float]:
    value = _require_mapping(raw, key)
    parsed: Dict[str, float] = {}
    for item_key, item_value in value.items():
        if not isinstance(item_value, (int, float)):
            raise ValueError(
                "Reward config value '{}.{}' must be numeric.".format(key, item_key)
            )
        parsed[str(item_key)] = float(item_value)
    return parsed


def _parse_simple_yaml(path: str) -> Dict[str, Any]:
    """Parse a small YAML subset used by reward_config.yaml.

    Supported features:
      - nested mappings via indentation
      - comments starting with '#'
      - numeric, boolean, quoted-string, and bare-string scalars
    """
    data: Dict[str, Any] = {}
    stack = [(-1, data)]

    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.split("#", 1)[0].rstrip("\n")
            if not line.strip():
                continue

            indent = len(line) - len(line.lstrip(" "))
            if indent % 2 != 0:
                raise ValueError(
                    "Reward config line {} uses unsupported indentation.".format(lineno)
                )

            content = line.strip()
            if ":" not in content:
                raise ValueError(
                    "Reward config line {} must contain a ':' separator.".format(lineno)
                )
            key, raw_value = content.split(":", 1)
            key = key.strip()
            value = raw_value.strip()

            while stack and indent <= stack[-1][0]:
                stack.pop()
            if not stack:
                raise ValueError(
                    "Reward config line {} has invalid indentation.".format(lineno)
                )

            parent = stack[-1][1]
            if value == "":
                child: Dict[str, Any] = {}
                parent[key] = child
                stack.append((indent, child))
            else:
                parent[key] = _parse_yaml_scalar(value)

    return data


def _parse_yaml_scalar(raw_value: str) -> Any:
    value = raw_value.strip()
    if not value:
        return ""
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return ast.literal_eval(value)
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value
