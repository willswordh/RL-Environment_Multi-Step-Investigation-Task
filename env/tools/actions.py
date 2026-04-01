"""
Action schema, parsing, and validation for BugInvestigationEnv.

Each agent action is a plain Python dict with at minimum a "type" key.
This module defines the valid action types, required/optional fields,
and provides a ``parse_action`` helper that normalises raw input and
returns a validated Action namedtuple or an error message.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Action type constants
# ---------------------------------------------------------------------------

LIST_FILES       = "list_files"
OPEN_FILE        = "open_file"
SEARCH           = "search"
RUN_TESTS        = "run_tests"
INSPECT_FUNCTION = "inspect_function"
SUBMIT_ANSWER    = "submit_answer"

VALID_TYPES = {
    LIST_FILES,
    OPEN_FILE,
    SEARCH,
    RUN_TESTS,
    INSPECT_FUNCTION,
    SUBMIT_ANSWER,
}


# ---------------------------------------------------------------------------
# Validated action dataclass
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """A fully validated agent action ready to be executed."""
    type: str
    # Optional fields populated depending on action type
    filename: str = ""
    keyword: str = ""
    function_name: str = ""
    root_cause: str = ""
    fix: str = ""
    bug_file: str = ""
    bug_function: str = ""
    mechanism: str = ""
    proposed_fix: str = ""

    def __str__(self) -> str:
        if self.type == LIST_FILES:
            return "list_files()"
        if self.type == OPEN_FILE:
            return f'open_file("{self.filename}")'
        if self.type == SEARCH:
            return f'search("{self.keyword}")'
        if self.type == RUN_TESTS:
            return "run_tests()"
        if self.type == INSPECT_FUNCTION:
            return f'inspect_function("{self.filename}", "{self.function_name}")'
        if self.type == SUBMIT_ANSWER:
            rc_preview = (self.root_cause[:60] + "...") if len(self.root_cause) > 60 else self.root_cause
            return (
                f'submit_answer(root_cause="{rc_preview}", fix=..., '
                f'bug_file="{self.bug_file or "(none)"}", '
                f'bug_function="{self.bug_function or "(none)"}")'
            )
        return f"unknown_action(type={self.type})"


# ---------------------------------------------------------------------------
# Schema: required and optional keys per action type
# ---------------------------------------------------------------------------

_SCHEMA: Dict[str, Dict[str, List[str]]] = {
    LIST_FILES:       {"required": [],                            "optional": []},
    RUN_TESTS:        {"required": [],                            "optional": []},
    OPEN_FILE:        {"required": ["filename"],                  "optional": []},
    SEARCH:           {"required": ["keyword"],                   "optional": []},
    INSPECT_FUNCTION: {"required": ["filename"],                  "optional": ["function", "function_name"]},
    SUBMIT_ANSWER:    {
        "required": ["root_cause"],
        "optional": ["fix", "bug_file", "bug_function", "mechanism", "proposed_fix"],
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_action(raw: Any) -> Tuple[Optional[Action], Optional[str]]:
    """Parse and validate a raw action dict from the agent.

    Args:
        raw: Anything the agent passes — expected to be a dict.

    Returns:
        (action, error):
          - If valid:   (Action instance, None)
          - If invalid: (None, human-readable error string)

    Examples:
        >>> action, err = parse_action({"type": "open_file", "filename": "repo/discount.py"})
        >>> assert err is None
        >>> action, err = parse_action({"type": "open_file"})
        >>> assert "filename" in err
    """
    # ── type check ──────────────────────────────────────────────────────
    if not isinstance(raw, dict):
        return None, (
            f"Action must be a dict, got {type(raw).__name__}. "
            f"Example: {{'type': 'list_files'}}"
        )

    raw_type = raw.get("type", "")
    if not isinstance(raw_type, str):
        return None, (
            "Action 'type' must be a string. "
            f"Got {type(raw_type).__name__}."
        )

    action_type = raw_type.strip().lower()

    if not action_type:
        return None, (
            "Action dict is missing the 'type' key. "
            f"Valid types: {sorted(VALID_TYPES)}"
        )

    if action_type not in VALID_TYPES:
        return None, (
            f"Unknown action type '{action_type}'. "
            f"Valid types: {sorted(VALID_TYPES)}"
        )

    schema = _SCHEMA[action_type]

    # ── required field check ─────────────────────────────────────────────
    for key in schema["required"]:
        if not raw.get(key, ""):
            return None, (
                f"Action '{action_type}' requires a non-empty '{key}' field. "
                f"Example: {_example(action_type)}"
            )

    # ── build validated Action ───────────────────────────────────────────
    action = Action(type=action_type)

    if action_type == OPEN_FILE:
        action.filename = str(raw["filename"]).strip()

    elif action_type == SEARCH:
        action.keyword = str(raw["keyword"]).strip()

    elif action_type == INSPECT_FUNCTION:
        action.filename      = str(raw["filename"]).strip()
        # Accept both "function" and "function_name" as the key
        fn = raw.get("function") or raw.get("function_name") or ""
        action.function_name = str(fn).strip()
        if not action.function_name:
            return None, (
                "Action 'inspect_function' requires a non-empty 'function' field. "
                f"Example: {_example(INSPECT_FUNCTION)}"
            )

    elif action_type == SUBMIT_ANSWER:
        action.root_cause = str(raw["root_cause"]).strip()
        action.fix        = str(raw.get("fix", "")).strip()
        action.bug_file = str(raw.get("bug_file", "")).strip()
        action.bug_function = str(raw.get("bug_function", "")).strip()
        action.mechanism = str(raw.get("mechanism", "")).strip()
        action.proposed_fix = str(raw.get("proposed_fix", "")).strip()

    return action, None


def action_help() -> str:
    """Return a human-readable description of all available actions."""
    return (
        "Available actions:\n"
        "  {\"type\": \"list_files\"}\n"
        "      → Lists all files in the repository.\n\n"
        "  {\"type\": \"open_file\", \"filename\": \"<path>\"}\n"
        "      → Returns the full source of the specified file.\n\n"
        "  {\"type\": \"search\", \"keyword\": \"<term>\"}\n"
        "      → Searches all files for lines containing the keyword.\n\n"
        "  {\"type\": \"run_tests\"}\n"
        "      → Runs the test suite and returns pass/fail output.\n\n"
        "  {\"type\": \"inspect_function\", \"filename\": \"<path>\", \"function\": \"<name>\"}\n"
        "      → Returns the source of a single named function.\n\n"
        "  {\"type\": \"submit_answer\", \"root_cause\": \"<explanation>\", \"fix\": \"<fix>\",\n"
        "   \"bug_file\": \"<path>\", \"bug_function\": \"<name>\",\n"
        "   \"mechanism\": \"<why>\", \"proposed_fix\": \"<summary>\"}\n"
        "      → Submits the root cause and ends the episode.\n"
        "        Structured fields are optional metadata; positive credit comes from\n"
        "        the free-text root cause and fix.\n"
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _example(action_type: str) -> str:
    examples = {
        LIST_FILES:       '{"type": "list_files"}',
        RUN_TESTS:        '{"type": "run_tests"}',
        OPEN_FILE:        '{"type": "open_file", "filename": "repo/discount.py"}',
        SEARCH:           '{"type": "search", "keyword": "round_currency"}',
        INSPECT_FUNCTION: '{"type": "inspect_function", "filename": "repo/math_utils.py", "function": "round_currency"}',
        SUBMIT_ANSWER:    (
            '{"type": "submit_answer", "root_cause": "...", "fix": "...", '
            '"bug_file": "repo/math_utils.py", "bug_function": "round_currency", "mechanism": "..."}'
        ),
    }
    return examples.get(action_type, '{"type": "' + action_type + '"}')
