# RL Environment — Plan & Design

> **Task:** Build a Reinforcement Learning environment for multi-step bug investigation.
> **Date:** March 30, 2026
> **Note:** This is the original pre-refactor planning document. Some file paths and layout examples below describe the earlier single-task repo structure rather than the current task-pack-based layout.

---

## Table of Contents

1. [High-Level Strategy](#1-high-level-strategy)
2. [Mock Codebase Design](#2-mock-codebase-design)
3. [Environment Architecture](#3-environment-architecture)
4. [State & Observation Design](#4-state--observation-design)
5. [Action Space Design](#5-action-space-design)
6. [Reward Function Design](#6-reward-function-design)
7. [Termination Conditions](#7-termination-conditions)
8. [Environment Design Document (Deliverable 2)](#8-environment-design-document-deliverable-2)
9. [File Structure](#9-file-structure)
10. [Implementation Checklist](#10-implementation-checklist)

---

## 1. High-Level Strategy

The core challenge is designing an environment that:

- Forces **genuine investigation** — the agent cannot brute-force the answer without exploring.
- Has a **clear ground truth** — so rewards are unambiguous.
- Is **tractable but not trivial** — requires 3–8 steps of meaningful reasoning.
- Is **resistant to reward hacking** — the agent must demonstrate understanding, not pattern matching.

**Chosen bug scenario:** An e-commerce order processing pipeline where a discount is calculated incorrectly because a helper utility function applies the wrong rounding mode, and the test only catches it under a specific edge case that requires reading both the business logic and the utility layer.

Why this bug type is ideal:
- It spans **multiple files** (order logic → discount calculator → math utils).
- The symptom (wrong test output) does not directly reveal the root cause.
- The fix requires understanding the **call chain**, not just reading one file.
- It is natural and believable.

---

## 2. Mock Codebase Design

### Files

| File | Role | Contains Bug? |
|------|------|---------------|
| `repo/order_processor.py` | Entry point; processes customer orders | No |
| `repo/discount.py` | Core discount calculation logic | No (calls buggy helper) |
| `repo/math_utils.py` | Shared math utilities | **YES** — `round_half_up` uses wrong rounding |
| `repo/config.py` | Business rule constants (discount rates, thresholds) | No |
| `repo/models.py` | Data classes: `Order`, `LineItem` | No |
| `tests/test_order_processor.py` | Failing test case | — |

### The Bug

**Location:** `repo/math_utils.py`, function `round_currency(value, decimals=2)`

**Bug:** Uses Python's default `round()` (banker's rounding — rounds half to even), instead of the business-required "round half up" behavior. This causes `$2.225` to round to `$2.22` instead of `$2.23`, making order totals off by one cent.

**Why multi-file investigation is required:**
1. The failing test is in `test_order_processor.py` — agent sees a wrong total.
2. The total is computed in `order_processor.py` — which calls `discount.apply_discount()`.
3. `discount.apply_discount()` calls `math_utils.round_currency()`.
4. The bug lives in `math_utils.round_currency()`.
5. The correct fix is to replace `round(value, decimals)` with `Decimal` ROUND_HALF_UP.

The agent **must trace the call chain** across at least 3 files.

---

## 3. Environment Architecture

```
┌─────────────────────────────────────────────────────┐
│                  BugInvestigationEnv                │
│                                                     │
│  ┌─────────────┐     ┌──────────────────────────┐  │
│  │   Agent     │────▶│  step(action) -> obs,    │  │
│  │  (LLM/RL)   │◀────│  reward, done, info      │  │
│  └─────────────┘     └──────────────────────────┘  │
│                              │                      │
│                    ┌─────────▼──────────┐           │
│                    │   Action Router    │           │
│                    └─────────┬──────────┘           │
│          ┌──────────┬────────┴──────────┬────────┐  │
│          ▼          ▼                   ▼        ▼  │
│     list_files  open_file(f)       search(kw)  ...  │
│          │          │                   │        │  │
│          └──────────┴──────── ──────────┘        │  │
│                    ┌─────────▼──────────┐         │  │
│                    │  Repository Store  │         │  │
│                    │  (in-memory files) │         │  │
│                    └────────────────────┘         │  │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │  Episode State                               │  │
│  │  - step_count                                │  │
│  │  - files_opened (set)                        │  │
│  │  - searches_performed (list)                 │  │
│  │  - correct_file_found (bool)                 │  │
│  │  - correct_function_found (bool)             │  │
│  │  - submitted (bool)                          │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### Interface

```python
env = BugInvestigationEnv()
obs = env.reset()           # returns initial prompt + file manifest hint
obs, reward, done, info = env.step(action)
```

`action` is a **structured dict**:

```python
# Examples:
{"type": "list_files"}
{"type": "open_file",     "filename": "repo/discount.py"}
{"type": "search",        "keyword": "round_currency"}
{"type": "run_tests"}
{"type": "inspect_function", "filename": "repo/math_utils.py", "function": "round_currency"}
{"type": "submit_answer", "root_cause": "...", "fix": "..."}
```

---

## 4. State & Observation Design

### Internal State (hidden from agent)

```python
@dataclass
class EpisodeState:
    step_count: int = 0
    max_steps: int = 15
    files_opened: set[str] = field(default_factory=set)
    functions_inspected: set[str] = field(default_factory=set)
    searches: list[str] = field(default_factory=list)
    tests_run: bool = False
    submitted: bool = False
    # Ground truth
    bug_file: str = "repo/math_utils.py"
    bug_function: str = "round_currency"
```

### Observation (returned to agent)

Each observation is a **structured dict** rendered as text:

```python
{
    "step": int,
    "max_steps": int,
    "action_taken": str,          # echo of what was done
    "result": str,                # primary content (file code, search results, etc.)
    "hint": str | None,           # optional nudge if agent is stuck (configurable)
}
```

**Design rationale:**
- Returning **raw text** (file contents, test output) mirrors real LLM tool-use scenarios.
- Including `step` / `max_steps` gives the agent awareness of budget.
- No "magic" state is exposed — the agent must infer the bug from code.

### Initial Observation (from `reset()`)

```
Bug Report:
  Tests in this repository are failing due to incorrect output in the order
  processing pipeline. Specifically, calculated order totals are sometimes
  off by $0.01. Determine the root cause of the issue and propose a fix.

Available actions:
  list_files, open_file(filename), search(keyword),
  run_tests, inspect_function(filename, function), submit_answer(root_cause, fix)

Files in repository: [HIDDEN — use list_files to discover]
Steps remaining: 15
```

The agent does NOT see the file list upfront. It must call `list_files` first.

---

## 5. Action Space Design

| Action | Arguments | Returns |
|--------|-----------|---------|
| `list_files` | — | List of all files in the repo |
| `open_file` | `filename` | Full source code of the file |
| `search` | `keyword` | List of `(filename, line_no, line_content)` matches |
| `run_tests` | — | Test runner output (pass/fail + assertion errors) |
| `inspect_function` | `filename`, `function` | Source of just that function (scoped view) |
| `submit_answer` | `root_cause` (str), `fix` (str) | Final reward + episode end |

**Design decisions:**

- `inspect_function` is more efficient than `open_file` — rewards focused agents.
- `search` is powerful but coarse — requires follow-up actions to confirm.
- `run_tests` is always available and is the logical first step — it surfaces the symptom.
- No `undo` or `backtrack` — the agent must commit to its investigation path.
- Actions outside this set return an error observation (not a crash).

---

## 6. Reward Function Design

### Summary Table

| Event | Reward | Rationale |
|-------|--------|-----------|
| Correct root cause identified | **+1.0** | Primary objective |
| Correct fix proposed | **+0.5** | Bonus for actionable answer |
| Identifies correct file (`math_utils.py`) | **+0.2** | Partial credit milestone |
| Identifies correct function (`round_currency`) | **+0.2** | Partial credit milestone |
| Per step taken | **-0.02** | Efficiency pressure |
| Submitting wrong answer | **-0.5** | Penalise confident wrong answers |
| Hitting max steps (no submission) | **-0.3** | Penalise failure to commit |

**Total possible range:** approximately `[-0.3, +1.9]`

**Normalised score** for comparison: `(reward - min) / (max - min)` → `[0, 1]`

### Correctness Evaluation

Correct root cause is determined by **keyword matching + semantic check**:

```python
ROOT_CAUSE_KEYWORDS = [
    "round_currency", "math_utils", "banker's rounding",
    "round half up", "rounding", "ROUND_HALF_UP", "decimal"
]

def evaluate_root_cause(submission: str) -> float:
    score = 0.0
    text = submission.lower()
    if "math_utils" in text:
        score += 0.5      # found the right file
    if "round_currency" in text or "round" in text:
        score += 0.3      # found the right function
    if any(k in text for k in ["half up", "banker", "round_half_up", "decimal"]):
        score += 0.2      # understands the mechanism
    return min(score, 1.0)
```

This gives **partial credit** to answers that are close but not perfect, making the reward signal denser and more useful for training.

### Reward Hacking Scenarios & Mitigations

| Hack | Description | Mitigation |
|------|-------------|------------|
| **Keyword spam** | Submit a huge list of all file/function names | Require coherent explanation; penalise submissions that are just keyword dumps |
| **Immediate submit** | Submit without exploring | Reward is 0 or negative without correct keywords; they cannot guess the exact function name without reading the code |
| **Exhaustive file reading** | Open every file sequentially | Step penalty makes this suboptimal vs. targeted search |
| **Search oracle** | Search for exact known keywords | Requires agent to already know `round_currency` — it won't know this without reading `discount.py` first |

---

## 7. Termination Conditions

| Condition | Done? | Notes |
|-----------|-------|-------|
| Agent calls `submit_answer` | ✅ | Episode ends; reward computed |
| `step_count >= max_steps` | ✅ | Partial reward still applied if any milestones hit |
| Environment error | ❌ | Returns error observation, agent continues |

`max_steps = 15` is chosen to allow approximately 2–3 file reads + a few searches + final submission, but not unlimited exploration.

---

## 8. Environment Design Document (Deliverable 2)

### 8.1 State Representation

The environment maintains a **private episode state** tracking:
- Steps taken (integer)
- Set of files opened (for partial rewards and anti-exploit tracking)
- Set of functions inspected
- Whether tests have been run
- Whether a submission has been made

The agent **never sees** this state directly. It must reconstruct its understanding from the text observations returned by actions.

This mirrors real agentic settings where the agent has no privileged access to ground truth.

### 8.2 Observation Design

Observations are **freeform text** structured as a short header + content block. This is ideal for LLM-based agents because:
1. It matches the format LLMs are trained on (code + prose).
2. It avoids over-structuring the signal — the agent must read and reason.
3. It is extensible (add metadata without breaking the interface).

The initial observation withholds the file list intentionally to test whether the agent takes a reasonable first action (`list_files` or `run_tests`) rather than blindly guessing.

### 8.3 Action Space

The action space is **discrete and structured** (a small set of named tool-calls with typed arguments). Reasons:
- Prevents prompt injection attacks via open-ended shell commands.
- Keeps the environment deterministic and reproducible.
- Is directly mappable to LLM function-calling / tool-use APIs.

Each action is idempotent — repeated calls return the same result. This prevents the agent from gaining information from side effects.

### 8.4 Reward Design

The reward function uses **three tiers**:

1. **Correctness** (primary, dense): Partial credit for identifying the file, function, and mechanism. This avoids the sparse reward problem where an agent that almost solved the task gets the same signal as one that gave a random answer.

2. **Efficiency** (secondary, continuous): A per-step penalty of `-0.02` creates a soft pressure to investigate concisely. It is small enough not to dominate but large enough to distinguish a 5-step solution from a 14-step one.

3. **Commitment** (tertiary, terminal): Penalising max-step episodes (`-0.3`) discourages agents that hedge by never submitting.

### 8.5 Why This Reward Design Works

- **Dense signal at every submission** means the agent gets a learning signal even when partially wrong.
- **Step penalty** shapes policy towards efficient investigation strategies (search before open, inspect_function before open_file).
- **Partial credit milestones** (`+0.2` for correct file, `+0.2` for correct function) provide intermediate rewards that help exploration in sparse-reward RL.

### 8.6 Potential Reward Hacking

The most likely exploit is a **"fire-hose submit"**: an agent that opens all files once and submits a response containing all file names and function names. Mitigation: the correctness evaluator checks for a **coherent explanation** (presence of both the mechanism keyword and the location), not just keyword presence alone.

A secondary risk is **step-padding**: an agent might learn to run exactly 1 step and immediately submit to minimise step penalties, even with a wrong answer. The `-0.5` wrong-answer penalty exceeds any step-saving benefit.

### 8.7 Failure Modes

| Mode | Description |
|------|-------------|
| **Shallow exploration** | Agent opens one file, never traces the call chain, submits wrong answer |
| **Infinite loops** | Agent repeatedly searches the same keyword — caught by max_steps |
| **Overconfidence** | Agent submits on step 1 with a plausible-sounding but wrong answer |
| **Distraction** | Agent spends all steps reading unrelated files (`models.py`, `config.py`) |
| **Correct file, wrong reason** | Agent finds `math_utils.py` for a wrong reason and submits a wrong mechanism |

### 8.8 What Behaviours This Environment Tests

1. **Call-chain tracing**: Can the agent follow a symptom back to its root cause through multiple files?
2. **Efficient tool use**: Does the agent use `search` and `inspect_function` rather than reading entire files?
3. **Hypothesis formation**: Does the agent update its belief about the bug location as it gathers evidence?
4. **Calibrated confidence**: Does the agent submit when it has sufficient evidence, rather than too early or too late?

### 8.9 Example Trajectory

```
RESET
Observation:
  Bug Report: Tests are failing due to incorrect order totals (off by $0.01).
  Steps remaining: 15

Step 1: run_tests
Observation:
  FAILED test_order_processor.py::test_discount_rounding
  AssertionError: assert 2.22 == 2.23
  Steps remaining: 14

Step 2: search("discount")
Observation:
  repo/discount.py:14:  def apply_discount(order, rate):
  repo/discount.py:22:      return math_utils.round_currency(discounted)
  tests/test_order_processor.py:8:  result = apply_discount(order, 0.10)
  Steps remaining: 13

Step 3: inspect_function("repo/discount.py", "apply_discount")
Observation:
  def apply_discount(order, rate):
      subtotal = sum(item.price * item.qty for item in order.items)
      discounted = subtotal * (1 - rate)
      return math_utils.round_currency(discounted)
  Steps remaining: 12

Step 4: inspect_function("repo/math_utils.py", "round_currency")
Observation:
  def round_currency(value, decimals=2):
      return round(value, decimals)   # uses banker's rounding
  Steps remaining: 11

Step 5: submit_answer(
    root_cause="The round_currency function in math_utils.py uses Python's
    built-in round(), which applies banker's rounding (round-half-to-even).
    For $2.225, this rounds down to $2.22 instead of the expected $2.23.",
    fix="Replace round(value, decimals) with Decimal(str(value)).quantize(
    Decimal('0.01'), rounding=ROUND_HALF_UP)"
)
Reward: +1.0 (correct) + 0.5 (fix) + 0.2 (file) + 0.2 (function) - 0.08 (4 steps) = +1.82
Episode ends.
```

---

## 9. File Structure

```
RL-env/
├── task.md                          # Original task spec
├── PLAN.md                          # This document
│
├── env/
│   ├── __init__.py
│   ├── bug_investigation_env.py     # Main Env class (reset, step, action router)
│   ├── episode_state.py             # EpisodeState dataclass
│   ├── actions.py                   # Action parsing & validation
│   ├── reward.py                    # Reward computation logic
│   └── repository.py               # In-memory repo store
│
├── repo/                            # Mock codebase (the "bug" lives here)
│   ├── models.py                    # Order, LineItem dataclasses
│   ├── config.py                    # Business constants
│   ├── math_utils.py                # BUG: round_currency uses wrong rounding
│   ├── discount.py                  # Calls math_utils.round_currency
│   └── order_processor.py          # Top-level entry point
│
├── tests/
│   └── test_order_processor.py     # Failing test case
│
├── example_run.py                   # Demo: manually scripted trajectory
└── design_doc_short.md              # Deliverable 2: Environment Design Document
```

---

## 10. Implementation Checklist

### Phase 1 — Mock Codebase
- [ ] Write `repo/models.py` — `Order`, `LineItem` dataclasses
- [ ] Write `repo/config.py` — discount rates, price thresholds
- [ ] Write `repo/math_utils.py` — **include the rounding bug**
- [ ] Write `repo/discount.py` — calls `round_currency`
- [ ] Write `repo/order_processor.py` — orchestrates order processing
- [ ] Write `tests/test_order_processor.py` — failing test that exposes the bug

### Phase 2 — Environment Core
- [ ] Write `env/episode_state.py` — internal state dataclass
- [ ] Write `env/repository.py` — loads repo files into memory, provides search/open APIs
- [ ] Write `env/actions.py` — action schema, parser, validator
- [ ] Write `env/reward.py` — reward computation with partial credit
- [ ] Write `env/bug_investigation_env.py` — main `Env` class with `reset()` and `step()`

### Phase 3 — Demo & Documentation
- [ ] Write `example_run.py` — scripted 5-step trajectory demonstrating the env
- [ ] Write `design_doc_short.md` — clean version of Section 8 above (Deliverable 2)

### Phase 4 — Quality & Polish
- [ ] Add type annotations throughout
- [ ] Add docstrings to all public methods
- [ ] Verify `run_tests` action accurately reflects the real test failure
- [ ] Add `README.md` with setup and usage instructions

---

## Design Principles Summary

| Principle | Decision |
|-----------|----------|
| **Minimal but sufficient** | 5 files, 1 bug, 6 actions — no unnecessary complexity |
| **Realistic** | Bug mirrors real-world floating point / rounding issues |
| **Dense rewards** | Partial credit at multiple milestones avoids sparse reward problem |
| **Anti-exploit** | Step penalty + wrong-answer penalty + keyword-only check prevention |
| **LLM-friendly** | Text observations match LLM training distribution |
| **Reproducible** | In-memory repo; no filesystem side effects; deterministic |
