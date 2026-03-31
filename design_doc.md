# Environment Design Document

**Project:** RL Environment for Multi-Step Bug Investigation  
**Date:** March 31, 2026

## 1. Overview

This repository implements `BugInvestigationEnv`, a reinforcement learning environment for multi-step software investigation. The agent is given a bug report, a limited action space, and a step budget. It must inspect the codebase through environment actions before submitting a root-cause analysis and proposed fix.

The current repo version is task-pack based rather than single-task hardcoded. Tasks live under `tasks/<task_id>/` and define:

- `instruction.md`: agent-facing prompt
- `task.toml`: task metadata, budget, accessible files, verifier target, and ground-truth scoring signals
- `repo/`: repository snapshot exposed to the agent
- `tests/`: verifier-side tests used by `run_tests`

The bundled task is `discount-rounding`, but the runtime is structured to support more than one task.

The repository also contains a top-level `tests/` directory, but those tests are author-side environment and reward tests rather than task verifiers. The failing task-side verifier is intentionally kept inside the task pack.

## 2. Environment Design

### 2.1 State Representation

The environment keeps an internal `EpisodeState` that is not directly visible to the agent. It tracks both generic rollout state and task-specific ground truth loaded from the active task pack.

Core state fields:

| Field | Purpose |
|---|---|
| `step_count`, `max_steps` | Step budget accounting |
| `files_opened`, `functions_inspected`, `keywords_searched` | Investigation trace |
| `tests_run` | Whether the agent ran the verifier |
| `invalid_action_count`, `repeated_action_count`, `parse_error_count` | Control-plane behavior |
| `trajectory` | Logged transition history for debugging |
| `correct_file_opened`, `correct_function_inspected` | Milestone evidence features |
| `submitted`, `submission_*` | Final answer payload |
| `stop_reason`, `elapsed_seconds`, `timed_out`, `max_invalid_reached` | Termination bookkeeping |
| `task_id`, `bug_file`, `bug_function`, `mechanism_keywords`, `fix_keywords`, `evidence_function_entries` | Task-scoring ground truth |

Why this design:

- The agent only sees textual observations, not hidden labels.
- The reward function can remain generic while still using task-specific metadata.
- The stored trajectory makes reward debugging and evaluator inspection straightforward.

### 2.2 Observation Design

Each environment step returns:

```python
{
    "step": int,
    "max_steps": int,
    "steps_remaining": int,
    "action_taken": str,
    "result": str,
}
```

Observations are text-first by design. File contents, search hits, pytest output, and submission confirmation are all returned as raw strings. This mirrors the shape of real tool-using LLM workflows better than an overly structured symbolic observation.

The initial observation is generated from the active task pack's `instruction.md` plus the action schema. It intentionally does not reveal the repository file list.

The separate `info` payload exposes rollout diagnostics that are useful for debugging and evaluation, including counters, trajectory length, terminal reward breakdown, `task_id`, and the active sandbox mode.

### 2.3 Action Space

The action space is a fixed validated dictionary interface:

| Action | Arguments | Result |
|---|---|---|
| `list_files` | — | Lists accessible files |
| `open_file` | `filename` | Returns full file contents |
| `search` | `keyword` | Returns matching `filename:lineno: line` hits |
| `run_tests` | — | Runs the task verifier target and returns pytest output |
| `inspect_function` | `filename`, `function` | Returns a single function body |
| `submit_answer` | `root_cause`, optional structured fields | Ends the episode |

Design choices:

- Actions are validated before execution to avoid open-ended shelling out.
- `inspect_function` rewards targeted hypotheses over brute-force full-file reading.
- The action set is intentionally small to keep the RL problem legible.
- Invalid actions count against the step budget instead of crashing the environment.

### 2.4 Task-Pack Architecture

The main structural change from the initial version is the introduction of task packs. A task pack defines the scenario through `task.toml` instead of scattering assumptions across the runtime.

Current `task.toml` fields include:

- task identity and title
- step budget and timeout defaults
- accessible file list
- benchmark pytest target
- ground-truth bug file and function
- mechanism and fix keywords for reward scoring
- evidence function entries that count as meaningful investigation

This moves the repo closer to a reusable benchmark layout without requiring a separate orchestration framework.

### 2.5 Sandbox Execution

`run_tests` does not execute against the live workspace. Instead, the environment:

1. creates a fresh temporary directory
2. copies the task pack's `repo/` and `tests/` into it
3. runs `pytest` against the configured benchmark target inside that directory
4. returns stdout/stderr as the observation

This sandbox is lightweight rather than containerized. It improves reproducibility relative to the original host-tree execution model because:

- task evaluation is based on the task snapshot, not the mutable top-level repo
- verifier tests do not mutate or depend on the live working directory
- each `run_tests` call starts from a fresh copy

Limitations:

- it is still process-level isolation, not full OS-level sandboxing
- it inherits the host Python and pytest installation
- it does not yet support per-task Docker images

## 3. Reward Design

### 3.1 Reward Components

The reward function combines correctness, evidence, efficiency, and control-plane behavior.

Terminal correctness terms:

| Term | Value |
|---|---|
| Correct root-cause score | up to `+1.00` |
| Correct fix bonus | `+0.50` |
| Correct file bonus | `+0.20` |
| Correct function bonus | `+0.20` |
| Wrong submission penalty | `-0.50` |
| Insufficient evidence penalty | `-0.25` |
| Contradiction penalty | `-0.15` |

Shaping and termination terms:

| Term | Value |
|---|---|
| Per-step penalty | `-0.02` |
| Invalid action penalty | `-0.03` |
| Immediate repeated action penalty | `-0.01` |
| Budget exhausted without submission | `-0.30` |
| Timeout | `-0.20` |
| Max invalid actions reached | `-0.20` |

### 3.2 How Correctness Is Computed

When the agent submits, the environment computes:

1. `reasoning_score` from text and structured answer fields
2. `evidence_score` from the observed investigation trace
3. `correctness_score = min(reasoning_score, evidence_score + 0.2)`

This evidence gate is the main anti-shortcut mechanism. A model that guesses the answer text without inspecting the repo cannot receive full correctness credit.

For the bundled task:

- file mention contributes `0.50`
- function mention contributes `0.30`
- mechanism understanding contributes `0.20`

Structured fields can also supply those signals, which makes the environment usable both for free-text and hybrid structured-answer agents.

### 3.3 Evidence Features

Evidence currently awards credit for:

- running tests
- performing at least one search
- inspecting the configured evidence function entry
- opening the correct file
- inspecting the correct function

This is intentionally not a pure shortest-path rubric. The goal is to reward grounded investigation, not merely final-string matching.

### 3.4 Reward-Hacking Considerations

Main failure modes and mitigations:

| Exploit | Mitigation |
|---|---|
| Guess the answer immediately | Evidence gate caps correctness |
| Spam keywords in submission | Text-only reasoning is insufficient without investigation evidence |
| Repeat cheap actions | Immediate repetition is penalized |
| Never submit | Budget exhaustion penalty |
| Contradict free text with structured fields | Explicit contradiction penalty and `solved=False` |

Residual weakness:

- the current bundled task still uses task-specific keyword scoring rather than semantic grading
- the environment is more robust than the first version, but it is not yet a benchmark-grade verifier

## 4. Termination Conditions

The episode ends when:

- the agent calls `submit_answer`
- the step budget is exhausted
- wall-clock timeout is reached
- too many invalid actions have accumulated

Any action after termination returns a no-op terminal observation and `done=True`.

## 5. Example Trajectory

The bundled `discount-rounding` task supports a short successful trajectory such as:

```text
Step 1: run_tests()
Observation: one targeted test fails with an off-by-one-cent assertion

Step 2: search("discount")
Observation: the call chain points toward the discount pipeline

Step 3: inspect_function("repo/discount.py", "apply_discount")
Observation: discount application delegates rounding to a helper

Step 4: inspect_function("repo/math_utils.py", "round_currency")
Observation: the helper implementation reveals the likely cause

Step 5: submit_answer(...)
Observation: episode ends and terminal reward is computed
```

The important property is not the exact file names; it is that the agent must trace a symptom through multiple steps before it can justify a submission.

## 6. What This Environment Tests

This environment is designed to measure:

- ability to investigate instead of answer immediately
- call-chain tracing across multiple files
- efficient use of search versus full-file inspection
- calibrated submission timing under a step budget
- ability to explain both location and mechanism

## 7. Current Strengths And Limitations

### Strengths

- clear `reset/step` RL interface
- explicit partial observability
- task-pack structure instead of fully hardcoded single-task assumptions
- fresh-copy sandbox execution for verifier calls
- reward shaping that discourages shortcut guessing
- unit tests for runtime control-plane and reward behavior

### Limitations

- sandboxing is tempdir-based, not container-based
- only one bundled task exists today
- correctness still depends on task-authored keyword signals
- `task.toml` parsing is intentionally lightweight rather than full TOML support
- author-facing docs still contain more task-specific detail than a benchmark release would

## 8. Conclusion

The current repo version is a meaningful step up from the original take-home implementation. It now separates task data from runtime logic, avoids leaking the answer in the main README, and evaluates tests in a fresh ephemeral workspace. The next major quality jump would be to expand the task set, move from keyword scoring toward verifier-driven correctness, and replace the tempdir sandbox with containerized per-task execution.
