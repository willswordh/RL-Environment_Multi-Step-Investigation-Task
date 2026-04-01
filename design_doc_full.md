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

The repository also contains a top-level `tests/` directory, but those tests are author-side environment and reward tests rather than task verifiers. The failing task-side verifier is intentionally kept inside the task pack and is only exposed to the agent through `run_tests`, not through `open_file` or `search`.

Reward policy values are separated from reward execution. Numerical weights and thresholds live in `env/reward_config.yaml`, while `env/reward.py` keeps the custom Python logic for semantic matching, evidence validation, solved gating, and anti-double-counting.

## 2. Environment Design

### 2.1 State Representation

The environment keeps an internal `EpisodeState` that is not directly visible to the agent. It tracks both generic rollout state and task-specific ground truth loaded from the active task pack.

Core state fields:

| Field | Purpose |
|---|---|
| `step_count`, `max_steps` | Step budget accounting |
| `files_listed_count`, `files_opened`, `functions_inspected`, `keywords_searched` | Investigation trace |
| `tests_run` | Whether the agent ran the verifier |
| `invalid_action_count`, `repeated_action_count`, `parse_error_count` | Control-plane behavior |
| `trajectory` | Logged transition history for debugging |
| `correct_file_opened`, `correct_function_inspected` | Milestone evidence features |
| `submitted`, `submission_*` | Final answer payload |
| `stop_reason`, `elapsed_seconds`, `timed_out`, `max_invalid_reached` | Termination bookkeeping |
| `task_id`, `accessible_files`, `bug_file`, `bug_function`, `mechanism_keywords`, `fix_keywords`, `evidence_function_entries` | Task-scoring ground truth |

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

Verifier-side tests are not part of the agent-visible file set. This reduces leakage from explanatory test comments and keeps the environment focused on debugging through observations rather than answer extraction from the verifier source.

The separate `info` payload exposes rollout diagnostics that are useful for debugging and evaluation, including counters, trajectory length, terminal reward breakdown, `task_id`, and the active sandbox mode.

### 2.3 Action Space

The action space is a fixed validated dictionary interface:

| Action | Arguments | Result |
|---|---|---|
| `list_files` | — | Lists accessible repository files |
| `open_file` | `filename` | Returns full file contents |
| `search` | `keyword` | Returns matching `filename:lineno: line` hits |
| `run_tests` | — | Runs the task verifier target and returns pytest output |
| `inspect_function` | `filename`, `function` | Returns a single function body |
| `submit_answer` | `root_cause`, optional structured fields for diagnostics | Ends the episode |

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
- accessible repository file list
- benchmark verifier target
- ground-truth bug file and function
- mechanism and fix keywords for reward scoring
- evidence function entries that count as meaningful investigation

This moves the repo closer to a reusable benchmark layout without requiring a separate orchestration framework.

### 2.5 Sandbox Execution

`run_tests` does not execute against the live workspace. Instead, the environment:

1. creates a fresh temporary directory
2. copies the task pack's `repo/` and `tests/` into it
3. runs `pytest` against the configured task verifier target inside that directory with bytecode generation disabled and third-party plugin autoload turned off
4. returns stdout/stderr as the observation

This sandbox is lightweight rather than containerized. It improves reproducibility relative to the original host-tree execution model because:

- task evaluation is based on the task snapshot, not the mutable top-level repo
- verifier tests do not mutate or depend on the live working directory
- copied cache artifacts such as `__pycache__/` and `*.pyc` are excluded
- each `run_tests` call starts from a fresh copy

Limitations:

- it is still process-level isolation, not full OS-level sandboxing
- it still inherits the host Python interpreter and pytest version
- it disables third-party pytest plugin autoload, but it is not yet a full container boundary
- it does not yet support per-task Docker images

## 3. Reward Design

### 3.1 Reward Components

The reward function combines correctness, evidence, efficiency, and control-plane behavior.

Terminal correctness terms:

| Term | Value |
|---|---|
| Correct root-cause score | up to `+1.00` |
| Correct mechanism bonus | `+0.25` |
| Correct fix bonus | `+0.20` |
| Correct file bonus | `+0.10` |
| Correct function bonus | `+0.10` |
| Wrong submission penalty | `-0.40` |
| Insufficient evidence penalty | `-0.20` |
| Contradiction penalty | `-0.15` |

Shaping, efficiency, and termination terms:

| Term | Value |
|---|---|
| Per-step penalty | `-0.01` |
| Invalid action penalty | `-0.05` |
| Immediate repeated action penalty | `-0.02` |
| First `run_tests` | `+0.05` |
| First useful search | `+0.03` |
| First correct-file observation | `+0.05` |
| First correct-function inspection | `+0.05` |
| Positive shaping cap | `+0.20` max total |
| Budget exhausted without submission | `-0.10` |
| Timeout | `-0.10` |
| Max invalid actions reached | `-0.10` |

`step()` returns incremental reward. The terminal `reward_breakdown.total` represents the full episode return after aggregating correctness, shaping, and terminal penalties.

### 3.2 How Correctness Is Computed

When the agent submits, the environment computes:

1. `reasoning_score` from the free-text root-cause explanation
2. `evidence_score` from the small set of one-shot shaping events
3. `direct_bug_evidence` from whether the agent inspected the true bug file or function
4. `required_path_evidence` from whether the agent inspected the configured call-chain hop
5. `correctness_score`, which is the main root-cause reward after gating is applied

The simplified policy intentionally uses only a few interpretable gates:

- If `reasoning_score < reasonable_answer`, the submission gets `wrong_submission` and no terminal bonuses.
- If `reasoning_score >= reasonable_answer` but evidence is weak, the episode gets an `insufficient_evidence` penalty and zero root-cause credit.
- Mechanism, fix, file, and function bonuses unlock only when `reasoning_score >= solved_correctness` and evidence is strong enough for full credit.

Structured fields are recorded for diagnostics and contradiction checks, but they do not provide positive correctness credit on their own. If the text says the bug is in the wrong file, the contradiction penalty applies and any otherwise available correctness credit is capped.

Reasoning is intentionally normalized instead of using a long list of task-specific thresholds. The bundled task scores three root-cause signals:

- correct bug file
- correct bug function
- correct mechanism

Those signals determine `reasoning_score`, while the actual terminal payout stays small and explicit in YAML.

### 3.3 Evidence Features

Evidence and shaping are now aligned. The same sparse events that produce small step-level shaping rewards also define the normalized evidence score:

- running tests
- performing a useful search
- opening the correct file
- inspecting the correct function

Each shaping event pays at most once, and total positive shaping is capped. This keeps shaping sparse and prevents an agent from farming reward by repeating cheap actions.

When a task defines `evidence_function_entries`, that call-chain evidence is still required to count the episode as solved. The final `solved` flag is stricter than partial credit: the agent must inspect the real bug location, traverse the configured evidence hop, clear the solved correctness threshold, meet the evidence threshold for full credit, and propose a positive fix.

### 3.4 Reward-Hacking Considerations

Main failure modes and mitigations:

| Exploit | Mitigation |
|---|---|
| Guess the answer immediately | Weak evidence removes root-cause credit and triggers `insufficient_evidence` |
| Jump straight to the helper bug without tracing the call chain | Full-credit bonuses and `solved=True` stay locked until the required path is inspected |
| Farm reward by repeating cheap actions | Per-step cost, repeated-action penalty, one-shot shaping events, and a shaping cap prevent accumulation |
| Spam keywords in submission | Text-only reasoning is not enough to unlock full terminal bonuses without evidence |
| Use structured fields to override a weak or wrong explanation | Structured fields do not add positive correctness credit; contradiction heuristic still penalizes conflicting free-text file claims |
| Mention the right fix keywords while explicitly refusing to change the code | Fix scoring rejects self-negating or documentation-only fix text |
| Never submit | Budget exhaustion penalty |

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
Observation: the full task verifier shows one off-by-one-cent failure while other cases pass

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
- efficient use of discovery actions and search versus full-file inspection
- calibrated submission timing under a step budget
- ability to explain both location and mechanism

## 7. Current Strengths And Limitations

### Strengths

- clear `reset/step` RL interface
- explicit partial observability
- task-pack structure instead of fully hardcoded single-task assumptions
- fresh-copy sandbox execution for verifier calls
- reward gating and shaping that discourage shortcut guessing
- unit tests for runtime control-plane and reward behavior

### Limitations

- sandboxing is tempdir-based, not container-based
- only one bundled task exists today
- correctness still depends on task-authored keyword signals
- `task.toml` parsing is intentionally lightweight rather than full TOML support
- author-facing docs still contain more task-specific detail than a benchmark release would

## 8. Conclusion

The current repo version is a meaningful step up from the original take-home implementation. It now separates task data from runtime logic, avoids leaking verifier hints into the agent-visible file set, evaluates tests in a fresh ephemeral workspace, and is harder to game with shortcut submissions. The next major quality jump would be to expand the task set, move more of correctness onto executable verification, and replace the tempdir sandbox with containerized per-task execution.
