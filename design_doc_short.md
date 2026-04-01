# Environment Design Document

**Project:** RL Environment for Multi-Step Bug Investigation  
**Date:** April 1, 2026

## 1. Overview

This project implements `BugInvestigationEnv`, an RL environment where an agent investigates a small Python repository to identify the root cause of a bug. The agent starts from a short bug report, cannot see the full repository immediately, and must gather evidence through environment actions before submitting an answer.

The environment is task-pack based. Each task lives under `tasks/<task_id>/` and contains:

- `instruction.md`: the agent-facing bug report
- `task.toml`: task metadata such as step budget, accessible files, verifier target, and ground-truth labels used for scoring
- `repo/`: the repository snapshot exposed through environment actions
- `tests/`: the hidden verifier used by `run_tests`

The bundled task, `discount-rounding`, is a small order-processing codebase with one failing test and a bug in a helper function. The failure cannot be justified from a single file alone: the agent must connect a failing pipeline output to discount application and then to the rounding helper.

## 2. Environment Design

### State Representation

The environment keeps a private `EpisodeState`. The agent never sees this object directly. It tracks:

- budget and timing: `step_count`, `max_steps`, `elapsed_seconds`
- investigation history: files opened, functions inspected, search terms, and whether tests were run
- control-plane behavior: invalid actions, repeated actions, and parse errors
- terminal data: submitted answer, proposed fix, and stop reason
- task ground truth: bug file, bug function, mechanism keywords, fix keywords, and required call-path evidence

This separation keeps the interaction partially observable while still allowing the evaluator to score both what the agent concluded and how it investigated.

### Observation Design

Each call to `step()` returns a small text-first observation:

```python
{
    "step": int,
    "max_steps": int,
    "steps_remaining": int,
    "action_taken": str,
    "result": str,
}
```

I chose a text observation rather than a symbolic state because the intended agent is an LLM using tool-like actions. File contents, search results, and pytest output are all returned as strings. The initial observation contains the bug report and the action schema, but not the repository file list. The agent must call `list_files()` or use more targeted actions to explore.

The `info` dictionary is richer and mainly serves debugging and evaluation. It includes counters, the trajectory length, the final reward breakdown, and a `solved` flag.

### Action Space

The action space is a validated dictionary interface:

- `list_files`
- `open_file(filename)`
- `search(keyword)`
- `run_tests()`
- `inspect_function(filename, function)`
- `submit_answer(root_cause, fix, ...)`

These actions were chosen to make the task clearly sequential without requiring an open-ended shell. `inspect_function` is especially useful because it rewards targeted investigation rather than brute-force reading of full files. Invalid actions consume a step and produce an error observation instead of crashing the environment.

### Termination

An episode ends when:

- the agent submits an answer
- the step budget is exhausted
- wall-clock timeout is reached
- too many invalid actions accumulate

Any later action returns a terminal no-op observation with `done=True`.

## 3. Reward Design

The reward combines correctness, efficiency, and evidence of real investigation.

### Correctness

The main terminal signal is whether the agent identifies the correct root cause. The environment scores the free-text `root_cause` explanation, not just optional structured metadata. For the bundled task, the answer is evaluated on three interpretable signals:

- mention of the correct bug file
- mention of the correct bug function
- mention of the correct mechanism, namely banker or half-even rounding

I also added a guard against keyword stuffing. A submission only counts as a reasoned explanation if it contains minimally explanatory natural language, not just a bag of correct tokens.

### Evidence And Gating

Full credit is intentionally harder than simply guessing the right sentence. The environment tracks sparse evidence from the trajectory:

- first test run
- first useful search
- opening the correct file
- inspecting the correct bug function

The final answer only counts as `solved` if the agent:

- inspects the actual buggy function
- inspects the required upstream call-path function
- clears the correctness threshold
- has enough evidence
- proposes a positive fix

If the answer text is strong but the investigation evidence is insufficient, the environment applies an evidence penalty and awards zero root-cause credit.

This design is meant to reward investigation rather than answer extraction or lucky guessing.

### Efficiency And Control

The environment applies:

- a per-step penalty
- an invalid-action penalty
- a repeated-action penalty
- penalties for timeout, budget exhaustion, and too many invalid actions

Positive shaping rewards are one-shot and capped, so agents cannot farm reward by repeating easy actions.

### Reward Hacking Considerations

The main failure modes I considered were:

- guessing immediately without investigating
- reading the right helper file but skipping the call chain
- spamming keywords in the final answer
- using structured fields to override a weak explanation
- repeating cheap actions to accumulate reward

The current implementation mitigates these with evidence gates, contradiction checks, repeated-action penalties, and the explanatory-text requirement. The remaining weakness is that the bundled task still uses task-authored keyword signals rather than a fully semantic or executable root-cause verifier.

## 4. Failure Modes And Example Trajectory

This environment is intended to test whether an agent can investigate a codebase step by step, trace a symptom across files, and decide when it has enough evidence to submit. It does not primarily test code editing; it tests diagnosis.

A short successful trajectory looks like this:

```text
Step 1: run_tests()
Observation: one verifier case fails with a one-cent discrepancy

Step 2: search("discount")
Observation: results point to the discount pipeline

Step 3: inspect_function("repo/discount.py", "apply_discount")
Observation: discounted totals are passed to a rounding helper

Step 4: inspect_function("repo/math_utils.py", "round_currency")
Observation: the helper uses Python's default round behavior

Step 5: submit_answer(...)
Observation: terminal reward is computed
```

The desired behavior is deliberate investigation followed by a justified answer. Likely failure behaviors include submitting too early, exploring too broadly under the step budget, or identifying the right area of code without explaining the mechanism.

## 5. Limitations

This is a strong take-home prototype, but it is not yet a benchmark-grade environment. `run_tests()` uses a fresh temp directory rather than a fully containerized sandbox, only one task is currently bundled, and root-cause grading still depends on task-authored heuristics. The next improvements would be a larger task set, stronger verifier isolation, and more execution-backed scoring.
