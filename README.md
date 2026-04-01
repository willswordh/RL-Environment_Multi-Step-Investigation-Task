# RL Environment — Multi-Step Bug Investigation

A task-pack-based reinforcement learning environment for repository investigation tasks.

## What Changed

- Task content now lives under `tasks/<task_id>/`.
- The environment loads task metadata from `task.toml` instead of hardcoding one bug across the codebase.
- Reward policy values now live in `env/reward_config.yaml`, while reward execution stays in Python.
- `run_tests` now executes the full task-side verifier in a fresh temporary workspace copied from the task pack, with bytecode writes and third-party pytest plugin autoload disabled.
- Agent-visible files now exclude verifier tests, so the main reward signal comes from investigation rather than reading explanatory assertions.
- Reward shaping is now small, sparse, one-shot, and capped; terminal correctness remains the dominant signal.
- Solved episodes now require strong investigation evidence, the required call path, a sufficiently correct root cause, and a non-negated fix.
- The top-level README no longer publishes the benchmark answer.

## Quick Start

```bash
pytest -q
python example_run.py
```

`pytest -q` runs the author-side environment and reward tests in `tests/`. The task-side failing verifier lives inside the task pack and is exercised through `run_tests` in the environment.

## Layout

```text
RL-env/
├── env/                             # Environment runtime
├── docs/
│   ├── PLAN.md                     # Original planning notes
│   └── PLAN2.md                    # Current forward plan
├── tasks/
│   └── discount-rounding/
│       ├── instruction.md           # Agent-facing task prompt
│       ├── task.toml                # Task metadata + ground-truth scoring signals
│       ├── repo/                    # Repository snapshot for the task
│       └── tests/                   # Task verifier inputs
├── tests/                           # Environment/unit tests
├── design_doc.md                    # Design write-up
├── pytest.ini                       # Pytest config for author-side tests
└── pyproject.toml                   # Project metadata
```

## Environment Interface

```python
from env import BugInvestigationEnv

env = BugInvestigationEnv(task_name="discount-rounding")
obs = env.reset()
obs, reward, done, info = env.step({"type": "run_tests"})
```

Available actions:

- `{"type": "list_files"}`
- `{"type": "open_file", "filename": "<path>"}`
- `{"type": "search", "keyword": "<term>"}`
- `{"type": "run_tests"}`
- `{"type": "inspect_function", "filename": "<path>", "function": "<name>"}`
- `{"type": "submit_answer", "root_cause": "<str>", "fix": "<str>", "bug_file": "<path>", "bug_function": "<name>", "mechanism": "<why>", "proposed_fix": "<summary>"}`

The structured `bug_file` / `bug_function` / `mechanism` / `proposed_fix` fields are optional diagnostic metadata. Positive terminal reward credit comes from the free-text `root_cause` and `fix` only when the agent also gathers sufficient investigation evidence; structured fields are mainly used for contradiction checks and debugging.

## Sandbox Execution

Every `run_tests` action copies the task pack's `repo/` and `tests/` directories into an ephemeral temp directory, drops cache artifacts such as `__pycache__/` and `*.pyc`, disables third-party pytest plugin autoload, and runs the configured task verifier there. This makes evaluation more deterministic relative to the task snapshot instead of the live working tree.

`step()` returns incremental reward, while `info["reward_breakdown"]["total"]` is the full episode return. Summing emitted step rewards across the trajectory should match that terminal total.

## Notes

- `design_doc.md`, `docs/PLAN.md`, and `docs/PLAN2.md` are author-facing docs; the agent-facing prompt is `tasks/<task_id>/instruction.md`.
