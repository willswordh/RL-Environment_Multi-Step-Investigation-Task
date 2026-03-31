# RL Environment — Multi-Step Bug Investigation

A task-pack-based reinforcement learning environment for repository investigation tasks.

## What Changed

- Task content now lives under `tasks/<task_id>/`.
- The environment loads task metadata from `task.toml` instead of hardcoding one bug across the codebase.
- `run_tests` now executes in a fresh temporary workspace copied from the task pack, so evaluation does not mutate or depend on the live repo tree.
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
│       ├── task.toml                # Task metadata + reward config
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

## Sandbox Execution

Every `run_tests` action copies the task pack's `repo/` and `tests/` directories into an ephemeral temp directory and runs `pytest` there. This makes evaluation deterministic relative to the task snapshot instead of the live working tree.

## Notes

- `design_doc.md`, `docs/PLAN.md`, and `docs/PLAN2.md` are author-facing docs; the agent-facing prompt is `tasks/<task_id>/instruction.md`.
