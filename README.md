# RL Environment — Multi-Step Bug Investigation

A task-pack-based reinforcement learning environment for multi-step repository investigation. An agent receives a bug report, explores a partially observable codebase through a small action set, and submits a root-cause analysis and proposed fix.

This project is built for the RL environment design take-home in `task.md`. It focuses on the parts that matter most for that assignment:

- partial observability over a small codebase rather than full upfront access
- a compact investigation-oriented action space with `reset()` / `step()`
- reward shaping that values grounded investigation over shortcut guessing
- a hidden verifier executed through `run_tests()` instead of direct test-file access
- author-side tests and design docs that explain the environment decisions

## Design Docs

- [Short Design Doc](design_doc_short.md)
- [Full Design Doc](design_doc_full.md)

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
├── design_doc_short.md              # Short design write-up
├── design_doc_full.md               # Full design write-up
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

- `design_doc_short.md`, `design_doc_full.md`, `docs/PLAN.md`, and `docs/PLAN2.md` are author-facing docs; the agent-facing prompt is `tasks/<task_id>/instruction.md`.
