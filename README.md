# RL Environment — Multi-Step Bug Investigation

A task-pack-based reinforcement learning environment for multi-step repository investigation. An agent receives a bug report, explores a partially observable codebase through a small action set, and submits a root-cause analysis and proposed fix. The environment is designed for take-home scale RL environment design: it emphasizes sequential investigation, sparse evidence-based reward shaping, and a hidden verifier executed through `run_tests()`.

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
- Detailed implementation notes and reward-design rationale live in `design_doc.md` and `design_doc_full.md`.
