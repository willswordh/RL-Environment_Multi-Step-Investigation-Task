# RL Environment — Plan 2

**Purpose:** document the current post-refactor state of the repo and the next improvements needed to make it closer to a reusable benchmark environment.

## 1. Current State

The repo has already moved beyond the original take-home version in four important ways:

1. Runtime logic and task data are separated.
   Task data now lives under `tasks/<task_id>/` with `instruction.md`, `task.toml`, `repo/`, and `tests/`.

2. The environment is no longer fully hardcoded to one bug.
   `BugInvestigationEnv` loads task configuration through `env/task_config.py`.

3. `run_tests` no longer executes against the live repo tree.
   It copies the task snapshot into a temporary workspace and runs pytest there.

4. The top-level README is no longer leaking the benchmark answer.

This is a solid assignment submission and a better base for future benchmark work.

## 2. What Still Needs Improvement

The repo is better, but it is not benchmark-grade yet. The main gaps are:

### A. More Than One Task

Right now the architecture supports multiple tasks, but the dataset contains only one bundled task. That means:

- no diversity of bug types
- no way to compare policies across different task families
- no split between development and held-out evaluation tasks

### B. Stronger Correctness Verification

The environment still uses task-authored keyword signals for most correctness scoring. That is acceptable for this assignment, but weak for a real benchmark because:

- the answer space is overfit to one task's wording
- semantic equivalents may not score well
- a clever keyword-stuffing policy can still partially exploit the rubric

### C. Stronger Sandbox Isolation

The current tempdir-copy sandbox is a good improvement, but it still inherits:

- the host Python interpreter
- the host pytest version
- the host package environment

For benchmark reliability, task execution should be pinned and isolated per task.

### D. Cleaner Task Schema

The current `task.toml` parser is intentionally minimal. That keeps the repo dependency-light, but it also means:

- limited schema validation
- limited future extensibility
- more custom parsing logic in the runtime

## 3. Recommended Next Milestones

### Milestone 1: Expand The Task Set

Add 3 to 5 more tasks with different failure mechanisms:

- wrong default value
- argument propagation bug
- stale config constant
- helper function misuse
- cross-file data-format mismatch

Each task should have:

- a neutral `instruction.md`
- a repository snapshot
- a targeted verifier test
- task metadata in `task.toml`

### Milestone 2: Split Reward Into Two Layers

Keep the current shaping signals for RL efficiency, but separate correctness into:

- generic environment shaping reward
- task verifier correctness reward

Preferred direction:

- use keyword scoring only for partial shaping
- add a task-level verifier that can grade the final answer or resulting patch more robustly

### Milestone 3: Containerize Task Execution

Move from tempdir-copy execution to per-task sandbox definitions.

Ideal end state:

- task config specifies a Docker image or build recipe
- verifier runs inside that sandbox
- Python and pytest versions are pinned per task
- file system and process isolation are stronger

### Milestone 4: Formalize The Task Schema

Upgrade `task.toml` into a more explicit schema with fields such as:

- metadata
- environment
- verifier
- scoring
- answer contract

Then add validation tests that fail fast when a task is malformed.

### Milestone 5: Add Benchmark Evaluation Scripts

Add a small runner that can:

- enumerate tasks
- run a policy over all tasks
- aggregate rewards
- emit per-task diagnostics and trajectory summaries

That would make the repo feel more like a reusable evaluation package rather than just a local env implementation.

## 4. Prioritized Action Order

If the goal is maximum quality gain per unit effort, the order should be:

1. add more tasks
2. improve verifier-backed correctness
3. containerize execution
4. formalize schema validation
5. add multi-task evaluation tooling

This order matters. Adding infrastructure before adding task diversity would improve engineering polish without materially improving benchmark quality.

## 5. Suggested Repo Direction

A good target shape for the next version is:

```text
RL-env/
├── env/
├── docs/
│   ├── PLAN.md
│   └── PLAN2.md
├── tasks/
│   ├── discount-rounding/
│   ├── config-default-mismatch/
│   ├── helper-arg-propagation/
│   └── ...
├── scripts/
│   ├── run_task.py
│   └── run_eval.py
├── tests/
├── design_doc.md
└── ...
```

## 6. Success Criteria For The Next Revision

The next version should aim to satisfy these checks:

- at least 4 distinct bundled tasks
- no benchmark answers leaked in top-level user-facing docs
- per-task verifier execution isolated from the host repo state
- reward tests covering shortcut behaviors across multiple tasks
- one command to run a small multi-task evaluation

## 7. Bottom Line

The repo is now in a good intermediate state:

- stronger than the original assignment submission
- clearly more modular
- less leaky
- more reproducible

The next real jump in quality is no longer about polishing the single task. It is about turning the architecture into a small but credible benchmark suite.
