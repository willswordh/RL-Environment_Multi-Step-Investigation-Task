"""Neutral demonstration of task-pack loading and sandboxed execution."""

import sys
import os

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from env import BugInvestigationEnv


def banner(text: str) -> None:
    width = 66
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


def print_obs(obs: dict) -> None:
    """Render an observation to stdout in a readable format."""
    step_info = (
        f"  Step {obs['step']} / {obs['max_steps']}"
        f"  (steps remaining: {obs['steps_remaining']})"
    )
    sep = "─" * 66
    print(sep)
    print(step_info)
    print(f"  Action → {obs['action_taken']}")
    print(sep)
    # Indent the result text for readability
    for line in obs["result"].splitlines():
        print(f"  {line}")
    print()


def main() -> None:
    env = BugInvestigationEnv(task_name="discount-rounding", max_steps=15)

    # ── Episode start ────────────────────────────────────────────────────
    banner("EPISODE START — reset()")
    obs = env.reset()
    print_obs(obs)

    # ── Step 1: run tests to see the failure ─────────────────────────────
    banner("Step 1: run_tests()")
    obs, reward, done, info = env.step({"type": "run_tests"})
    print_obs(obs)
    print(f"  Immediate reward: {reward:+.3f}   done={done}")

    # ── Step 2: search for the discount function ─────────────────────────
    banner('Step 2: search("discount")')
    obs, reward, done, info = env.step({"type": "search", "keyword": "discount"})
    print_obs(obs)
    print(f"  Immediate reward: {reward:+.3f}   done={done}")

    # ── Step 3: inspect a likely function to continue the investigation ──
    banner('Step 3: inspect_function("repo/discount.py", "apply_discount")')
    obs, reward, done, info = env.step({
        "type": "inspect_function",
        "filename": "repo/discount.py",
        "function": "apply_discount",
    })
    print_obs(obs)
    print(f"  Immediate reward: {reward:+.3f}   done={done}")

    banner("Episode Summary")
    print(f"  Files opened:          {info['files_opened']}")
    print(f"  Functions inspected:   {info['functions_inspected']}")
    print(f"  Keywords searched:     {info['keywords_searched']}")
    print(f"  Tests run:             {info['tests_run']}")
    print(f"  Task id:               {info['task_id']}")
    print(f"  Sandbox mode:          {info['sandbox_mode']}")
    print()


if __name__ == "__main__":
    main()
