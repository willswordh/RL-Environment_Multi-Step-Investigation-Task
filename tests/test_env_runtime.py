"""Regression tests for environment runtime compatibility paths."""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env import load_task
from env.tools.actions import parse_action
from env.tools.repository import Repository


def test_default_task_loads_from_task_pack():
    task = load_task()
    assert task.task_id == "discount-rounding"
    assert task.benchmark_test_target.endswith("test_vip_discount_rounding")
    assert os.path.exists(task.instruction_path)


def test_parse_action_accepts_function_name_alias():
    action, err = parse_action({
        "type": "inspect_function",
        "filename": "repo/discount.py",
        "function_name": "apply_discount",
    })
    assert err is None
    assert action is not None
    assert action.function_name == "apply_discount"


def test_parse_action_type_must_be_string():
    action, err = parse_action({"type": 123})
    assert action is None
    assert err is not None
    assert "must be a string" in err


def test_parse_action_submit_answer_supports_structured_fields():
    action, err = parse_action({
        "type": "submit_answer",
        "root_cause": "Root cause identified.",
        "bug_file": "repo/math_utils.py",
        "bug_function": "round_currency",
        "mechanism": "banker rounding",
        "proposed_fix": "Use Decimal ROUND_HALF_UP",
    })
    assert err is None
    assert action is not None
    assert action.bug_file == "repo/math_utils.py"
    assert action.bug_function == "round_currency"
    assert action.mechanism == "banker rounding"
    assert action.proposed_fix == "Use Decimal ROUND_HALF_UP"


def test_repository_inspect_function_returns_source():
    repo = Repository(task=load_task())
    ok, content = repo.inspect_function("repo/discount.py", "apply_discount")
    assert ok is True
    assert "def apply_discount" in content


def test_repository_run_tests_uses_compatible_pytest_flags(monkeypatch):
    captured = {}

    class FakeResult:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs.get("cwd")
        captured["repo_exists"] = os.path.exists(
            os.path.join(captured["cwd"], "repo", "order_processor.py")
        )
        captured["tests_exists"] = os.path.exists(
            os.path.join(captured["cwd"], "tests", "test_task_order_processor.py")
        )
        return FakeResult()

    monkeypatch.setattr(subprocess, "run", fake_run)
    repo = Repository(task=load_task())
    passed, output = repo.run_tests()

    assert passed is True
    assert output == "ok"
    assert captured["cwd"] != repo.task.task_root
    assert captured["repo_exists"] is True
    assert captured["tests_exists"] is True
    assert "--no-header" not in captured["cmd"]
    assert "tests/" not in captured["cmd"]
    assert (
        "tests/test_task_order_processor.py::TestDiscountRounding::test_vip_discount_rounding"
        in captured["cmd"]
    )
