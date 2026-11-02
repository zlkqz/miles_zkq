"""验证 SETA 部分进度奖励及缺失验证结果的处理。"""

import json

import pytest

from examples.seta.progress import progress_reward, read_test_results


def test_pytest_partial_progress(tmp_path):
    (tmp_path / "test-stdout.txt").write_text(
        "\x1b[32mPASSED\x1b[0m tests/test_task.py::test_a\n"
        "FAILED tests/test_task.py::test_b - AssertionError\n"
        "ERROR tests/test_task.py::test_c\n"
        "SKIPPED tests/test_task.py::test_d\n"
    )
    assert progress_reward(read_test_results(tmp_path), full_pass_bonus=0.2) == 0.25


def test_ctrf_takes_precedence_and_adds_bonus(tmp_path):
    (tmp_path / "test-stdout.txt").write_text("FAILED stale::test\n")
    tests = [{"name": "test_a", "status": "passed"}, {"name": "test_b", "status": "passed"}]
    (tmp_path / "ctrf.json").write_text(json.dumps({"results": {"tests": tests}}))
    assert progress_reward(read_test_results(tmp_path), full_pass_bonus=0.2) == 1.2


def test_binary_score_does_not_hide_missing_test_results(tmp_path):
    (tmp_path / "test-stdout.txt").write_text("pytest installation failed")
    (tmp_path / "reward.txt").write_text("0")
    with pytest.raises(ValueError, match="没有逐测试结果"):
        read_test_results(tmp_path)
