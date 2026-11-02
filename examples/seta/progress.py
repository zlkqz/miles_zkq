"""从 Harbor 保存的验证日志计算部分通过率，不退化成二元奖励。"""

import json
import re
from pathlib import Path

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_RESULT = re.compile(r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(\S+)", re.MULTILINE)


def read_test_results(verifier_dir: Path) -> dict[str, bool]:
    """优先读取 CTRF，否则解析 pytest -rA 摘要；缺失结果时明确报错。"""
    ctrf = verifier_dir / "ctrf.json"
    if ctrf.is_file():
        tests = json.loads(ctrf.read_text())["results"]["tests"]
        results = {test["name"]: test["status"] == "passed" for test in tests}
    else:
        stdout = verifier_dir / "test-stdout.txt"
        text = _ANSI.sub("", stdout.read_text(errors="replace"))
        results = {name: status == "PASSED" for status, name in _RESULT.findall(text)}
    if not results:
        raise ValueError(f"没有逐测试结果：{verifier_dir}；不能把二元 reward.txt 当作通过率")
    return results


def progress_reward(results: dict[str, bool], full_pass_bonus: float) -> float:
    """跳过、预期失败和错误均计为未通过，避免把未执行测试当作成功。"""
    if not results:
        raise ValueError("逐测试结果为空")
    passed = sum(results.values())
    return passed / len(results) + (full_pass_bonus if passed == len(results) else 0.0)
