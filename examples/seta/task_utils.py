"""SETA CPU 沙盒任务路径及设备配置检查，不负责 HTTP 或调度。"""

import re
import tomllib
from pathlib import Path

import yaml


def validate_task(root: Path, instance_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", instance_id):
        raise ValueError("非法 task ID")
    task = (root / instance_id).resolve()
    if root.resolve() not in task.parents or not (task / "task.toml").is_file():
        raise FileNotFoundError(f"远端缺少任务：{instance_id}")
    config = tomllib.loads((task / "task.toml").read_text())
    if config.get("environment", {}).get("privileged"):
        raise ValueError("CPU 沙盒拒绝 privileged 任务")
    for compose in (task / "environment").glob("*compose.y*ml"):
        for service in (yaml.safe_load(compose.read_text()) or {}).get("services", {}).values():
            reservations = service.get("deploy", {}).get("resources", {}).get("reservations", {})
            if any(service.get(key) for key in ("privileged", "gpus", "devices")) or reservations.get("devices"):
                raise ValueError("CPU 沙盒拒绝 privileged/GPU/设备透传任务")
            if service.get("runtime") == "nvidia":
                raise ValueError("CPU 沙盒拒绝 NVIDIA runtime")
    return task
