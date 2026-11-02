"""停止本次 Ray 作业，再按 run_id 回收它在所有远端上的 episode。

不带参数读取当前运行记录；run_seta.sh 通过 --run-file 指定自己的记录，
避免另一轮启动覆盖 active_run.json 后，退出清理误取消其他运行。
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.seta.remote_config import load_endpoints


def _cancel_remote(active):
    succeeded = True
    with httpx.Client(timeout=15, trust_env=False) as client:
        for endpoint in load_endpoints(active["remote_config"]):
            try:
                result = client.post(f"{endpoint.url}/flush", json={"run_id": active["run_id"]}, headers=endpoint.headers())
                result.raise_for_status()
                print(f"已请求回收远端运行 {active['run_id']}：{endpoint.url} {result.json()}", flush=True)
            except Exception as error:
                succeeded = False
                print(f"远端 flush 未送达，请恢复连接后重试停止命令：{endpoint.url} {type(error).__name__}", flush=True)
    return succeeded


def _stop_producer(active):
    if active.get("ray_submission_id"):
        command = [sys.executable, "-m", "ray.scripts.scripts", "job", "stop", "--address", active["ray_address"], active["ray_submission_id"]]
    else:
        # 兼容旧启动器的记录：当时没有保存 submission_id，只能停专用容器的 Ray。
        command = [sys.executable, "-m", "ray.scripts.scripts", "stop", "--force"]
    try:
        result = subprocess.run(command, check=False, timeout=60)
        if result.returncode:
            print("Ray 停止命令未成功（作业或服务可能已经退出），继续请求远端回收。", flush=True)
    except subprocess.TimeoutExpired:
        print("等待 Ray 停止超时，继续请求远端回收；请检查 GPU 作业状态。", flush=True)


def main(argv=None):
    runtime = Path(os.environ.get("RUNTIME_DIR", Path(__file__).resolve().parents[1] / "seta_miles_runtime"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-file", type=Path)
    args = parser.parse_args(argv)
    marker = args.run_file or runtime / "remote/active_run.json"
    try:
        active = json.loads(marker.read_text())
    except (OSError, ValueError, TypeError) as error:
        if args.run_file is None:
            # 兼容旧版手动停止入口；独立记录损坏时绝不扩大到其他 Ray 作业。
            _stop_producer({})
        print(f"无法读取运行记录：{marker} {type(error).__name__}")
        return 1
    try:
        # 先停生产者，再 flush，避免取消后又创建新 episode。每个节点独立尝试。
        _stop_producer(active)
    finally:
        succeeded = _cancel_remote(active)
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
