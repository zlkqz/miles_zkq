"""用真实 Bash/信号/HTTP 验证训练入口退出，隔离 Ray 与 GPU。"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def launch(tmp_path):
    events = tmp_path / "events.jsonl"
    events.touch()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with events.open("a") as file:
                file.write(json.dumps({"flush": body}) + "\n")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"cancelled":1}')

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "my_scripts").mkdir()
    (tmp_path / "my_scripts/stop_seta.py").symlink_to(_REPO / "my_scripts/stop_seta.py")
    (tmp_path / "scripts/run_qwen3_8b_seta.py").write_text(
        "import os, signal, time\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGTERM, lambda *_: exit(143))\n"
        "Path(os.environ['READY']).write_text('ready')\n"
        "if 'FINISH' in os.environ: exit(int(os.environ['FINISH']))\n"
        "while True: time.sleep(1)\n"
    )
    # 假 Ray 仅记录 job stop；测试绝不停止真实训练或 Ray 集群。
    ray = tmp_path / "ray/scripts"
    ray.mkdir(parents=True)
    for path in (ray / "__init__.py", ray.parent / "__init__.py"):
        path.touch()
    (ray / "scripts.py").write_text(
        "import json, os, sys\n"
        "with open(os.environ['EVENTS'], 'a') as f: f.write(json.dumps({'ray':sys.argv[1:]})+'\\n')\n"
    )
    endpoint_file = tmp_path / "endpoints.json"
    endpoint_file.write_text(json.dumps({"endpoints": [{"url": f"http://127.0.0.1:{server.server_port}", "max_concurrent": 1}]}))
    env = {**os.environ, "REPO_DIR": str(tmp_path), "RUNTIME_DIR": str(tmp_path / "runtime"), "OUTPUT_DIR": str(tmp_path / "output"),
           "MODE": "train", "RUN_ID": "own-run", "EXP_NAME": "test", "PYTHON_BIN": sys.executable,
           "WANDB_MODE": "disabled", "WANDB_PROXY": "", "REMOTE_CONFIG": str(endpoint_file),
           "PYTHONPATH": f"{tmp_path}:{_REPO}", "READY": str(tmp_path / "ready"), "EVENTS": str(events)}
    processes = []

    def start(**overrides):
        log = (tmp_path / "console.log").open("w")
        process = subprocess.Popen(["bash", str(_REPO / "my_scripts/run_seta.sh")], env={**env, **overrides}, stdout=log, stderr=log, start_new_session=True)
        processes.append((process, log))
        deadline = time.monotonic() + 5
        while not (tmp_path / "ready").exists():
            if process.poll() is not None or time.monotonic() > deadline:
                pytest.fail((tmp_path / "console.log").read_text())
            time.sleep(0.01)
        return process

    try:
        yield start, tmp_path, events
    finally:
        for process, log in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            log.close()
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("signum,status", [(signal.SIGINT, 130), (signal.SIGTERM, 143)])
def test_signal_stops_own_job_and_flushes_own_run(launch, signum, status):
    start, directory, events = launch
    process = start()
    # 模拟后续启动覆盖全局记录；退出仍必须清理自己的独立记录。
    (directory / "runtime/remote/active_run.json").write_text('{"run_id":"other-run"}')
    if signum == signal.SIGINT:
        # 终端 Ctrl+C 会发给整个前台进程组，包括日志转发进程。
        os.killpg(process.pid, signum)
    else:
        process.send_signal(signum)
    assert process.wait(timeout=10) == status
    recorded = [json.loads(line) for line in events.read_text().splitlines()]
    assert recorded[0]["ray"][-1] == "own-run"
    assert recorded[0]["ray"][:2] == ["job", "stop"]
    assert recorded[1] == {"flush": {"run_id": "own-run"}}


@pytest.mark.parametrize("status", [0, 7])
def test_completion_and_failure_also_flush(launch, status):
    start, _, events = launch
    process = start(FINISH=str(status))
    assert process.wait(timeout=10) == status
    assert json.loads(events.read_text().splitlines()[-1]) == {"flush": {"run_id": "own-run"}}


def test_plan_does_not_stop_or_flush(launch):
    start, _, events = launch
    process = start(MODE="plan", FINISH="0")
    assert process.wait(timeout=10) == 0
    assert events.read_text() == ""
