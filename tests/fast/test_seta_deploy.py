"""部署配置应正确生成 endpoint，并在活动任务存在时拒绝更新。"""

import copy
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("seta_deploy", _REPO / "seta_miles_runtime/seta-remote/setup.py")
deploy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(deploy)


@pytest.fixture
def configuration():
    return json.loads((_REPO / "seta_miles_runtime/seta-remote/config.json").read_text())


def test_service_args_do_not_require_gpu_hosts(configuration):
    node = configuration["nodes"][0]
    args = deploy._install_args(configuration, node)
    assert args[args.index("--root") + 1] == node["root"]
    assert "--callback-host" not in args
    assert args[args.index("--max-concurrent") + 1] == str(node["max_concurrent"])
    assert not {"--cpu-quota", "--memory-max", "--min-free-gb"}.intersection(args)


@pytest.mark.parametrize("concurrency,cpu,memory,disk", [(1, 200, "8G", 15), (16, 1600, "64G", 16), (32, 3200, "128G", 32)])
def test_installer_derives_service_budget_without_writing_files(tmp_path, concurrency, cpu, memory, disk):
    root = tmp_path / "sandbox"
    result = subprocess.run(
        [sys.executable, str(_REPO / "my_scripts/configure_seta_remote.py"), "--root", str(root), "--max-concurrent", str(concurrency), "--dry-run"],
        check=True, capture_output=True, text=True,
    )
    configuration, unit = result.stdout.split("[Unit]", 1)
    config = json.loads(configuration)
    assert (config["cpu_quota"], config["memory_max"], config["min_free_gb"]) == (cpu, memory, disk)
    assert f"CPUQuota={cpu}%" in unit and f"MemoryMax={memory}" in unit
    assert not root.exists()


def test_deployment_rejects_obsolete_manual_budgets(tmp_path, configuration):
    configuration["nodes"][0]["memory_max"] = "64G"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(configuration))
    with pytest.raises(ValueError, match="自动推导"):
        deploy._load(path)


def test_endpoint_file_uses_gpu_local_tokens(tmp_path, configuration):
    node = configuration["nodes"][0]
    node["token_file"] = "secrets/a.token"
    token = tmp_path / node["token_file"]
    token.parent.mkdir()
    token.write_text("test-token")
    configuration["endpoints_file"] = "remote/endpoints.json"
    deploy._write_endpoints(tmp_path, configuration)
    endpoint = json.loads((tmp_path / "remote/endpoints.json").read_text())["endpoints"][0]
    assert endpoint["token_file"] == str(token)
    assert endpoint["url"] == f"http://{node['host']}:{node['port']}"
    assert "test-token" not in (tmp_path / "remote/endpoints.json").read_text()


def test_unprepared_node_does_not_replace_endpoint_file(tmp_path, configuration):
    path = tmp_path / "endpoints.json"
    path.write_text("previous")
    configuration["endpoints_file"] = str(path)
    configuration["nodes"][0]["token_file"] = "missing.token"
    with pytest.raises(ValueError, match="尚无"):
        deploy._write_endpoints(tmp_path, configuration)
    assert path.read_text() == "previous"


def test_deployment_rejects_shared_remote_root_and_duplicate_names(tmp_path, configuration):
    path = tmp_path / "config.json"
    configuration["nodes"].append(copy.deepcopy(configuration["nodes"][0]))
    path.write_text(json.dumps(configuration))
    with pytest.raises(ValueError, match="重复"):
        deploy._load(path)
    configuration["nodes"] = configuration["nodes"][:1]
    configuration["nodes"][0]["root"] = "/ceph_home/shared"
    path.write_text(json.dumps(configuration))
    with pytest.raises(ValueError, match="非共享盘"):
        deploy._load(path)


def test_active_remote_service_is_not_interrupted(tmp_path, monkeypatch):
    (tmp_path / "server.json").write_text('{"port":18080}')

    class Opener:
        def open(self, *args, **kwargs):
            return io.BytesIO(b'{"active":1}')

    monkeypatch.setattr(deploy, "build_opener", lambda *args: Opener())
    with pytest.raises(RuntimeError, match="活动 episode"):
        deploy._assert_idle(tmp_path)
