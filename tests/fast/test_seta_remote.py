"""官方 agent-server 的 SETA 集成：并发、run-scoped flush、鉴权与部署边界。"""

import asyncio
import importlib.util
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from examples.seta.remote_config import Endpoint, load_endpoints
from examples.seta import remote_pool
from examples.seta.remote_images import ImageCache, image_sources
from examples.seta.docker_adapter import proxy_defaults, proxy_environment
from examples.seta.task_utils import validate_task
from my_scripts import stop_seta

_HARBOR = Path(__file__).resolve().parents[2] / "seta_miles_runtime/harbor"
sys.path.insert(0, str(_HARBOR))
sys.path.insert(0, str(_HARBOR / "server-dashboard"))
_SPEC = importlib.util.spec_from_file_location("seta_agent_server_test", _HARBOR / "miles_agent_server.py")
server = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(server)
background = importlib.import_module("agent_server.background")


@pytest.mark.asyncio
async def test_session_heartbeat_bypasses_download_proxy(monkeypatch):
    for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(key, "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setattr(background, "_HEARTBEAT_INTERVAL", 0.01)
    monkeypatch.setattr(background, "_HEARTBEAT_FAILURE_THRESHOLD", 1)
    monkeypatch.setattr(background, "_HEARTBEAT_TIMEOUT", 0.2)
    received = asyncio.Event()

    async def health(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        body = b'{"session_server_instance_id":"heartbeat-unit"}'
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        await writer.drain()
        writer.close()
        await writer.wait_closed()
        received.set()

    listener = await asyncio.start_server(health, "127.0.0.1", 0)
    port = listener.sockets[0].getsockname()[1]
    work = asyncio.create_task(asyncio.sleep(3600))
    server._register_active_task("heartbeat-unit", f"127.0.0.1:{port}", "heartbeat-unit", work)
    checker = asyncio.create_task(background._health_checker_loop())
    try:
        await asyncio.wait_for(received.wait(), timeout=2)
        await asyncio.sleep(0.02)
        assert not work.done()
    finally:
        checker.cancel()
        work.cancel()
        await asyncio.gather(checker, work, return_exceptions=True)
        server._state.active_tasks.pop("heartbeat-unit", None)
        listener.close()
        await listener.wait_closed()


@pytest.fixture
def remote(tmp_path, monkeypatch):
    server._state.active_tasks.clear()
    server._state.inflight_tasks.clear()
    server._state.run_tasks.clear()
    server._state.trials_dir = tmp_path
    monkeypatch.setenv("HARBOR_DOCKER_PRUNE_ENABLED", "false")
    monkeypatch.setenv("AGENT_MAX_CONCURRENT", "1")
    monkeypatch.setattr(server, "_RUN_SECRET", "unit-test-token")
    monkeypatch.setattr(server, "_ADMIN_SECRET", "unit-test-token")
    active = {"now": 0, "peak": 0}

    async def hold(*args):
        active["now"] += 1
        active["peak"] = max(active["peak"], active["now"])
        try:
            await asyncio.sleep(3600)
        finally:
            active["now"] -= 1

    monkeypatch.setattr(server, "run_trial_in_subprocess", hold)
    with TestClient(server.app) as client:
        client.headers["Authorization"] = "Bearer unit-test-token"
        yield client, active


def request(run_id="unit-run"):
    return {"base_url": "http://127.0.0.1:30000/sessions/x/v1", "model": "openai/Qwen3-8B", "instance_id": "one", "agent_name": "terminus-2", "run_id": run_id}


def _wait_active(client, count):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if client.get("/health").json()["active"] == count:
            return
        time.sleep(0.01)
    pytest.fail(f"服务没有达到预期活动数 {count}")


def test_run_scoped_flush_and_capacity(remote):
    client, active = remote
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(client.post, "/run", json=request("run-a"))
        _wait_active(client, 1)
        second = pool.submit(client.post, "/run", json=request("run-b"))
        _wait_active(client, 2)
        assert client.post("/flush", json={"run_id": "run-a"}).json()["cancelled"] == 1
        assert first.result(timeout=2).json()["exit_status"] == "Flushed"
        assert not second.done()
        assert client.post("/flush", json={"run_id": "run-b"}).json()["cancelled"] == 1
        assert second.result(timeout=2).json()["exit_status"] == "Flushed"
        _wait_active(client, 0)
        assert active["peak"] == 1


def test_run_and_admin_auth(remote):
    client, _ = remote
    assert client.post("/run", json=request(), headers={"Authorization": "wrong"}).status_code == 403
    assert client.post("/flush", json={"run_id": "unit-run"}, headers={"Authorization": "wrong"}).status_code == 403
    assert client.get("/health").json()["protocol"] == "harbor-agent-server-v1"


def test_cpu_task_validation(tmp_path):
    task = tmp_path / "one"
    (task / "environment").mkdir(parents=True)
    (task / "task.toml").write_text('version = "1.0"\n')
    assert validate_task(tmp_path, "one") == task
    with pytest.raises(ValueError):
        validate_task(tmp_path, "../escape")
    (task / "environment/docker-compose.yaml").write_text("services:\n  main:\n    privileged: true\n")
    with pytest.raises(ValueError, match="CPU 沙盒"):
        validate_task(tmp_path, "one")


@pytest.mark.asyncio
async def test_endpoint_slots_enforce_each_node_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(remote_pool.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(remote_pool, "current_scope", lambda: "unit-slots")
    endpoints = (Endpoint("http://one:18080", 2, ""), Endpoint("http://two:18080", 1, ""))
    active = {endpoint.url: 0 for endpoint in endpoints}
    peak = dict(active)

    async def episode():
        async with remote_pool.endpoint_slot(endpoints) as endpoint:
            active[endpoint.url] += 1
            peak[endpoint.url] = max(peak[endpoint.url], active[endpoint.url])
            assert active[endpoint.url] <= endpoint.max_concurrent
            await asyncio.sleep(0.02)
            active[endpoint.url] -= 1

    await asyncio.gather(*(episode() for _ in range(12)))
    assert peak == {"http://one:18080": 2, "http://two:18080": 1}


def test_endpoint_config_rejects_duplicate_or_invalid_limits(tmp_path):
    path = tmp_path / "endpoints.json"
    endpoint = {"url": "http://localhost:18080", "max_concurrent": 2}
    path.write_text(json.dumps({"endpoints": [endpoint]}))
    assert load_endpoints(str(path))[0].max_concurrent == 2
    path.write_text(json.dumps({"endpoints": [endpoint, endpoint]}))
    with pytest.raises(ValueError):
        load_endpoints(str(path))
    endpoint["max_concurrent"] = 0
    path.write_text(json.dumps({"endpoints": [endpoint]}))
    with pytest.raises(ValueError):
        load_endpoints(str(path))


def test_explicit_task_proxy_wins_over_both_default_variants(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "proxies": {
                    "default": {
                        "httpsProxy": "http://proxy:7890",
                        "httpProxy": "http://proxy:7890",
                    }
                }
            }
        )
    )
    defaults = proxy_environment(str(tmp_path))
    values = proxy_defaults(defaults, {"HTTPS_PROXY"})
    assert "HTTPS_PROXY" not in values and "https_proxy" not in values
    assert values["HTTP_PROXY"] == values["http_proxy"] == "http://proxy:7890"


def test_registry_mirrors_preserve_non_docker_hub_images():
    assert image_sources("python:3.13-slim", ["mirror.example"]) == ("mirror.example/library/python:3.13-slim", "python:3.13-slim")
    assert image_sources("rocker/r-ver:4", ["mirror.example"])[0] == "mirror.example/rocker/r-ver:4"
    assert image_sources("docker.io/ubuntu:24.04", ["mirror.example"])[0] == "mirror.example/library/ubuntu:24.04"
    assert image_sources("mcr.microsoft.com/dotnet/sdk:8.0", ["mirror.example"]) == ("mcr.microsoft.com/dotnet/sdk:8.0",)


@pytest.mark.asyncio
async def test_mirror_download_is_tagged_back_to_task_reference(tmp_path, monkeypatch):
    run = AsyncMock(return_value="")
    monkeypatch.setattr("examples.seta.remote_images.command", run)
    cache = ImageCache({"crane": "/tools/crane", "registry_mirrors": ["mirror.example"]}, tmp_path)
    await cache.pull("python:3.13-slim", tmp_path / "image.tar")
    assert run.call_args_list[0].args[4] == "mirror.example/library/python:3.13-slim"
    assert run.call_args_list[-1].args == ("docker", "tag", "mirror.example/library/python:3.13-slim", "python:3.13-slim")


def test_stop_still_stops_ray_when_active_config_is_corrupt(tmp_path, monkeypatch):
    marker = tmp_path / "remote/active_run.json"
    marker.parent.mkdir()
    marker.write_text("invalid json")
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path))
    calls = []
    monkeypatch.setattr(stop_seta.subprocess, "run", lambda argv, **kwargs: calls.append(argv) or SimpleNamespace(returncode=0))
    stop_seta.main([])
    assert calls[0][-2:] == ["stop", "--force"]


def test_stop_uses_own_job_and_flushes_even_after_ray_timeout(tmp_path, monkeypatch):
    marker = tmp_path / "run.json"
    active = {"run_id": "own-run", "ray_submission_id": "own-job", "ray_address": "http://head:8265", "remote_config": "own.json"}
    marker.write_text(json.dumps(active))
    calls = []

    def timeout(argv, **kwargs):
        calls.append(argv)
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(stop_seta.subprocess, "run", timeout)
    monkeypatch.setattr(stop_seta, "_cancel_remote", lambda record: calls.append(record) or True)
    assert stop_seta.main(["--run-file", str(marker)]) == 0
    assert calls[0][-3:] == ["--address", "http://head:8265", "own-job"]
    assert calls[1] == active


def test_corrupt_explicit_run_file_does_not_stop_other_jobs(tmp_path, monkeypatch):
    marker = tmp_path / "run.json"
    marker.write_text("invalid json")
    stop = AsyncMock()
    monkeypatch.setattr(stop_seta.subprocess, "run", stop)
    assert stop_seta.main(["--run-file", str(marker)]) == 1
    stop.assert_not_called()


def test_stop_attempts_every_remote_when_one_fails(monkeypatch):
    endpoints = (Endpoint("http://one:18080", 1), Endpoint("http://two:18080", 1))
    calls = []

    def handle(request):
        calls.append((str(request.url), json.loads(request.content)))
        return httpx.Response(503 if request.url.host == "one" else 200, json={"cancelled": 1})

    client = httpx.Client(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(stop_seta, "load_endpoints", lambda path: endpoints)
    monkeypatch.setattr(stop_seta.httpx, "Client", lambda **kwargs: client)
    assert not stop_seta._cancel_remote({"run_id": "own-run", "remote_config": "own.json"})
    assert calls == [(endpoint.url + "/flush", {"run_id": "own-run"}) for endpoint in endpoints]
