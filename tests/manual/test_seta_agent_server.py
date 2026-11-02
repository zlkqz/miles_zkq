"""在远端 Harbor 依赖环境中验证 SETA runner；不使用 GPU 或 Docker。"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_server.models import RunRequest
from examples.seta import harbor_trial


@pytest.fixture
def trial(tmp_path, monkeypatch):
    task = tmp_path / "tasks/one"
    (task / "environment").mkdir(parents=True)
    (task / "task.toml").write_text('version = "1.0"\n')
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    config = {"service_id": "unit", "tasks_dir": str(tmp_path / "tasks"), "state_dir": str(tmp_path), "max_trial_timeout": 10, "min_free_gb": 0}
    path = tmp_path / "server.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("SETA_SERVER_CONFIG", str(path))
    monkeypatch.setenv("HARBOR_ENV_TYPE", "docker")
    monkeypatch.setenv("SETA_FULL_PASS_BONUS", "0.2")
    monkeypatch.setenv("AGENT_TRIAL_TIMEOUT", "10")
    monkeypatch.setattr(harbor_trial.ImageCache, "prepare", AsyncMock())
    monkeypatch.setattr(harbor_trial.ImageCache, "completed", AsyncMock())
    monkeypatch.setattr(harbor_trial, "configure_docker", lambda trial: "image")
    monkeypatch.setattr(harbor_trial, "_trial_compose_project", lambda trial: "project")
    result = SimpleNamespace(verifier_result=object(), exception_info=None)
    fake = SimpleNamespace(paths=SimpleNamespace(trial_dir=tmp_path, verifier_dir=verifier), run=AsyncMock(return_value=result))
    monkeypatch.setattr(harbor_trial.Trial, "create", AsyncMock(return_value=fake))
    monkeypatch.setattr(harbor_trial, "trial_result_to_metadata", lambda result: {"reward": 99.0, "eval_report": {"weighted": 99.0}, "agent_metrics": {}, "exit_status": "Submitted"})
    request = RunRequest(base_url="http://127.0.0.1:30000/v1", model="openai/Qwen3-8B", instance_id="one", agent_name="terminus-2", request_id="a" * 32, run_id="unit-run")
    return request, verifier, result


@pytest.mark.parametrize("url,host", [("http://model-anywhere:30000/session/v1", "model-anywhere"), ("https://model.example/v1", "model.example"), ("http://[::1]:30000/v1", "::1")])
@pytest.mark.asyncio
async def test_any_model_host_bypasses_proxy_and_preserves_other_entries(trial, monkeypatch, url, host):
    request, verifier, _ = trial
    monkeypatch.setenv("NO_PROXY", "localhost,old-host")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setenv("HTTPS_PROXY", "http://download-proxy:7890")
    (verifier / "ctrf.json").write_text(json.dumps({"results": {"tests": [{"name": "one", "status": "passed"}]}}))
    output = await harbor_trial.run(request.model_copy(update={"base_url": url}))
    assert output["metadata"]["seta_valid_reward"]
    assert harbor_trial.os.environ["NO_PROXY"] == harbor_trial.os.environ["no_proxy"]
    assert set(harbor_trial.os.environ["NO_PROXY"].split(",")) == {"localhost", "old-host", "127.0.0.1", host}
    assert harbor_trial.os.environ["HTTPS_PROXY"] == "http://download-proxy:7890"
    config = harbor_trial.Trial.create.call_args.args[0]
    assert config.agent.kwargs["api_base"] == url


@pytest.mark.parametrize("passed,expected", [(0, 0.0), (1, 0.5), (2, 1.2)])
@pytest.mark.asyncio
async def test_progress_reward_overrides_generic_scalar(trial, passed, expected):
    request, verifier, _ = trial
    tests = [{"name": str(i), "status": "passed" if i < passed else "failed"} for i in range(2)]
    (verifier / "ctrf.json").write_text(json.dumps({"results": {"tests": tests}}))
    result = await harbor_trial.run(request)
    assert result["reward"] == expected
    assert result["metadata"]["seta_valid_reward"]
    assert result["eval_report"]["passed"] == passed


@pytest.mark.asyncio
async def test_missing_verifier_is_invalid_not_normal_zero(trial):
    request, _, result = trial
    result.verifier_result = None
    output = await harbor_trial.run(request)
    assert not output["metadata"]["seta_valid_reward"]


@pytest.mark.asyncio
async def test_request_cannot_override_remote_proxy(trial):
    request, _, _ = trial
    request = request.model_copy(update={"seta_settings": {"DOCKER_CONFIG": "/gpu/wrong"}})
    output = await harbor_trial.run(request)
    assert not output["metadata"]["seta_valid_reward"]
    assert "不能覆盖" in output["metadata"]["seta_error"]


@pytest.mark.parametrize("url", ["file:///tmp/model", "http:///missing-host"])
@pytest.mark.asyncio
async def test_model_url_requires_http_address(trial, url):
    request, _, _ = trial
    output = await harbor_trial.run(request.model_copy(update={"base_url": url}))
    assert not output["metadata"]["seta_valid_reward"]
    assert "HTTP(S)" in output["metadata"]["seta_error"]
