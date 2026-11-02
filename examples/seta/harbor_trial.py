"""官方 agent-server 的 SETA trial runner 扩展；只处理实验特有行为。

由 HARBOR_TRIAL_RUNNER=examples.seta.harbor_trial.run 在官方每轨迹子进程中加载。
生产数据使用 Terminus-2；oracle/nop 仅用于不调用模型的环境检查。
"""

import asyncio
import json
import os
import re
import shutil
from pathlib import Path
from urllib.parse import urlsplit

from harbor.trial.trial import Trial
from agent_server.trial_runner import _trial_compose_project

from examples.experimental.harbor.harbor_agent_function import build_trial_config, trial_result_to_metadata
from examples.seta.docker_adapter import configure_docker
from examples.seta.progress import progress_reward, read_test_results
from examples.seta.remote_config import EPISODE_ENV
from examples.seta.remote_images import ImageCache
from examples.seta.task_utils import validate_task


def _validate(request, config):
    data = request.model_dump()
    settings = data.get("seta_settings", {})
    if set(settings) - set(EPISODE_ENV) or not all(isinstance(value, str) for value in settings.values()):
        raise ValueError("请求不能覆盖远端路径、代理或资源配置")
    if request.agent_name not in {"terminus-2", "oracle", "nop"}:
        raise ValueError("SETA 仅支持 Terminus-2 和环境检查 agent")
    request_id = data.get("request_id", "")
    if not re.fullmatch(r"[a-f0-9]{32}", request_id):
        raise ValueError("request_id 必须是 32 位十六进制 ID")
    if shutil.disk_usage(config["state_dir"]).free < config.get("min_free_gb", 15) * 1024**3:
        raise ValueError("远端磁盘剩余空间不足")
    task = validate_task(Path(config["tasks_dir"]), request.instance_id)
    return data, settings, task


def _configure_model_access(base_url):
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("模型回连地址必须是有效的 HTTP(S) URL")
    # 每条轨迹独立进程：直接访问 Miles 提供的模型地址，其他下载仍保留服务代理。
    entries = [entry.strip() for key in ("NO_PROXY", "no_proxy") for entry in os.getenv(key, "").split(",") if entry.strip()]
    value = ",".join(dict.fromkeys([*entries, parsed.hostname]))
    os.environ.update(NO_PROXY=value, no_proxy=value)


async def _trial(request, data, config, task):
    state = Path(config["state_dir"])
    cache = ImageCache(config, state)
    await cache.prepare(task)
    trial_config = build_trial_config(data, request.base_url, request.sampling_params)
    trial_config.trial_name = f"seta-{config['service_id']}-{data['request_id']}"
    if request.agent_name == "terminus-2":
        trial_config.agent.name = None
        trial_config.agent.import_path = "examples.seta.terminus:SetaTerminus2"
        trial_config.agent.kwargs["record_terminal_session"] = os.getenv("SETA_RECORD_TERMINAL_SESSION", "false") == "true"
    trial = await Trial.create(trial_config)
    image = configure_docker(trial)
    try:
        result = await trial.run()
        output = trial_result_to_metadata(result)
        metadata = {"seta_valid_reward": False, "trial_dir": str(trial.paths.trial_dir), "remote_request_id": data["request_id"]}
        if result.verifier_result is None:
            metadata["seta_error"] = str(result.exception_info or "验证器未返回结果")
        else:
            try:
                tests = read_test_results(trial.paths.verifier_dir)
                output["reward"] = progress_reward(tests, float(os.environ["SETA_FULL_PASS_BONUS"]))
                output["eval_report"] = {"passed": sum(tests.values()), "total": len(tests), "tests": tests}
                metadata["seta_valid_reward"] = True
            except (OSError, ValueError, KeyError, TypeError) as error:
                metadata["seta_error"] = str(error)
        return {**output, "metadata": metadata, "trial_dir": str(trial.paths.trial_dir), "compose_project": _trial_compose_project(trial)}
    finally:
        try:
            await cache.completed(image)
        except Exception as error:
            trial.logger.warning("镜像缓存回收失败：%s", error)


async def run(request):
    """官方 RunRequest -> RunResponse 字段；基础设施异常不作为正常零分。"""
    try:
        config = json.loads(Path(os.environ["SETA_SERVER_CONFIG"]).read_text())
        data, settings, task = _validate(request, config)
        os.environ.update(settings)  # 官方每轨迹独立子进程，不改父服务或兄弟轨迹的环境。
        _configure_model_access(request.base_url)
        os.environ.update(HARBOR_TASKS_DIR=config["tasks_dir"], HARBOR_TRIALS_DIR=str(Path(config["state_dir"]) / "trials"))
        timeout = min(float(os.environ["AGENT_TRIAL_TIMEOUT"]), config["max_trial_timeout"])
        return await asyncio.wait_for(_trial(request, data, config, task), timeout=timeout)
    except Exception as error:
        return {"reward": 0.0, "exit_status": f"Error: {type(error).__name__}", "eval_report": {}, "agent_metrics": {}, "metadata": {"seta_valid_reward": False, "seta_error": str(error)}}
