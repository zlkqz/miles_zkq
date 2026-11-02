"""为固定 Harbor Docker 后端补齐 Compose 版本差异、稳定镜像名和跨进程构建锁。"""

import asyncio
import fcntl
import json
import os
from pathlib import Path

import yaml


def proxy_environment(config_dir: str) -> dict[str, str]:
    path = Path(config_dir or Path.home() / ".docker") / "config.json"
    if not path.exists():
        return {}
    proxy = json.loads(path.read_text()).get("proxies", {}).get("default", {})
    values = {}
    for key, variable in (("httpProxy", "HTTP_PROXY"), ("httpsProxy", "HTTPS_PROXY"), ("noProxy", "NO_PROXY"), ("allProxy", "ALL_PROXY")):
        if value := proxy.get(key):
            values[variable] = str(value)
            values[variable.lower()] = str(value)
    return values


def _keys(values):
    if isinstance(values, dict):
        return set(values)
    return {item.split("=", 1)[0] for item in values or []}


def proxy_defaults(defaults, explicit):
    # 一个大小写变体被任务显式设置时，不再注入另一个变体来盖过它。
    explicit = {key.upper() for key in explicit}
    return {key: value for key, value in defaults.items() if key.upper() not in explicit}


def configure_docker(trial):
    environment = trial.agent_environment
    defaults = proxy_environment(os.environ.get("DOCKER_CONFIG", ""))
    task_compose = environment.environment_dir / "docker-compose.yaml"
    authored = yaml.safe_load(task_compose.read_text()) if task_compose.exists() else {}
    services = (authored or {}).get("services", {})
    names = {"main", *services}
    overlay = {}
    for name in names:
        service = services.get(name, {})
        explicit_env = _keys(service.get("environment")) | set(environment.task_env_config.env or {})
        config = {"environment": proxy_defaults(defaults, explicit_env), "runtime": "runc"}
        build = service.get("build", {})
        if isinstance(build, dict) and ((name == "main" and not environment.task_env_config.docker_image) or "build" in service):
            config["build"] = {"args": proxy_defaults(defaults, _keys(build.get("args")))}
        overlay[name] = config
    image = None
    if not environment.task_env_config.docker_image and not services.get("main", {}).get("image"):
        service_id = os.environ.get("SETA_SERVICE_ID", "remote")
        image = f"seta-cache-{service_id}-{environment.environment_id}"
        overlay["main"]["image"] = image
        overlay["main"]["build"]["labels"] = {"org.miles.seta.service": service_id}
        # 与实际 Compose image 保持一致，供 Harbor 的 OS 检查和本服务缓存管理使用。
        environment._env_vars.main_image_name = image
    path = trial.paths.trial_dir / "seta-docker-defaults.json"
    path.write_text(json.dumps({"services": overlay}, indent=2))
    environment.extra_docker_compose_paths.append(path)
    if image:
        _guard_build(environment, image, trial.config.trials_dir)
    return image


def _guard_build(environment, image, trials_dir):
    from harbor.environments.base import ExecResult

    from examples.seta.remote_images import command

    original = environment._run_docker_compose_command
    directory = Path(trials_dir).parent / "build-locks"
    directory.mkdir(exist_ok=True)

    async def guarded(arguments, **kwargs):
        if arguments != ["build"]:
            return await original(arguments, **kwargs)
        fd = os.open(directory / f"{image}.lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    await asyncio.sleep(0.2)
            try:
                await command("docker", "image", "inspect", image)
            except RuntimeError:
                return await original(arguments, **kwargs)
            return ExecResult(return_code=0, stdout="SETA image cache hit", stderr="")
        finally:
            os.close(fd)

    environment._run_docker_compose_command = guarded
