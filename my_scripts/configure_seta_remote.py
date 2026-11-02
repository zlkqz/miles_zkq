"""在远端裸机安装 SETA systemd 服务；不安装/重启 Docker，不分配 GPU。

先准备本地磁盘上的 Miles、tasks、Harbor .venv 和可选 Crane。
用 --dry-run 查看配置；正式执行需要 sudo。真实配置与 token 放在 --root 下。
"""

import argparse
import json
import os
import pwd
import re
import secrets
import subprocess
from pathlib import Path


def _configuration(args):
    root = args.root.resolve()
    # 服务进程的保护预算；任务容器仍按各自 task.toml 限制资源。
    concurrency = args.max_concurrent
    return {
        "service_id": args.service_id or args.service_name,
        "host": args.host,
        "port": args.port,
        "tasks_dir": str(root / "tasks"),
        "state_dir": str(root / "state"),
        "max_concurrent": args.max_concurrent,
        "token_file": str(root / "state/remote-token"),
        "cpu_quota": max(200, 100 * concurrency),
        "memory_max": f"{max(8, 4 * concurrency)}G",
        "min_free_gb": max(15, concurrency),
        "max_trial_timeout": 7200,
        "image_cache_keep": args.image_cache_keep,
        "crane": str(root / "tools/crane") if (root / "tools/crane").exists() else "",
        "registry_mirrors": args.registry_mirror,
    }


def _no_proxy(args):
    return ",".join(["localhost", "127.0.0.1", "::1", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", *args.no_proxy])


def _service_environment(args, config):
    root = args.root.resolve()
    no_proxy = _no_proxy(args)
    values = {
        "PYTHONPATH": str(root / "miles"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "DOCKER_CONFIG": str(root / "docker_config"),
        "DOCKER_BUILDKIT": "0",
        "COMPOSE_DOCKER_CLI_BUILD": "0",
        "HTTP_PROXY": args.proxy,
        "HTTPS_PROXY": args.proxy,
        "http_proxy": args.proxy,
        "https_proxy": args.proxy,
        "NO_PROXY": no_proxy,
        "no_proxy": no_proxy,
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "void",
        "HF_HOME": str(root / "home/.cache/huggingface"),
        "XDG_CACHE_HOME": str(root / "home/.cache"),
        "XDG_CONFIG_HOME": str(root / "home/.config"),
        "UV_CACHE_DIR": str(root / "home/.cache/uv"),
        "LITELLM_LOCAL_MODEL_COST_MAP": "true",
        "HARBOR_TASKS_DIR": config["tasks_dir"],
        "HARBOR_ENV_TYPE": "docker",
        "HARBOR_TRIAL_RUNNER": "examples.seta.harbor_trial.run",
        "SETA_SERVER_CONFIG": str(root / "server.json"),
        "SETA_SERVICE_ID": config["service_id"],
        "HARBOR_ADMIN_SECRET_FILE": config["token_file"],
        "HARBOR_RUN_SECRET_FILE": config["token_file"],
        "HARBOR_DOCKER_PRUNE_ENABLED": "false",
        "HARBOR_REQUIRE_DOCKER_LOGIN": "false",
        "HARBOR_WORKER_CANCEL_TIMEOUT_SEC": "120",
        "HEARTBEAT_INTERVAL_SEC": "10",
        "HEARTBEAT_FAILURE_THRESHOLD": "3",
        "HEARTBEAT_TIMEOUT_SEC": "5",
        "DASHBOARD_LOG_PATH": str(root / "state/dashboard/requests.jsonl"),
    }
    if any("\n" in value or '"' in value for value in values.values()):
        raise ValueError("环境变量值不能含换行或双引号")
    return "".join(f'{name}="{value}"\n' for name, value in values.items())


def _unit(args, config):
    root = args.root.resolve()
    python = root / "miles/seta_miles_runtime/harbor/.venv/bin/python"
    return f"""[Unit]
Description=SETA remote CPU Harbor sandbox service
After=network-online.target docker.service
Requires=docker.service

[Service]
Type=simple
User={args.service_user}
SupplementaryGroups=docker
WorkingDirectory={root}/miles
EnvironmentFile={root}/service.env
ExecStart={python} -m examples.seta.remote_server --config {root}/server.json
Restart=on-failure
RestartSec=5
TimeoutStopSec=180
KillMode=mixed
InaccessiblePaths=-/ceph_home
PrivateTmp=true
DevicePolicy=closed
NoNewPrivileges=true
CPUQuota={config['cpu_quota']}%
MemoryMax={config['memory_max']}

[Install]
WantedBy=multi-user.target
"""


def _install(args, config):
    root = args.root.resolve()
    python = root / "miles/seta_miles_runtime/harbor/.venv/bin/python"
    if not python.exists() or not (root / "tasks").is_dir():
        raise ValueError("先准备 root/miles 的 Harbor 虚拟环境和 root/tasks")
    try:
        account = pwd.getpwnam(args.service_user)
    except KeyError:
        subprocess.run(["useradd", "--system", "--user-group", "--home-dir", str(root / "home"), "--create-home", "--shell", "/usr/sbin/nologin", args.service_user], check=True)
        account = pwd.getpwnam(args.service_user)
    subprocess.run(["usermod", "-aG", "docker", args.service_user], check=True)
    for directory in (root / "state", root / "docker_config", root / "home"):
        directory.mkdir(parents=True, exist_ok=True)
        os.chown(directory, account.pw_uid, account.pw_gid)
    token = Path(config["token_file"])
    if not token.exists():
        token.write_text(args.token_source.read_text() if args.token_source else secrets.token_urlsafe(32) + "\n")
    os.chown(token, account.pw_uid, account.pw_gid)
    token.chmod(0o600)
    if args.export_client_token:
        # 将实际服务 token 安全交给部署账号，供 SSH 复制；不输出到终端或 argv。
        fd = os.open(args.export_client_token, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as output:
            output.write(token.read_text())
        os.chown(args.export_client_token, int(os.environ.get("SUDO_UID", "0")), int(os.environ.get("SUDO_GID", "0")))
        args.export_client_token.chmod(0o600)
    docker_config = root / "docker_config/config.json"
    if not docker_config.exists() or args.update_proxy:
        data = json.loads(docker_config.read_text()) if docker_config.exists() else {}
        if args.proxy:
            data.setdefault("proxies", {})["default"] = {"httpProxy": args.proxy, "httpsProxy": args.proxy, "noProxy": _no_proxy(args)}
        elif "proxies" in data:
            data["proxies"].pop("default", None)
        docker_config.write_text(json.dumps(data, indent=2) + "\n")
        os.chown(docker_config, account.pw_uid, account.pw_gid)
        docker_config.chmod(0o600)
    (root / "server.json").write_text(json.dumps(config, indent=2) + "\n")
    (root / "service.env").write_text(_service_environment(args, config))
    (root / "service.env").chmod(0o600)
    unit = Path("/etc/systemd/system") / f"{args.service_name}.service"
    content = _unit(args, config)
    changed = not unit.exists() or unit.read_text() != content
    if changed:
        unit.write_text(content)
    # 共享 GPU 宿主机上的旧 NVIDIA runtime 可能受 daemon-reload 影响。
    # 配置 JSON/env 改变不需要重载整个 systemd；unit 真正变化时才执行一次。
    subprocess.run(["systemctl", "enable", "--no-reload", args.service_name], check=True)
    if changed:
        subprocess.run(["systemctl", "daemon-reload"], check=True)
    if subprocess.run(["systemctl", "is-active", "--quiet", args.service_name]).returncode:
        subprocess.run(["systemctl", "start", args.service_name], check=True)
    print(f"服务配置已写入：{root}；{args.service_name}。已运行服务不会被自动重启。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--service-name", default="seta-harbor")
    parser.add_argument("--service-user", default="seta-sandbox")
    parser.add_argument("--service-id", default="")
    parser.add_argument("--proxy", default="")
    parser.add_argument("--registry-mirror", action="append", default=[])
    parser.add_argument("--no-proxy", action="append", default=[])
    parser.add_argument("--image-cache-keep", type=int, default=16)
    parser.add_argument("--token-source", type=Path)
    parser.add_argument("--export-client-token", type=Path)
    parser.add_argument("--update-proxy", action="store_true", help="更新 Docker 默认代理，保留仓库认证等其他字段")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.service_name + args.service_user):
        parser.error("服务名与账号只支持字母、数字、下划线、连字符")
    if args.max_concurrent < 1:
        parser.error("max-concurrent 必须为正整数")
    config = _configuration(args)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        print(_unit(args, config))
    elif os.geteuid() != 0:
        parser.error("请使用 sudo 安装，或 --dry-run 只查看配置")
    else:
        _install(args, config)


if __name__ == "__main__":
    main()
