"""从 GPU 宿主机部署远端 CPU Harbor 服务；配置集中于同目录 config.json。

plan 只展示；check 只检查；apply 创建/同步文件、安装并重启选定的空闲服务。
不会启动训练、安装或重启 Docker，不分配远端 GPU。
"""

import argparse
import json
import os
import platform
import re
import shlex
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener


def _run(argv, *, capture=False, env=None):
    return subprocess.run(argv, check=True, text=True, env=env, stdout=subprocess.PIPE if capture else None).stdout


def _ssh(node, argv, *, capture=False, interactive=False):
    return _run(["ssh", *(["-t"] if interactive else []), node["ssh"], shlex.join(map(str, argv))], capture=capture)


def _path(repo, value):
    path = Path(value)
    return path if path.is_absolute() else repo / path


def _load(path):
    config = json.loads(path.read_text())
    names = []
    for node in config["nodes"]:
        obsolete = {"cpu_quota", "memory_max", "min_free_gb"} & node.keys()
        if obsolete:
            raise ValueError(f"{', '.join(sorted(obsolete))} 已由 max_concurrent 自动推导，请删除这些旧配置项")
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", node["name"]):
            raise ValueError("节点 name 只能包含字母、数字、下划线和连字符")
        if not Path(node["root"]).is_absolute() or node["root"] == "/ceph_home" or node["root"].startswith("/ceph_home/"):
            raise ValueError("root 必须是远端非共享盘的绝对路径")
        if type(node["max_concurrent"]) is not int or node["max_concurrent"] < 1:
            raise ValueError("max_concurrent 必须为正整数")
        names.append(node["name"])
    if not names or len(set(names)) != len(names):
        raise ValueError("至少配置一个节点，且 name 不可重复")
    return config


def _install_args(config, node):
    root = Path(node["root"])
    result = ["--root", str(root)]
    for field in ("max_concurrent", "port", "service_name", "service_user", "service_id", "proxy", "image_cache_keep"):
        result += ["--" + field.replace("_", "-"), str(node[field])]
    for value in node["no_proxy"]:
        result += ["--no-proxy", value]
    for value in node["registry_mirrors"]:
        result += ["--registry-mirror", value]
    return result


def _assert_idle(root):
    previous = root / "server.json"
    if previous.exists():
        previous_config = json.loads(previous.read_text())
        url = f"http://127.0.0.1:{previous_config['port']}/health"
        # 已停止的旧服务可以重新部署；在线服务有任务时拒绝打断它。
        try:
            with build_opener(ProxyHandler({})).open(url, timeout=3) as response:
                health = json.load(response)
        except ConnectionError:
            return
        except OSError as error:
            if isinstance(getattr(error, "reason", None), ConnectionRefusedError):
                return
            raise
        if health["active"]:
            raise RuntimeError("远端仍有活动 episode；先停止对应训练再部署")


def _bootstrap(config, node):
    """在远端部署账号下执行；仅安装 systemd 的步骤通过 sudo 提权。"""
    root = Path(node["root"])
    repo = root / "miles"
    _assert_idle(root)
    for command in (["docker", "info"], ["docker", "compose", "version"], ["docker", "buildx", "version"]):
        _run(command, capture=True)
    if platform.machine() != "x86_64":
        raise ValueError("当前工具下载配置仅支持 Linux x86_64；其他架构需调整下载地址")
    environment = dict(os.environ)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        environment[name] = node["proxy"]
    environment.update(UV_INSTALL_DIR=str(root / "tools"), UV_NO_MODIFY_PATH="1", UV_PYTHON_INSTALL_DIR=str(root / "python"), UV_CACHE_DIR=str(root / "uv-cache"))
    tools = root / "tools"
    tools.mkdir(exist_ok=True)
    uv = tools / "uv"
    if not uv.exists():
        installer = tools / "install-uv.sh"
        _run(["curl", "-fLsS", "https://astral.sh/uv/install.sh", "-o", str(installer)], env=environment)
        _run(["sh", str(installer)], env=environment)
    _run([str(uv), "python", "install", config["python_version"]], env=environment)
    _run([str(uv), "sync", "--project", str(repo / "seta_miles_runtime/harbor"), "--python", config["python_version"], "--locked", "--no-dev"], env=environment)
    crane = tools / "crane"
    if not crane.exists():
        archive = tools / "crane.tar.gz"
        url = f"https://github.com/google/go-containerregistry/releases/download/{config['crane_version']}/go-containerregistry_Linux_x86_64.tar.gz"
        _run(["curl", "-fL", url, "-o", str(archive)], env=environment)
        _run(["tar", "-xzf", str(archive), "-C", str(tools), "crane"])
        archive.unlink()
    _run(["sudo", "python3", str(repo / "my_scripts/configure_seta_remote.py"), *_install_args(config, node), "--update-proxy", "--export-client-token", str(root / "client-token")])
    # apply 明确部署此专用服务；不重启 Docker，也不影响其他服务。
    _run(["sudo", "systemctl", "restart", node["service_name"]])


def _copy(source, node, destination, excludes=()):
    _run(["rsync", "-az", *(f"--exclude={value}" for value in excludes), str(source).rstrip("/") + "/", f"{node['ssh']}:{shlex.quote(str(destination))}/"])


def _prepare_node(repo, config_path, config, node):
    root = Path(node["root"])
    remote_repo = root / "miles"
    uid = _ssh(node, ["id", "-u"], capture=True).strip()
    gid = _ssh(node, ["id", "-g"], capture=True).strip()
    # 提权仅创建指定目录；不递归 chown 已有任务或服务的 state。
    _ssh(node, ["sudo", "install", "-d", "-o", uid, "-g", gid, str(root), str(remote_repo), str(remote_repo / "seta_miles_runtime"), str(root / "tasks")], interactive=True)
    bundle = remote_repo / "seta_miles_runtime/seta-remote"
    _copy(Path(__file__).resolve().parent, node, bundle, ("__pycache__",))
    # --config 支持自定义部署输入；远端统一读取其独立副本，不访问 GPU 原文件。
    _run(["scp", str(config_path), f"{node['ssh']}:{shlex.quote(str(bundle / 'config.json'))}"])
    _ssh(node, ["python3", str(bundle / "setup.py"), "--mode", "check", "--node", node["name"], "--remote"])
    excludes = (".git", ".venv", "__pycache__", "node_modules", "seta_miles_runtime/***", "my_scripts/run_seta.sh", ".env", "/models/***", "/data/***", "/datasets/***", "/outputs/***", "/checkpoints/***", "/saved_models/***", "/wandb/***")
    _copy(repo, node, remote_repo, excludes)
    _copy(repo / "seta_miles_runtime/harbor", node, remote_repo / "seta_miles_runtime/harbor", (".venv", "__pycache__"))
    _copy(_path(repo, config["tasks_source"]), node, root / "tasks")
    _ssh(node, ["python3", str(bundle / "setup.py"), "--mode", "apply", "--node", node["name"], "--remote"], interactive=True)
    token = _ssh(node, ["cat", str(root / "client-token")], capture=True).strip()
    if not token or any(character.isspace() for character in token):
        raise ValueError("远端未返回有效 token")
    local_token = _path(repo, node["token_file"])
    local_token.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(local_token, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as output:
        output.write(token + "\n")
    local_token.chmod(0o600)
    _ssh(node, ["rm", "-f", str(root / "client-token")])


def _write_endpoints(repo, config):
    for node in config["nodes"]:
        if not _path(repo, node["token_file"]).is_file():
            raise ValueError(f"节点 {node['name']} 尚无 GPU 本地 token，请先部署此节点")
    endpoints = [{"url": f"http://{node['host']}:{node['port']}", "max_concurrent": node["max_concurrent"], "token_file": str(_path(repo, node["token_file"]).resolve())} for node in config["nodes"]]
    path = _path(repo, config["endpoints_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"endpoints": endpoints}, indent=2) + "\n")
    temporary.replace(path)


def _check(repo, node, *, wait=False):
    token = _path(repo, node["token_file"]).read_text().strip()
    request = Request(f"http://{node['host']}:{node['port']}/health", headers={"Authorization": f"Bearer {token}"})
    deadline = time.monotonic() + (30 if wait else 0)
    while True:
        try:
            with build_opener(ProxyHandler({})).open(request, timeout=10) as response:
                health = json.load(response)
            break
        except HTTPError:
            raise
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.5)
    probe = Request(f"http://{node['host']}:{node['port']}/clients", headers={"Authorization": f"Bearer {token}"})
    with build_opener(ProxyHandler({})).open(probe, timeout=10):
        pass
    if health.get("protocol") != "harbor-agent-server-v1" or node["max_concurrent"] > health["max_concurrent"]:
        raise ValueError(f"{node['name']}：协议或容量不匹配")
    root = Path(node["root"])
    for field, expected in (("tasks_dir", root / "tasks"), ("docker_config", root / "docker_config")):
        if health[field] != str(expected):
            raise ValueError(f"{node['name']}：正在运行的 {field} 与部署配置不同")
    _ssh(node, ["systemctl", "is-active", "--quiet", node["service_name"]])
    print(f"{node['name']}：服务健康；容量 {health['max_concurrent']}，活动 {health['active']}，磁盘可用 {health['free_gb']} GB；Docker 配置 {health['docker_config']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--mode", choices=("plan", "check", "apply"), default="plan")
    parser.add_argument("--node", action="append", help="仅操作指定 name；不传则操作全部节点")
    parser.add_argument("--remote", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    config = _load(args.config)
    nodes = [node for node in config["nodes"] if not args.node or node["name"] in args.node]
    if not nodes or (args.node and set(args.node) - {node["name"] for node in nodes}):
        parser.error("指定的节点 name 不存在")
    if args.remote:
        if args.mode == "plan" or len(nodes) != 1:
            parser.error("远端 bootstrap 仅允许单节点 check/apply")
        if args.mode == "check":
            _assert_idle(Path(nodes[0]["root"]))
        else:
            _bootstrap(config, nodes[0])
        return
    if args.mode == "apply" and not _path(repo, config["tasks_source"]).is_dir():
        parser.error("tasks_source 不存在；先准备数据")
    for node in nodes:
        if args.mode == "plan":
            print(f"{node['name']}：SSH={node['ssh']}，本地盘 root={node['root']}，容量={node['max_concurrent']}")
            print("服务参数：" + shlex.join(_install_args(config, node)))
        elif args.mode == "apply":
            _prepare_node(repo, args.config.resolve(), config, node)
        else:
            _check(repo, node)
    if args.mode == "apply":
        _write_endpoints(repo, config)
        for node in nodes:
            _check(repo, node, wait=True)
        print("部署完成，endpoint/token 已写入 GPU 本地；没有启动训练。")


if __name__ == "__main__":
    main()
