"""在宿主机为原始 Miles 镜像增加 Docker 工具和独立 Python 解释器。

示例：PROXY_URL=http://124.16.138.150:7890 python3 my_scripts/build_seta_image.py
构建上下文为临时白名单目录；构建记录保存在忽略的 runtime/image_build 下。
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _inspect(image):
    return json.loads(subprocess.check_output(["docker", "image", "inspect", image]))[0]


def _build(args, repo, artifacts):
    sources = {
        "docker": Path(args.docker_bin),
        "docker-compose": Path(args.compose_bin),
        "docker-buildx": Path(args.buildx_bin),
    }
    for source in sources.values():
        if not source.is_file():
            raise FileNotFoundError(f"找不到工具，请通过命令行指定路径：{source}")
    base = _inspect(args.base_image)
    with tempfile.TemporaryDirectory(prefix="context-", dir=artifacts) as temporary:
        context = Path(temporary)
        for name in ("Dockerfile", ".dockerignore"):
            shutil.copy2(repo / "docker/seta-tools" / name, context / name)
        checksums = {}
        for name, source in sources.items():
            shutil.copy2(source, context / name)
            (context / name).chmod(0o755)
            checksums[name] = hashlib.sha256((context / name).read_bytes()).hexdigest()
        command = [
            "docker", "build", "--pull=false", "--network=host",
            "--build-arg", f"BASE_IMAGE={args.base_image}", "-t", args.image,
        ]
        if args.proxy:
            for variable in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                command.extend(["--build-arg", f"{variable}={args.proxy}"])
        command.append(str(context))
        # 本机 Docker 23 的传统 builder 可以完全复用本地基础镜像，无需查询镜像源。
        environment = dict(os.environ, DOCKER_BUILDKIT="0")
        with (artifacts / "build.log").open("w") as log:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env=environment,
            )
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait():
                raise RuntimeError(f"镜像构建失败，日志：{log.name}")
    image = _inspect(args.image)
    manifest = {
        "image": args.image, "image_id": image["Id"],
        "base_image": args.base_image, "base_image_id": base["Id"],
        "base_repo_digests": base.get("RepoDigests", []),
        "tool_sha256": checksums,
        "dockerfile_sha256": hashlib.sha256((repo / "docker/seta-tools/Dockerfile").read_bytes()).hexdigest(),
    }
    destination = artifacts / "manifest.json"
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"已构建 {args.image}；构建记录：{destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="miles-seta:tools-v1")
    parser.add_argument("--base-image", default="radixark/miles:latest")
    parser.add_argument("--proxy", default=os.environ.get("PROXY_URL", ""))
    parser.add_argument("--docker-bin", default="/usr/bin/docker")
    parser.add_argument("--compose-bin", default="/usr/libexec/docker/cli-plugins/docker-compose")
    parser.add_argument("--buildx-bin", default="/usr/libexec/docker/cli-plugins/docker-buildx")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    artifacts = repo / "seta_miles_runtime/image_build"
    artifacts.mkdir(parents=True, exist_ok=True)
    _build(args, repo, artifacts)


if __name__ == "__main__":
    main()
