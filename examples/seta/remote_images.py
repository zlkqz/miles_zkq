"""只管理本服务创建的镜像；预取基础镜像不需要重启共享 Docker daemon。"""

import asyncio
import fcntl
import json
import os
import re
import time
from pathlib import Path
from contextlib import asynccontextmanager


@asynccontextmanager
async def _cache_lock(root):
    fd = os.open(root / "image-cache.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(0.2)
        yield
    finally:
        os.close(fd)


async def command(*argv):
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), 900)
    except BaseException:
        if process.returncode is None:
            process.terminate()
            await process.wait()
        raise
    if process.returncode:
        raise RuntimeError(f"命令失败 {argv[:3]}：{stdout.decode(errors='replace')[-2000:]}")
    return stdout.decode()


def base_images(task: Path) -> tuple[str, ...]:
    dockerfile = task / "environment/Dockerfile"
    if not dockerfile.exists():
        return ()
    stages = set()
    images = []
    # SETA 的 Dockerfile 使用大写 FROM；只匹配指令，避开内嵌 Python 的 from import。
    for line in dockerfile.read_text().splitlines():
        match = re.match(r"^FROM\s+(\S+)(?:\s+AS\s+(\S+))?", line.strip())
        if match:
            image, stage = match.groups()
            if image not in stages and image != "scratch":
                images.append(image)
            if stage:
                stages.add(stage)
    return tuple(dict.fromkeys(images))


def image_sources(image, mirrors):
    """仅映射 Docker Hub 镜像；其他 registry 保持原地址。"""
    parts = image.split("/")
    if len(parts) == 1:
        repository = "library/" + image
    elif parts[0] in {"docker.io", "index.docker.io", "registry-1.docker.io"}:
        repository = "/".join(parts[1:])
        if "/" not in repository:
            repository = "library/" + repository
    elif "." in parts[0] or ":" in parts[0] or parts[0] == "localhost":
        return (image,)
    else:
        repository = image
    return tuple(dict.fromkeys([*(f"{mirror.rstrip('/')}/{repository}" for mirror in mirrors), image]))


class ImageCache:
    def __init__(self, config, state_dir):
        self.crane = config.get("crane", "")
        self.mirrors = tuple(config.get("registry_mirrors", []))
        self.keep = int(config.get("image_cache_keep", 16))
        if self.keep < 0:
            raise ValueError("image_cache_keep 不能为负数")
        self.prefix = f"seta-{config.get('service_id', 'remote')}-"
        self.image_prefix = f"seta-cache-{config.get('service_id', 'remote')}-"
        self.service_id = config.get("service_id", "remote")
        self.root = state_dir
        self.lock = asyncio.Lock()
        self.owned_file = state_dir / "owned_images.json"
        self.owned = {}

    async def prepare(self, task):
        if not self.crane:
            return
        async with self.lock, _cache_lock(self.root):
            for image in base_images(task):
                try:
                    await command("docker", "image", "inspect", image)
                    continue
                except RuntimeError:
                    pass
                archive = self.root / "base-image.tar"
                try:
                    await self.pull(image, archive)
                finally:
                    archive.unlink(missing_ok=True)

    async def pull(self, image, archive):
        errors = []
        for source in image_sources(image, self.mirrors):
            try:
                await command(self.crane, "pull", "--platform", "linux/amd64", source, str(archive))
                await command("docker", "load", "-i", str(archive))
                if source != image:
                    await command("docker", "tag", source, image)
                return
            except RuntimeError as error:
                errors.append(str(error))
                archive.unlink(missing_ok=True)
        raise RuntimeError("基础镜像的所有下载地址均失败：" + "\n".join(errors))

    async def completed(self, image):
        if not image or not image.startswith(self.image_prefix):
            return
        async with self.lock, _cache_lock(self.root):
            self.owned = json.loads(self.owned_file.read_text()) if self.owned_file.exists() else {}
            try:
                info = json.loads(await command("docker", "image", "inspect", image))[0]
            except RuntimeError:
                return
            labels = info["Config"].get("Labels") or {}
            if labels.get("org.miles.seta.service") != self.service_id:
                return
            self.owned[image] = time.time()
            for candidate in sorted(self.owned, key=self.owned.get)[: -self.keep or None]:
                if len(self.owned) <= self.keep:
                    break
                try:
                    # 不使用 --force；仍被任何容器引用的镜像保留。
                    await command("docker", "image", "rm", candidate)
                except RuntimeError:
                    continue
                self.owned.pop(candidate, None)
            temporary = self.owned_file.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.owned, indent=2))
            temporary.replace(self.owned_file)
