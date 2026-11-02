"""在多个 agent 子进程间共享每个远端的并发槽，不依赖 GPU 侧模块内信号量。"""

import asyncio
import fcntl
import hashlib
import os
import random
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from miles.rollout.agentic.rollout_limits import current_scope

from examples.seta.remote_config import Endpoint


@asynccontextmanager
async def endpoint_slot(endpoints: tuple[Endpoint, ...]):
    root = Path(tempfile.gettempdir()) / f"seta-remote-slots-{current_scope()}"
    root.mkdir(exist_ok=True)
    slots = [(endpoint, index) for endpoint in endpoints for index in range(endpoint.max_concurrent)]
    fd = None
    while fd is None:
        for endpoint, index in random.sample(slots, len(slots)):
            key = hashlib.sha256(endpoint.url.encode()).hexdigest()[:20]
            candidate = os.open(root / f"{key}-{index}.lock", os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(candidate)
                continue
            fd = candidate
            break
        if fd is None:
            await asyncio.sleep(random.uniform(0.1, 0.3))
    try:
        yield endpoint
    finally:
        os.close(fd)
