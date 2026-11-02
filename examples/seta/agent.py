"""SETA 的薄客户端：复用官方 Harbor /run，增加多节点限流与奖励有效性检查。"""

import importlib
import logging
import os
import uuid

import httpx

from miles.rollout.agentic.rollout_limits import rollout_semaphore
from examples.seta.remote_config import EPISODE_ENV, load_endpoints
from examples.seta.remote_pool import endpoint_slot

_official = importlib.import_module("examples.swe-agent-harbor-docker.swe_agent_function")
logger = logging.getLogger(__name__)


async def run(base_url, prompt, request_kwargs, metadata, **kwargs):
    async with rollout_semaphore("seta-harbor", int(os.environ["SETA_MAX_CONCURRENT_TRIALS"])):
        async with endpoint_slot(load_endpoints(os.environ["SETA_REMOTE_CONFIG"])) as endpoint:
            request_id = uuid.uuid4().hex
            meta = {**metadata, "run_id": os.environ["SETA_RUN_ID"], "request_id": request_id, "seta_settings": {key: os.environ[key] for key in EPISODE_ENV if key in os.environ}}
            result = await _official.run(
                base_url,
                prompt,
                request_kwargs,
                meta,
                agent_server_url=endpoint.url,
                headers=endpoint.headers(),
                trial_timeout_s=float(os.environ["AGENT_TRIAL_TIMEOUT"]) + 180,
            )
            if result is None:
                result = {"seta_valid_reward": False, "seta_error": "官方 agent-server 请求失败或已取消"}
            logger.info("远端 episode 完成：%s / %s / reward=%s", endpoint.url, request_id, result.get("reward"))
            return {**result, "remote_endpoint": endpoint.url, "remote_request_id": request_id}


async def abort(args):
    """Miles 原生 abort hook：只 flush 当前训练的任务，不影响同服务的其他运行。"""
    async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
        for endpoint in load_endpoints(os.environ["SETA_REMOTE_CONFIG"]):
            response = await client.post(f"{endpoint.url}/flush", json={"run_id": os.environ["SETA_RUN_ID"]}, headers=endpoint.headers())
            response.raise_for_status()


async def reward_func(args, samples, **kwargs):
    if isinstance(samples, list):
        return [await reward_func(args, sample, **kwargs) for sample in samples]
    if not samples.metadata.get("seta_valid_reward", False):
        samples.status = samples.Status.ABORTED
        logger.error("丢弃无效验证轨迹：%s", samples.metadata.get("seta_error"))
        return 0.0
    return float(samples.metadata["reward"])
