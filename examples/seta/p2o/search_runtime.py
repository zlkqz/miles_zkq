"""GEPA episodes use native session v2 and the existing SETA sandbox admission path."""

import asyncio
import hashlib
import json
import logging
import time
from collections import Counter
from pathlib import Path
from statistics import fmean

import httpx

from examples.seta.agent import run
from examples.seta.p2o.checkpoint import atomic_json
from examples.seta.p2o.evidence import EpisodeEvidence, TaskKey
from examples.seta.p2o.history import TemplateEvaluation
from examples.seta.p2o.reflection_files import build_bundle
from examples.seta.p2o.scoring import aggregate_dev
from examples.seta.p2o.templates import Template
from examples.seta.p2o.timing import GepaTimings

logger = logging.getLogger(__name__)


async def gather_episodes(coroutines):
    pending = [asyncio.create_task(coroutine) for coroutine in coroutines]
    try:
        return await asyncio.gather(*pending)
    finally:
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


class SearchRuntime:
    def __init__(self, *, args, panel, endpoints, directory, config, tokenizer, timings=None):
        self.args, self.panel, self.config, self.tokenizer = args, panel, config, tokenizer
        self.directory = Path(directory)
        self.timings = timings if timings is not None else GepaTimings(directory)
        self.policy = [e for e in endpoints if e["role"] == "policy"]
        self.reflection = [e for e in endpoints if e["role"] == "reflection"]
        if not self.policy or not self.reflection:
            raise ValueError("GEPA needs policy and initial-reference session endpoints")
        self.version = self.policy[0]["model_version"]
        if any(e["model_version"] != self.version for e in self.policy):
            raise ValueError("GEPA policy versions differ")
        self._next = {"policy": 0, "reflection": 0}
        self.metrics = Counter()

    def validate_template(self, template):
        template.validate_token_budget(self.tokenizer, max_tokens=self.config.max_template_tokens)

    def evidence(self, episode):
        result = episode["result"]
        common = dict(
            task=TaskKey.model_validate(episode["task"]),
            attempt_id=episode["attempt_id"],
            policy_versions=(self.version,),
            template_id=episode.get("template_id"),
            **{key: result.get(key) for key in ("remote_request_id", "remote_endpoint", "trial_dir")},
        )
        if result.get("seta_valid_reward"):
            try:
                report = result["eval_report"]
                return EpisodeEvidence(
                    **common, valid=True, passed=report["passed"], total=report["total"], reward=result["reward"]
                )
            except (KeyError, ValueError, TypeError) as error:
                return EpisodeEvidence(**common, valid=False, error=f"invalid verifier result: {error}")
        return EpisodeEvidence(
            **common, valid=False, error=result.get("seta_error") or result.get("exit_status") or "invalid episode"
        )

    def mean(self, episodes):
        scores = [self.evidence(e) for e in episodes]
        return fmean(e.reward if e.valid else 0.0 for e in scores)

    async def minibatch(self, text, tasks, seeds, exploration, stage):
        with self.timings.measure("minibatch_" + stage, exploration=exploration):
            return await gather_episodes(
                self._episode(task, seed, text, f"exploration_{exploration}/{stage}_{i}", export=True)
                for i, (task, seed) in enumerate(zip(tasks, seeds, strict=True))
            )

    async def evaluate(self, text, key):
        with self.timings.measure("bootstrap_dev" if key == "bootstrap" else "dev", key=key):
            episodes = await gather_episodes(
                self._episode(plan.task, seed, text, f"{key}/dev_{i}_{j}")
                for i, plan in enumerate(self.panel.dev)
                for j, seed in enumerate(plan.seeds)
            )
            scores = aggregate_dev(
                [p.task for p in self.panel.dev],
                [self.evidence(e) for e in episodes],
                expected_repeats=len(self.panel.dev[0].seeds),
            )
            return TemplateEvaluation(text=text, scores=scores, policy_version=self.version)

    async def _episode(self, task, seed, text, key, export=False):
        metadata = dict(
            instance_id=task.instance_id,
            agent_name="terminus-2",
            max_seq_len=self.args.max_seq_len,
            p2o_export_evidence=export,
        )
        if text:
            metadata["p2o_template"] = Template(text).to_dict()
        identity = dict(
            task=task.model_dump(mode="json"), seed=seed, template=text, policy_version=self.version, export=export
        )
        raw = await self._cached_request(key, "policy", identity, metadata, seed)
        return dict(
            **raw,
            task=task.model_dump(mode="json"),
            task_id=task.instance_id,
            seed=seed,
            attempt_id=f"cycle_{self.panel.cycle_id}/{key}",
            template_id=Template(text).template_id if text else None,
        )

    async def reflect(self, parent, episodes, exploration):
        with self.timings.measure("reflection", exploration=exploration):
            bundle = build_bundle(parent, episodes)
            key = f"exploration_{exploration}/reflection"
            identity = dict(
                bundle_hash=hashlib.sha256(json.dumps(bundle, sort_keys=True).encode()).hexdigest(),
                checkpoint=self.reflection[0]["checkpoint"],
                turns=self.config.reflection_max_turns,
            )
            metadata = dict(
                instance_id="p2o-reflection",
                agent_name="terminus-2",
                max_seq_len=self.args.max_seq_len,
                p2o_reflection=bundle,
                p2o_reflection_max_turns=self.config.reflection_max_turns,
            )
            raw = await self._cached_request(key, "reflection", identity, metadata, exploration)
            return raw["result"].get("p2o_proposal") or ""

    async def _cached_request(self, key, role, identity, metadata, seed):
        path = self.directory / "episodes" / f"{key}.json"
        if path.exists():
            value = json.loads(path.read_text())
            if value["identity"] != identity:
                raise ValueError(f"Episode cache identity differs: {key}")
            self.metrics["episode_cache_hits"] += 1
            return value["output"]
        endpoints = self.policy if role == "policy" else self.reflection
        endpoint = endpoints[self._next[role] % len(endpoints)]
        self._next[role] += 1
        session = endpoint["session_url"].rstrip("/")
        started = time.monotonic()
        self.metrics[f"{role}_requests_started"] += 1
        logger.info("P2O GEPA episode start role=%s key=%s task=%s seed=%s", role, key, metadata["instance_id"], seed)
        sampling = dict(
            temperature=self.args.rollout_temperature, top_p=self.args.rollout_top_p, top_k=self.args.rollout_top_k
        )
        async with httpx.AsyncClient(trust_env=False, timeout=120) as client:
            response = await client.post(session + "/sessions", json=sampling)
            response.raise_for_status()
            session_url = session + "/sessions/" + response.json()["session_id"]
            try:
                request = dict(
                    max_tokens=self.args.rollout_max_response_len,
                    seed=seed,
                    **sampling,
                    extra_body={"chat_template_kwargs": self.args.apply_chat_template_kwargs or {}},
                )
                result = await run(session_url, "", request, metadata)
                response = await client.get(session_url)
                response.raise_for_status()
                output = dict(result=result, records=response.json())
                self.metrics[f"{role}_requests_completed"] += 1
                self.metrics[f"{role}_seconds"] += time.monotonic() - started
                self.metrics[f"{role}_turns"] += len(output["records"].get("records", []))
                if role == "policy" and not result.get("seta_valid_reward"):
                    self.metrics["policy_invalid_episodes"] += 1
                atomic_json(path, dict(identity=identity, output=output))
                logger.info(
                    "P2O GEPA episode done role=%s key=%s seconds=%.2f turns=%d valid=%s reward=%s proposal_chars=%d error=%s",
                    role,
                    key,
                    time.monotonic() - started,
                    len(output["records"].get("records", [])),
                    result.get("seta_valid_reward"),
                    result.get("reward"),
                    len(result.get("p2o_proposal") or ""),
                    result.get("seta_error") or result.get("p2o_reflection_error"),
                )
                return output
            finally:
                response = await client.delete(session_url)
                response.raise_for_status()
