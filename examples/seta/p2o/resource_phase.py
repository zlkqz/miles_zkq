"""Drained GEPA inference phase with original Miles placements preserved.

The caller owns producer draining and pre/post checkpoints. This object owns
only resource lifetime; exceptions always propagate, including failed restore.
"""

import asyncio
import logging
import time
from dataclasses import asdict
from pathlib import Path

import httpx

from examples.seta.p2o.resource_specs import phase_specs, training_engine_config
from examples.seta.p2o.resources import plan_gepa_placement
from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient, wait_server_healthy
from miles.ray.specs.inference import compute_engine_pool_ids
from miles.utils.workers.launch_gate import activate_launch_gate

logger = logging.getLogger(__name__)


class GepaResourcePhase:
    def __init__(
        self,
        args,
        *,
        manager,
        actor,
        inference,
        sync_weights,
        cycle_id,
        policy_checkpoint,
        reflection_checkpoint,
    ):
        if args.cluster_backend != "ray" or not args.offload_train or not args.starts_inference_engines:
            raise ValueError("GEPA phase requires native Ray engines and --offload-train")
        if args.use_critic or args.rollout_external:
            raise ValueError("GEPA phase currently supports a single actor and managed inference engines")
        self.args, self.manager, self.actor, self.inference = args, manager, actor, inference
        self.sync_weights = sync_weights
        self.cycle_id = cycle_id
        self.policy_checkpoint = str(policy_checkpoint)
        self.reflection_checkpoint = str(reflection_checkpoint)
        self.engine_config = training_engine_config(args)
        self.pool_ids = []
        self.normal_cells = []
        self.endpoints = []
        self.phase = "training"
        self._actor_offloaded = False
        self._serving_disposed = False
        self._normal_stopped = False

    async def enter(self, rollout_id):
        if self.phase != "training":
            raise RuntimeError(f"Cannot enter GEPA from {self.phase}")
        slots = await self.manager.gpu_slots.remote("actor")
        placements = plan_gepa_placement(slots, rollout_gpus_per_engine=self.engine_config.num_gpus_per_engine)
        specs, engines = phase_specs(
            self.args,
            placements,
            engine_config=self.engine_config,
            cycle_id=self.cycle_id,
            policy_checkpoint=self.policy_checkpoint,
            reflection_checkpoint=self.reflection_checkpoint,
        )
        infos = await self.manager.get_cell_infos.remote(pool_ids=compute_engine_pool_ids(self.args))
        self.normal_cells = [cell_id for cell_id, info in infos.items() if info.alive]
        if len(self.normal_cells) != len(infos):
            raise RuntimeError("Cannot start GEPA with missing normal inference cells")
        started = time.monotonic()
        self.phase = "entering"
        logger.info("P2O GEPA resource enter cycle=%s layout=%s", self.cycle_id, [asdict(p) for p in placements])
        try:
            # Export collectively while the old trainer still has live process groups.
            Path(self.policy_checkpoint).parent.mkdir(parents=True, exist_ok=True)
            await self.actor.export_hf(rollout_id, self.policy_checkpoint)
            # onload() is idempotent per rank, including after partial offload.
            self._actor_offloaded = True
            await self.actor.offload()
            await self.inference.dispose()
            self._serving_disposed = True
            self._normal_stopped = True  # Also restore after a partially failed stop.
            await self.manager.stop_cells.remote(self.normal_cells)
            cells = await self.manager.register_pools.remote(specs)
            self.pool_ids = [spec.name for spec in specs]
            await self.manager.start_cells.remote(cells)
            addresses = await self.manager.get_addrs.remote()
            pending = [asyncio.create_task(self._ready(engine, addresses)) for engine in engines]
            try:
                self.endpoints = list(await asyncio.gather(*pending))
            finally:
                for task in pending:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
            self.phase = "gepa"
            logger.info(
                "P2O GEPA resources ready cycle=%s seconds=%.2f engines=%d",
                self.cycle_id,
                time.monotonic() - started,
                len(engines),
            )
            return self.endpoints
        except BaseException:
            logger.exception("P2O GEPA entry failed cycle=%s; restoring original resources", self.cycle_id)
            await self.restore()
            raise

    async def _ready(self, engine, addresses):
        (backend,) = addresses[engine["pool_id"]]
        (session,) = addresses[engine["session_pool_id"]]
        await activate_launch_gate(backend["gate"].addr)
        await asyncio.wait_for(wait_server_healthy(backend["primary"].addr, self.args.sglang_api_key), timeout=600)
        client = SGLangApiClient(backend["primary"].addr, api_key=self.args.sglang_api_key)
        info = await client.get_server_info()
        model_path = info.get("model_path") or info.get("server_args", {}).get("model_path")
        if model_path is None or Path(model_path).resolve() != Path(engine["checkpoint"]).resolve():
            raise RuntimeError(f"GEPA checkpoint acknowledgement mismatch: {model_path!r}")
        version = f"p2o-{self.cycle_id}-{engine['role']}"
        await client.update_weight_version(version)
        if str(await client.get_weight_version()) != version:
            raise RuntimeError("GEPA model version acknowledgement mismatch")
        async with httpx.AsyncClient(trust_env=False, timeout=5) as http:
            deadline = time.monotonic() + 120
            while True:
                try:
                    response = await http.get(session["primary"].addr + "/health")
                    response.raise_for_status()
                    break
                except httpx.HTTPError:
                    if time.monotonic() >= deadline:
                        raise
                    await asyncio.sleep(1)
        result = {
            "role": engine["role"],
            "model_version": version,
            "checkpoint": engine["checkpoint"],
            "engine_url": backend["primary"].addr,
            "session_url": f"http://{session['primary'].external_netloc}",
            "placement": asdict(engine["placement"]),
        }
        logger.info("P2O GEPA engine ready %s", result)
        return result

    async def restore(self):
        if self.phase == "training":
            return
        self.phase = "restoring"
        needs_weight_sync = self._normal_stopped or self._serving_disposed
        try:
            if self.pool_ids:
                await self.manager.remove_pools.remote(self.pool_ids)
                self.pool_ids = []
            if self._normal_stopped:
                await self.manager.start_cells.remote(self.normal_cells)
                self._normal_stopped = False
            if self._serving_disposed:
                await self.inference.resume_after_phase()
                self._serving_disposed = False
            if self._actor_offloaded:
                # Colocated rollout resumes with the trainer asleep, exactly as
                # in Miles' normal rollout phase. Its next train() wakes it.
                if not self.args.colocate:
                    await self.actor.onload()
                self._actor_offloaded = False
            if needs_weight_sync:
                await self.sync_weights()
            if needs_weight_sync and self.args.offload_rollout:
                await self.inference.onload_kv()
            self.phase = "training"
            self.endpoints = []
            logger.info("P2O original training layout restored cycle=%s", self.cycle_id)
        except BaseException:
            self.phase = "failed_restore"
            logger.exception("P2O restore failed; training must not continue cycle=%s", self.cycle_id)
            raise
