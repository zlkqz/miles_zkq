"""Native async trainer phase adapter; search never overlaps an optimizer update."""

import hashlib
import json
import logging
from pathlib import Path

from examples.seta.p2o.checkpoint import CheckpointLedger, atomic_json
from examples.seta.p2o.hard_tasks import CyclePanel
from examples.seta.p2o.resource_phase import GepaResourcePhase
from examples.seta.p2o.resource_specs import training_engine_config
from examples.seta.p2o.resources import plan_gepa_placement
from examples.seta.p2o.scoring import CollectionConfig
from examples.seta.p2o.search import Search, SearchConfig
from examples.seta.p2o.search_metrics import SearchMetrics
from examples.seta.p2o.search_runtime import SearchRuntime
from examples.seta.p2o.timing import GepaTimings
from miles.ray.placement_group import update_weights
from miles.utils.processing_utils import load_tokenizer

logger = logging.getLogger(__name__)


def search_config(args):
    return SearchConfig(**{name: getattr(args, "gepa_" + name) for name in SearchConfig.__dataclass_fields__})


def checkpoint_identity(args):
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    checkpoint = Path(args.hf_checkpoint).resolve()
    files = ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json")
    return dict(
        hf_checkpoint=str(checkpoint),
        prompt_data=str(Path(args.prompt_data).resolve()),
        data_sha256=digest(args.prompt_data),
        model_metadata={name: digest(checkpoint / name) for name in files if (checkpoint / name).is_file()},
        template_schema=1,
    )


def prepare_resume(args):
    """Resolve the paired --load root before native components copy their args."""
    if not args.save or not args.fully_async or args.colocate or args.use_critic:
        raise ValueError("P2O requires --save and disaggregated fully-async actor training")
    if args.max_seq_len is None or args.max_seq_len <= 0:
        raise ValueError("P2O requires a finite max-seq-len")
    collection = CollectionConfig(**{name: getattr(args, name) for name in CollectionConfig.model_fields})
    config = search_config(args)
    if args.rollout_batch_size < 2 or collection.gepa_trigger_unique_hard_tasks < args.rollout_batch_size:
        raise ValueError("P2O requires batch>=2 and trigger>=batch so a phase follows a completed update")
    if collection.gepa_trigger_unique_hard_tasks - collection.gepa_dev_task_count < config.minibatch:
        raise ValueError("GEPA trigger must leave enough search-train tasks for the minibatch")
    if args.cluster_backend != "ray" or not args.offload_train or args.rollout_external:
        raise ValueError("P2O requires native Ray inference and --offload-train")
    if not args.use_checkpoint_opt_param_scheduler:
        raise ValueError("P2O requires --use-checkpoint-opt-param-scheduler to resume the saved schedule exactly")
    # Reject unsupported logical tiling before model allocation; runtime planning
    # later checks the actual node-qualified physical slots as well.
    total = args.actor_num_nodes * args.actor_num_gpus_per_node + args.rollout_num_gpus
    slots = [
        dict(
            slot=i,
            bundle_index=i,
            node_id=str(i // args.num_gpus_per_node),
            node_ip="",
            physical_gpu_id=i % args.num_gpus_per_node,
        )
        for i in range(total)
    ]
    engine_config = training_engine_config(args)
    plan_gepa_placement(slots, rollout_gpus_per_engine=engine_config.num_gpus_per_engine)
    # Fail before GPU allocation if local plot export is unavailable.
    from matplotlib.figure import Figure

    _ = Figure
    return select_paired_checkpoint(args)


def select_paired_checkpoint(args):
    """Run before native validation can turn a fresh --load into ref-load/finetune."""
    ledger = CheckpointLedger(args.save, identity=checkpoint_identity(args))
    if latest := ledger.latest():
        args.load = latest["directory"]
        logger.info("P2O resume paired checkpoint: %s", latest)
    elif Path(getattr(args, "requested_load", args.load) or args.save, "latest_checkpointed_iteration.txt").exists():
        raise ValueError("P2O cannot resume an unpaired baseline checkpoint; use a new SAVE directory")
    return ledger


class PhaseHandler:
    def __init__(self, args, *, manager, actor, inference, executor):
        self.args, self.manager, self.actor, self.inference, self.executor = args, manager, actor, inference, executor
        self.ledger = CheckpointLedger(args.save, identity=checkpoint_identity(args))
        self.config = search_config(args)
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)

    async def save(self, step, *, cycle_id=None, phase="periodic"):
        if cycle_id is None:
            cycle_id = (await self.executor.rollout_phase("status"))["cycle_id"]
        # A restored trainer can be asleep before its first update. Saving and HF
        # export require resident tensors, even when the preceding step is reused.
        await self.actor.onload()
        return await self.ledger.save(
            rollout_id=step,
            cycle_id=cycle_id,
            phase=phase,
            save_model=self.actor.save_model,
            save_rollout=self.executor.save,
        )

    async def run(self, step):
        panel = CyclePanel.model_validate(await self.executor.rollout_phase("freeze"))
        directory = Path(self.args.save) / "gepa" / f"cycle_{panel.cycle_id:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "panel.json", panel.model_dump(mode="json"))
        metrics = SearchMetrics(args=self.args, directory=directory, cycle_id=panel.cycle_id, training_step=step)
        timings = GepaTimings(directory, emit=metrics.timing)
        try:
            with timings.measure("phase_total", training_step=step):
                await self._run_phase(step, panel, directory, metrics, timings)
        finally:
            timings.plot()

    async def _run_phase(self, step, panel, directory, metrics, timings):
        with timings.measure("pre_checkpoint"):
            await self.save(step, cycle_id=panel.cycle_id, phase="pre_gepa")

        async def sync():
            await update_weights(self.args, self.actor, self.executor, self.inference, rollout_id=step)

        resources = GepaResourcePhase(
            self.args,
            manager=self.manager,
            actor=self.actor,
            inference=self.inference,
            sync_weights=sync,
            cycle_id=panel.cycle_id,
            policy_checkpoint=directory / "policy_hf",
            reflection_checkpoint=self.args.hf_checkpoint,
        )
        search = None
        try:
            with timings.measure("resource_enter"):
                endpoints = await resources.enter(step)
            atomic_json(directory / "resources.json", endpoints)
            runtime = SearchRuntime(
                args=self.args,
                panel=panel,
                endpoints=endpoints,
                directory=directory,
                config=self.config,
                tokenizer=self.tokenizer,
                timings=timings,
            )
            search = Search(
                panel=panel,
                config=self.config,
                runtime=runtime,
                directory=directory,
                policy_version=runtime.version,
                seed=self.args.rollout_seed + panel.cycle_id,
                emit=metrics,
            )
            with timings.measure("search_total"):
                cycle = await search.run()
            atomic_json(directory / "templates.json", cycle.model_dump(mode="json"))
            await self.executor.rollout_phase("commit", cycle.model_dump(mode="json"))
        finally:
            # No producer admission, post snapshot, or training if restore fails.
            with timings.measure("resource_restore"):
                await resources.restore()
            if search is not None:
                metrics.finish(search.state)
        with timings.measure("post_checkpoint"):
            await self.save(step, cycle_id=panel.cycle_id, phase="post_gepa")
        await self.executor.rollout_phase("resume")
        logger.info("P2O GEPA cycle=%d committed; resumed native training at step=%d", panel.cycle_id, step + 1)
