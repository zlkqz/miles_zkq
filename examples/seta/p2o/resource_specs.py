"""Temporary GEPA workers expressed as native Miles command-worker specs."""

import shlex
from copy import copy

from miles.backends.sglang_utils.sglang_api_client import WorkerType
from miles.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig
from miles.ray.specs.inference import _compute_spec_inference_engine
from miles.rollout.session.config import compute_session_server_config
from miles.utils.workers.argv_utils import config_to_argv, python_argv_prefix
from miles.utils.workers.worker_spec import CommandWorkerSpec, PortInfo, SchedulingSpec


def phase_specs(args, placements, *, cycle_id, policy_checkpoint, reflection_checkpoint):
    """Use existing engine arguments/launchers, with phase-local immutable overrides."""
    phase_args = copy(args)
    phase_args.offload_rollout = False
    specs, engines = [], []
    for index, placement in enumerate(placements):
        path = policy_checkpoint if placement.role == "policy" else reflection_checkpoint
        name = f"p2o-{cycle_id}-{placement.role}-{index}"
        config = ServerGroupConfig(
            worker_type=WorkerType.REGULAR,
            num_gpus=placement.tp,
            num_gpus_per_engine=placement.tp,
            gpu_offset=placement.slot_offset,
            engine_offset=index,
            overrides={"model_path": str(path), "enable_memory_saver": False},
            needs_offload=False,
        )
        model = ModelConfig(name=placement.role, model_path=str(path), server_groups=[config], update_weights=False)
        base = _compute_spec_inference_engine(phase_args, index, 0, model, config)
        engine = base.model_copy(
            update={
                "name": name,
                "scheduling": base.scheduling.model_copy(update={"pg_name": "actor"}),
            }
        )
        session = _session_spec(phase_args, engine_pool=name, instance_id=f"{args.run_uuid}-{name}")
        specs.extend((engine, session))
        engines.append(
            {
                "pool_id": name,
                "session_pool_id": session.name,
                "role": placement.role,
                "checkpoint": str(path),
                "placement": placement,
            }
        )
    return specs, engines


def _session_spec(args, *, engine_pool, instance_id):
    interpreter = python_argv_prefix()

    def launch(ctx):
        (engine,) = ctx.pool_addrs[engine_pool]
        config = compute_session_server_config(
            args,
            host="0.0.0.0",
            port=ctx.self_addrs["primary"].port,
            instance_id=instance_id,
            backend_url=engine["primary"].addr,
        )
        return shlex.join([*interpreter, "-m", "miles.rollout.session.server", *config_to_argv(config)])

    return CommandWorkerSpec(
        name=f"{engine_pool}-session",
        port_infos=[PortInfo(name="primary", static_port=8000, allow_dynamic=True)],
        env_var=lambda _: {},
        scheduling=SchedulingSpec.single(num_gpus_per_worker=0, pin_to_head=True),
        launch_command=launch,
    )
