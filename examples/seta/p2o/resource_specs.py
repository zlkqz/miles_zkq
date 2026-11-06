"""Temporary GEPA workers expressed as native Miles command-worker specs."""

import shlex
from copy import copy

from miles.backends.sglang_utils.sglang_api_client import WorkerType
from miles.backends.sglang_utils.sglang_config import ModelConfig, resolve_sglang_config
from miles.ray.specs.inference import _compute_spec_inference_engine
from miles.rollout.session.config import compute_session_server_config
from miles.utils.workers.argv_utils import config_to_argv, python_argv_prefix
from miles.utils.workers.worker_spec import CommandWorkerSpec, PortInfo, SchedulingSpec


def training_engine_config(args):
    """Resolve the training engine recipe, including per-pool YAML overrides.

    A single reflection instance needs one unambiguous recipe. Heterogeneous or
    disaggregated recipes are rejected before allocation, never silently flattened.
    """
    models = [model for model in resolve_sglang_config(args).models if model.update_weights]
    if len(models) != 1 or not models[0].server_groups:
        raise ValueError("GEPA requires one managed training policy engine recipe")
    groups = models[0].server_groups
    first = groups[0]
    for group in groups:
        if group.worker_type != WorkerType.REGULAR:
            raise ValueError("GEPA requires regular rollout engines, not prefill/decode or placeholder pools")
        if group.num_gpus_per_engine != first.num_gpus_per_engine or group.overrides != first.overrides:
            raise ValueError("GEPA requires homogeneous training rollout engine settings")
    if first.overrides.get("tp_size", first.num_gpus_per_engine) != first.num_gpus_per_engine:
        raise ValueError("GEPA rollout TP must agree with the native engine GPU allocation")
    return first


def phase_specs(args, placements, *, engine_config, cycle_id, policy_checkpoint, reflection_checkpoint):
    """Use existing engine arguments/launchers, with phase-local immutable overrides."""
    phase_args = copy(args)
    phase_args.offload_rollout = False
    specs, engines = [], []
    for index, placement in enumerate(placements):
        path = policy_checkpoint if placement.role == "policy" else reflection_checkpoint
        name = f"p2o-{cycle_id}-{placement.role}-{index}"
        if placement.tp != engine_config.num_gpus_per_engine:
            raise ValueError("GEPA placement must match the training rollout engine size")
        config = engine_config.model_copy(
            deep=True,
            update=dict(
                num_gpus=placement.tp,
                gpu_offset=placement.slot_offset,
                engine_offset=index,
                # Phase ownership wins only for model identity and memory lifetime:
                # these fresh engines remain resident until GEPA teardown.
                overrides={**engine_config.overrides, "model_path": str(path), "enable_memory_saver": False},
                needs_offload=False,
            ),
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
