import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from examples.seta.p2o import resource_phase
from examples.seta.p2o.resource_specs import training_engine_config
from examples.seta.p2o.resources import GpuSlot, plan_gepa_placement

from miles.utils.workers.naming import compute_cell_id
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.worker_spec import CommandWorkerSpec, SchedulingSpec


def _slots(nodes=1, per_node=8):
    return tuple(
        GpuSlot(n * per_node + i, n * per_node + i, f"node{n}", f"10.0.0.{n+1}", i)
        for n in range(nodes)
        for i in range(per_node)
    )


@pytest.mark.parametrize("nodes,tp,expected", [(1, 1, 8), (1, 2, 4), (2, 2, 8), (2, 4, 4), (2, 8, 2)])
def test_all_owned_devices_are_used_once_with_node_local_tp(nodes, tp, expected):
    slots = _slots(nodes)
    plan = plan_gepa_placement(slots, rollout_gpus_per_engine=tp)
    assert len(plan) == expected
    assert tuple(d for engine in plan for d in engine.devices) == slots
    assert sum(p.role == "reflection" for p in plan) == 1
    assert all(p.tp == tp for p in plan)
    assert all(len({d.node_id for d in engine.devices}) == 1 for engine in plan)


@pytest.mark.parametrize("tp", [0, 8, 3, True, 16])
def test_invalid_layouts_are_not_silently_shrunk(tp):
    with pytest.raises(ValueError):
        plan_gepa_placement(_slots(), rollout_gpus_per_engine=tp)


def test_distinct_node_gpu_zero_is_valid_but_duplicate_physical_device_is_not():
    slots = _slots(2)
    assert len(plan_gepa_placement(slots, rollout_gpus_per_engine=1)) == 16
    bad = list(slots)
    bad[1] = replace(bad[1], physical_gpu_id=0)
    with pytest.raises(ValueError, match="Duplicate"):
        plan_gepa_placement(bad, rollout_gpus_per_engine=1)


def test_noncontiguous_physical_devices_refuse_tp_but_allow_single_gpu_engines():
    slots = tuple(replace(d, physical_gpu_id=d.physical_gpu_id * 2) for d in _slots())
    assert len(plan_gepa_placement(slots, rollout_gpus_per_engine=1)) == 8
    with pytest.raises(ValueError, match="contiguous"):
        plan_gepa_placement(slots, rollout_gpus_per_engine=2)


@pytest.fixture
def phase(monkeypatch, tmp_path):
    events = []

    def method(name, result=None):
        async def call(*args, **kwargs):
            events.append(name)
            return result

        return AsyncMock(side_effect=call)

    manager = SimpleNamespace(
        **{
            name: SimpleNamespace(remote=method(name, result))
            for name, result in {
                "gpu_slots": _slots(),
                "get_cell_infos": {"normal-0": SimpleNamespace(alive=True)},
                "stop_cells": None,
                "register_pools": ["temp-0"],
                "start_cells": None,
                "get_addrs": {},
                "remove_pools": None,
            }.items()
        }
    )
    actor = SimpleNamespace(**{name: method(name) for name in ("export_hf", "offload", "onload")})
    inference = SimpleNamespace(**{name: method(name) for name in ("dispose", "resume_after_phase", "onload_kv")})
    monkeypatch.setattr(resource_phase, "compute_engine_pool_ids", lambda _: ["normal"])
    monkeypatch.setattr(
        resource_phase, "phase_specs", lambda *a, **k: ([SimpleNamespace(name="temp")], [{"role": "policy"}])
    )
    from tests.fast.ray.rollout.conftest import make_args

    args = make_args(
        cluster_backend="ray",
        offload_train=True,
        starts_inference_engines=True,
        use_critic=False,
        rollout_external=False,
        colocate=False,
        offload_rollout=False,
    )
    obj = resource_phase.GepaResourcePhase(
        args,
        manager=manager,
        actor=actor,
        inference=inference,
        sync_weights=method("sync_weights"),
        cycle_id=1,
        policy_checkpoint=tmp_path / "policy",
        reflection_checkpoint=tmp_path / "initial",
    )
    obj._ready = method("ready", {"role": "policy"})
    return obj, events


async def test_phase_order_restores_original_services_before_training(phase):
    obj, events = phase
    assert await obj.enter(0) == [{"role": "policy"}]
    assert obj.phase == "gepa"
    await obj.restore()
    assert obj.phase == "training"
    assert events == [
        "gpu_slots",
        "get_cell_infos",
        "export_hf",
        "offload",
        "dispose",
        "stop_cells",
        "register_pools",
        "start_cells",
        "get_addrs",
        "ready",
        "remove_pools",
        "start_cells",
        "resume_after_phase",
        "onload",
        "sync_weights",
    ]
    await obj.restore()
    assert events.count("sync_weights") == 1


async def test_colocated_restore_keeps_trainer_asleep_until_training(phase):
    obj, events = phase
    obj.args.colocate = obj.args.offload_rollout = True
    await obj.enter(0)
    await obj.restore()
    assert "onload" not in events
    assert events[-2:] == ["sync_weights", "onload_kv"]


async def test_failed_engine_start_restores_then_raises(phase):
    obj, events = phase
    obj._ready = AsyncMock(side_effect=ValueError("engine failed"))
    with pytest.raises(ValueError, match="engine failed"):
        await obj.enter(0)
    assert obj.phase == "training"
    assert events[-5:] == ["remove_pools", "start_cells", "resume_after_phase", "onload", "sync_weights"]


async def test_failed_restore_is_not_reported_as_ready(phase):
    obj, _ = phase
    await obj.enter(0)
    obj.inference.resume_after_phase = AsyncMock(side_effect=RuntimeError("restore failed"))
    with pytest.raises(RuntimeError, match="restore failed"):
        await obj.restore()
    assert obj.phase == "failed_restore"
    obj.sync_weights.assert_not_awaited()


def _spec(name, offset=0):
    return CommandWorkerSpec(
        name=name,
        port_infos=[],
        env_var=lambda _: {},
        launch_command=lambda _: "true",
        scheduling=SchedulingSpec(
            num_cells=1,
            num_workers_per_cell=1,
            num_gpus_per_worker=0.2,
            num_gpu_slots_per_worker=1,
            pg_name="actor",
            pg_slot_offset=offset,
        ),
    )


async def test_temporary_pool_registration_is_atomic_and_cannot_remove_normal_pools():
    manager = RayWorkerManager()
    manager._pools = {"normal": object()}
    manager._temporary_pool_ids = set()
    manager._membership_lock = asyncio.Lock()
    manager.pgs = {"actor": SimpleNamespace(pg_reordered_gpu_ids=list(range(8)))}
    with pytest.raises(ValueError, match="new and unique"):
        await manager.register_pools([_spec("normal")])
    with pytest.raises(ValueError, match="exceeds"):
        await manager.register_pools([_spec("ok"), _spec("invalid", 8)])
    assert set(manager._pools) == {"normal"}
    assert await manager.register_pools([_spec("temp", 7)]) == [compute_cell_id(pool_id="temp", cell_index=0)]
    with pytest.raises(ValueError, match="temporary"):
        await manager.remove_pools(["normal"])
    await manager.remove_pools(["temp"])
    assert set(manager._pools) == {"normal"}


def test_phase_specs_preserve_launch_args_and_use_owned_slots():
    from examples.seta.p2o.resource_specs import phase_specs
    from tests.fast.ray.rollout.conftest import make_args

    args = make_args(offload_rollout=True)
    before = vars(args).copy()
    specs, engines = phase_specs(
        args,
        plan_gepa_placement(_slots(2), rollout_gpus_per_engine=1),
        engine_config=training_engine_config(args),
        cycle_id=3,
        policy_checkpoint="/policy",
        reflection_checkpoint="/initial",
    )
    assert vars(args) == before
    assert len(specs) == 32
    assert len({s.name for s in specs}) == 32
    for i, engine in enumerate(engines):
        spec = specs[2 * i]
        assert spec.scheduling.pg_name == "actor"
        assert spec.scheduling.pg_slot_offset == i
        assert spec.scheduling.num_gpu_slots_per_worker == 1
        assert engine["checkpoint"] == ("/initial" if i == 15 else "/policy")
        assert specs[2 * i + 1].scheduling.num_gpus_per_worker == 0


async def test_engine_readiness_rejects_wrong_model_before_publishing_version(phase, monkeypatch):
    from miles.utils.workers.worker_spec import HostAndPort

    obj, _ = phase
    obj.args.sglang_api_key = None
    monkeypatch.setattr(resource_phase, "activate_launch_gate", AsyncMock())
    monkeypatch.setattr(resource_phase, "wait_server_healthy", AsyncMock())
    client = SimpleNamespace(
        get_server_info=AsyncMock(return_value={"model_path": "/wrong"}), update_weight_version=AsyncMock()
    )
    monkeypatch.setattr(resource_phase, "SGLangApiClient", lambda *a, **k: client)
    addr = HostAndPort(host="127.0.0.1", port=1234)
    with pytest.raises(RuntimeError, match="checkpoint acknowledgement"):
        await resource_phase.GepaResourcePhase._ready(
            obj,
            dict(pool_id="engine", session_pool_id="session", checkpoint="/initial"),
            {"engine": [dict(gate=addr, primary=addr)], "session": [dict(primary=addr)]},
        )
    client.update_weight_version.assert_not_awaited()


def test_gpu_slots_use_bundle_node_mapping_not_global_device_ids(monkeypatch):
    from miles.utils.workers import ray_worker_manager as module

    manager = RayWorkerManager()
    manager.pgs = {
        "actor": SimpleNamespace(
            pg=object(), pg_reordered_gpu_ids=[0, 1, 0, 1], pg_reordered_bundle_indices=[2, 3, 0, 1]
        )
    }
    monkeypatch.setattr(
        module.ray.util, "placement_group_table", lambda _: {"bundles_to_node_id": {0: "b", 1: "b", 2: "a", 3: "a"}}
    )
    monkeypatch.setattr(
        module.ray,
        "nodes",
        lambda: [dict(NodeID="a", NodeManagerAddress="10.0.0.1"), dict(NodeID="b", NodeManagerAddress="10.0.0.2")],
    )
    slots = manager.gpu_slots("actor")
    assert [s["node_id"] for s in slots] == ["a", "a", "b", "b"]
    assert [s["physical_gpu_id"] for s in slots] == [0, 1, 0, 1]


async def test_export_failure_does_not_cycle_engines_or_publish_new_weights(phase):
    obj, events = phase
    obj.actor.export_hf = AsyncMock(side_effect=OSError("checkpoint filesystem failed"))
    with pytest.raises(OSError, match="filesystem failed"):
        await obj.enter(0)
    assert obj.phase == "training"
    assert events == ["gpu_slots", "get_cell_infos"]


async def test_partial_offload_failure_attempts_onload_without_restarting_engines(phase):
    obj, events = phase
    obj.actor.offload = AsyncMock(side_effect=RuntimeError("rank offload failed"))
    with pytest.raises(RuntimeError, match="rank offload failed"):
        await obj.enter(0)
    assert obj.phase == "training"
    assert events == ["gpu_slots", "get_cell_infos", "export_hf", "onload"]


def test_gepa_inherits_yaml_recipe_and_cli_engine_settings(monkeypatch):
    from examples.seta.p2o import resource_specs
    from tests.fast.ray.rollout.conftest import make_args

    from miles.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, SglangConfig

    recipe = ServerGroupConfig(
        worker_type="regular",
        num_gpus=4,
        num_gpus_per_engine=2,
        gpu_offset=0,
        engine_offset=0,
        overrides={
            "model_path": "/initial",
            "mem_fraction_static": 0.57,
            "attention_backend": "triton",
            "disable_cuda_graph": True,
        },
        needs_offload=False,
    )
    config = SglangConfig(
        models=[ModelConfig(name="actor", model_path="/initial", server_groups=[recipe], update_weights=True)]
    )
    monkeypatch.setattr(resource_specs, "resolve_sglang_config", lambda _: config)
    args = make_args(
        rollout_num_gpus_per_engine=1, sglang_context_length=8192, sglang_dp_size=1, sglang_ep_size=1, sglang_pp_size=1
    )
    inherited = training_engine_config(args)
    assert inherited.num_gpus_per_engine == 2  # YAML wins over the global fallback.
    captured = []
    original = resource_specs._compute_spec_inference_engine

    def capture(phase_args, index, group_index, model, group):
        captured.append((phase_args, group))
        return original(phase_args, index, group_index, model, group)

    monkeypatch.setattr(resource_specs, "_compute_spec_inference_engine", capture)
    _, engines = resource_specs.phase_specs(
        args,
        plan_gepa_placement(_slots(2), rollout_gpus_per_engine=2),
        engine_config=inherited,
        cycle_id=1,
        policy_checkpoint="/current",
        reflection_checkpoint="/initial",
    )
    assert len(engines) == 8 and sum(e["role"] == "reflection" for e in engines) == 1
    for phase_args, group in captured:
        assert phase_args.sglang_context_length == 8192
        assert group.num_gpus_per_engine == 2
        assert group.overrides["mem_fraction_static"] == 0.57
        assert group.overrides["attention_backend"] == "triton"
        assert group.overrides["disable_cuda_graph"] is True
        assert group.overrides["enable_memory_saver"] is False
    assert captured[-1][1].model_path == "/initial"
    assert all(group.model_path == "/current" for _, group in captured[:-1])
    assert "enable_memory_saver" not in recipe.overrides


@pytest.mark.parametrize(
    "change", [{"num_gpus_per_engine": 2}, {"overrides": {"model_path": "/x"}}, {"worker_type": "prefill"}]
)
def test_ambiguous_engine_recipes_are_rejected_before_start(monkeypatch, change):
    from examples.seta.p2o import resource_specs

    from miles.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, SglangConfig

    first = ServerGroupConfig(
        worker_type="regular",
        num_gpus=2,
        num_gpus_per_engine=1,
        gpu_offset=0,
        engine_offset=0,
        overrides={"model_path": "/initial"},
        needs_offload=False,
    )
    second = first.model_copy(update=change)
    config = SglangConfig(
        models=[ModelConfig(name="actor", model_path="/initial", server_groups=[first, second], update_weights=True)]
    )
    monkeypatch.setattr(resource_specs, "resolve_sglang_config", lambda _: config)
    with pytest.raises(ValueError, match="homogeneous|regular"):
        training_engine_config(None)
