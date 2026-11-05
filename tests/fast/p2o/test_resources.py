import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from examples.seta.p2o import resource_phase
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


@pytest.mark.parametrize("nodes,reflection,tp,expected", [(1, 1, 1, 8), (2, 1, 1, 16), (2, 2, 2, 15), (2, 4, 2, 14)])
def test_all_owned_devices_are_used_once_with_node_local_tp(nodes, reflection, tp, expected):
    slots = _slots(nodes)
    plan = plan_gepa_placement(slots, reflection_num_gpus=reflection, reflection_tp=tp)
    assert len(plan) == expected
    assert tuple(d for engine in plan for d in engine.devices) == slots
    assert sum(p.tp for p in plan if p.role == "reflection") == reflection
    assert all(len({d.node_id for d in engine.devices}) == 1 for engine in plan)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(reflection_num_gpus=0),
        dict(reflection_num_gpus=8),
        dict(reflection_num_gpus=3, reflection_tp=2),
        dict(policy_tp=2),
        dict(policy_tp=True),
    ],
)
def test_invalid_layouts_are_not_silently_shrunk(kwargs):
    with pytest.raises(ValueError):
        plan_gepa_placement(_slots(), **kwargs)


def test_distinct_node_gpu_zero_is_valid_but_duplicate_physical_device_is_not():
    slots = _slots(2)
    assert len(plan_gepa_placement(slots)) == 16
    bad = list(slots)
    bad[1] = replace(bad[1], physical_gpu_id=0)
    with pytest.raises(ValueError, match="Duplicate"):
        plan_gepa_placement(bad)


def test_noncontiguous_physical_devices_refuse_tp_but_allow_single_gpu_engines():
    slots = tuple(replace(d, physical_gpu_id=d.physical_gpu_id * 2) for d in _slots())
    assert len(plan_gepa_placement(slots)) == 8
    with pytest.raises(ValueError, match="contiguous"):
        plan_gepa_placement(slots, reflection_num_gpus=2, reflection_tp=2)


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
    args = SimpleNamespace(
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
        plan_gepa_placement(_slots(2)),
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
