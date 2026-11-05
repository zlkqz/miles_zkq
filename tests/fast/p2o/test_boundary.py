import asyncio
import json
from pathlib import Path

import pytest
from examples.seta.p2o.checkpoint import CheckpointLedger
from examples.seta.p2o.history import TemplateCycle
from tests.fast.p2o.test_rollout import make_fn

from miles.rollout.base_types import RolloutFnTrainInput, RolloutPhaseBoundary
from miles.rollout.fully_async_data_buffer import DataBufferInput
from miles.utils.types import Sample


def complete(batch):
    for sample in batch:
        sample.status = Sample.Status.COMPLETED
        sample.tokens = [1, 2]
        sample.response_length = 1
        sample.loss_mask = [1]
        sample.rollout_log_probs = [-0.2]
        sample.reward = 0.0
        sample.metadata.update(seta_valid_reward=True, reward=0.0, eval_report={"passed": 0, "total": 4})
    return DataBufferInput(prompt_group=batch, group=batch)


async def test_trigger_drains_slow_inflight_full_batches_then_signals_tail_without_waiting(monkeypatch, tmp_path):
    fn = make_fn(
        monkeypatch,
        rollout_batch_size=4,
        async_max_concurrent_samples=10,
        gepa_trigger_unique_hard_tasks=3,
        gepa_dev_task_count=1,
        async_data_buffer_capacity_factor=1,
    )
    release = asyncio.Event()
    submitted = []

    async def generate(batch):
        submitted.append(batch[0].group_index)
        if len(submitted) == 5:
            await release.wait()
        return complete(batch)

    fn._generate_group = generate
    first = await asyncio.wait_for(fn(RolloutFnTrainInput(rollout_id=0, weight_version=0)), 2)
    assert len(first.samples) == 4
    waiting = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=1, weight_version=0)))
    await asyncio.sleep(0.01)
    assert not waiting.done() and len(submitted) == 5
    release.set()
    with pytest.raises(RolloutPhaseBoundary):
        await asyncio.wait_for(waiting, 2)
    assert not fn._worker_error_cancel_targets
    assert len(fn._output.state_dict()) == 1
    panel = await fn.phase("freeze", {})
    assert len(panel["hard_tasks"]) == 5  # include the draining overshoot
    assert await fn.phase("freeze", {}) == panel
    fn.data_source.save(tmp_path / "data")
    fn.save(tmp_path / "fn")
    recovered = make_fn(
        monkeypatch,
        rollout_batch_size=4,
        async_max_concurrent_samples=10,
        gepa_trigger_unique_hard_tasks=3,
        gepa_dev_task_count=1,
        async_data_buffer_capacity_factor=1,
    )
    recovered.data_source.load(tmp_path / "data")
    recovered.load(tmp_path / "fn")
    try:
        with pytest.raises(RolloutPhaseBoundary):
            await asyncio.wait_for(recovered(RolloutFnTrainInput(rollout_id=1, weight_version=0)), 2)
        assert await recovered.phase("freeze", {}) == panel
        assert len(recovered._output.state_dict()) == 1
        with pytest.raises(RuntimeError, match="committed"):
            await recovered.phase("resume", {})
        committed = TemplateCycle(cycle_id=1, evaluations=()).model_dump(mode="json")
        assert await recovered.phase("commit", committed) == {"committed": True}
        assert not recovered.windows.ready
        assert await recovered.phase("commit", committed) == {"committed": True}
        recovered._generate_group = lambda batch: asyncio.sleep(0, result=complete(batch))
        await recovered.phase("resume", {})
        output = await asyncio.wait_for(recovered(RolloutFnTrainInput(rollout_id=1, weight_version=0)), 2)
        assert len(output.samples) == 4
        assert any(s.metadata["p2o"]["cycle_id"] == 1 for batch in output.samples for s in batch)
    finally:
        await fn.dispose()
        await recovered.dispose()


async def test_draining_can_exceed_normal_buffer_capacity_but_does_not_submit_more(monkeypatch):
    fn = make_fn(
        monkeypatch,
        rollout_batch_size=2,
        async_max_concurrent_samples=12,
        gepa_trigger_unique_hard_tasks=3,
        gepa_dev_task_count=1,
        async_data_buffer_capacity_factor=1,
    )
    fn._generate_group = lambda batch: asyncio.sleep(0, result=complete(batch))
    try:
        await fn(RolloutFnTrainInput(rollout_id=0, weight_version=0))
        for _ in range(100):
            if (await fn.phase("status", {}))["ready"]:
                break
            await asyncio.sleep(0.001)
        assert not fn._running_tasks
        assert fn.owner.collector.hard_task_count == 6
        assert len(fn._output.state_dict()) == 4
    finally:
        await fn.dispose()


async def test_pre_post_share_frozen_model_but_keep_distinct_native_loadable_rollout_states(tmp_path):
    ledger = CheckpointLedger(tmp_path, identity={"model": "initial", "data": "abc"})
    calls = []

    async def model(step, *, force_sync):
        assert force_sync
        calls.append(step)
        folder = tmp_path / f"iter_{step:07d}"
        folder.mkdir()
        (folder / "weights").write_text("same optimizer and rng")

    async def rollout(step, *, directory):
        Path(directory).mkdir(parents=True)
        (Path(directory) / "state").write_text(str(len(list(tmp_path.glob("p2o_checkpoints/*")))))

    pre = await ledger.save(rollout_id=3, cycle_id=1, phase="pre_gepa", save_model=model, save_rollout=rollout)
    post = await ledger.save(rollout_id=3, cycle_id=1, phase="post_gepa", save_model=model, save_rollout=rollout)
    assert calls == [3] and pre["directory"] != post["directory"]
    assert ledger.latest() == post
    for checkpoint in (pre, post):
        path = Path(checkpoint["directory"])
        assert (path / "iter_0000003" / "weights").read_text() == "same optimizer and rng"
        assert (path / "latest_checkpointed_iteration.txt").read_text() == "3"
    restored = CheckpointLedger(tmp_path, identity=ledger.identity)
    assert restored.latest() == post

    async def fail(*a, **k):
        raise OSError("disk full")

    with pytest.raises(OSError):
        await ledger.save(rollout_id=3, cycle_id=1, phase="post_gepa", save_model=model, save_rollout=fail)
    assert restored.latest() == post
    with pytest.raises(ValueError, match="identity"):
        CheckpointLedger(tmp_path, identity={"model": "wrong"})


async def test_native_driver_runs_phase_only_between_updates_and_uses_paired_saves(monkeypatch):
    from functools import partial
    from types import SimpleNamespace

    import train_async
    from tests.fast.test_train_async import _install_driver_fakes, _make_args

    from miles.utils.async_utils import with_disposer

    events = []
    args = _make_args(fully_async=True, num_rollout=3, update_weights_interval=2, save_interval=1)
    components = _install_driver_fakes(monkeypatch, args, events)
    original_get = components.rollout_executor.get
    needs_phase = True

    async def get(step):
        nonlocal needs_phase
        if step == 1 and needs_phase:
            needs_phase = False
            events.append("boundary")
            return SimpleNamespace(phase_boundary=True)
        result = await original_get(step)
        return SimpleNamespace(phase_boundary=False, original=result)

    components.rollout_executor.get = get

    class Handler:
        def __init__(self, args, **handles):
            assert handles["actor"] is components.actor_model

        async def run(self, step):
            assert step == 0 and components.actor_model.trained == [0]
            events.append("phase")
            await asyncio.sleep(0)

        async def save(self, step):
            events.append(f"paired_save:{step}")

    await with_disposer(partial(train_async.train, phase_factory=Handler), args)
    assert components.actor_model.trained == [0, 1, 2]
    assert (
        events.index("actor_train:0")
        < events.index("boundary")
        < events.index("phase")
        < events.index("actor_train:1")
    )
    assert events.index("actor_train:1") < events.index("generate_start:2")
    assert sum(e == "phase" for e in events) == 1
    assert all(f"paired_save:{step}" in events for step in range(3))
    assert "generate_start:3" not in events


async def test_executor_boundary_is_not_captured_as_an_empty_training_batch(monkeypatch, tmp_path):
    from tests.fast.ray.rollout.test_rollout_executor import _CountingRolloutFn, _make_executor

    executor = _make_executor(tmp_path, _CountingRolloutFn())

    async def boundary(**kwargs):
        raise RolloutPhaseBoundary()

    monkeypatch.setattr(executor, "_generate_rollout_data", boundary)
    output = await executor.get(rollout_id=4)
    assert output.phase_boundary and output.data_ref is None
    assert executor._output_snapshotter.get(trainer_model_id=None, rollout_id=4) is None


async def test_phase_rollout_checkpoint_is_readable_by_native_loader(tmp_path):
    from tests.fast.ray.rollout.test_rollout_executor import _CountingRolloutFn, _make_executor

    executor = _make_executor(tmp_path, _CountingRolloutFn())
    executor._output_snapshotter.capture(trainer_model_id=None, rollout_id=4, data=[Sample(index=17)], metadata={})
    pair = tmp_path / "pair"
    await executor.save(3, directory=str(pair / "rollout" / "3"))
    restored = _make_executor(pair, _CountingRolloutFn())
    await restored.load(3)
    replay = restored._output_snapshotter.get(trainer_model_id=None, rollout_id=4)
    assert replay.data[0].index == 17
