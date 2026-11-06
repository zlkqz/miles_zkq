import argparse
import asyncio

import pytest
from examples.seta.p2o.rollout import P2ORolloutFn
from tests.fast.p2o.test_sampling import completed, finish_cycle, source
from tests.fast.rollout.test_fully_async_rollout import FakeGenerateState, make_args

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles.rollout.fully_async_data_buffer import DataBufferInput
from miles.utils.types import Sample


def args(**overrides):
    parser = argparse.ArgumentParser()
    P2ORolloutFn.add_arguments(parser)
    values = vars(parser.parse_args([]))
    values.update(
        rollout_batch_size=8,
        rollout_seed=1,
        rollout_shuffle=True,
        async_data_buffer_capacity_factor=2,
        custom_async_data_buffer_path="examples.seta.p2o.sampling.P2ODataBuffer",
    )
    values.update(overrides)
    return make_args(**values)


def make_fn(monkeypatch, **overrides):
    monkeypatch.setattr("miles.rollout.fully_async_rollout.GenerateState", FakeGenerateState)
    return P2ORolloutFn(RolloutFnConstructorInput(args=args(**overrides), data_source=source(128)))


async def test_native_worker_collects_all_failure_groups_before_buffer_and_exposes_metrics(monkeypatch):
    fn = make_fn(monkeypatch)

    async def generate(batch):
        # Completion order differs from submission; a task is still a K-episode group.
        await asyncio.sleep(0.002 * (batch[0].group_index % 3))
        for sample in batch:
            sample.status = Sample.Status.COMPLETED
            sample.response = "failed action"
            sample.tokens = [1, 2]
            sample.response_length = 1
            sample.loss_mask = [1]
            sample.rollout_log_probs = [-0.2]
            sample.reward = 0.0
            sample.metadata.update(seta_valid_reward=True, reward=0.0, eval_report={"passed": 0, "total": 4})
        return DataBufferInput(prompt_group=batch, group=batch)

    fn._generate_group = generate
    try:
        result = await asyncio.wait_for(fn(RolloutFnTrainInput(rollout_id=0, weight_version=0)), timeout=2)
        assert len(result.samples) == 8 and all(len(batch) == 2 for batch in result.samples)
        assert all(s.reward == 0.0 for batch in result.samples for s in batch)
        assert result.metrics["p2o/hard_tasks"] >= 8
        assert result.metrics["p2o/collection/added"] >= 8
        assert result.metrics["p2o/history_templates"] == 0
    finally:
        await fn.dispose()
        await asyncio.gather(*(entry.task for entry in fn._running_tasks), return_exceptions=True)


def test_rollout_state_roundtrip_includes_prefetch_rng_and_native_cursors(monkeypatch, tmp_path):
    fn = make_fn(monkeypatch)
    fn._next_prompt_groups()
    fn.data_source.save(tmp_path / "data")
    fn.save(tmp_path / "fn")
    other = make_fn(monkeypatch)
    other.data_source.load(tmp_path / "data")
    other.load(tmp_path / "fn")
    for _ in range(17):
        assert [[s.to_dict() for s in b] for b in fn._next_prompt_groups()] == [
            [s.to_dict() for s in b] for b in other._next_prompt_groups()
        ]


async def test_checkpoint_restores_all_cycles_scores_assignments_pending_tasks_and_buffer(monkeypatch, tmp_path):
    config = dict(gepa_trigger_unique_hard_tasks=2, gepa_dev_task_count=1)
    fn = make_fn(monkeypatch, **config)
    finish_cycle(fn.owner, [100, 101], texts=("first low", "first high"))
    finish_cycle(fn.owner, [102, 103], texts=("second low", "second high"))
    # An unsuccessful cycle keeps old assignments and still records hard identity.
    finish_cycle(fn.owner, [100, 104], texts=())
    pending = completed(fn.data_source, fn.owner, 105, templated=False)
    assert fn.owner.observe(pending).added
    await fn._output.put(pending)
    retry = completed(fn.data_source, fn.owner, 106, templated=False)
    fn._retry_buffer.append(retry.prompt_group)
    fn._next_prompt_groups()  # drains retry first
    fn._next_prompt_groups()  # leaves prefetched and displaced natural tasks
    assert fn.windows.ready and fn.windows.displaced
    fn._retry_buffer.append(retry.prompt_group)
    fn.data_source.save(tmp_path / "data")
    fn.save(tmp_path / "fn")

    other = make_fn(monkeypatch, **config)
    other.data_source.load(tmp_path / "data")
    other.load(tmp_path / "fn")
    assert other.owner.snapshot() == fn.owner.snapshot()
    assert len(other.owner.history) == 4
    assert len(other.owner.history.snapshot().cycles) == 3
    assert other.owner.collector.hard_task_count == 1
    assert other.windows.state_dict() == fn.windows.state_dict()
    assert other._output.state_dict() == fn._output.state_dict()
    assert [[s.to_dict() for s in b] for b in other._retry_buffer] == [
        [s.to_dict() for s in b] for b in fn._retry_buffer
    ]
    # Derived normalization/solver eligibility is rebuilt from the saved evidence.
    for task in fn.owner.catalog:
        assert other.owner.history.probabilities(task) == fn.owner.history.probabilities(task)
        assert other.owner.priority(task) == fn.owner.priority(task)
    for _ in range(24):
        assert [[s.to_dict() for s in b] for b in fn._next_prompt_groups()] == [
            [s.to_dict() for s in b] for b in other._next_prompt_groups()
        ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"rollout_batch_size": 1},
        {"rollout_shuffle": False},
        {"custom_async_data_buffer_path": None},
        {"dynamic_sampling_filter_path": "miles.rollout.filter_hub.common_filters.apply_reward_nonzero_std_filter"},
    ],
)
def test_incompatible_launches_are_rejected(monkeypatch, overrides):
    with pytest.raises(ValueError):
        make_fn(monkeypatch, **overrides)
