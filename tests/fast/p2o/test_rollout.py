import argparse
import asyncio

import pytest
from examples.seta.p2o.rollout import P2ORolloutFn
from tests.fast.p2o.test_sampling import source
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
