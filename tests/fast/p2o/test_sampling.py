import asyncio
import copy
import random
from argparse import Namespace
from collections import deque

import pytest

from examples.seta.p2o.coordinator import CoordinatorState, SamplingCoordinator
from examples.seta.p2o.evidence import EpisodeEvidence, Origin, TaskObservation
from examples.seta.p2o.history import TemplateCycle, TemplateEvaluation
from examples.seta.p2o.mixing import MixConfig
from examples.seta.p2o.sampling import P2ODataBuffer, SubmissionWindows
from examples.seta.p2o.scoring import CollectionConfig, aggregate_dev
from miles.rollout.data_source import RolloutDataSource
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput, first_sample
from miles.utils.types import Sample


class Catalog:
    def __init__(self, size):
        self.origin_samples = [
            Sample(prompt=f"solve task {i}", metadata={"instance_id": f"task_{i}"}) for i in range(size)
        ]
        self.samples = list(self.origin_samples)

    def __len__(self):
        return len(self.samples)

    def shuffle(self, epoch):
        self.samples = list(self.origin_samples)
        random.Random(epoch).shuffle(self.samples)


def source(size=32, repeats=2):
    data = object.__new__(RolloutDataSource)
    data.args = Namespace(n_samples_per_prompt=repeats, rollout_shuffle=False, rollout_global_dataset=True)
    data.dataset = Catalog(size)
    data.sample_offset = data.epoch_id = data.sample_group_index = data.sample_index = 0
    data.metadata = {}
    return data


def coordinator(data):
    return SamplingCoordinator(
        samples=data.dataset.origin_samples,
        config=CollectionConfig(
            n_samples_per_prompt=data.args.n_samples_per_prompt,
            gepa_trigger_unique_hard_tasks=2,
            gepa_dev_task_count=1,
            gepa_dev_repeats=4,
        ),
        seed=123,
    )


def finish_cycle(owner, indices, *, texts=("template a", "template b")):
    tasks = [owner.task(owner.catalog[t]) for t in list(owner.catalog) if int(t.instance_id.split("_")[1]) in indices]
    for task in tasks:
        episodes = tuple(
            EpisodeEvidence(task=task, attempt_id=f"{task.instance_id}:{i}", valid=True, passed=0, total=10, reward=0)
            for i in range(owner.collector.config.n_samples_per_prompt)
        )
        owner.collector.observe(
            TaskObservation(
                task=task,
                cycle_id=owner.collector.cycle_id,
                observation_id=task.instance_id,
                origin=Origin.NATURAL,
                episodes=episodes,
            )
        )
    panel = owner.collector.freeze()
    entries = []
    for i, text in enumerate(texts):
        episodes = [
            EpisodeEvidence(
                task=d.task,
                attempt_id=f"{d.task.instance_id}:{j}",
                valid=True,
                passed=0,
                total=10,
                reward=i / len(texts),
            )
            for d in panel.dev
            for j in range(4)
        ]
        entries.append(
            TemplateEvaluation(
                text=text,
                scores=aggregate_dev([d.task for d in panel.dev], episodes, expected_repeats=4),
                policy_version="v1",
            )
        )
    owner.finish_cycle(TemplateCycle(cycle_id=panel.cycle_id, evaluations=tuple(entries)))


def window(data, owner):
    return SubmissionWindows(source=data, coordinator=owner, size=8, config=MixConfig())


def pop_window(windows, retries=None, in_flight=None):
    retries = retries if retries is not None else deque()
    return [windows.pop(retries=retries, in_flight=in_flight or set()) for _ in range(8)]


@pytest.mark.parametrize("natural_hits", range(6))
def test_natural_hits_and_replay_respect_soft_target_and_cap(natural_hits):
    data = source()
    owner = coordinator(data)
    finish_cycle(owner, [*range(natural_hits), 20, 21, 22])
    windows = window(data, owner)
    selected = pop_window(windows)
    templated = [b for b in selected if b[0].metadata["p2o"]["templated"]]
    assert len(templated) == min(4, max(2, natural_hits))
    assert len(windows.displaced) == max(0, 2 - natural_hits)
    assert data.sample_offset == 8  # replay allocated IDs, not natural rows
    for batch in selected:
        assert len(batch) == 2 and len({s.group_index for s in batch}) == 1
        assert len({s.index for s in batch}) == 2
        assert len({s.metadata["p2o"]["templated"] for s in batch}) == 1
    assert len({s.index for b in selected for s in b}) == 16


def test_three_natural_hits_all_templated_and_all_hard_uses_newest_four():
    data = source()
    owner = coordinator(data)
    finish_cycle(owner, list(range(8)))
    finish_cycle(owner, [0, 1, 2, 3])
    selected = pop_window(window(data, owner))
    assert [b[0].metadata["instance_id"] for b in selected if b[0].metadata["p2o"]["templated"]] == [
        f"task_{i}" for i in range(4)
    ]
    assert [b[0].metadata["instance_id"] for b in selected] == [f"task_{i}" for i in range(8)]


def test_no_template_keeps_hard_identity_and_generates_plain():
    data = source()
    owner = coordinator(data)
    finish_cycle(owner, [0, 1], texts=())
    selected = pop_window(window(data, owner))
    assert not any(b[0].metadata["p2o"]["templated"] for b in selected)
    assert owner.priority(owner.task(selected[0][0])) == 1


def test_insufficient_nonconflicting_replay_falls_back_without_partial_adjustment():
    data = source()
    owner = coordinator(data)
    finish_cycle(owner, [20, 21])
    windows = window(data, owner)
    selected = pop_window(windows, in_flight={list(owner.catalog)[20]})
    assert not any(b[0].metadata["p2o"]["templated"] for b in selected)
    assert not windows.displaced and data.sample_offset == 8
    assert owner.metrics["submission/fallback/insufficient_available_hard_tasks"] == 1


def test_displaced_natural_tasks_are_returned_fifo_without_cursor_advance():
    data = source()
    owner = coordinator(data)
    finish_cycle(owner, [20, 21])
    windows = window(data, owner)
    first = pop_window(windows)
    pending = [b[0].metadata["instance_id"] for b in windows.displaced]
    # Conflict prevents another replacement, making the deferred prefix visible.
    second = pop_window(windows, in_flight=set(owner.collector.participated_tasks))
    assert [b[0].metadata["instance_id"] for b in second[:2]] == pending == ["task_6", "task_7"]
    assert data.sample_offset == 14 and not windows.displaced
    assert {s.index for b in first for s in b}.isdisjoint(s.index for b in second for s in b)


def test_same_task_can_use_different_templates_with_one_identity_per_episode():
    data = source(repeats=16)
    owner = coordinator(data)
    finish_cycle(owner, [0, 1], texts=tuple(f"t{i}" for i in range(20)))
    [batch] = data.materialize_samples([data.dataset.origin_samples[0]])
    owner.stamp(batch, origin=Origin.NATURAL, inject=True)
    # Draw many pairs instead of depending on a lucky first random draw.
    ids = set()
    varied_batch = False
    for _ in range(30):
        owner.stamp(batch, origin=Origin.NATURAL, inject=True)
        ids.update(s.metadata["p2o_template"]["template_id"] for s in batch)
        varied_batch |= len({s.metadata["p2o_template"]["template_id"] for s in batch}) > 1
    assert varied_batch and len(ids) > 1 and len({s.group_index for s in batch}) == 1


def test_native_source_and_window_checkpoint_preserve_draws_and_pending(tmp_path):
    data = source()
    owner = coordinator(data)
    finish_cycle(owner, [20, 21])
    windows = window(data, owner)
    windows.pop(retries=deque(), in_flight=set())
    data.save(tmp_path / "source")
    state = windows.state_dict()
    owner_state = CoordinatorState.model_validate_json(owner.snapshot().model_dump_json())
    other_source = source()
    other_source.load(tmp_path / "source")
    other_owner = coordinator(other_source)
    other_owner.load_state(owner_state)
    other = window(other_source, other_owner)
    other.load_state_dict(copy.deepcopy(state))
    for _ in range(24):
        left = windows.pop(retries=deque(), in_flight=set())
        right = other.pop(retries=deque(), in_flight=set())
        assert [s.to_dict() for s in left] == [s.to_dict() for s in right]
    assert data.sample_offset == other_source.sample_offset


def buffer(owner):
    args = Namespace(
        async_data_buffer_capacity_factor=10,
        rollout_batch_size=8,
        dynamic_sampling_filter_path=None,
        max_weight_staleness=None,
        reward_key=None,
        enable_sample_ownership_checker=False,
    )
    result = P2ODataBuffer(DataBufferConstructorInput(args=args, unused_handler_fn=lambda *_: None))
    result.bind(owner, MixConfig())
    return result


def completed(data, owner, index, *, templated):
    [batch] = data.materialize_samples([data.dataset.origin_samples[index]])
    owner.stamp(batch, origin=Origin.NATURAL, inject=templated)
    for sample in batch:
        sample.status = Sample.Status.COMPLETED
        sample.reward = 0.0
        sample.metadata.update(seta_valid_reward=True, reward=0.0, eval_report={"passed": 0, "total": 10})
    return DataBufferInput(prompt_group=batch, group=batch)


async def test_buffer_out_of_order_completion_keeps_all_groups_and_actual_template_marks():
    data = source()
    owner = coordinator(data)
    finish_cycle(owner, list(range(16)))
    replay = buffer(owner)
    entries = [completed(data, owner, i, templated=i < 8) for i in range(16)]
    shuffled = list(entries)
    random.Random(12).shuffle(shuffled)
    for entry in shuffled:
        await replay.put(entry)
    before = copy.deepcopy([e.group[0].metadata for e in entries])
    left = await asyncio.wait_for(replay.get(num_groups=8), timeout=1)
    right = await asyncio.wait_for(replay.get(num_groups=8), timeout=1)
    assert 2 <= sum(first_sample(e.group).metadata["p2o"]["templated"] for e in left) <= 4
    assert {id(e) for e in left + right} == {id(e) for e in entries} and not replay._buffer
    assert before == [e.group[0].metadata for e in entries]


@pytest.mark.parametrize("templated", [False, True])
async def test_buffer_infeasible_quota_does_not_wait_or_discard(templated):
    data = source()
    owner = coordinator(data)
    finish_cycle(owner, list(range(16)))
    replay = buffer(owner)
    for i in range(8):
        await replay.put(completed(data, owner, i, templated=templated))
    selected = await asyncio.wait_for(replay.get(num_groups=8), timeout=1)
    assert len(selected) == 8 and not replay._buffer
    reason = "insufficient_plain" if templated else "insufficient_templated"
    assert owner.metrics[f"consumption/fallback/{reason}"] == 1


def test_bonus_free_collection_observes_only_complete_natural_groups():
    data = source()
    owner = coordinator(data)
    entry = completed(data, owner, 0, templated=False)
    for sample in entry.group:
        sample.reward = sample.metadata["reward"] = 1.2
        sample.metadata["eval_report"] = {"passed": 1, "total": 1}
    assert not owner.observe(entry).added
    for sample in entry.group:
        sample.reward = sample.metadata["reward"] = 0.0
        sample.metadata["eval_report"] = {"passed": 0, "total": 1}
    assert owner.observe(entry).added
    other = completed(data, owner, 1, templated=False)
    owner.stamp(other.prompt_group, origin=Origin.REPLAY, inject=False)
    assert owner.observe(other).reason == "not_natural"
    assert owner.collector.hard_task_count == 1


def test_continuous_replay_does_not_starve_deferred_natural_tasks_or_lose_ids():
    data = source(64)
    owner = coordinator(data)
    finish_cycle(owner, [60, 61])
    windows = window(data, owner)
    submitted = set()
    deferred_previous = set()
    for _ in range(6):
        selected = pop_window(windows)
        ids = {b[0].group_index for b in selected}
        assert not (submitted & ids)
        assert deferred_previous <= ids
        submitted.update(ids)
        deferred_previous = {b[0].group_index for b in windows.displaced}
        assert submitted | deferred_previous == set(range(data.sample_group_index))
        assert not (submitted & deferred_previous)


def test_catalog_revision_and_sampling_configuration_checked_on_resume():
    data = source()
    owner = coordinator(data)
    finish_cycle(owner, [0, 1])
    state = owner.snapshot()
    changed = source()
    changed.dataset.origin_samples[0].prompt = "different task input"
    with pytest.raises(ValueError, match="catalog"):
        coordinator(changed).load_state(state)
    other = SamplingCoordinator(
        samples=data.dataset.origin_samples, config=owner.collector.config, seed=9, temperature=2
    )
    with pytest.raises(ValueError, match="temperature"):
        other.load_state(state)


def test_invalid_episode_remains_evidence_without_advancing_hard_count():
    data = source()
    owner = coordinator(data)
    entry = completed(data, owner, 0, templated=False)
    entry.group[0].status = Sample.Status.ABORTED
    entry.group[0].metadata["seta_error"] = "verifier timeout"
    result = owner.observe(entry)
    assert not result.added and result.reason == "invalid_episode"
    assert owner.collector.hard_task_count == 0


def test_resubmission_clears_old_context_even_when_new_attempt_is_plain():
    data = source()
    owner = coordinator(data)
    finish_cycle(owner, [0, 1])
    [batch] = data.get_samples(1)
    owner.stamp(batch, origin=Origin.NATURAL, inject=True)
    for sample in batch:
        sample.metadata["p2o_context"] = {"plain_prompt_token_ids": [99]}
        sample.reset_for_retry()
    owner.stamp(batch, origin=Origin.NATURAL, inject=False)
    assert all("p2o_context" not in sample.metadata and "p2o_template" not in sample.metadata for sample in batch)
