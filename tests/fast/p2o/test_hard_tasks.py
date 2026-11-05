import json

import pytest
from examples.seta.p2o.evidence import Origin, TaskKey
from examples.seta.p2o.hard_tasks import CollectorState, HardTaskCollector
from examples.seta.p2o.scoring import CollectionConfig
from pydantic import ValidationError


@pytest.fixture
def catalog():
    return tuple(TaskKey(dataset_id="revision", instance_id=f"task_{i:03}") for i in range(80))


@pytest.fixture
def config():
    return CollectionConfig(n_samples_per_prompt=16)


def _roundtrip(collector):
    # Exercise real JSON conversion, including the nested tuple Random state.
    state = CollectorState.model_validate(json.loads(collector.snapshot().model_dump_json()))
    return HardTaskCollector.restore(state, config=collector.config, training_tasks=collector.training_tasks)


def _fill(collector, catalog, make_observation, count):
    for i, task in enumerate(catalog[:count]):
        collector.observe(make_observation(task_key=task, cycle=collector.cycle_id, observation_id=f"batch-{i}"))


def test_trigger_is_unique_natural_tasks_and_latches_through_drain(catalog, config, make_observation):
    collector = HardTaskCollector(config, catalog, seed=11)
    for origin in (Origin.REPLAY, Origin.SEARCH, Origin.REFLECTION, Origin.EVALUATION):
        result = collector.observe(make_observation(task_key=catalog[0], origin=origin))
        assert not result.added and result.reason == "not_natural"
    _fill(collector, catalog, make_observation, 63)
    assert collector.hard_task_count == 63 and not collector.trigger_pending
    duplicate = collector.observe(make_observation(task_key=catalog[0], observation_id="another-batch"))
    assert not duplicate.added and collector.hard_task_count == 63
    with pytest.raises(ValueError, match="not reached"):
        collector.freeze()
    assert collector.observe(make_observation(task_key=catalog[63])).trigger_pending
    _fill(collector, catalog[64:], make_observation, 7)
    assert collector.hard_task_count == 71
    # Later improvement does not remove the first hard observation or unlatch the trigger.
    collector.observe(make_observation(task_key=catalog[0], passed=16))
    panel = collector.freeze()
    assert len(panel.hard_tasks) == 71 and len(panel.dev) == 16 and len(panel.search_train) == 55
    assert {d.task for d in panel.dev}.isdisjoint(o.task for o in panel.search_train)
    assert {o.task for o in panel.hard_tasks} == set(catalog[:71])
    assert collector.freeze() == panel
    for origin in (Origin.REPLAY, Origin.SEARCH, Origin.REFLECTION, Origin.EVALUATION):
        assert collector.observe(make_observation(task_key=catalog[71], origin=origin)).reason == "not_natural"
    assert collector.hard_task_count == 71
    assert all(len(d.seeds) == len(set(d.seeds)) == 4 for d in panel.dev)
    with pytest.raises(ValueError, match="frozen"):
        collector.observe(make_observation(task_key=catalog[71]))


def test_typical_64_task_panel_has_48_search_and_16_dev(catalog, config, make_observation):
    collector = HardTaskCollector(config, catalog, seed=1)
    _fill(collector, catalog, make_observation, 64)
    assert len(collector.freeze().search_train) == 48


def test_invalid_incomplete_and_nonhard_never_advance_trigger(catalog, config, make_observation):
    collector = HardTaskCollector(config, catalog, seed=1)
    for kwargs in ({"valid": False}, {"count": 15}, {"passed": 4}):
        result = collector.observe(make_observation(task_key=catalog[0], **kwargs))
        assert not result.added and collector.hard_task_count == 0
    assert collector.observe(make_observation(task_key=catalog[0])).added


def test_training_catalog_isolates_final_eval_and_dataset_revision(catalog, config, make_observation):
    collector = HardTaskCollector(config, catalog, seed=1)
    for unknown in (
        TaskKey(dataset_id="final-test", instance_id=catalog[0].instance_id),
        TaskKey(dataset_id="revision", instance_id="held-out"),
    ):
        with pytest.raises(ValueError, match="outside"):
            collector.observe(make_observation(task_key=unknown))
    assert collector.hard_task_count == 0


def test_resume_before_trigger_before_freeze_and_mid_gepa(catalog, config, make_observation):
    uninterrupted = HardTaskCollector(config, catalog, seed=2026)
    _fill(uninterrupted, catalog, make_observation, 31)
    resumed = _roundtrip(uninterrupted)
    for collector in (uninterrupted, resumed):
        _fill(collector, catalog[31:], make_observation, 38)
    resumed = _roundtrip(resumed)
    assert uninterrupted.freeze() == resumed.freeze()
    mid_search = _roundtrip(resumed)
    assert mid_search.freeze() == uninterrupted.freeze()
    assert mid_search.snapshot() == uninterrupted.snapshot()


def test_empty_search_result_keeps_history_and_recollects_next_cycle(catalog, config, make_observation):
    collector = HardTaskCollector(config, catalog, seed=2026)
    with pytest.raises(ValueError, match="no matching"):
        collector.finish_cycle(0)
    _fill(collector, catalog, make_observation, 70)
    panel = collector.freeze()
    collector.finish_cycle(1)  # No templates are required to finish a cycle.
    assert collector.cycle_id == 2 and collector.hard_task_count == 0 and not collector.trigger_pending
    assert {h.task for h in collector.snapshot().history} == {o.task for o in panel.hard_tasks}
    assert all(h.last_participated_cycle == 1 for h in collector.snapshot().history)
    collector.finish_cycle(1)
    assert collector.cycle_id == 2
    restored = _roundtrip(collector)
    for c in (collector, restored):
        with pytest.raises(ValueError, match="different cycle"):
            c.observe(make_observation(task_key=catalog[0], cycle=1))
        _fill(c, catalog[10:], make_observation, 64)
    assert collector.freeze() == restored.freeze()
    collector.finish_cycle(2)
    history = {h.task: h.last_participated_cycle for h in collector.snapshot().history}
    assert history[catalog[0]] == 1 and history[catalog[10]] == 2 and len(history) == 74


def test_restore_rejects_incompatible_config_catalog_and_schema(catalog, config):
    collector = HardTaskCollector(config, catalog, seed=0)
    state = collector.snapshot()
    with pytest.raises(ValueError, match="differs"):
        HardTaskCollector.restore(state, config=CollectionConfig(n_samples_per_prompt=8), training_tasks=catalog)
    with pytest.raises(ValueError, match="differs"):
        HardTaskCollector.restore(state, config=config, training_tasks=catalog[:-1])
    raw = state.model_dump(mode="json")
    raw["schema_version"] = 2
    with pytest.raises(ValidationError):
        CollectorState.model_validate(raw)


def test_restore_rejects_corrupt_partition_and_nonhard_evidence(catalog, config, make_observation):
    collector = HardTaskCollector(config, catalog, seed=0)
    _fill(collector, catalog, make_observation, 64)
    collector.freeze()
    raw = collector.snapshot().model_dump(mode="json")
    raw["pending"][0]["episodes"][0]["passed"] = 16
    raw["pending"][0]["episodes"][1]["passed"] = 16
    with pytest.raises(ValueError, match="ineligible"):
        HardTaskCollector.restore(CollectorState.model_validate(raw), config=config, training_tasks=catalog)
    raw = collector.snapshot().model_dump(mode="json")
    raw["panel"]["hard_tasks"].pop()
    with pytest.raises(ValueError):
        HardTaskCollector.restore(CollectorState.model_validate(raw), config=config, training_tasks=catalog)


def test_tunable_small_panel_and_repeat_count(catalog, make_observation):
    config = CollectionConfig(
        n_samples_per_prompt=4, gepa_trigger_unique_hard_tasks=5, gepa_dev_task_count=2, gepa_dev_repeats=3
    )
    collector = HardTaskCollector(config, catalog, seed=2)
    for task in catalog[:6]:
        collector.observe(make_observation(task_key=task, count=4))
    panel = collector.freeze()
    assert len(panel.dev) == 2 and len(panel.search_train) == 4
    assert all(len(task.seeds) == 3 for task in panel.dev)
