"""CPU wall-clock timing, concurrency, durable restart and W&B field contracts."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from examples.seta.p2o import timing
from examples.seta.p2o.search import SearchConfig
from examples.seta.p2o.search_metrics import SearchMetrics
from examples.seta.p2o.search_runtime import SearchRuntime
from tests.fast.p2o.test_search import panel


def test_nested_stage_durations_are_separate_and_persist_on_failure(tmp_path, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(timing.time, "monotonic", lambda: clock[0])
    events = []
    recorder = timing.GepaTimings(tmp_path, emit=events.append)
    with recorder.measure("phase_total"):
        clock[0] = 12
        with recorder.measure("reflection", exploration=0):
            clock[0] = 17
        with pytest.raises(RuntimeError):
            with recorder.measure("dev", key="e0"):
                clock[0] = 20
                raise RuntimeError("sandbox failure")
        clock[0] = 22
    summary = recorder.summary()
    assert summary["phase_total"]["seconds"] == 12
    assert summary["reflection"]["seconds"] == 5
    assert summary["dev"]["seconds"] == 3 and summary["dev"]["failed"] == 1
    assert [e["event"] for e in events] == [0, 1, 2]
    assert json.loads((tmp_path / "timing.json").read_text())["summary"] == summary
    assert len((tmp_path / "timing.jsonl").read_text().splitlines()) == 3
    recorder.plot()
    assert (tmp_path / "timing.png").stat().st_size > 1000
    assert (tmp_path / "timing.svg").exists()


async def test_concurrent_intervals_report_completion_order_without_double_counting(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(timing.time, "monotonic", lambda: clock[0])
    events = []
    recorder = timing.GepaTimings(tmp_path, emit=events.append)
    a, b = asyncio.Event(), asyncio.Event()

    async def run(index, release):
        with recorder.measure("dev", exploration=index):
            await release.wait()

    first = asyncio.create_task(run(0, a))
    second = asyncio.create_task(run(1, b))
    await asyncio.sleep(0)
    clock[0] = 3
    b.set()
    await second
    clock[0] = 5
    a.set()
    await first
    assert [e["exploration"] for e in events] == [1, 0]
    assert [e["event"] for e in events] == [0, 1]
    assert recorder.summary()["dev"]["seconds"] == 8  # overlapping work, not total phase wall time
    restored = timing.GepaTimings(tmp_path)
    assert restored.records == recorder.records
    assert restored.summary() == recorder.summary()


def test_restart_marks_unfinished_intervals_unknown_and_appends_new_attempts(tmp_path):
    recorder = timing.GepaTimings(tmp_path)
    recorder.records.append(dict(interval=0, stage="phase_total", status="running", seconds=None, started_unix=0))
    recorder._persist()
    restored = timing.GepaTimings(tmp_path)
    assert restored.summary()["phase_total"] == dict(seconds=0, completed=0, failed=0, interrupted=1, running=0)
    with restored.measure("phase_total"):
        pass
    assert len(restored.records) == 2 and restored.records[0]["seconds"] is None
    assert restored.summary()["phase_total"]["completed"] == 1


def test_timing_wandb_axis_and_values_match_local_record(tmp_path, monkeypatch):
    from miles.utils.tracking_utils import tracking

    calls = []
    monkeypatch.setattr(tracking, "define_step_key_metric_group", lambda *a: None)
    monkeypatch.setattr(tracking, "log", lambda args, metrics, step_key: calls.append((metrics, step_key)))
    reporter = SearchMetrics(args=None, directory=tmp_path, cycle_id=4, training_step=12)
    recorder = timing.GepaTimings(tmp_path, emit=reporter.timing)
    with recorder.measure("reflection", exploration=1):
        pass
    metrics, axis = calls[0]
    assert axis == "gepa/cycle_4/timing/event"
    assert metrics[axis] == 0
    assert metrics[axis] == json.loads((tmp_path / "timing.json").read_text())["records"][0]["event"]
    assert metrics["gepa/cycle_4/timing/reflection_seconds"] == recorder.records[0]["seconds"]
    assert metrics["gepa/cycle_4/timing/training_step"] == 12


async def test_runtime_times_both_minibatches_reflection_dev_and_cached_repeats(tmp_path):
    args = SimpleNamespace(max_seq_len=4096)
    runtime = SearchRuntime(
        args=args,
        panel=panel(),
        endpoints=[dict(role=r, model_version="frozen", checkpoint="initial") for r in ("policy", "reflection")],
        directory=tmp_path,
        config=SearchConfig(),
        tokenizer=None,
    )

    async def cached(key, role, identity, metadata, seed):
        return dict(
            result=dict(
                seta_valid_reward=True, reward=0.2, eval_report=dict(passed=0, total=1), p2o_proposal="new template"
            ),
            records={"records": []},
        )

    runtime._cached_request = cached
    tasks = [o.task for o in runtime.panel.search_train[:3]]
    before = await runtime.minibatch("", tasks, [1, 2, 3], 0, "before")
    assert await runtime.reflect("", before, 0) == "new template"
    await runtime.minibatch("new template", tasks, [1, 2, 3], 0, "after")
    await runtime.evaluate("", "bootstrap")
    await runtime.evaluate("new template", "exploration_0")
    await runtime.evaluate("new template", "exploration_0")
    summary = runtime.timings.summary()
    assert set(summary) == {"minibatch_before", "minibatch_after", "reflection", "bootstrap_dev", "dev"}
    assert summary["dev"]["completed"] == 2
    assert all(v["seconds"] >= 0 for v in summary.values())
