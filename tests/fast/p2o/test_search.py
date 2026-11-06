"""GEPA scheduling tests exercise asynchronous completions and durable reservations."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from examples.seta.p2o.evidence import EpisodeEvidence, Origin, TaskKey, TaskObservation
from examples.seta.p2o.hard_tasks import CyclePanel, DevTaskPlan
from examples.seta.p2o.history import TemplateEvaluation
from examples.seta.p2o.scoring import aggregate_dev
from examples.seta.p2o.search import Search, SearchConfig, pareto_indices


def panel():
    tasks = [TaskKey(dataset_id="test", instance_id=f"task_{i}") for i in range(8)]
    observations = tuple(
        TaskObservation(
            cycle_id=1,
            observation_id=str(i),
            task=t,
            origin=Origin.NATURAL,
            episodes=(EpisodeEvidence(task=t, attempt_id=str(i), valid=True, passed=0, total=1, reward=0.0),),
        )
        for i, t in enumerate(tasks)
    )
    return CyclePanel(
        cycle_id=1, hard_tasks=observations, dev=tuple(DevTaskPlan(task=t, seeds=(1, 2, 3, 4)) for t in tasks[:2])
    )


def evaluation(text, scores=(0.0, 0.0)):
    p = panel()
    evidence = [
        EpisodeEvidence(task=d.task, attempt_id=f"{i}:{j}", valid=True, passed=int(score >= 1), total=1, reward=score)
        for i, (d, score) in enumerate(zip(p.dev, scores, strict=True))
        for j in range(4)
    ]
    return TemplateEvaluation(
        text=text, policy_version="frozen", scores=aggregate_dev([d.task for d in p.dev], evidence, expected_repeats=4)
    )


class Runtime:
    def __init__(self, reject=False):
        self.reject = reject
        self.active = 0
        self.peak = 0
        self.calls = []
        self.failed = False

    async def minibatch(self, text, tasks, seeds, exploration, stage):
        self.calls.append((exploration, stage, tuple(tasks), tuple(seeds), text))
        if stage == "before":
            self.active += 1
            self.peak = max(self.peak, self.active)
        await asyncio.sleep(0)
        if stage == "after":
            self.active -= 1
        return [
            SimpleNamespace(
                score=float(stage == "after" and not self.reject), task=t, attempt_id=f"{exploration}:{stage}:{i}"
            )
            for i, t in enumerate(tasks)
        ]

    async def reflect(self, parent, before, exploration):
        await asyncio.sleep(0)
        return f"template {exploration}"

    def validate_template(self, template):
        pass

    def mean(self, episodes):
        return episodes[0].score

    def evidence(self, episode):
        return EpisodeEvidence(
            task=episode.task, attempt_id=episode.attempt_id, valid=True, passed=1, total=1, reward=1.2
        )

    async def evaluate(self, text, key):
        self.calls.append(("dev", key))
        await asyncio.sleep(0)
        return evaluation(text, (0.2, 0.1) if text else (0.4, 0.4))


def make_search(tmp_path, runtime=None, **options):
    return Search(
        panel=panel(),
        config=SearchConfig(**options),
        runtime=runtime or Runtime(),
        directory=tmp_path,
        policy_version="frozen",
        seed=12,
    )


def test_pareto_uses_task_mean_vectors_and_keeps_ties():
    candidates = [
        evaluation("a", (1.0, 0.0)),
        evaluation("b", (0.0, 1.0)),
        evaluation("c", (0.5, 0.5)),
        evaluation("d", (0.1, 0.1)),
        evaluation("e", (0.5, 0.5)),
    ]
    assert pareto_indices(candidates) == [0, 1, 2, 4]


async def test_admission_is_minibatch_improvement_not_dev(tmp_path):
    runtime = Runtime()
    search = make_search(tmp_path, runtime, budget=4)
    cycle = await search.run()
    assert len(cycle.evaluations) == 5
    assert max(c.scores.mean for c in cycle.evaluations) == 0.4
    assert all(e["done"] for e in search.state["explorations"])
    assert search.state["stop"] == "budget" and search.state["failures"] == 0  # warmup
    for i in range(4):
        before = next(c for c in runtime.calls if c[:2] == (i, "before"))
        after = next(c for c in runtime.calls if c[:2] == (i, "after"))
        assert before[2:4] == after[2:4]


async def test_rejected_minibatches_use_budget_and_no_dev_or_warmup_stop(tmp_path):
    runtime = Runtime(reject=True)
    search = make_search(tmp_path, runtime, budget=19, patience=2)
    cycle = await search.run()
    assert len(cycle.evaluations) == 1 and runtime.peak == 1
    assert len(search.state["explorations"]) == 19
    assert sum(c[0] == "dev" for c in runtime.calls) == 1
    assert search.state["failures"] == 0


@pytest.mark.parametrize("count", [1, 16, 17])
def test_candidate_count_controls_parent_parallelism(tmp_path, count):
    search = make_search(tmp_path, budget=40)
    search.state["candidates"] = [
        evaluation(str(i), (float(i == 0), float(i == 0))).model_dump(mode="json") for i in range(count)
    ]
    busy = []
    for _ in range(16):
        e = search._reserve(set(busy))
        if e:
            busy.append(e["parent"])
    if count <= 16:
        assert len(busy) == count and len(set(busy)) == count
    else:
        assert busy == [0] * 16  # only the actual Pareto parent can repeat


async def test_early_stop_counts_all_failures_after_warmup_and_drains(tmp_path):
    runtime = Runtime(reject=True)
    search = make_search(tmp_path, runtime, budget=100, patience=3)
    search.state["candidates"] = [evaluation(str(i)).model_dump(mode="json") for i in range(17)]
    await search.run()
    assert search.state["stop"] == "patience"
    assert 3 <= len(search.state["explorations"]) <= 18
    assert all(e["done"] for e in search.state["explorations"])
    assert runtime.peak == 16 and runtime.active == 0


async def test_resume_reserved_work_does_not_charge_budget_twice(tmp_path):
    search = make_search(tmp_path, budget=2)
    search.state["candidates"] = [evaluation("").model_dump(mode="json")]
    entry = search._reserve(set())
    resumed = make_search(tmp_path, budget=2)
    await resumed.run()
    assert len(resumed.state["explorations"]) == 2
    assert resumed.state["explorations"][0]["tasks"] == entry["tasks"]
    count = len(resumed.state["candidates"])
    resumed._commit(resumed.state["explorations"][0], {"reason": "no_template"})
    assert len(resumed.state["candidates"]) == count
    saved = json.loads((tmp_path / "search_state.json").read_text())
    assert saved == resumed.state


def test_resume_rejects_changed_frozen_policy_or_search_config(tmp_path):
    make_search(tmp_path, budget=2)
    with pytest.raises(ValueError, match="journal"):
        make_search(tmp_path, budget=3)


async def test_runtime_caches_requests_and_aggregates_partial_invalid_dev(tmp_path, monkeypatch):
    import httpx
    from examples.seta.p2o import search_runtime as module

    calls = []
    session_count = 0

    def transport(request):
        nonlocal session_count
        if request.method == "POST":
            session_count += 1
            return httpx.Response(200, json={"session_id": str(session_count)})
        if request.method == "GET":
            return httpx.Response(200, json={"records": []})
        return httpx.Response(200, json={})

    client = httpx.AsyncClient
    monkeypatch.setattr(
        module.httpx, "AsyncClient", lambda **kwargs: client(transport=httpx.MockTransport(transport), **kwargs)
    )

    async def run(url, prompt, request, metadata):
        calls.append((url, request, metadata))
        if request["seed"] == 4:
            return dict(seta_valid_reward=False, seta_error="failed verifier")
        return dict(
            seta_valid_reward=True, reward=1.2, eval_report=dict(passed=1, total=1), remote_request_id=str(len(calls))
        )

    monkeypatch.setattr(module, "run", run)
    args = SimpleNamespace(
        max_seq_len=4096,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_max_response_len=512,
        apply_chat_template_kwargs={"enable_thinking": False},
    )
    endpoints = [
        dict(role=role, session_url="http://model", model_version="frozen", checkpoint="initial")
        for role in ("policy", "reflection")
    ]
    runtime = module.SearchRuntime(
        args=args, panel=panel(), endpoints=endpoints, directory=tmp_path, config=SearchConfig(), tokenizer=None
    )
    evaluated = await runtime.evaluate("hello", "candidate")
    assert evaluated.scores.vector == (1.2, 1.2)
    assert all(s.valid_count == 3 and s.attempted_count == 4 for s in evaluated.scores.tasks)
    assert len(calls) == 8
    assert all(c[1]["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False} for c in calls)
    assert all(c[2]["p2o_template"]["text"] == "hello" for c in calls)
    assert await runtime.evaluate("hello", "candidate") == evaluated
    assert len(calls) == 8
    with pytest.raises(ValueError, match="cache identity"):
        await runtime.evaluate("different", "candidate")


def test_local_metrics_and_custom_wandb_axis(tmp_path, monkeypatch):
    from examples.seta.p2o.search_metrics import SearchMetrics

    from miles.utils.tracking_utils import tracking

    events = []
    monkeypatch.setattr(tracking, "define_step_key_metric_group", lambda *args: events.append(args))
    monkeypatch.setattr(tracking, "log", lambda *args, **kwargs: events.append(kwargs))
    reporter = SearchMetrics(args=None, directory=tmp_path, cycle_id=2, training_step=3)
    event = dict(event=0, kind="completed", best_dev=0.3, candidates=2, pareto=1, started=1, completed=1, patience=0)
    reporter(event)
    reporter.finish({"events": [event]})
    assert events[0] == ("gepa/cycle_2", "gepa/cycle_2/event")
    assert events[1] == ("gepa/cycle_2/timing", "gepa/cycle_2/timing/event")
    assert events[2]["step_key"] == "gepa/cycle_2/event"
    assert json.loads((tmp_path / "metrics.json").read_text()) == [event]
    assert (tmp_path / "metrics.png").stat().st_size > 1000 and (tmp_path / "metrics.svg").exists()


def test_patience_compares_cycle_best_and_never_unlatches_stop(tmp_path):
    search = make_search(tmp_path, budget=100, patience=2)
    search.state["candidates"] = [evaluation(str(i), (0.6, 0.6)).model_dump(mode="json") for i in range(17)]
    for i, score in enumerate([0.2, 0.5, 0.9]):
        entry = dict(id=i, parent=0, tasks=[0, 1, 2], seeds=[1, 2, 3], done=False)
        search.state["explorations"].append(entry)
        search._commit(
            entry, dict(reason="admitted", evaluation=evaluation(f"new_{i}", (score, score)).model_dump(mode="json"))
        )
        if i == 1:
            assert search.state["stop"] == "patience" and search.state["failures"] == 2
    assert search.state["failures"] == 0 and search.state["stop"] == "patience"


async def test_failed_worker_leaves_resumable_reservation_and_cancels_siblings(tmp_path):
    runtime = Runtime()

    async def failure(*args):
        raise OSError("interrupted request")

    runtime.reflect = failure
    search = make_search(tmp_path, runtime, budget=2)
    with pytest.raises(OSError, match="interrupted"):
        await search.run()
    assert len(search.state["explorations"]) == 1 and not search.state["explorations"][0]["done"]
    resumed = make_search(tmp_path, budget=2)
    cycle = await resumed.run()
    assert len(resumed.state["explorations"]) == 2 and len(cycle.evaluations) == 3
