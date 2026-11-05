"""Phase transactions and launcher integration without allocating GPUs."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from examples.seta.p2o import driver
from examples.seta.p2o.search import SearchConfig
from tests.fast.p2o.test_search import panel


def args(tmp_path):
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text("{}")
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n")
    return SimpleNamespace(
        save=str(tmp_path / "save"),
        load=str(tmp_path / "save"),
        hf_checkpoint=str(model),
        prompt_data=str(data),
        fully_async=True,
        colocate=False,
        use_critic=False,
        max_seq_len=4096,
        rollout_batch_size=2,
        n_samples_per_prompt=2,
        hard_reward_threshold=None,
        gepa_trigger_unique_hard_tasks=4,
        gepa_dev_task_count=1,
        gepa_dev_repeats=4,
        cluster_backend="ray",
        offload_train=True,
        use_checkpoint_opt_param_scheduler=True,
        rollout_external=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=4,
        rollout_num_gpus=4,
        num_gpus_per_node=8,
        gepa_reflection_num_gpus=1,
        gepa_reflection_tp=1,
        gepa_policy_tp=1,
        rollout_seed=12,
        **{"gepa_" + k: v for k, v in vars(SearchConfig()).items()},
    )


def test_preflight_validates_before_gpu_allocation(tmp_path):
    a = args(tmp_path)
    assert driver.prepare_resume(a).latest() is None
    a.gepa_trigger_unique_hard_tasks = 3
    with pytest.raises(ValueError, match="search-train"):
        driver.prepare_resume(a)
    a.gepa_trigger_unique_hard_tasks = 4
    a.gepa_policy_tp = 2
    with pytest.raises(ValueError, match="divisible"):
        driver.prepare_resume(a)
    a.gepa_policy_tp = 1
    a.use_checkpoint_opt_param_scheduler = False
    with pytest.raises(ValueError, match="saved schedule"):
        driver.prepare_resume(a)


async def test_resume_selects_exact_paired_native_load_root(tmp_path):
    a = args(tmp_path)
    ledger = driver.prepare_resume(a)

    async def model(step, force_sync):
        (ledger.root / f"iter_{step:07d}").mkdir(parents=True)

    async def rollout(step, directory):
        from pathlib import Path

        Path(directory).mkdir(parents=True)

    saved = await ledger.save(rollout_id=2, cycle_id=1, phase="pre_gepa", save_model=model, save_rollout=rollout)
    driver.prepare_resume(a)
    assert a.load == saved["directory"]
    assert a.save == str(tmp_path / "save")
    (tmp_path / "train.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="identity"):
        driver.prepare_resume(a)


@pytest.mark.parametrize("failure", [False, True])
async def test_phase_order_restore_and_only_resume_after_post_checkpoint(tmp_path, monkeypatch, failure):
    events = []
    a = args(tmp_path)
    actor = SimpleNamespace(onload=AsyncMock(side_effect=lambda: events.append("onload")), save_model=AsyncMock())

    class Executor:
        save = AsyncMock()

        async def rollout_phase(self, op, payload=None):
            events.append(op)
            return panel().model_dump(mode="json") if op == "freeze" else {}

    class Ledger:
        def __init__(self, *a, **k):
            pass

        async def save(self, **kwargs):
            events.append(kwargs["phase"])

    class Resources:
        def __init__(self, *a, **k):
            pass

        async def enter(self, step):
            events.append("enter")
            return []

        async def restore(self):
            events.append("restore")

    class Runtime:
        version = "frozen"

        def __init__(self, **kwargs):
            pass

    class Search:
        state = {"events": []}

        def __init__(self, **kwargs):
            pass

        async def run(self):
            events.append("search")
            if failure:
                raise OSError("search interrupted")
            return SimpleNamespace(model_dump=lambda **kwargs: {"cycle_id": 1, "evaluations": []})

    class Metrics:
        def __init__(self, **kwargs):
            pass

        def finish(self, state):
            events.append("metrics")

    for name, value in dict(
        CheckpointLedger=Ledger,
        GepaResourcePhase=Resources,
        SearchRuntime=Runtime,
        Search=Search,
        SearchMetrics=Metrics,
    ).items():
        monkeypatch.setattr(driver, name, value)
    monkeypatch.setattr(driver, "load_tokenizer", lambda *a, **k: None)
    handler = driver.PhaseHandler(a, manager=None, actor=actor, inference=None, executor=Executor())
    if failure:
        with pytest.raises(OSError, match="interrupted"):
            await handler.run(2)
        assert "resume" not in events and "post_gepa" not in events and "restore" in events
    else:
        await handler.run(2)
        assert events == [
            "freeze",
            "onload",
            "pre_gepa",
            "enter",
            "search",
            "commit",
            "restore",
            "metrics",
            "onload",
            "post_gepa",
            "resume",
        ]


def test_fresh_native_reference_fallback_is_not_a_baseline_resume(tmp_path):
    a = args(tmp_path)
    original_load = a.load
    from pathlib import Path

    (Path(a.hf_checkpoint) / "latest_checkpointed_iteration.txt").write_text("1")
    a.load = a.hf_checkpoint
    a.requested_load = original_load
    a.finetune = True
    assert driver.prepare_resume(a).latest() is None
    assert a.load == a.hf_checkpoint and a.finetune
