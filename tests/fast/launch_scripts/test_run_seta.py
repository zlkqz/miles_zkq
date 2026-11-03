"""CPU checks for SETA driver selection and global multi-node GPU allocation."""

import json
import os
import shlex
import subprocess
import sys

import pytest

from tests.fast.launch_scripts.py_harness import freeze_environment, import_launch_script, install_command_recorder
from tests.fast.launch_scripts.sh_harness import REPO_ROOT


@pytest.fixture
def launcher():
    return import_launch_script(REPO_ROOT / "scripts/run_qwen3_8b_seta.py")


def _flag(argv, flag):
    return argv[argv.index(flag) + 1]


@pytest.mark.parametrize(
    ("overrides", "actor_nodes", "actor_gpus", "infer_gpus"),
    [
        ({}, 1, 4, 4),
        ({"num_nodes": 2}, 1, 8, 8),
        ({"num_nodes": 3}, 1, 8, 16),
        ({"num_nodes": 4}, 2, 8, 16),
        ({"train_num_gpus": 6, "infer_num_gpus": 2, "rollout_batch_size": 6}, 1, 6, 2),
        ({"num_nodes": 3, "train_num_gpus": 16, "infer_num_gpus": 8}, 2, 8, 8),
        ({"num_nodes": 3, "infer_num_gpus": 8}, 2, 8, 8),
        ({"num_nodes": 3, "train_num_gpus": 8}, 1, 8, 16),
        ({"num_nodes": 3, "train_num_gpus": 8, "infer_num_gpus": 8}, 1, 8, 8),
        ({"num_nodes": 2, "train_num_gpus": 4, "infer_num_gpus": 12}, 1, 4, 12),
        ({"num_nodes": 3, "rollout_gpus_per_engine": 16}, 1, 8, 16),
    ],
)
def test_async_driver_reserves_global_counts_without_overallocating(
    launcher, monkeypatch, overrides, actor_nodes, actor_gpus, infer_gpus
):
    freeze_environment(monkeypatch)
    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    recording = install_command_recorder(monkeypatch)
    args = launcher.ScriptArgs(**overrides)
    launcher.execute(args)
    (command,) = [command for command in recording.commands if "ray job submit" in command]
    argv = shlex.split(command)
    assert any(word.endswith("/train_async.py") for word in argv)
    assert "--fully-async" in argv and "--colocate" not in argv
    assert int(_flag(argv, "--actor-num-nodes")) == actor_nodes
    assert int(_flag(argv, "--actor-num-gpus-per-node")) == actor_gpus
    assert int(_flag(argv, "--rollout-num-gpus")) == infer_gpus
    assert actor_nodes * actor_gpus + infer_gpus <= args.num_nodes * args.num_gpus_per_node
    assert _flag(argv, "--pause-generation-mode") == "in_place"


@pytest.mark.parametrize("num_nodes", [1, 2, 4])
@pytest.mark.parametrize("mode,training_mode", [("train", "sync"), ("rollout", "async"), ("rollout", "sync")])
def test_sync_and_rollout_use_all_gpus_with_the_sync_driver(launcher, monkeypatch, num_nodes, mode, training_mode):
    freeze_environment(monkeypatch)
    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    recording = install_command_recorder(monkeypatch)
    # Explicit async counts do not reserve extra resources in shared-GPU modes.
    args = launcher.ScriptArgs(
        num_nodes=num_nodes, mode=mode, training_mode=training_mode, train_num_gpus=4, infer_num_gpus=4
    )
    launcher.execute(args)
    (command,) = [command for command in recording.commands if "ray job submit" in command]
    argv = shlex.split(command)
    assert any(word.endswith("/train.py") for word in argv)
    assert "--colocate" in argv and "--fully-async" not in argv and "--rollout-num-gpus" not in argv
    assert int(_flag(argv, "--actor-num-nodes")) == num_nodes
    assert int(_flag(argv, "--actor-num-gpus-per-node")) == 8
    assert ("--debug-rollout-only" in argv) == (mode == "rollout")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"training_mode": "invalid"}, "training-mode"),
        ({"num_nodes": 0}, "num-nodes"),
        ({"num_gpus_per_node": 0}, "num-gpus-per-node"),
        ({"train_num_gpus": -1}, ">= 0"),
        ({"infer_num_gpus": -1}, ">= 0"),
        ({"train_num_gpus": 8}, "总卡数"),
        ({"train_num_gpus": 4, "infer_num_gpus": 8}, "总卡数"),
        ({"num_nodes": 2, "train_num_gpus": 12}, "整数倍"),
        ({"num_nodes": 2, "train_num_gpus": 8, "infer_num_gpus": 4}, "节点边界"),
        ({"train_num_gpus": 6, "tensor_parallel_size": 4}, "tensor-parallel-size"),
        ({"num_nodes": 4, "train_num_gpus": 24, "tensor_parallel_size": 4}, "数据并行度"),
        ({"rollout_gpus_per_engine": 0}, "推理总卡数"),
        ({"train_num_gpus": 4, "infer_num_gpus": 2, "rollout_gpus_per_engine": 4}, "推理总卡数"),
        ({"num_nodes": 2, "train_num_gpus": 4, "infer_num_gpus": 12, "rollout_gpus_per_engine": 6}, "整数个节点"),
        ({"num_nodes": 2, "train_num_gpus": 4, "infer_num_gpus": 8, "rollout_gpus_per_engine": 8}, "节点边界"),
    ],
)
def test_invalid_topologies_fail_before_any_ray_submission(launcher, overrides, message):
    with pytest.raises(ValueError, match=message):
        launcher.ScriptArgs(**overrides)


def test_multinode_requires_a_prejoined_ray_cluster(launcher, monkeypatch):
    monkeypatch.delenv("MILES_SCRIPT_EXTERNAL_RAY", raising=False)
    with pytest.raises(ValueError, match="MILES_SCRIPT_EXTERNAL_RAY=1"):
        launcher.execute(launcher.ScriptArgs(num_nodes=2))


@pytest.mark.parametrize(
    ("settings", "driver", "actor_nodes", "actor_gpus", "infer_gpus"),
    [
        ({}, "train_async.py", 1, 4, 4),
        ({"TRAINING_MODE": "sync"}, "train.py", 1, 8, 8),
        ({"NUM_NODES": "2", "TRAIN_NUM_GPUS": "8", "INFER_NUM_GPUS": "8"}, "train_async.py", 1, 8, 8),
        ({"NUM_NODES": "3", "TRAIN_NUM_GPUS": "16", "INFER_NUM_GPUS": "8"}, "train_async.py", 2, 8, 8),
    ],
)
def test_shell_plan_passes_mode_and_global_counts_without_starting_a_job(
    tmp_path, settings, driver, actor_nodes, actor_gpus, infer_gpus
):
    env = {key: value for key, value in os.environ.items() if not key.startswith("MILES_SCRIPT_")}
    env.update(
        MODE="plan",
        PYTHON_BIN=sys.executable,
        WANDB_MODE="disabled",
        RUN_ID="cpu-test",
        RUNTIME_DIR=str(tmp_path),
        OUTPUT_DIR=str(tmp_path / "output"),
        TRAINING_MODE="async",
        NUM_NODES="1",
        NUM_GPUS_PER_NODE="8",
        TRAIN_NUM_GPUS="0",
        INFER_NUM_GPUS="0",
        TENSOR_PARALLEL_SIZE="4",
        ROLLOUT_GPUS_PER_ENGINE="2",
        EXTRA_ARGS="",
        EXTRA_ENV_VARS="{}",
    )
    env.update(settings)
    result = subprocess.run(
        ["bash", "my_scripts/run_seta.sh"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    start = result.stdout.index('{\n  "train_script"')
    plan, _ = json.JSONDecoder().raw_decode(result.stdout[start:])
    assert plan["train_script"] == driver
    assert plan["gpu_layout"]["actor_num_nodes"] == actor_nodes
    assert plan["gpu_layout"]["actor_num_gpus_per_node"] == actor_gpus
    assert plan["gpu_layout"]["infer_num_gpus"] == infer_gpus
    assert not (tmp_path / "remote/active_run.json").exists()
