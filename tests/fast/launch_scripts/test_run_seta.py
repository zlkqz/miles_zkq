"""CPU checks for SETA driver selection and global multi-node GPU allocation."""

import argparse
import json
import os
import shlex
import subprocess
import sys

import pytest
from tests.fast.launch_scripts.py_harness import freeze_environment, import_launch_script, install_command_recorder
from tests.fast.launch_scripts.sh_harness import REPO_ROOT

from miles.backends.megatron_utils.megatron_config import get_megatron_arg_parser
from miles.backends.training_utils.loss.hub.corrections import icepop_function
from miles.utils.arguments import get_miles_extra_args_provider, miles_validate_args
from miles.utils.function_registry import load_function


@pytest.fixture
def launcher():
    return import_launch_script(REPO_ROOT / "scripts/run_qwen3_8b_seta.py")


def _flag(argv, flag):
    return argv[argv.index(flag) + 1]


def _parse_training_args(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["train", *argv])
    parser = get_miles_extra_args_provider()(argparse.ArgumentParser())
    args, _ = parser.parse_known_args(argv)
    return args


@pytest.mark.parametrize("algorithm", ["grpo", "dapo"])
@pytest.mark.parametrize("training_mode", ["async", "sync"])
def test_baselines_resolve_icepop_and_asymmetric_ppo_clip(launcher, monkeypatch, algorithm, training_mode):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    launcher.execute(launcher.ScriptArgs(algorithm=algorithm, training_mode=training_mode))
    (command,) = [command for command in recording.commands if "ray job submit" in command]
    argv = shlex.split(command)
    driver_index = next(i for i, arg in enumerate(argv) if arg.endswith(("/train.py", "/train_async.py")))
    argv = argv[driver_index + 1 :]
    args = _parse_training_args(monkeypatch, argv)
    assert args.use_session_server == "v2"
    assert args.session_sample_postprocessor_path == "examples.seta.agent.postprocess_session_samples"
    assert callable(load_function(args.session_sample_postprocessor_path, sync_required=True))
    assert args.use_tis
    assert load_function(args.custom_tis_function_path) is icepop_function
    assert (args.tis_clip_low, args.tis_clip) == (0, 2)
    assert (1 - args.eps_clip, 1 + args.eps_clip_high) == pytest.approx((0.8, 1.28))
    assert bool(args.dynamic_sampling_filter_path) == (algorithm == "dapo")


def test_baselines_differ_only_in_reward_filter(launcher):
    grpo = shlex.split(launcher._train_args(launcher.ScriptArgs(algorithm="grpo")))
    dapo = shlex.split(launcher._train_args(launcher.ScriptArgs(algorithm="dapo")))
    index = dapo.index("--dynamic-sampling-filter-path")
    assert dapo[index + 1] == "miles.rollout.filter_hub.common_filters.apply_reward_nonzero_std_filter"
    assert dapo[:index] + dapo[index + 2 :] == grpo
    assert launcher.ScriptArgs().algorithm == "dapo"


@pytest.mark.parametrize("training_mode", ["async", "sync"])
def test_session_v2_passes_full_miles_validation_with_native_cache_defaults(launcher, monkeypatch, training_mode):
    config = launcher.ScriptArgs(training_mode=training_mode, num_rollout=2)
    argv = shlex.split(launcher._train_args(config))
    monkeypatch.setattr(sys, "argv", ["train", *argv])
    args = get_megatron_arg_parser().parse_args(argv)
    args.world_size = config.gpu_layout.actor_num_nodes * config.gpu_layout.actor_num_gpus_per_node
    miles_validate_args(args)
    assert args.use_session_server == "v2"
    assert args.namespaced_radix_cache == (training_mode == "async")


@pytest.mark.parametrize("algorithm", ["grpo", "dapo"])
def test_rollout_debug_does_not_filter_uniform_rewards(launcher, algorithm):
    argv = shlex.split(launcher._train_args(launcher.ScriptArgs(algorithm=algorithm, mode="rollout")))
    assert "--dynamic-sampling-filter-path" not in argv
    assert _flag(argv, "--use-session-server") == "v2"


def test_extra_args_override_clip_defaults(launcher, monkeypatch):
    argv = shlex.split(
        launcher._train_args(
            launcher.ScriptArgs(extra_args="--eps-clip 0.1 --eps-clip-high 0.15 --tis-clip-low 0.5 --tis-clip 3")
        )
    )
    args = _parse_training_args(monkeypatch, argv)
    assert (args.eps_clip, args.eps_clip_high, args.tis_clip_low, args.tis_clip) == (0.1, 0.15, 0.5, 3)
    assert load_function(args.custom_tis_function_path) is icepop_function


@pytest.mark.parametrize(
    "extra_args", ["--dynamic-sampling-filter-path pkg.fn", "--dynamic-sampling-filter-path=pkg.fn"]
)
def test_filter_override_requires_the_algorithm_switch(launcher, extra_args):
    with pytest.raises(ValueError, match="algorithm"):
        launcher.ScriptArgs(extra_args=extra_args)


def test_unknown_algorithm_is_rejected(launcher):
    with pytest.raises(ValueError, match="algorithm"):
        launcher.ScriptArgs(algorithm="unknown")


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
    ("entrypoint", "algorithm"),
    [("run_seta.sh", "dapo"), ("run_seta_grpo.sh", "grpo"), ("run_seta_dapo.sh", "dapo")],
)
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
    tmp_path, settings, driver, actor_nodes, actor_gpus, infer_gpus, entrypoint, algorithm
):
    env = {key: value for key, value in os.environ.items() if not key.startswith("MILES_SCRIPT_")}
    env.update(
        MODE="plan",
        PYTHON_BIN=sys.executable,
        WANDB_MODE="disabled",
        EXP_NAME="",
        SAVE_PATH="",
        SETA_ALGORITHM="dapo",
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
        ["bash", f"my_scripts/{entrypoint}"],
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
    argv = shlex.split(plan["train_args"])
    assert ("--dynamic-sampling-filter-path" in argv) == (algorithm == "dapo")
    assert f"_{env['TRAINING_MODE']}_{algorithm}_" in _flag(argv, "--save")
    assert not (tmp_path / "remote/active_run.json").exists()


def test_p2o_uses_native_async_topology_and_exact_algorithm_hooks(launcher, monkeypatch):
    config = launcher.ScriptArgs(algorithm="p2o", num_rollout=3)
    argv = shlex.split(launcher._train_args(config))
    args = _parse_training_args(monkeypatch, argv)
    assert config.train_script == "examples/seta/p2o/train.py"
    assert args.rollout_function_path == "examples.seta.p2o.rollout.P2ORolloutFn"
    assert args.custom_async_data_buffer_path == "examples.seta.p2o.sampling.P2ODataBuffer"
    assert args.custom_megatron_actor_scoring_path == "examples.seta.p2o.training.score_contexts"
    assert args.dynamic_sampling_filter_path is None and args.offload_train
    assert (args.gepa_trigger_unique_hard_tasks, args.gepa_dev_task_count, args.gepa_dev_repeats) == (64, 16, 4)
    assert (args.gepa_concurrency, args.gepa_budget, args.gepa_patience) == (16, 256, 32)
    assert (args.gepa_reflection_num_gpus, args.gepa_reflection_tp, args.gepa_policy_tp) == (1, 1, 1)
    assert (args.p2o_task_mix_ratio, args.p2o_max_template_ratio) == (0.25, 0.5)
    assert args.fully_async and args.actor_num_gpus_per_node == 4 and args.rollout_num_gpus == 4
    assert (args.tis_clip_low, args.tis_clip) == (0, 2)


def test_p2o_multinode_keeps_training_layout(launcher):
    config = launcher.ScriptArgs(algorithm="p2o", num_nodes=2, num_gpus_per_node=8)
    assert config.gpu_layout.actor_num_nodes == 1
    assert config.gpu_layout.actor_num_gpus_per_node == 8 and config.gpu_layout.infer_num_gpus == 8


def test_p2o_rejects_sync_and_rollout_only(launcher):
    with pytest.raises(ValueError, match="async"):
        launcher.ScriptArgs(algorithm="p2o", training_mode="sync")
    with pytest.raises(ValueError, match="rollout-only"):
        launcher.ScriptArgs(algorithm="p2o", mode="rollout")


def test_p2o_passes_full_native_validation(launcher, monkeypatch):
    config = launcher.ScriptArgs(algorithm="p2o", num_rollout=3)
    argv = shlex.split(launcher._train_args(config))
    monkeypatch.setattr(sys, "argv", ["train", *argv])
    args = get_megatron_arg_parser().parse_args(argv)
    args.world_size = config.gpu_layout.actor_num_nodes * config.gpu_layout.actor_num_gpus_per_node
    miles_validate_args(args)
    assert args.use_checkpoint_opt_param_scheduler
    assert args.rollout_function_path == "examples.seta.p2o.rollout.P2ORolloutFn"
    assert args.eval_function_path == args.rollout_function_path
    assert args.namespaced_radix_cache and args.fully_async
