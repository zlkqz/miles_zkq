"""Qwen3-8B 非思考模式，在完整 SETA-Env 上使用 Terminus-2 + GRPO，默认纯异步。

运行前需要已转换的 torch_dist 模型、训练 JSONL 和已运行的远端 Harbor 服务。
使用 session server v2，支持纯异步权重更新时的版本隔离 KV cache。
默认仅打印计划；正式训练必须显式选择 --mode train。

Args:
  --mode: plan（打印）、prepare（转换模型）、rollout（只采样）、train（训练）。
  --model-dir / --hf-checkpoint: 模型目录及可选的原始 HF 模型路径。
  --data-dir / --remote-config / --output-dir: 数据、远端服务池配置、输出目录。
  --save-path: checkpoint 保存和恢复的根目录；空值时使用 output-dir/checkpoints。
  --num-nodes / --num-gpus-per-node: 节点与 GPU 数；双机预先加入 Ray。
  --training-mode: async（默认，独立 GPU 持续采样）或 sync（全部 GPU 分时复用）。
  --train-num-gpus / --infer-num-gpus: 异步训练/推理的集群总卡数；0 表示自动分配。
  --algorithm: grpo 关闭同分过滤；dapo 开启（默认）；p2o 在 GRPO 上启用难任务/CRM/GEPA。
  --gepa-trigger-unique-hard-tasks / --gepa-budget: GEPA 难任务触发门槛与探索预算。
  GEPA policy/reflection 均继承训练 rollout engine 的 GPU 数、TP 和 SGLang 配置；反思固定一个实例。
  --num-rollout: 非零时覆盖 --num-epoch；小规模验证应使用该参数。

示例：python scripts/run_qwen3_8b_seta.py --mode plan
"""

import json
import os
import shlex
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import typer

from miles.utils.external_utils import command_utils as U

# 直接以 scripts/file.py 执行时，editable 安装只暴露 miles 包；适配代码位于仓库 examples。
sys.path.insert(0, str(U.repo_base_dir))
from examples.seta.p2o.scoring import CollectionConfig
from examples.seta.p2o.search import SearchConfig
from examples.seta.remote_config import load_endpoints

_COLLECTION_DEFAULTS = CollectionConfig(n_samples_per_prompt=1)
_SEARCH_DEFAULTS = SearchConfig()


@dataclass(frozen=True)
class _GpuLayout:
    actor_num_nodes: int
    actor_num_gpus_per_node: int
    infer_num_gpus: int
    fully_async: bool

    @property
    def train_script(self) -> str:
        return "train_async.py" if self.fully_async else "train.py"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=lambda: U.create_run_id())
    mode: Literal["plan", "prepare", "rollout", "train"] = "plan"
    model_dir: str = "/root/models"
    hf_checkpoint: str = ""
    data_dir: str = "/root/datasets/seta"
    remote_config: str = ""
    save_path: str = ""
    megatron_path: str = "/root/Megatron-LM"
    num_gpus_per_node: int = 8
    algorithm: Literal["grpo", "dapo", "p2o"] = "dapo"
    hard_reward_threshold: float | None = None
    gepa_trigger_unique_hard_tasks: int = _COLLECTION_DEFAULTS.gepa_trigger_unique_hard_tasks
    gepa_dev_task_count: int = _COLLECTION_DEFAULTS.gepa_dev_task_count
    gepa_dev_repeats: int = _COLLECTION_DEFAULTS.gepa_dev_repeats
    gepa_concurrency: int = _SEARCH_DEFAULTS.concurrency
    gepa_budget: int = _SEARCH_DEFAULTS.budget
    gepa_patience: int = _SEARCH_DEFAULTS.patience
    gepa_minibatch: int = _SEARCH_DEFAULTS.minibatch
    gepa_reflection_max_turns: int = _SEARCH_DEFAULTS.reflection_max_turns
    gepa_max_template_tokens: int = _SEARCH_DEFAULTS.max_template_tokens
    training_mode: Literal["async", "sync"] = "async"
    train_num_gpus: int = 0
    infer_num_gpus: int = 0
    tensor_parallel_size: int = 2
    rollout_gpus_per_engine: int = 2
    max_tokens_per_gpu: int = 32768
    attention_backend: Literal["flash", "fused", "unfused", "auto"] = "flash"
    mem_fraction: float = 0.65
    num_epoch: int = 40
    num_rollout: int = 0
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 16
    max_seq_len: int = 28672
    max_response_len: int = 4096
    agent_max_iterations: int = 30
    agent_timeout: int = 900
    trial_timeout: int = 3600
    max_concurrent_trials: int = 8
    record_terminal_session: bool = False
    full_pass_bonus: float = 0.2
    save_interval: int = 50
    lr: float = 1e-6
    weight_decay: float = 0.01
    max_weight_staleness: int = 4
    session_server_port: int = 30000
    session_server_workers: int = 8
    sglang_context_length: int = 32768
    router_port: int = 31000
    session_server_external_host: str = ""
    pin_rollout_manager_to_head: bool = True
    extra_args: str = ""
    gpu_layout: _GpuLayout = field(init=False)

    def __post_init__(self) -> None:
        if self.algorithm not in {"grpo", "dapo", "p2o"}:
            raise ValueError("algorithm 必须为 grpo、dapo 或 p2o")
        if self.algorithm == "p2o" and (self.training_mode != "async" or self.mode == "rollout"):
            raise ValueError("P2O requires async training; use grpo for rollout-only debugging")
        if any(token.split("=", 1)[0] == "--dynamic-sampling-filter-path" for token in shlex.split(self.extra_args)):
            raise ValueError("请用 --algorithm 选择同分过滤，不要通过 extra-args 覆盖 dynamic-sampling-filter-path")
        self.gpu_layout = _resolve_gpu_layout(self)

    @property
    def hf_path(self) -> str:
        return self.hf_checkpoint or f"{self.model_dir}/Qwen3-8B"

    @property
    def global_batch_size(self) -> int:
        return self.rollout_batch_size * self.n_samples_per_prompt

    @property
    def train_script(self) -> str:
        return "examples/seta/p2o/train.py" if self.algorithm == "p2o" else self.gpu_layout.train_script


def _resolve_gpu_layout(args: ScriptArgs) -> _GpuLayout:
    if args.training_mode not in {"async", "sync"}:
        raise ValueError("training-mode 必须为 async 或 sync")
    if args.num_nodes <= 0 or args.num_gpus_per_node <= 0:
        raise ValueError("num-nodes 和 num-gpus-per-node 必须为正整数")
    if args.train_num_gpus < 0 or args.infer_num_gpus < 0:
        raise ValueError("train-num-gpus 和 infer-num-gpus 必须 >= 0；0 表示自动分配")
    capacity = args.num_nodes * args.num_gpus_per_node
    fully_async = args.training_mode == "async" and args.mode not in {"rollout", "prepare"}
    if fully_async:
        train_gpus, infer_gpus = args.train_num_gpus, args.infer_num_gpus
        if train_gpus == 0:
            if infer_gpus:
                train_gpus = capacity - infer_gpus
            else:
                # 多机默认按整节点划分，避免把每节点 4 卡误当成 Ray 的实际分配。
                train_gpus = (
                    args.num_gpus_per_node // 2
                    if args.num_nodes == 1
                    else (args.num_nodes // 2) * args.num_gpus_per_node
                )
        infer_gpus = infer_gpus or capacity - train_gpus
        if train_gpus <= 0 or infer_gpus <= 0 or train_gpus + infer_gpus > capacity:
            raise ValueError("异步 train/infer 总卡数必须均为正数，且之和不能超过 num-nodes × num-gpus-per-node")
        requested_gpus = train_gpus + infer_gpus
        # Ray 先 PACK 再按 IP 排序；末尾半个节点排序到前面会打乱 trainer 的本地 rank。
        if requested_gpus > args.num_gpus_per_node and requested_gpus % args.num_gpus_per_node:
            raise ValueError("跨节点申请的 train+infer 总卡数必须按整节点对齐，以保持训练/推理的节点边界")
        actor_gpus_per_node = min(train_gpus, args.num_gpus_per_node)
        if train_gpus % actor_gpus_per_node:
            raise ValueError("跨机训练的 train-num-gpus 必须为 num-gpus-per-node 的整数倍；单机可使用部分 GPU")
        actor_nodes = train_gpus // actor_gpus_per_node
    else:
        # 同步/仅采样模式由全部 GPU 分时复用；异步专用卡数不参与此模式的分配。
        actor_nodes, actor_gpus_per_node, infer_gpus = args.num_nodes, args.num_gpus_per_node, capacity
    if args.mode not in {"rollout", "prepare"}:
        if args.tensor_parallel_size <= 0 or actor_gpus_per_node % args.tensor_parallel_size:
            raise ValueError("每节点训练 GPU 数必须能被 tensor-parallel-size 整除")
        dp_size = actor_nodes * actor_gpus_per_node // args.tensor_parallel_size
        if args.global_batch_size % dp_size:
            raise ValueError("rollout-batch-size × n-samples-per-prompt 必须能被训练数据并行度整除")
    engine_gpus = args.rollout_gpus_per_engine
    if engine_gpus <= 0 or infer_gpus % engine_gpus:
        raise ValueError("推理总卡数必须能被 rollout-gpus-per-engine 整除")
    if max(engine_gpus, args.num_gpus_per_node) % min(engine_gpus, args.num_gpus_per_node):
        raise ValueError("rollout-gpus-per-engine 必须整除每节点 GPU 数，或使用整数个节点")
    if (
        fully_async
        and requested_gpus > args.num_gpus_per_node
        and train_gpus % min(engine_gpus, args.num_gpus_per_node)
    ):
        raise ValueError("训练卡数使推理引擎跨越节点边界；请将 train-num-gpus 对齐推理引擎或整节点")
    return _GpuLayout(actor_nodes, actor_gpus_per_node, infer_gpus, fully_async)


def _execution_args(layout: _GpuLayout) -> str:
    topology = (
        f"--actor-num-nodes {layout.actor_num_nodes} " f"--actor-num-gpus-per-node {layout.actor_num_gpus_per_node} "
    )
    if layout.fully_async:
        return topology + f"--fully-async --pause-generation-mode in_place --rollout-num-gpus {layout.infer_num_gpus} "
    return topology + "--colocate "


def _train_args(args: ScriptArgs) -> str:
    save_path = args.save_path or args.output_dir + "/checkpoints"
    checkpoint = f"--hf-checkpoint {shlex.quote(args.hf_path)} --ref-load {shlex.quote(args.model_dir + '/Qwen3-8B_torch_dist')} --load {shlex.quote(save_path)} "
    if args.mode != "rollout":
        checkpoint += f"--save {shlex.quote(save_path)} --save-interval {args.save_interval} "
    count = f"--num-rollout {args.num_rollout} " if args.num_rollout else f"--num-epoch {args.num_epoch} "
    rollout = (
        f"--prompt-data {shlex.quote(args.data_dir + '/train.jsonl')} --input-key prompt --metadata-key metadata "
        f"{count}--rollout-shuffle --rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} --global-batch-size {args.global_batch_size} "
        f"--max-weight-staleness {args.max_weight_staleness} "
        f"--rollout-max-response-len {args.max_response_len} --max-seq-len {args.max_seq_len} "
        "--rollout-temperature 1 --rollout-top-p 1 --balance-data "
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path examples.seta.agent.run --custom-rm-path examples.seta.agent.reward_func "
        "--use-session-server v2 --tito-model qwen3 "
        "--session-sample-postprocessor-path examples.seta.agent.postprocess_session_samples "
        f"--session-server-workers {args.session_server_workers} --session-server-port {args.session_server_port} "
        "--apply-chat-template-kwargs '{\"enable_thinking\":false}' "
    )
    # 仅采样验证不筛选同分组，避免小样本调试因缺少奖励方差不断补采。
    if args.algorithm == "dapo" and args.mode != "rollout":
        rollout += (
            "--dynamic-sampling-filter-path miles.rollout.filter_hub.common_filters.apply_reward_nonzero_std_filter "
        )
    if args.session_server_external_host:
        rollout += f"--session-server-external-host {shlex.quote(args.session_server_external_host)} "
    if args.pin_rollout_manager_to_head:
        rollout += "--pin-rollout-manager-to-head "
    perf = (
        f"--tensor-model-parallel-size {args.tensor_parallel_size} --sequence-parallel "
        "--pipeline-model-parallel-size 1 --context-parallel-size 1 --expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 --recompute-granularity full --recompute-method uniform "
        "--recompute-num-layers 1 --use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
    )
    algorithm = (
        "--advantage-estimator grpo --eps-clip 0.2 --eps-clip-high 0.28 "
        "--kl-coef 0 --kl-loss-coef 0 --entropy-coef 0 --use-tis "
        "--custom-tis-function-path miles.backends.training_utils.loss.hub.corrections.icepop_function "
    )
    optimizer = f"--optimizer adam --lr {args.lr} --lr-decay-style constant --lr-warmup-fraction 0 --weight-decay {args.weight_decay} --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-8 --clip-grad 1 "
    sglang = f"--rollout-num-gpus-per-engine {args.rollout_gpus_per_engine} --sglang-mem-fraction-static {args.mem_fraction} --sglang-attention-backend triton --sglang-context-length {args.sglang_context_length} --sglang-router-port {args.router_port} "
    misc = (
        "--seed 1 --bf16 --attention-dropout 0 --hidden-dropout 0 "
        f"--attention-backend {args.attention_backend} "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 "
        f"--num-gpus-per-node {args.num_gpus_per_node} --dump-details {shlex.quote(args.output_dir + '/dump_details')} "
    )
    debug = "--debug-rollout-only " if args.mode == "rollout" else ""
    return (
        checkpoint
        + rollout
        + perf
        + algorithm
        + optimizer
        + sglang
        + misc
        + _execution_args(args.gpu_layout)
        + _p2o_args(args)
        + debug
        + args.extra_args
    ).strip()


def _p2o_args(args: ScriptArgs) -> str:
    if args.algorithm != "p2o":
        return ""
    values = {name: value for name, value in vars(args).items() if name.startswith("gepa_")}
    if args.hard_reward_threshold is not None:
        values["hard_reward_threshold"] = args.hard_reward_threshold
    return (
        "--offload-train --rollout-function-path examples.seta.p2o.rollout.P2ORolloutFn "
        "--use-checkpoint-opt-param-scheduler "
        "--custom-async-data-buffer-path examples.seta.p2o.sampling.P2ODataBuffer "
        "--custom-megatron-actor-scoring-path examples.seta.p2o.training.score_contexts "
        + " ".join(f"--{key.replace('_', '-')} {value}" for key, value in values.items())
        + " "
    )


def _worker_env(args: ScriptArgs) -> dict[str, str]:
    env = {
        "PYTHONPATH": f"{U.repo_base_dir}:{args.megatron_path}",
        "SETA_REMOTE_CONFIG": args.remote_config,
        "SETA_RUN_ID": args.run_id,
        "SETA_FULL_PASS_BONUS": str(args.full_pass_bonus),
        "SETA_MAX_CONCURRENT_TRIALS": str(args.max_concurrent_trials),
        "SETA_RECORD_TERMINAL_SESSION": str(args.record_terminal_session).lower(),
        "HARBOR_AGENT_MAX_ITERATIONS": str(args.agent_max_iterations),
        "HARBOR_MAX_SEQ_LEN": str(args.max_seq_len),
        "AGENT_MAX_INPUT_TOKENS": str(args.max_seq_len),
        "AGENT_MAX_OUTPUT_TOKENS": str(args.max_response_len),
        "AGENT_MODEL_NAME": "Qwen3-8B",
        "AGENT_TIMEOUT": str(args.agent_timeout),
        "AGENT_TRIAL_TIMEOUT": str(args.trial_timeout),
        "HARBOR_TIMEOUT_MULTIPLIER": "1.0",
        "HARBOR_VERIFIER_TIMEOUT_SEC": "600",
        "HARBOR_RESPONSE_LENGTH_POLICY": "abort",
        "HARBOR_TERMINUS_2_ENABLE_SUMMARIZE": "false",
        "HARBOR_TERMINUS_2_LINEAR_HISTORY": "true",
    }
    return env


def prepare(args: ScriptArgs):
    args.create_backend().convert_checkpoint(
        model_name="Qwen3-8B",
        megatron_model_type="qwen3-8B",
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=args.hf_path,
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    if (
        args.num_nodes > 1
        and args.cluster_backend.value == "ray"
        and os.environ.get("MILES_SCRIPT_EXTERNAL_RAY") != "1"
    ):
        raise ValueError("多机运行需先将所有节点加入 Ray，再设置 MILES_SCRIPT_EXTERNAL_RAY=1；只在 head 启动本脚本")
    wandb = U.get_default_wandb_args(__file__, run_id=args.run_id)
    args.create_backend().execute_train(
        train_args=_train_args(args) + " " + wandb,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="qwen3-8B",
        train_script=args.train_script,
        megatron_path=args.megatron_path,
        extra_env_vars=_worker_env(args),
        # Ctrl+C/TERM 必须停止 Ray 作业，而不只是断开日志跟随客户端。
        job_lifetime="launcher",
    )


def _preflight(args: ScriptArgs) -> None:
    for value in (args.hf_path, args.remote_config, args.data_dir + "/train.jsonl"):
        if not Path(value).exists():
            raise FileNotFoundError(value)
    _check_remote_services(args.remote_config)
    if args.mode == "train":
        if args.n_samples_per_prompt < 2:
            raise ValueError("GRPO 训练至少需要两条同任务轨迹")
        tracker = Path(args.model_dir) / "Qwen3-8B_torch_dist/latest_checkpointed_iteration.txt"
        if not tracker.is_file():
            raise FileNotFoundError(f"先用 --mode prepare 转换模型：{tracker}")
    with (Path(args.data_dir) / "train.jsonl").open() as file:
        for row in map(json.loads, file):
            if row["metadata"].get("agent_name") != "terminus-2":
                raise ValueError("请重新转换数据，指定 --agent-name terminus-2")


def _check_remote_services(config_path: str) -> None:
    # 在预检查阶段导入网络客户端，计划/快照生成不进行任何网络访问。
    import httpx

    with httpx.Client(timeout=10, trust_env=False) as client:
        for endpoint in load_endpoints(config_path):
            response = client.get(f"{endpoint.url}/health", headers=endpoint.headers())
            response.raise_for_status()
            health = response.json()
            if health.get("protocol") != "harbor-agent-server-v1":
                raise ValueError(f"不是 SETA 远端服务：{endpoint.url}")
            if endpoint.max_concurrent > health["max_concurrent"]:
                raise ValueError(f"客户端并发超过远端上限：{endpoint.url}")


@U.dataclass_cli
def main(args: ScriptArgs):
    if args.mode == "plan":
        print(
            json.dumps(
                {
                    "train_script": args.train_script,
                    "gpu_layout": asdict(args.gpu_layout),
                    "train_args": _train_args(args),
                    "worker_env": _worker_env(args),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.mode == "prepare":
        prepare(args)
    else:
        _preflight(args)
        execute(args)


if __name__ == "__main__":
    typer.run(main)
