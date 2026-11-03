#!/usr/bin/env bash
# SETA + Qwen3-8B + Terminus-2，默认纯异步；通过 HTTP 使用远端 Harbor 沙盒。
# 默认正式训练并启用 W&B；只查看配置：MODE=plan bash my_scripts/run_seta.sh
# 外部 export 覆盖下列默认值；更多 Python 启动参数也可作为脚本参数追加。
set -euo pipefail
set +x
# 长任务忽略终端挂断；此设置也由随后启动的 Ray 控制进程继承。
trap '' HUP

# ── 运行入口 ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REPO_DIR="${REPO_DIR:-$(dirname -- "$SCRIPT_DIR")}"
export MODE="${MODE:-train}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONUNBUFFERED=1
cd "$REPO_DIR"

# ── 数据、模型与输出路径 ────────────────────────────────────────────────────
export RUNTIME_DIR="${RUNTIME_DIR:-$REPO_DIR/seta_miles_runtime}"
export MODEL_DIR="${MODEL_DIR:-$RUNTIME_DIR/models}"
export HF_MODEL="${HF_MODEL:-/ceph_home/arknet/hf_models/Qwen/Qwen3-8B}"
export DATA_DIR="${DATA_DIR:-/ceph_home/zhangkaiqi2024/seta_miles_data}"
export REMOTE_CONFIG="${REMOTE_CONFIG:-$RUNTIME_DIR/remote/endpoints.json}"

# ── GPU 与并行 ──────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_NODES="${NUM_NODES:-1}"
export NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
# async：训练/推理独占不同卡；sync：全部 GPU 分时复用，忽略下面两个异步卡数。
export TRAINING_MODE="${TRAINING_MODE:-async}"
# 集群总卡数，不是每节点卡数；0 由 Python 启动器自动分配。
# 全为 0：单机默认 4+4，多机按整节点划分，训练占 floor(NUM_NODES/2) 台。
# 只指定一个：另一个使用剩余卡；也可显式指定两个数，允许留卡不用。
# 例：两台各 8 卡，TRAIN_NUM_GPUS=8 INFER_NUM_GPUS=8；三台可设 16+8。
export TRAIN_NUM_GPUS="${TRAIN_NUM_GPUS:-0}"
export INFER_NUM_GPUS="${INFER_NUM_GPUS:-0}"
# 多机：所有 worker 先加入 head 的 Ray；仅在 head 启动本脚本并设置此值为 1。
export MILES_SCRIPT_EXTERNAL_RAY="${MILES_SCRIPT_EXTERNAL_RAY:-0}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
export ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-2}"
# rollout 协调器固定在 head，endpoint/token 只需放在 head 的训练容器。
export PIN_ROLLOUT_MANAGER_TO_HEAD="${PIN_ROLLOUT_MANAGER_TO_HEAD:-1}"

# ── 当前 A100 PCIe 主机的通信与 NCCL 库 ───────────────────────────────────────
# 仅允许 NVLink P2P；避免当前 IOMMU 配置下未经验证的 PCIe P2P。
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export SGLANG_DISABLE_CUSTOM_ALL_REDUCE="${SGLANG_DISABLE_CUSTOM_ALL_REDUCE:-1}"
# tools-v1 同时含系统与 PyTorch NCCL，优先后者，避免 TE 导入顺序导致符号冲突。
nccl_library_dir="$("$PYTHON_BIN" - <<'PY_NCCL'
import importlib.util
from pathlib import Path
spec = importlib.util.find_spec('nvidia.nccl')
print(str(Path(next(iter(spec.submodule_search_locations))) / 'lib') if spec else '')
PY_NCCL
)"
if [[ -d $nccl_library_dir ]]; then
  export LD_LIBRARY_PATH="$nccl_library_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi


# ── 采样预算与任务并发 ──────────────────────────────────────────────────────
export NUM_EPOCH="${NUM_EPOCH:-40}"
# 0：由 NUM_EPOCH 换算总批次；正整数：直接指定总 rollout 批次数。
export NUM_ROLLOUT="${NUM_ROLLOUT:-0}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
# 同时运行的 Harbor episode 数；不是任务组数，也不是每批总轨迹数。
export MAX_CONCURRENT_TRIALS="${MAX_CONCURRENT_TRIALS:-32}"

# ── Agent 轮数、序列长度与超时 ──────────────────────────────────────────────
export AGENT_MAX_ITERATIONS="${AGENT_MAX_ITERATIONS:-30}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-4096}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-28672}"
export AGENT_TIMEOUT="${AGENT_TIMEOUT:-900}"
export TRIAL_TIMEOUT="${TRIAL_TIMEOUT:-3600}"

# ── 训练与推理的显存预算 ────────────────────────────────────────────────────
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"
# A100 需按 Quick Start 移除不支持其反向的 FA4，保留镜像已有的 FA2。
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"
export MEM_FRACTION="${MEM_FRACTION:-0.65}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-32768}"

# ── 奖励、优化与保存 ────────────────────────────────────────────────────────
export FULL_PASS_BONUS="${FULL_PASS_BONUS:-0.2}"
export LR="${LR:-1.7e-5}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-30}"

# ── 模型服务与端口 ──────────────────────────────────────────────────────────
export GPU_HOST="${GPU_HOST:-124.16.138.150}"
export MASTER_ADDR="${MASTER_ADDR:-$GPU_HOST}"
export SESSION_SERVER_WORKERS="${SESSION_SERVER_WORKERS:-8}"
export SESSION_SERVER_PORT="${SESSION_SERVER_PORT:-30000}"
export ROUTER_PORT="${ROUTER_PORT:-31000}"
export SESSION_SERVER_EXTERNAL_HOST="${SESSION_SERVER_EXTERNAL_HOST:-$GPU_HOST}"

# ── 实验名、checkpoint 与日志输出 ───────────────────────────────────────────
# 实验名和 checkpoint 路径不含时间戳，相同配置重跑时自动读取同一目录。
export EXP_NAME="${EXP_NAME:-qwen3_8b_seta_remote_${TRAINING_MODE}_grpo_${ROLLOUT_BATCH_SIZE}x${N_SAMPLES_PER_PROMPT}_len${MAX_SEQ_LEN}+${MAX_RESPONSE_LEN}_turn${AGENT_MAX_ITERATIONS}_lr${LR}}"
export SAVE_PATH="${SAVE_PATH:-/ceph_home/zhangkaiqi2024/saved_models/p2o-agent/$EXP_NAME}"
export OUTPUT_DIR="${OUTPUT_DIR:-$RUNTIME_DIR/runs/$EXP_NAME}"

# ── W&B 与运行标识 ──────────────────────────────────────────────────────────
# key 直接配置在此，外部 export WANDB_API_KEY 仍可覆盖。
# RUN_ID 只区分每次启动的日志和凭据文件，不参与实验名和 checkpoint 路径。
export RUN_ID="${RUN_ID:-$("$PYTHON_BIN" -c 'from miles.utils.external_utils import command_utils as U; print(U.create_run_id())')}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_PROJECT="${WANDB_PROJECT:-miles_seta_p2o}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-$EXP_NAME}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"

# ── 代理 ────────────────────────────────────────────────────────────────────
# 任务沙盒代理由远端服务配置；这里仅控制训练进程/W&B 网络。
# 外部 WANDB_PROXY="" 表示直连。
export WANDB_PROXY="${WANDB_PROXY-http://124.16.138.150:7890}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,127.0.1.1,::1,0.0.0.0,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,169.254.0.0/16,124.16.138.150,124.16.138.144,.local,.lan,.internal,.cluster.local}"
export no_proxy="$NO_PROXY"

# ── 其他 Miles 参数与 worker 环境 ────────────────────────────────────────────
export EXTRA_ARGS="${EXTRA_ARGS:-}"
export EXTRA_ENV_VARS="${EXTRA_ENV_VARS:-}"
if [[ -z $EXTRA_ENV_VARS ]]; then
  EXTRA_ENV_VARS='{}'
fi

# ── 准备本次运行的日志与配置 ────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR/logs"
LOG_PREFIX="${EXP_NAME}_${RUN_ID}"
LOG_FILE="$OUTPUT_DIR/logs/$LOG_PREFIX.log"

# W&B 参数在 Bash 中配置；Python 只处理凭据文件和需要可靠转义的配置文本。
wandb_args=()
if [[ $WANDB_MODE != disabled ]]; then
  wandb_args=(
    --use-wandb --wandb-project "$WANDB_PROJECT"
    --wandb-group "$WANDB_RUN_NAME" --wandb-mode "$WANDB_MODE"
    --wandb-dir "$OUTPUT_DIR/wandb" --disable-wandb-random-suffix
  )
fi
export WORKER_ENV_FILE="$OUTPUT_DIR/logs/$LOG_PREFIX.worker-env.json"
export MILES_EXTRA_ARGS_FILE="$OUTPUT_DIR/logs/$LOG_PREFIX.extra-args.txt"
export RUN_STATE_FILE="$OUTPUT_DIR/logs/$LOG_PREFIX.run.json"
if [[ $WANDB_MODE == online ]]; then
  export NETRC="$RUNTIME_DIR/secrets/$RUN_ID.netrc"
fi

runtime_args=("${wandb_args[@]}")
if [[ $SGLANG_DISABLE_CUSTOM_ALL_REDUCE == 1 ]]; then
  runtime_args+=(--sglang-disable-custom-all-reduce)
fi
"$PYTHON_BIN" - "${runtime_args[@]}" <<'PY'
import json
import os
import shlex
import sys
from pathlib import Path
from urllib.parse import urlsplit

env = os.environ
if env['MODE'] not in {'plan', 'prepare', 'rollout', 'train'}:
    raise SystemExit('MODE 只能是 plan、prepare、rollout 或 train')
if env['WANDB_MODE'] not in {'online', 'offline', 'disabled'}:
    raise SystemExit('WANDB_MODE 只能是 online、offline 或 disabled')

if env['WANDB_MODE'] == 'online':
    credentials = Path(env['NETRC'])
    key = env.get('WANDB_API_KEY', '').strip()
    if not key or any(c.isspace() for c in key):
        raise SystemExit('W&B API key 为空或含空白字符')
    host = urlsplit(env['WANDB_BASE_URL']).hostname
    if not host:
        raise SystemExit('WANDB_BASE_URL 需要包含 https:// 或 http://')
    credentials.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(credentials, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as file:
        file.write(f'machine {host}\n  login user\n  password {key}\n')
    credentials.chmod(0o600)
worker_env = json.loads(env['EXTRA_ENV_VARS'])
if not isinstance(worker_env, dict) or not all(isinstance(v, str) for v in worker_env.values()):
    raise SystemExit('EXTRA_ENV_VARS 必须是字符串值的 JSON 对象')
if 'WANDB_API_KEY' in worker_env:
    raise SystemExit('请用 WANDB_API_KEY 提供 key，不要放进 EXTRA_ENV_VARS')
worker_env.update(WANDB_BASE_URL=env['WANDB_BASE_URL'], NO_PROXY=env['NO_PROXY'], no_proxy=env['NO_PROXY'])
for name in ('NCCL_P2P_DISABLE', 'NCCL_P2P_LEVEL', 'NCCL_IB_DISABLE', 'LD_LIBRARY_PATH'):
    if name in env:
        worker_env[name] = env[name]
if env['WANDB_MODE'] == 'online':
    worker_env['NETRC'] = env['NETRC']
for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    worker_env[name] = env['WANDB_PROXY']
# 显式 W&B 参数优先于 EXTRA_ARGS 中的同名字段，保持本脚本的记录配置一致。
Path(env['WORKER_ENV_FILE']).write_text(json.dumps(worker_env))
Path(env['MILES_EXTRA_ARGS_FILE']).write_text(shlex.join(shlex.split(env['EXTRA_ARGS']) + sys.argv[1:]))
if env['MODE'] in {'train', 'rollout'}:
    marker = Path(env['RUNTIME_DIR']) / 'remote/active_run.json'
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix('.tmp')
    record = json.dumps({'run_id': env['RUN_ID'], 'exp_name': env['EXP_NAME'],
                         'remote_config': env['REMOTE_CONFIG'], 'output_dir': env['OUTPUT_DIR'],
                         'ray_submission_id': env['RUN_ID'],
                         'ray_address': env.get('RAY_ADDRESS', 'http://127.0.0.1:8265')})
    Path(env['RUN_STATE_FILE']).write_text(record)
    temporary.write_text(record)
    temporary.replace(marker)
PY

# SDK 从 NETRC 认证，避免默认 helper 将 API key 拼进训练命令和日志。
unset WANDB_API_KEY
export HTTP_PROXY="$WANDB_PROXY"
export HTTPS_PROXY="$WANDB_PROXY"
export http_proxy="$WANDB_PROXY"
export https_proxy="$WANDB_PROXY"
WORKER_ENV_JSON="$(<"$WORKER_ENV_FILE")"
MILES_EXTRA_ARGS="$(<"$MILES_EXTRA_ARGS_FILE")"

# 日志进程留在外层；启动器独立进程组，Ctrl+C/TERM 由外层转发并触发清理。
# 后台 wait 可被信号打断，避免 Bash 等整个前台流水线结束后才执行 trap。
exec {log_fd}> >(exec setsid tee --output-error=warn "$LOG_FILE")
logger_pid=$!
launcher_pid=""
_finish_seta() {
  local status="$1"
  local watchdog_pid=""
  local cleanup_status=0
  local logger_status=0
  trap - EXIT INT TERM
  set +e
  if [[ -n $launcher_pid ]] && kill -0 "$launcher_pid" 2>/dev/null; then
    kill -TERM -- "-$launcher_pid" 2>/dev/null || kill -TERM "$launcher_pid" 2>/dev/null
    # 给原生 Ray job stop 留出时间，但不让启动器无限阻塞退出。
    "$PYTHON_BIN" -c 'import os, signal, sys, time; time.sleep(90); os.killpg(int(sys.argv[1]), signal.SIGKILL)' "$launcher_pid" {log_fd}>&- >/dev/null 2>&1 &
    watchdog_pid=$!
  fi
  if [[ $MODE == train || $MODE == rollout ]]; then
    printf '\n训练退出，停止本次 Ray 作业并请求远端回收……\n' >&"$log_fd"
    "$PYTHON_BIN" "$REPO_DIR/my_scripts/stop_seta.py" --run-file "$RUN_STATE_FILE" >&"$log_fd" 2>&1
    cleanup_status=$?
  fi
  if [[ -n $launcher_pid ]]; then
    wait "$launcher_pid" 2>/dev/null
  fi
  if [[ -n $watchdog_pid ]]; then
    kill "$watchdog_pid" 2>/dev/null
    wait "$watchdog_pid" 2>/dev/null
  fi
  printf '\n启动器退出码：%s；远端清理命令退出码：%s\n' "$status" "$cleanup_status" >&"$log_fd"
  exec {log_fd}>&-
  wait "$logger_pid"
  logger_status=$?
  printf '日志转发退出码：%s\n' "$logger_status" >> "$LOG_FILE"
  if [[ $status == 0 && $cleanup_status != 0 ]]; then
    status="$cleanup_status"
  fi
  exit "$status"
}
trap '_finish_seta "$?"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf '模式：%s；节点：%s；每节点 GPU：%s\n' "$MODE" "$NUM_NODES" "$NUM_GPUS_PER_NODE" >&"$log_fd"
printf '训练调度：%s；异步总卡数 train=%s / infer=%s（0 表示自动）\n' "$TRAINING_MODE" "$TRAIN_NUM_GPUS" "$INFER_NUM_GPUS" >&"$log_fd"
printf '实验名：%s；checkpoint：%s\n' "$EXP_NAME" "$SAVE_PATH" >&"$log_fd"
printf 'W&B：%s / %s / %s\n' "$WANDB_MODE" "$WANDB_PROJECT" "$WANDB_RUN_NAME" >&"$log_fd"
printf '输出目录：%s；终端日志：%s\n' "$OUTPUT_DIR" "$LOG_FILE" >&"$log_fd"

_start_seta() {
  pin_args=(--pin-rollout-manager-to-head)
  if [[ $PIN_ROLLOUT_MANAGER_TO_HEAD == 0 ]]; then
    pin_args=(--no-pin-rollout-manager-to-head)
  fi
  exec setsid "$PYTHON_BIN" "$REPO_DIR/scripts/run_qwen3_8b_seta.py" \
    --mode "$MODE" "${pin_args[@]}" \
    --model-dir "$MODEL_DIR" --hf-checkpoint "$HF_MODEL" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" --save-path "$SAVE_PATH" \
    --num-nodes "$NUM_NODES" --num-gpus-per-node "$NUM_GPUS_PER_NODE" \
    --training-mode "$TRAINING_MODE" --train-num-gpus "$TRAIN_NUM_GPUS" --infer-num-gpus "$INFER_NUM_GPUS" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --rollout-gpus-per-engine "$ROLLOUT_GPUS_PER_ENGINE" \
    --num-epoch "$NUM_EPOCH" --num-rollout "$NUM_ROLLOUT" \
    --rollout-batch-size "$ROLLOUT_BATCH_SIZE" --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT" \
    --max-concurrent-trials "$MAX_CONCURRENT_TRIALS" --agent-max-iterations "$AGENT_MAX_ITERATIONS" \
    --max-response-len "$MAX_RESPONSE_LEN" --max-seq-len "$MAX_SEQ_LEN" \
    --agent-timeout "$AGENT_TIMEOUT" --trial-timeout "$TRIAL_TIMEOUT" \
    --mem-fraction "$MEM_FRACTION" --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU" \
    --attention-backend "$ATTENTION_BACKEND" \
    --sglang-context-length "$SGLANG_CONTEXT_LENGTH" \
    --session-server-workers "$SESSION_SERVER_WORKERS" \
    --session-server-port "$SESSION_SERVER_PORT" --router-port "$ROUTER_PORT" \
    --session-server-external-host "$SESSION_SERVER_EXTERNAL_HOST" \
    --full-pass-bonus "$FULL_PASS_BONUS" --save-interval "$SAVE_INTERVAL" --lr "$LR" \
    --extra-env-vars "$WORKER_ENV_JSON" --extra-args "$MILES_EXTRA_ARGS" \
    "$@" \
    --run-id "$RUN_ID" --ray-submission-id "$RUN_ID" --remote-config "$REMOTE_CONFIG"
}
_start_seta "$@" >&"$log_fd" 2>&1 &
launcher_pid=$!
set +e
wait "$launcher_pid"
launcher_status=$?
set -e
exit "$launcher_status"
