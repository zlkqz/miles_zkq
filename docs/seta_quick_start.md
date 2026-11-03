# SETA 远端沙盒训练 Quick Start

GPU 容器运行 Miles / Megatron / SGLang；独立沙盒裸机运行 Harbor agent-server 和 Docker。两端无需共享盘。使用已有镜像 `miles-seta:tools-v1`，本文不涉及镜像构建。

## 1. 已配好当前环境：直接启动

**GPU 宿主机**检查沙盒服务并进入容器：

```bash
cd /ceph_home/zhangkaiqi2024/miles
python3 seta_miles_runtime/seta-remote/setup.py --mode check
docker exec -it miles_seta_zhangkaiqi2024 bash
```

**GPU 容器内**启动，长时间运行可放在 tmux 中：

```bash
cd /ceph_home/zhangkaiqi2024/miles
bash my_scripts/run_seta.sh
```

默认全量 4,567 条数据和 W&B online。外部 export 可覆盖脚本默认值；只看配置用 `MODE=plan bash my_scripts/run_seta.sh`。同一专用容器只运行一份训练。

训练默认纯异步，单机 8 卡自动分成 4 卡训练 + 4 卡推理。同步/异步、卡数和多机配置见 [第 4 节](#4-常用调整与多机注意)。默认实验名包含 `async_grpo` 或 `sync_grpo`，两种模式使用独立 checkpoint 目录；显式设置 EXP_NAME 或 SAVE_PATH 可指定恢复路径。`MODE=rollout` 仍使用全部 GPU 做采样验证。

- checkpoint：SAVE_PATH，默认 `/ceph_home/zhangkaiqi2024/saved_models/p2o-agent/$EXP_NAME`；相同实验名重跑加载已有 checkpoint。
- 训练日志：`seta_miles_runtime/runs/$EXP_NAME/logs/${EXP_NAME}_${RUN_ID}.log`。EXP_NAME 不含时间戳，RUN_ID 区分每次启动。
- 沙盒轨迹：远端 `nodes[].root/state/trials/`；请求日志在 `state/dashboard/requests.jsonl`。

前台运行时按 **Ctrl+C** 即可停止训练并自动回收本次运行的远端沙盒；SIGTERM、报错退出和正常结束也执行清理。清理需等待 Ray 停止及 Harbor 删除容器，通常需要数十秒。

也可在另一个 **GPU 容器内**终端停止当前训练：

```bash
python my_scripts/stop_seta.py
```

新启动的运行会保存独立的 `logs/${EXP_NAME}_${RUN_ID}.run.json`；停止指定运行可用 `python my_scripts/stop_seta.py --run-file <该文件>`。取消按 run_id 隔离，不会 flush 其他运行。已在运行的旧脚本不会自动获得新版退出逻辑，仍可用上面的手动命令停止。

以下首次部署步骤在现有环境无需重复执行。训练验收结果和限制见 [详细说明第 10 节](seta_miles.md#10-本次实际部署与验证记录)。

## 2. 首次准备 GPU 环境

`/.../miles` 等为占位路径；宿主机与容器保持相同绝对路径最方便。**GPU 宿主机**需有 Docker、NVIDIA Container Toolkit 和镜像：

```bash
nvidia-smi
docker image inspect miles-seta:tools-v1 >/dev/null
```

替换以下挂载占位符，覆盖代码、数据、模型和 checkpoint 路径：

```bash
# Bash：显式登记设备，避免部分 NVIDIA Toolkit + systemd 组合在 daemon-reload 后丢失 GPU。
gpu_device_args=()
for device in /dev/nvidia*; do
  [[ -c "$device" ]] && gpu_device_args+=(--device "$device")
done

docker run -d \
  --init \
  --restart unless-stopped \
  --name miles-seta \
  --gpus all \
  "${gpu_device_args[@]}" \
  --network host \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /...:/... \
  -w /.../miles \
  --entrypoint /bin/bash \
  miles-seta:tools-v1 -lc 'sleep infinity'

docker exec -it miles-seta bash
```

不挂 Docker socket，任务沙盒在远端创建。restart 策略保持容器运行，不会恢复 docker exec 启动的训练。

**GPU 容器内**安装 Miles：

```bash
cd /.../miles
python -m pip install -e . --no-deps
```

**仅 A100/SM80** 对当前 tools-v1 镜像执行下列命令，保留已有 FA2；**H20/H200/H100 跳过**。当前 A100 容器已完成。

```bash
python -m pip uninstall -y flash-attn-4
```

已有数据、转换权重可跳过；否则仍在 **GPU 容器内**执行：

```bash
python examples/seta/prepare_dataset.py \
  --dataset-root /.../SETA-Env \
  --output-dir /.../seta-data \
  --agent-name terminus-2
wc -l /.../seta-data/train.jsonl

python scripts/run_qwen3_8b_seta.py \
  --mode prepare \
  --model-dir /.../models \
  --hf-checkpoint /.../Qwen3-8B
```

生成 train.jsonl 与 tasks/，不做论文的静态过滤。转换权重已有 `Qwen3-8B_torch_dist/latest_checkpointed_iteration.txt` 时复用。GPU 侧不安装 Harbor 虚拟环境。

## 3. 首次准备沙盒裸机并部署

在**沙盒机裸机**检查，已有工具直接复用：

```bash
docker version
docker compose version
docker buildx version
docker info --format '{{.DockerRootDir}} {{.DefaultRuntime}}'
command -v curl rsync python3
free -h
df -h /var/lib/docker
```

SSH 部署账号需能调用 Docker 并具有 sudo 权限；CPU 沙盒使用 runc。uv、Python 3.13.16、Harbor 虚拟环境和 Crane 由部署工具安装，无需另行 clone Harbor。

<details>
<summary>全新 Ubuntu 缺少 Docker 时：安装步骤（已有 Docker 则跳过）</summary>

在**沙盒机裸机**执行；其他系统参考 [Docker 官方安装文档](https://docs.docker.com/engine/install/)。

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl rsync python3
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
```

```bash
sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF_DOCKER
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF_DOCKER
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
```

部署账号如无 Docker 权限，执行 `sudo usermod -aG docker "$USER"` 后重新 SSH 登录。已有 Docker 只缺基础命令时，仅安装 ca-certificates/curl/rsync/python3，不重装或重启共享 daemon。

</details>

在 **GPU 宿主机**修改 [seta-remote/config.json](../seta_miles_runtime/seta-remote/config.json)：

| 配置 | 需要修改什么 |
| --- | --- |
| `tasks_source` | GPU 宿主机上完整 tasks/ 的位置 |
| `nodes[].name / ssh / host / port` | 节点名、SSH 目标及 GPU 可访问的服务地址 |
| `nodes[].root` | 沙盒机非共享盘目录，工具自动创建 |
| `nodes[].max_concurrent` | 每台沙盒的并发容量 |
| `nodes[].proxy / no_proxy / registry_mirrors` | 任务及下载代理；不用代理写空串，无额外直连地址写空列表 |

其余字段通常不改。同机多套部署需使用不同 service_name/service_user/service_id/port；增加沙盒可复制 nodes 中的完整对象，分别设置路径、代理和并发。

资源保护预算按 max_concurrent 自动推导，无需填写 CPU、内存或磁盘阈值。规则与作用范围见 [资源预算](seta_miles.md#11-按并发自动推导资源预算)；单任务资源仍读取 task.toml。

先停止使用待更新沙盒的训练，再在 **GPU 宿主机**执行：

```bash
cd /.../miles
python3 seta_miles_runtime/seta-remote/setup.py --mode plan
python3 seta_miles_runtime/seta-remote/setup.py --mode apply
python3 seta_miles_runtime/seta-remote/setup.py --mode check
```

工具自动复制代码和完整任务、安装远端依赖、生成持久服务及代理配置、同步 token，并生成 GPU 的 endpoints.json。sudo 密码在提示时输入；不启动训练、不使用沙盒机 GPU、不安装或重启已有 Docker。只更新一台可加 `--node <name>`。

只维护 config.json。`endpoints_file` 指定自动生成文件的位置，不用手工同步节点/并发；远端拥有自己的 docker_config，不读取 GPU 上的配置路径。

## 4. 常用调整与多机注意

**GPU 容器内**可覆盖脚本默认值，再执行同一入口：

```bash
export HF_MODEL=/.../Qwen3-8B
export MODEL_DIR=/.../models
export DATA_DIR=/.../seta-data
export GPU_HOST=<GPU_IP>
export EXP_NAME=my-experiment
export WANDB_API_KEY='<自己的 W&B API key>'
export WANDB_PROXY='<PROXY_URL>'
bash my_scripts/run_seta.sh
```

直接编辑 `my_scripts/run_seta.sh` 的“GPU 与并行”块，或用同名环境变量覆盖：

| 变量 | 含义 |
| --- | --- |
| `TRAINING_MODE=async` | 默认纯异步，训练与推理使用独立 GPU；设为 `sync` 后使用全部 GPU 分时复用 |
| `NUM_NODES` / `NUM_GPUS_PER_NODE` | 集群节点数、每节点可用 GPU 数；假设各节点卡数一致 |
| `TRAIN_NUM_GPUS` / `INFER_NUM_GPUS` | 异步训练/推理的**集群总卡数**；同步和仅采样模式忽略它们 |
| `TENSOR_PARALLEL_SIZE` | 训练 TP，需整除实际每节点训练卡数；global batch 必须能被训练 DP 整除 |
| `ROLLOUT_GPUS_PER_ENGINE` | 每个推理引擎的卡数，推理总卡数需是其整数倍 |

两个总卡数都为 0 时自动分配：单机各一半，多机将 `floor(NUM_NODES/2)` 个完整节点给训练，其余给推理。只指定一个总卡数时，另一个使用集群剩余卡；同时指定则允许留卡不用，但两者之和不能超过集群容量。申请跨越一台机器时，train+infer 总卡数须为每节点卡数的整数倍，可以留下整台机器；这是为了保持 Ray PACK 后按 IP 排序的节点边界。训练仅占一台机器时可使用部分卡；跨机训练必须使用整数个完整节点。推理引擎可以使用单节点的整数份或整数个节点，跨节点分配时训练卡数还必须对齐推理引擎的节点边界，避免一个本应在单节点内的引擎横跨两机。

```bash
# 单机：默认异步 4+4；改为同步只需切换 TRAINING_MODE。
TRAINING_MODE=sync bash my_scripts/run_seta.sh

# 单机异步：6 卡训练 + 2 卡推理，TP=2 / DP=3，global batch 改为 6×16=96。
TRAINING_MODE=async TRAIN_NUM_GPUS=6 INFER_NUM_GPUS=2 TENSOR_PARALLEL_SIZE=2 \
  ROLLOUT_BATCH_SIZE=6 bash my_scripts/run_seta.sh
```

多机需先在各机容器内启动并加入**同一个 Ray 集群**，再只在 head 启动一次本脚本。各节点保持相同容器环境、仓库/数据/模型/checkpoint 绝对路径，并确保节点间 Ray/NCCL 及沙盒回连端口可达。例如两机各 8 卡：

```bash
# head 容器（替换 HEAD_IP）。
ray start --head --node-ip-address <HEAD_IP> --port 6379 --num-gpus 8

# worker 容器（替换 HEAD_IP 和 WORKER_IP）。
ray start --address <HEAD_IP>:6379 --node-ip-address <WORKER_IP> --num-gpus 8

# 回到 head 容器：一个完整节点训练，另一个完整节点推理。
MILES_SCRIPT_EXTERNAL_RAY=1 NUM_NODES=2 NUM_GPUS_PER_NODE=8 \
  TRAINING_MODE=async TRAIN_NUM_GPUS=8 INFER_NUM_GPUS=8 bash my_scripts/run_seta.sh

# 三机各 8 卡，两个完整节点训练、一个完整节点推理。
MILES_SCRIPT_EXTERNAL_RAY=1 NUM_NODES=3 NUM_GPUS_PER_NODE=8 \
  TRAINING_MODE=async TRAIN_NUM_GPUS=16 INFER_NUM_GPUS=8 bash my_scripts/run_seta.sh
```

Ray 根据节点 IP/GPU 排序后分配连续资源，上述配置不指定哪台物理机承担训练角色。多机运行未设置 `MILES_SCRIPT_EXTERNAL_RAY=1` 会报错，避免只启动本机 Ray 后无限等待其他节点。可给上述运行命令增加 `MODE=plan` 查看实际 `train_script`、训练节点数和推理卡数，不启动 Ray 或训练。多机的真实 GPU/NCCL 运行尚待资源空闲后验证。

- W&B 默认 online，无需单独 login；`WANDB_PROXY=""` 表示训练端直连，不控制沙盒代理。
- 修改沙盒代理或并发：改 config.json 后 apply。Dockerfile RUN、任务命令、verifier 在未显式覆盖时使用节点默认代理。
- `MAX_CONCURRENT_TRIALS` 限制训练总并发；设为不小于各节点容量之和时，实际只受节点限制。
- `NUM_ROLLOUT=0` 使用 epoch 预算，正数指定总批次数；每个任务的轨迹数由 N_SAMPLES_PER_PROMPT 控制。
- `SESSION_SERVER_PORT` / `SESSION_SERVER_WORKERS` 控制从 30000 起的会话端口；ROUTER_PORT 默认 31000。GPU 和沙盒需双向可达，模型地址由 Miles 随请求传入，不配置主机白名单。
- 默认 `PIN_ROLLOUT_MANAGER_TO_HEAD=1`，协调器、router 和会话服务固定在 Ray head；endpoint/token 只需 head 可读。其他节点的代码、模型、checkpoint 仍需准备，Ray 不自动复制本地文件。已有多机 Ray 集群通过 MILES_SCRIPT_EXTERNAL_RAY=1 接入；单机验收不代表双机已验证。
- 当前 A100 机器默认限制 NCCL P2P 为 NVLink、关闭 IB 和 SGLang custom all-reduce，并优先使用 PyTorch 配套 NCCL；这些设置可通过脚本中的环境变量覆盖。更换机器或使用多机 RDMA 时应按实际网络调整。

实现、奖励口径与通信配置见 [seta_miles.md](seta_miles.md)。
