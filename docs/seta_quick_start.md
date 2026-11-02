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

- W&B 默认 online，无需单独 login；`WANDB_PROXY=""` 表示训练端直连，不控制沙盒代理。
- 修改沙盒代理或并发：改 config.json 后 apply。Dockerfile RUN、任务命令、verifier 在未显式覆盖时使用节点默认代理。
- `MAX_CONCURRENT_TRIALS` 限制训练总并发；设为不小于各节点容量之和时，实际只受节点限制。
- `NUM_ROLLOUT=0` 使用 epoch 预算，正数指定总批次数；每个任务的轨迹数由 N_SAMPLES_PER_PROMPT 控制。
- `SESSION_SERVER_PORT` / `SESSION_SERVER_WORKERS` 控制从 30000 起的会话端口；ROUTER_PORT 默认 31000。GPU 和沙盒需双向可达，模型地址由 Miles 随请求传入，不配置主机白名单。
- 默认 `PIN_ROLLOUT_MANAGER_TO_HEAD=1`，协调器、router 和会话服务固定在 Ray head；endpoint/token 只需 head 可读。其他节点的代码、模型、checkpoint 仍需准备，Ray 不自动复制本地文件。已有多机 Ray 集群通过 MILES_SCRIPT_EXTERNAL_RAY=1 接入；单机验收不代表双机已验证。
- 当前 A100 机器默认限制 NCCL P2P 为 NVLink、关闭 IB 和 SGLang custom all-reduce，并优先使用 PyTorch 配套 NCCL；这些设置可通过脚本中的环境变量覆盖。更换机器或使用多机 RDMA 时应按实际网络调整。

实现、奖励口径与通信配置见 [seta_miles.md](seta_miles.md)。
