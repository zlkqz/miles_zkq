# GPU Reboot / CUDA 初始化失败交接

记录日期：2026-10-28 起，宿主机时区 Asia/Harbin（UTC+8）。本文件保留故障交接与后续检查记录。**用户于 2026-10-29 13:52 自行重启后的最新验证见第 13 节：GPU 检查通过，但仍有 Python/NCCL 环境问题及 IOMMU 配置风险。** 更早的在线恢复操作见第 12 节，跨节点定位见第 10 节。Agent 未执行宿主机重启。

## 1. 问题结论与影响

以下记录恢复前的故障现象；最新状态以第 13 节为准。训练节点 **124.16.138.150 的全部 8 张 A100 80GB 曾报告 `GPU Recovery Action: Reboot`**。宿主机内核记录了 Xid 31 和 Xid 154 / Node Reboot Required；故障后的训练容器无法显式创建 CUDA 张量。因此问题涉及宿主机 GPU/驱动状态，不能仅按容器缺少 Python 包处理。

另有一个需要分别检查的环境差异：宿主机用户 Python 的 PyTorch CUDA 13 构建报告原生驱动过旧；训练容器加载 CUDA 13 兼容驱动库，报的是 CUDA unknown / unavailable。**两处都存在失败，但错误及加载的用户态驱动库不同。根因尚未确定。**

第 10 节通过正常节点 144 的对照，进一步区分了 PyTorch 版本不兼容与 150 的内核 UVM 设备打开失败，并发现故障前的 IOMMU 异常。`Reboot` 是驱动给出的恢复动作，不是触发故障的根因，也不能证明重启后不会复发。

SETA 远端 rollout 曾成功完成，但最小训练检查没有完成优化器更新及 checkpoint 保存；正式训练尚未启动。不要把 CPU 矩阵乘法、`nvidia-smi` 能列卡或单独 FA2 检查成功当作当前训练环境可用的证据。

## 2. 机器、容器及环境

| 项目 | 本次实际位置 / 值 |
| --- | --- |
| 故障 GPU 宿主机 | `124.16.138.150`，8×NVIDIA A100 80GB |
| 宿主机 NVIDIA 内核驱动 | `570.195.03` |
| Miles 仓库 | `/ceph_home/zhangkaiqi2024/miles` |
| GPU 训练容器 | `miles_seta_zhangkaiqi2024` |
| 容器镜像 | `miles-seta:tools-v1`，基于 `radixark/miles` |
| 宿主机用户 Python | 用户在宿主机运行 `python3`；PyTorch 位于 `~/.local/lib/python3.10/site-packages`，版本 `2.14.0+cu130` |
| 训练容器 Python | Python `3.12.15`，PyTorch `2.14.1+cu130`；`python` 入口在 `/opt/sglang/bin/`，底层解释器在 `/opt/python/cpython-3.12.15-linux-x86_64-gnu/` |
| 容器实际加载的 libcuda | `/usr/local/cuda-13.0/compat/libcuda.so.580.126.20`；曾通过进程 maps 和原生栈确认 |
| GPU 训练 systemd 服务 | `seta-training-zkq`，刻意保持 `disabled` / `inactive` |
| Harbor 沙盒机 | `124.16.138.144`，只运行 CPU 沙盒；**其 GPU 被其他人使用，不属于本次故障处理目标** |

容器配置包含 `--init`、`--restart unless-stopped`、`--gpus all`、host network、host IPC；挂载 `/ceph_home` 与 `/141nfs`。另外显式传入了 `/dev/nvidia*` 字符设备。GPU 容器没有挂载 Docker socket，任务容器在远端 Harbor 机运行。

## 3. 宿主机中的失败与 CPU 对照

### 3.1 宿主机内核与 nvidia-smi

最近一次只读检查仍返回：

```text
0, 570.195.03, Reboot
1, 570.195.03, Reboot
2, 570.195.03, Reboot
3, 570.195.03, Reboot
4, 570.195.03, Reboot
5, 570.195.03, Reboot
6, 570.195.03, Reboot
7, 570.195.03, Reboot
```

对应命令在 **GPU 宿主机**执行：

```bash
nvidia-smi --query-gpu=index,driver_version,gpu_recovery_action --format=csv,noheader
```

早期归档记录了本地时间 **2026-10-28 16:34:15** 的 Xid 31；第 9 节重新导出的宿主机 journal 还包含 **16:33:59** 的 Xid 31 及全部 8 卡的 Xid 154。以各归档的原始时间戳为准：

```text
Xid ... 31 ... MMU Fault ... FAULT_INFO_TYPE_UNSUPPORTED_KIND
Xid ... 154, GPU recovery action changed from 0x0 (None) to 0x2 (Node Reboot Required)
```

日志中 Xid 31 的进程名字段是 `modprobe`、pid=10760；不能据此直接断定故障由哪个训练 kernel、模块加载操作或硬件部件引起。完整原文见第 6 节归档。

### 3.2 宿主机用户 Python

用户提供的宿主机实测：

```python
import torch
print(torch.__version__)           # 2.14.0+cu130
print(torch.cuda.is_available())   # False
```

警告为：

```text
The NVIDIA driver on your system is too old (found version 12080).
```

这里 12080 是 CUDA driver API 版本信息，对应 12.8，不是 NVIDIA 软件包的版本号。此环境使用宿主机原生驱动；应独立检查 PyTorch / CUDA / 驱动兼容性。

用户执行的下列矩阵乘法成功，但输入未指定 CUDA，实际是 **CPU 运算**：

```python
torch.matmul(
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
)
```

`torch.is_cuda_avalable()` 是拼写及 API 路径错误；正确 API 是 `torch.cuda.is_available()`。前者的 AttributeError 与 GPU 故障无关。

## 4. 训练容器中的失败

**容器内** CPU 张量乘法成功；显式创建 `device="cuda:0"` 张量失败。已见错误包括：

```text
CUDA initialization: CUDA unknown error ... Setting the available devices to be zero.
RuntimeError: CUDA unknown error ...
```

以及另一份探测日志中的：

```text
torch.AcceleratorError: CUDA-capable device(s) is/are busy or unavailable
Returning 999 (CUDA_ERROR_UNKNOWN) from cuDevicePrimaryCtxRetain
```

这些探测在创建 CUDA 张量时就失败，尚未进入用于探测的 TE L2 norm 运算。因此不能把这些探测结果解读为“已证明 TE L2 norm 算子本身有 bug”。

容器实际加载 `libcuda.so.580.126.20` 兼容库，原生训练栈也出现该库；不能仅凭宿主机 Python 的“驱动过旧”警告，就断言容器没有加载兼容库。兼容库的存在同样不能证明 Reboot 状态已恢复。

## 5. 故障前的训练进度与已做操作

1. 真实 Qwen3-8B 多轮 rollout 已经跨 GPU / 远端 Harbor 两台机器执行，产生有效奖励和 TITO 训练数组。
2. A100 上镜像原有 FA4 反向曾报不支持 SM80；已经卸载容器内 `flash-attn-4`，保留 FA2。单独 FA2 CUDA 前向/反向当时通过。该处理只针对当前 A100 镜像，不能推广到 H20/H200。
3. `remote-smoke-06` 最小检查进入 Megatron 训练，采集栈停在优化器 `get_grad_norm_fp32` → TE `multi_tensor_l2norm` → `cuLaunchKernel`。这只是停滞位置的证据，不等于已经定位根因。
4. 之后宿主机记录上述 Xid；新的 CUDA 进程无法初始化。训练回放检查也未完成。
5. 已停止本项目测试 Ray / 训练进程，训练 systemd 服务保持 disabled/inactive。没有擅自执行 GPU reset、驱动升级、模块卸载重载或宿主机重启。
6. 曾遇到另一个容器 NVML 问题：宿主机 `systemctl daemon-reload` 后旧 NVIDIA runtime 的设备权限丢失。容器已重建并增加显式 NVIDIA 设备映射。它与当前 Xid / Reboot 状态是两条需区分的线索，不能混为一谈。

当前 Harbor 已迁移到官方 agent-server；远端 CPU 参考解、取消清理及固定模拟模型的 Terminus-2 检查通过。GPU 故障修复无需重做 Harbor 部署，也不要操作远端其他人的 GPU。

## 6. 原始证据位置

所有相对路径均以 GPU 仓库 `/ceph_home/zhangkaiqi2024/miles` 为基准。以下诊断目录被 Git 忽略，**新 clone 不会自动带上日志；交接时需保留当前工作区或单独复制相应文件**。

主目录：`seta_miles_runtime/diagnostics/remote_migration_20261028/`。

| 文件 | 用途 |
| --- | --- |
| `gpu_driver_xid.log` | 宿主机 Xid 31 / 154 原文及时间 |
| `stack_smoke06_rank0.txt`、`stack_smoke06_rank4.txt` | 训练停滞时 Python 栈 |
| `native_smoke06_rank0.txt` | 含 libcuda、cuLaunchKernel、TE L2 norm 的原生栈 |
| `smoke_bootstrap_06.log` | 故障前最小训练入口及 Ray 输出 |
| `replay_bootstrap_07.log` | 之后尝试读取既有 rollout 数组的训练回放日志 |
| `te_l2norm_probe.log`、`te_l2norm_probe_eager.log` | 故障后 CUDA 张量初始化失败的探测记录 |
| `a100_fa2_backward.log` | 故障前独立 FA2 前向/反向成功记录 |
| `a100_fa4_remove.log` | 容器内卸载 FA4 记录 |
| `real_rollout_summary.json` | 故障前有效 rollout 的 token、mask、奖励及远端轨迹位置 |

部分容器日志时间是 UTC，例如 08:34 对应宿主机本地 16:34；对齐事件时注意时区。`smoke_bootstrap_06.log` 尾部还包含当时修改运行脚本引起的 shell EOF 错误，这是另一条历史记录，不能解释宿主机 Xid。当前入口语法可用 `bash -n my_scripts/run_seta.sh` 单独检查，勿为排查语法直接启动正式训练。

## 7. 后续 Agent 可用的分层检查入口

以下是原始交接用命令；后续实际执行结果见第 9 节。不要把宿主机与容器 Python 结果混用。

**GPU 宿主机：只读采集。**

```bash
cd /ceph_home/zhangkaiqi2024/miles
hostname
date -Is
cat /proc/driver/nvidia/version
nvidia-smi --query-gpu=index,name,driver_version,gpu_recovery_action --format=csv
sudo journalctl -k --since '2026-10-28 16:20:00' --until '2026-10-28 16:45:00' --no-pager
systemctl is-active seta-training-zkq
systemctl is-enabled seta-training-zkq
docker inspect miles_seta_zhangkaiqi2024
```

**GPU 宿主机用户 Python：确认实际解释器和包，先不分配 CUDA 张量。**

```bash
python3 - <<'PY'
import sys
import torch
print('python:', sys.executable)
print('torch:', torch.__version__, torch.__file__)
print('torch CUDA:', torch.version.cuda)
PY
```

**GPU 训练容器：确认其独立环境。**

```bash
docker exec -i miles_seta_zhangkaiqi2024 python - <<'PY'
import ctypes
import sys
import torch
print('python:', sys.executable)
print('torch:', torch.__version__, torch.__file__)
print('torch CUDA:', torch.version.cuda)
driver = ctypes.CDLL('libcuda.so.1')
for line in open('/proc/self/maps'):
    if 'libcuda.so' in line:
        print(line.strip())
PY
```

协调恢复后，先在 **训练容器内**做显式 CUDA 分配、乘法、同步和反向检查，再推进最小训练；示例从 GPU 宿主机进入容器执行：

```bash
docker exec -i miles_seta_zhangkaiqi2024 python - <<'PY'
import sys
import torch
print(sys.executable, torch.__version__, torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
x = torch.randn(32, 32, device='cuda:0', requires_grad=True)
y = x @ x
y.square().mean().backward()
torch.cuda.synchronize()
print(x.device, x.is_cuda, y.device, y.is_cuda)
print('CUDA forward/backward passed')
PY
```

此单卡小检查通过仍不代表 8 卡训练已验证。最终还需在无 Reboot 状态下完成本项目小批次优化器更新、checkpoint 保存及恢复检查，再按 [Quick Start](seta_quick_start.md) 启动正式训练。不要提前 enable/start 训练服务。

## 8. 恢复范围与交付要求

恢复方案由接手 Agent 与用户协调；本交接不预先断言必须升级哪个驱动版本，也不授权立即重启共享宿主机。若需要重启、GPU reset、模块重载或升级驱动，先核实其他任务并说明影响。

交付时分别报告宿主机驱动状态、宿主机 Python 兼容性、容器 libcuda 加载路径、容器实际 CUDA 运算，以及训练更新/checkpoint 的验证结果；保留恢复前后日志。当前仓库包含尚未提交的 SETA 适配工作，请勿 reset/clean 或覆盖现有脚本、凭据和模型目录。

参考：[NVIDIA GPU recovery action](https://docs.nvidia.com/deploy/nvidia-smi/index.html#gpu-recovery-action)、[CUDA forward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)。

## 9. 2026-10-28 20:17–20:22 复核：尚未恢复，禁止宿主机重启

### 实测结果与训练影响

**故障尚未修复，当前节点不能用于本项目 GPU 训练。** 用户允许 sudo 诊断，但明确禁止重启宿主机。本轮未执行 GPU reset、模块卸载/重载、驱动升级、进程终止或服务启停。

| 检查 | 本次结果 |
| --- | --- |
| 宿主机 GPU 状态 | 8 卡均为 `Reboot`，GPU 利用率均为 0%；显存使用为 8–25 MiB |
| 驱动模块一致性 | 运行中的 `nvidia`、`nvidia_uvm` 与磁盘上的模块均为 `570.195.03` |
| 宿主机原生 CUDA Driver API | 加载 `/usr/lib/x86_64-linux-gnu/libcuda.so.570.195.03`，API 版本 `12080`，`cuInit(0)` 返回 `999 / CUDA_ERROR_UNKNOWN` |
| 训练容器 CUDA Driver API | 加载 `/usr/local/cuda-13.0/compat/libcuda.so.580.126.20`，API 版本 `13000`，`cuInit(0)` 同样返回 `999 / CUDA_ERROR_UNKNOWN` |
| 训练容器 PyTorch | `2.14.1+cu130`；`torch.cuda.is_available()` 为 `False`；显式创建 CUDA 张量失败，探测进程退出码为 1 |
| 训练更新与 checkpoint | 本次 CUDA 分配失败，未进入前向、反向、优化器更新、保存或恢复检查 |
| 训练服务 | `seta-training-zkq` 仍为 `inactive` / `disabled` |

原生宿主机驱动库也无法初始化 CUDA，因此不能把故障仅归结为容器 CUDA 13 兼容库。宿主机 PyTorch CUDA 13 与原生 CUDA 12.8 Driver API 的版本差异仍是单独的环境问题；调整 PyTorch 版本不会自动清除当前的 `Reboot` 状态。故障原因仍未定位到某个训练算子或具体硬件部件。

### 在线恢复的阻塞

root 权限下的 `fuser -v /dev/nvidia*` 发现，另一位用户 `zhangjun2025` 的 **16 个非僵尸 Python 进程仍持有 `/dev/nvidia-uvm` 与 GPU 设备**。PID 为：

```text
1676437 1676440 1676444 1676445 1676446 1676447
1677155 1677158 1677288 1677291 1677426 1677429
2334251 2334324 2334978 2894129
```

此外，`dcgm-exporter` 持有 UVM/GPU 设备，`Xorg` 和 `nvidia-persistenced` 持有 GPU 设备。**利用率为 0%、显存很少或进程启动时间较早，均不能证明这些进程可以终止。** 尚未取得它们可以停止的确认，不得自行终止其他用户进程、停止共享监控/桌面，或强制卸载模块。

本次核对 NVIDIA 官方恢复说明：`Reboot` 表示故障可能使操作系统处于不一致状态，官方要求重启操作系统；UVM fatal error 是文档给出的示例，并非本机根因的确诊。该状态不同于官方建议执行 `nvidia-smi -r` 的 `Reset` 状态。GPU reset 本身还要求目标 GPU 没有应用占用，当前不满足条件。

在“禁止重启宿主机”和当前共享设备占用的条件下，**尚无经过验证、可安全执行的在线恢复方案**。如需评估模块重载等在线尝试，先由进程所有者协调释放相关设备并明确授权具体操作；它们不保证能恢复此类 `Reboot` 故障。恢复前不要启动正式训练，未来即使小型 CUDA 检查通过，也仍需完成第 7 节的训练验收。

### 新增证据归档

目录：`seta_miles_runtime/diagnostics/gpu_recovery_20261028_2017/`，同样被 Git 忽略，交接时需保留当前工作区或单独复制。

| 文件 | 内容 |
| --- | --- |
| `gpu_before.txt`、`gpu_before.xml`、`topology.txt`、`modules.txt` | GPU 完整状态、拓扑及模块列表 |
| `kernel_since_fault.log` | root 导出的 16:20 起宿主机内核 journal；含 16:33:59 的 Xid 31 / 154 |
| `gpu_clients.txt`、`foreign_clients.txt` | root 设备占用检查及 16 个其他用户进程的状态、启动时间 |
| `driver_probe.py`、`host_driver_probe.log`、`container_driver_probe.log` | 独立进程加载的实际驱动库、API 版本及 `cuInit` 返回值；脚本退出码 0 仅表示诊断执行结束，初始化结果均为 999 |
| `cuda_probe.py`、`container_cuda_probe.log` | 容器显式 CUDA 分配失败的完整异常，退出码 1 |
| `nvidia_recovery_reference.txt`、`cuda_compat_reference.txt` | 本次读取的 NVIDIA 官方页面文本与来源 URL |
| `final_state.json` | 20:22 的最终 GPU 状态、模块版本、训练服务状态及各检查退出码 |

## 10. 2026-10-28 跨节点与 Python 环境对照：UVM 打开失败，优先排查 IOMMU

### 四组 Python 结果及各自原因

按用户提供的新对照，20:38 起在 150 和 144 上以同一用户、同一绝对解释器路径执行相同探测。未更改两个 Python 环境、驱动、服务或系统配置；144 仅执行环境、设备查询和 CUDA 初始化，未分配 CUDA 张量或运行训练。144 当时 8 卡均有其他任务，恢复动作均为 `None`。

| 节点 | Python 环境 | 实际 PyTorch / CUDA 构建 | `torch.cuda.is_available()` | 直接使用原生 `libcuda` 的 `cuInit(0)` |
| --- | --- | --- | --- | --- |
| 150 | `/usr/bin/python3` | `2.14.0+cu130` / CUDA 13.0 | `False`，警告驱动过旧 | `999 / CUDA_ERROR_UNKNOWN` |
| 144 | `/usr/bin/python3` | `2.14.0+cu130` / CUDA 13.0 | `False`，警告驱动过旧 | `0 / CUDA_SUCCESS` |
| 150 | 用户提供的 `qwen35_neutral_v1` | `2.10.0+cu128` / CUDA 12.8 | `False`，警告 CUDA unknown error | `999 / CUDA_ERROR_UNKNOWN` |
| 144 | 用户提供的 `qwen35_neutral_v1` | `2.10.0+cu128` / CUDA 12.8 | **`True`**，底层 CUDA runtime 设备数为 8 | `0 / CUDA_SUCCESS` |

用户提供的完整解释器路径是 `/ceph_home/caoshijie2025/projects/242/.envs/qwen35_neutral_v1/bin/python`。两台机器的系统解释器都导入 `/ceph_home/zhangkaiqi2024/.local/lib/python3.10/site-packages/torch`，并不是 Ubuntu 自带的 PyTorch。四组均未设置 `CUDA_VISIBLE_DEVICES`、`LD_LIBRARY_PATH`、`LD_PRELOAD` 或 `PYTORCH_NVML_BASED_CUDA_CHECK`；144 的 `True` 也通过底层 CUDA runtime 设备枚举核实，并非只依赖 NVML 枚举。

**这是两个同时存在的问题：**

1. 系统 Python 的 CUDA 13.0 构建加载原生 CUDA 12.8 Driver API，导致两台机器都报告版本不兼容。144 的这个 `False` 不代表其 GPU 故障。该环境应使用兼容的 PyTorch 构建，或在满足官方条件时显式使用对应 CUDA 兼容库；本轮未改动共享包。
2. CUDA 12.8 环境排除了上述版本差异后，150 依然失败。两台原生 `libcuda.so.570.195.03` 及 `libnvidia-ml.so.570.195.03` 的 SHA256 分别一致，因此需要继续定位 150 的节点状态。

### 已定位的当前阻断：内核 UVM 全局错误

在完全不导入 PyTorch 的独立进程中直接打开设备：

```text
150 普通用户及 root：
  /dev/nvidiactl       open 成功
  /dev/nvidia0         open 成功
  /dev/nvidia-uvm      open 失败，errno=5 / EIO
  /dev/nvidia-uvm-tools open 失败，errno=5 / EIO

144 同一普通用户：以上四个设备均可 open。
```

两台设备均为字符设备且普通 UVM/GPU 节点权限为 `0666`，UVM 主设备号与各自 `/proc/devices` 一致；150 的失败不是普通文件权限不足。`strace` 同时确认 150 在 CUDA 初始化时 `openat("/dev/nvidia-uvm", O_RDWR...) = -1 EIO`，144 成功进入后续 UVM ioctl。

进一步核对本机 `/usr/src/nvidia-570.195.03/nvidia-uvm/` 源码：`uvm.c:193` 的 `uvm_open` 检查 `uvm_global_get_status()`；`uvm_tools.c:644` 的 `uvm_tools_open` 直接将同一全局状态转为 errno；`uvm_common.c:151` 的转换函数将 `NV_ERR_RC_ERROR` 或 `NV_ERR_ECC_ERROR` 转为 `EIO`。因此，设备打开失败结合该版本源码将当前阻断定位到 **UVM 全局错误状态**；尚未直接读取出上述两个具体 NV_STATUS 中的哪一个，不能据此确诊 ECC 硬件故障。

限制 `CUDA_VISIBLE_DEVICES` 为 0、1 或 7 时，150 原生 `cuInit` 仍均返回 999，符合 UVM 全局阻断，当前不能通过换一张卡或换 Python 绕过。源码的 `uvm_global_reset_fatal_error` 仅供主动触发错误的内置测试使用；本机内置测试未启用，不能将测试重置入口当作生产修复方法。

### 更早的异常及两机差异：IOMMU / GPU 间通信

重新回溯内核日志发现了早于第 3 节 Xid 的异常，时间顺序如下：

| 本地时间 UTC+8 | 证据 |
| --- | --- |
| 16:31:04.467 | `smoke_bootstrap_06.log` 的 `08:31:04.467` UTC：`Timer actor_train start` |
| 16:31:19 | AMD-Vi 对全部 8 个 GPU PCI 地址共记录 **80 条 `IO_PAGE_FAULT`**，另有 `IOMMU Event log restarting` |
| 16:33:59 | Xid 31 / `FAULT_INFO_TYPE_UNSUPPORTED_KIND`，随后全部 8 卡 Xid 154 / Node Reboot Required |
| 本次复测 | UVM 设备 open 返回 EIO；新进程 CUDA 初始化失败 |

150 的全部 8 张 GPU 都挂在 IOMMU group 中，group 的活动类型为 **`DMA-FQ`**；144 的对应 8 张 GPU **均没有 `iommu_group`**。两台 CPU 均为 AMD EPYC 7H12，NVIDIA 驱动均为 `570.195.03`；运行内核分别为 150 的 `6.8.0-124-generic` 和 144 的 `6.8.0-94-generic`。还观察到驱动参数 `EnableGpuFirmware` 分别为 0 和 18；未将任何一个差异单独认定为根因。

[NVIDIA CUDA 多 GPU 文档](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html#host-iommu-hardware-pci-access-control-services-and-vms) 明确说明 Linux 裸机不支持启用 IOMMU 的 PCIe P2P 内存传输，可能产生静默设备内存损坏。[NCCL GPU 故障排查](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html#linux-bare-metal-iommu-and-pcie-peer-to-peer) 同样要求检查活动 IOMMU 模式，并建议以限制/禁用 NCCL P2P 的对照实验判断传输路径相关性。

**IOMMU 地址转换与 GPU 间通信是目前有日志及正常节点对照支持的优先根因候选，尚未完成因果验证。** 不能把时间相近视作已经证明故障由 NCCL、TE L2 norm 或某个训练 kernel 引起，也不能从当前证据断言驱动重启或系统重启后就不会复发。

### 后续处理边界

当前仍禁止宿主机重启，未执行模块重载、GPU reset、IOMMU/ACS 动态修改、驱动升级或停止其他用户进程。当前 UVM 的 EIO 不会因设置 `NCCL_P2P_DISABLE=1` 自动清除；不要在此状态下直接反复启动训练。

如第 9 节的设备占用由所有者协调释放，并获准开展具体在线恢复尝试，先恢复设备 open / CUDA 初始化，再做最小单卡训练及受控的多卡传输对照。恢复后可用 `NCCL_P2P_DISABLE=1` 做 NCCL 路径的诊断对照，但它仅影响 NCCL P2P，不能保证覆盖其他组件的直接 GPU 访问，也不保证纠正系统 IOMMU 配置。涉及持久 IOMMU/BIOS/内核配置的更改须另行评估，不在本轮执行。

### 本轮证据

归档目录：`seta_miles_runtime/diagnostics/gpu_python_matrix_20261028_2038/`，被 Git 忽略。包括四组环境日志 `150_system.log`、`144_system.log`、`150_qwen35.log`、`144_qwen35.log`；两机 `*_node_state.txt`、`*_iommu_state.txt`、`*_device_open.log`、`*_cuinit.strace`；`150_root_device_open.log`；`150_kernel_gpu_boot.log`、`150_kernel_fault_window.log`、`training_kernel_timeline.txt`；`uvm_source_evidence.txt` 及本次核实的 NVIDIA 官方页面文本。探测脚本亦保存在同目录；以日志内的 CUDA 返回值和异常为判断依据，诊断脚本自身退出码 0 不代表 CUDA 可用。

## 11. 2026-10-29 cu130 实际运算复测

用户要求尝试 cu130，并询问是否必须更新整个系统。本轮先确认两个现有环境均已安装 cu130 构建，因此直接复测，未重复安装 PyTorch，也未修改他人的 `qwen35_neutral_v1` 环境。

| 位置 | 已有 PyTorch | 实际驱动 API / 库 | 本次结果 |
| --- | --- | --- | --- |
| 宿主机 `/usr/bin/python3` | `2.14.0+cu130` | CUDA 12.8 / 原生 `libcuda.so.570.195.03` | CUDA 可用性为 False；PyTorch 报驱动过旧；原生 `cuInit` 返回 999 |
| 训练容器 | `2.14.1+cu130` | CUDA 13.0 / `compat/libcuda.so.580.126.20` | `cuInit` 返回 999；CUDA 可用性为 False；显式创建 CUDA 张量失败，退出码 1，未进入矩阵运算和反向 |

150 的全部 8 卡仍报告 `Reboot`；普通用户直接打开 `/dev/nvidia-uvm` 仍返回 `EIO`。因此本次已覆盖“cu130 PyTorch + CUDA 13 用户态兼容驱动”的组合，尚不能运行 CUDA 运算。版本切换不能单独清除当前第 10 节定位的 UVM 全局错误。

安装范围需区分：cu130 版 PyTorch 及其配套 CUDA 运行时可以安装在独立 Python 环境或容器中，通常无需系统安装完整 CUDA Toolkit；编译 CUDA 扩展是否需要 Toolkit 则取决于项目。NVIDIA 内核驱动由宿主机提供，pip 无法升级或恢复它。原生 CUDA 13 驱动路线通常需要 R580 或更新驱动；本机 A100/R570 的另一条受支持路径是 NVIDIA CUDA forward compatibility，训练容器已加载该兼容库。两条路径都不能据此视作已修复 UVM 故障。

证据目录：`seta_miles_runtime/diagnostics/cu130_retest_20261029_1254/`，被 Git 忽略。保留 `cu130_probe.py`、`host_cu130.log`、`container_cu130.log`、`gpu_after.csv`、`uvm_open.log`。本轮未重启宿主机、重载模块、升级驱动或启动正式训练。

## 12. 2026-10-29 在线恢复成功：清理占用并重载 NVIDIA 驱动栈

### 授权与实际操作

用户明确授权清理 GPU 占用进程、使用 root，并尝试此前提出的其他恢复方案；禁止宿主机重启的限制继续有效。操作仅针对 150，未操作 144 的 GPU，也未启动正式训练服务。

1. 重新归档 GPU、模块及设备占用。暂时停止 `dcgm-exporter`，向当前持有 UVM 的 16 个 Python 进程发送 SIGTERM；全部退出并释放占用，无需 SIGKILL。UVM 模块引用数降为 0。
2. 正常卸载、重新加载 `nvidia_uvm`，均退出码 0。`/dev/nvidia-uvm` 与 tools 设备恢复可打开，原生 `cuInit` 恢复为 0，训练容器 cu130 单卡前向/反向通过；但 8 卡 NVML 恢复动作仍显示 `Reboot`。
3. 为清除剩余状态，暂时停止显示管理器和 `nvidia-persistenced`。首次检查仍发现旧 Xorg 占用，未强制卸载模块，先恢复服务；再次停止服务后 Xorg 已退出，确认 GPU 设备无客户端后继续。
4. 正常卸载 NVIDIA DRM、UVM 及其依赖，确认 `/sys/module/nvidia` 已不存在；随后重新加载 `nvidia`、`nvidia_uvm`、`nvidia_modeset`、`nvidia_drm`。驱动版本保持 `570.195.03`，全部 8 卡恢复动作变为 **`None`**。未执行 GPU reset、强制模块卸载、驱动升级或宿主机重启。
5. 恢复显示管理器、`nvidia-persistenced` 和 `dcgm-exporter`，均恢复运行。`seta-training-zkq` 保持 `inactive` / `disabled`。

本次证明在线重载可以清除当前故障状态；前几节“尚无经过验证的在线恢复方案”描述的是恢复前的调查阶段，不再代表本机最新状态。该操作需先清空相关设备占用，不能在其他 GPU 任务仍运行时照搬。

### 验证结果与范围

| 检查 | 结果 |
| --- | --- |
| 原生 CUDA Driver API | `cuInit(0)` 返回 0，UVM 设备打开成功 |
| 用户提供的 cu128 Python | `torch 2.10.0+cu128`，`torch.cuda.is_available()` 为 **True** |
| 训练容器 cu130 Python | `torch 2.14.1+cu130`，CUDA 可用性为 **True** |
| 全部 8 张 GPU 独立小模型检查 | 每张卡均通过前向、反向、有限梯度检查、SGD 参数更新 |
| 全部 8 张 GPU 小模型 checkpoint | 每张卡保存并重新加载模型及优化器状态，输出与动量逐项一致 |
| 8 rank NCCL 通信 | all-reduce 均得到预期的 36，barrier 正常结束，退出码 0 |
| 通信路径 | 本次显式设置 `NCCL_P2P_DISABLE=1`、`NCCL_IB_DISABLE=1`；日志确认 `SHM/direct` 路径 |
| 验证后 GPU 状态 | 8 卡均为 `None`，利用率回落至 0%，显存约 10 MiB/卡 |
| 宿主机启动时间 | 仍为 `2026-09-28 12:00:00`，本轮未重启 |

cu128 首次复核在与 NCCL 检查同时运行时达到 30 秒超时；随后带阶段输出重新检查，导入及 CUDA 可用性均成功。未将超时解释为新的 GPU 故障。

完整驱动重载前的 13:01:07，旧 Xorg 仍出现过 Xid 31 / IOMMU fault，保留在日志中。13:02:38 起的验证窗口未发现新的 Xid、`IO_PAGE_FAULT` 或 UVM global fatal error。NCCL 日志包含禁用 IB 时的 `lib wrapper not initialized` 警告，但该次 8 rank 检查完整通过；本轮没有验证 IB/GDR 通信。

**以上 checkpoint 属于诊断用小模型，并非 Qwen3-8B/SETA checkpoint。** 本轮未重新启动项目训练、执行真实 rollout 或验收 Qwen3-8B 的优化器更新及 checkpoint 保存/恢复；这些仍是正式启动前的剩余验收项。

### 尚未解决与下一步

- IOMMU 活动配置未改变，仍需按第 10 节调查复发原因；当前恢复不证明根因已消除，也未验证启用 PCIe P2P 的稳定性。
- NCCL 的上述环境变量只作用于本次诊断进程，未持久修改训练启动脚本。后续项目最小训练可先采用本次已通过的通信配置，但这不保证覆盖其他组件的直接 GPU 访问。
- 系统 `/usr/bin/python3` 的 cu130 构建与原生 CUDA 12.8 Driver API 的兼容性问题仍独立存在。应使用兼容环境或已配置兼容库的训练容器，不能据该系统环境的 False 判断本轮 GPU 恢复失败。
- 共享监控、显示及持久化服务均已恢复；被终止的其他用户 Python 任务未自动重启。

### 操作与证据归档

目录：`seta_miles_runtime/diagnostics/online_recovery_20261029_1257/`，被 Git 忽略。包括：

- `authorization.txt`、`clients_before.txt`、`uvm_clients.json`、`cleanup_actions.json`：授权范围、占用及 16 个进程的退出记录。
- `uvm_reload_actions.json`、`after_uvm_host_driver.log`、`after_uvm_container_cuda.log`：仅重载 UVM 的操作与初步成功结果。
- `driver_stack_attempt1.json`、`driver_stack_actions.json`、`base_driver_unloaded.txt`：完整驱动栈重载、首次因占用退出及最终基驱动卸载证明。
- `all_gpus_validation.log`、`all_gpus_validation.json`、`toy_checkpoint_gpu*.pt`：8 卡独立小模型更新与 checkpoint 检查。
- `nccl_validation.log`、`nccl_validation_command.txt`：8 rank 通信结果与确切环境变量。
- `host_cu128_after.log`、`native_driver_after_full_reload.log`、`kernel_after_driver_reload.log`、`kernel_validation_window.log`、`final_state.json`：Python 复核、内核证据及最终状态。

归档中的清理及重载脚本是本次实际执行记录，包含停止 GPU 客户端/显示服务的操作；不是无需核实机器占用即可运行的通用修复入口。

## 13. 2026-10-29 用户重启后的复核：GPU 检查通过，仍有环境及配置问题

### 当前机器与本轮操作

用户告知自行重启并修复了部分问题。本次宿主机启动时间为 **2026-10-29 13:52:20**；14:09 起检查，内核仍为 `6.8.0-124-generic`，驱动仍为 `570.195.03`。开始检查时 8 卡均为 `None`，无 CUDA 计算进程占用。

本轮未重载驱动、清理进程、GPU reset、再次重启或更改系统配置。原训练容器处于 `exited` 状态，检查确认其命令仅为 `sleep infinity` 后启动，用于下列探测；未启动正式训练服务。未操作 144。

### 已通过的检查

| 检查 | 结果 |
| --- | --- |
| 原生驱动初始化 | `cuInit(0)` 为 0，设备数为 8 |
| 用户提供的 cu128 环境 | `torch 2.10.0+cu128`，CUDA 可用性为 True |
| 训练容器 cu130 环境 | `torch 2.14.1+cu130`，加载 `compat/libcuda.so.580.126.20`，API 版本 13000，CUDA 可用性为 True |
| 容器实际训练导入路径 | 不加环境覆盖，在独立进程中以非 `__main__` 名称加载 `train.py`，再导入 Megatron actor，均成功；仅加载 PyTorch 配套 NCCL，随后 CUDA 小张量反向计算通过，退出码 0；未启动训练 |
| 8 卡独立小模型 | 全部通过前向、反向、SGD 更新、模型及优化器 checkpoint 保存/恢复 |
| TE `multi_tensor_l2norm` | 每张卡对 `[3,4]`、`[12]` 计算，均得到预期的 13 |
| FA2 | 每张卡均通过 bf16 causal attention 前向/反向，输出与梯度有限 |
| 8 rank NCCL，禁用 P2P | all-reduce 均为预期的 36，barrier 通过，日志确认 SHM 路径 |
| 8 rank NCCL，只允许 NVLink P2P | 设置 `NCCL_P2P_DISABLE=0`、`NCCL_P2P_LEVEL=NVL`，all-reduce/barrier 通过；日志同时出现 NVLink 对的 `P2P/CUMEM/read` 和其他连接的 `SHM/direct` |
| 原先停滞的 Megatron 梯度范数路径 | 8 rank 实际调用 `get_grad_norm_fp32`，均得到 `14.142135620117188`，与预期 `sqrt(8*25)` 一致 |
| 内核与最终 GPU 状态 | 本次启动的检查前后完整内核日志均未发现 Xid、`IO_PAGE_FAULT`、global fatal 或 BUG；验证后 8 卡仍全部为 `None` |

上述两类通信及 Megatron 梯度范数检查均设置 `NCCL_IB_DISABLE=1`，未验证 IB/GDR；只允许 NVLink P2P 的检查不代表不限制 PCIe P2P 时也稳定。小模型 checkpoint 不是 Qwen3-8B/SETA checkpoint，仍未完成真实项目最小训练验收。

### 仍存在的三个问题

1. **系统 Python 的 cu130 兼容性仍未解决。** `/usr/bin/python3` 仍导入用户目录中的 `torch 2.14.0+cu130`，加载原生 CUDA 12.8 Driver API，CUDA 可用性为 False，警告驱动过旧。但同进程直接调用原生 `cuInit` 已成功，因此该 False 属于版本组合问题，不能解释为当前 GPU 再次故障。
2. **容器中 NCCL 库存在导入顺序冲突，已检查的正常入口未触发。** 在先导入 Transformer Engine、后导入 PyTorch 的独立进程中，进程加载系统 `/usr/lib/x86_64-linux-gnu/libnccl.so.2.28.3`，随后 PyTorch 报 `undefined symbol: ncclCommResume`；先导入 PyTorch 则本次计算检查均正常。14:26 起进一步复核实际导入路径：默认容器环境下，以非 `__main__` 名称加载 `train.py` 并导入 Megatron actor 成功，CUDA 可用性为 True，反向计算通过；同时，另一个独立进程直接导入 TE 仍复现该错误。因此不能将此冲突表述为正常训练启动必然失败；Ray worker 等完整训练启动路径仍未验收。将 TE-first 探测进程的 `LD_LIBRARY_PATH` 前置 `/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib` 后，仅加载 PyTorch 配套 NCCL，版本查询为 **2.30.7**，TE 先导入也成功，退出码 0。该临时修正已验证，但未改动容器持久环境、共享库或训练启动脚本。
3. **IOMMU 风险配置仍在。** 本次启动日志明确为 `iommu: Default domain type: Translated`，8 张 GPU 的 IOMMU group 仍全部为 `DMA-FQ`；拓扑与之前归档一致。当前没有 fault，不证明此前第 10 节的配置风险已消除。本轮采用禁用 P2P 或限制为 NVLink 的通信方式，未验证不受限的裸机 PCIe P2P。

下一步是在已验证的通信配置下完成 Qwen3-8B 小批次更新及项目 checkpoint 验收；本轮未自动启动长期训练。容器默认环境已通过上述正常入口导入及计算检查；若其他入口触发 TE-first 冲突，可对启动进程前置上述 NCCL 目录，并保留原 `LD_LIBRARY_PATH` 的其他条目，不需要更新系统驱动。统一该库优先级可消除已复现的导入顺序隐患。

### 原始证据

目录：`seta_miles_runtime/diagnostics/post_reboot_20261029_1409/`，被 Git 忽略。保存当前 boot ID、检查前后完整内核日志、IOMMU 状态、拓扑/P2P 能力矩阵、三个 Python 环境探测、8 卡小模型结果及 checkpoint、TE/FA2 算子结果、两种 NCCL 模式日志与命令、Megatron 梯度范数结果，以及 NCCL 导入冲突/临时修正的对照日志。

实际训练入口与 TE-first 的独立进程对照脚本及日志：`seta_miles_runtime/diagnostics/container_import_check_20261029_1426/`，被 Git 忽略。
