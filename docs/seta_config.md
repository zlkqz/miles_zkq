下面按论文中的 **Qwen3-8B + GRPO on SETA-Env** 训练配方总结成一份可复现 recipe。核心是：**Qwen3-8B non-thinking + CAMEL Terminal Agent + SETA-Env 子采样 560 环境 + AReal/GRPO + DAPO 动态采样 + 部分进度奖励**。

---

## 1. 模型与训练目标

- **Base model**：`Qwen3-8B`
- **模式**：**non-thinking mode**  
  论文明确说 thinking mode 经常在前几轮就超过最大响应长度，导致大量轨迹无效，所以主实验用 non-thinking。
- **算法**：GRPO
- **RL 框架**：AReal
- **训练数据**：`SETA-Env`
  - 全量：4,567 个可验证 terminal 环境
  - 实际训练：先过滤掉 base model 太难的环境，再 **uniformly subsample 560 个环境**
  - 这样做的原因是算力约束，同时保持任务类型多样性、稳定 RL 优化。

---

## 2. Agent / Rollout 环境

- **Agent framework**：CAMEL `ChatAgent` + Terminal Toolkit
- **执行环境**：每个任务在独立 Docker container 中执行
- **Terminal toolkit 特性**：
  - 命令通过 Docker SDK 进入容器执行
  - ANSI 清洗
  - 输出超过 1000 chars 截断为 head 500 + tail 500
  - 完整输出保存在容器内 `/tmp/full_output_{id}_{ts}.txt`
  - 支持 blocking / non-blocking shell session
- **全局 agent 配置**：
  - Max agent iterations：`30`
  - Max completion tokens：`4096`
  - Max total tokens / context：`28672`
  - Max new tokens generation：`10240`
- **超时配置**：
  - Environment reset：`300s`
  - Agent reset：`120s`
  - Agent step：`900s`
  - Evaluation：`600s`
- **Runtime**：Docker

---

## 3. GRPO 训练算法配方

### 3.1 GRPO + DAPO 动态采样

- 使用 **GRPO** 计算 group-relative advantage。
- 按 **DAPO** 做 dynamic sampling：**过滤掉同一组内 reward 全部相同 / 无方差的样本**，因为这些样本对 GRPO 没有有效学习信号。
- GRPO objective 中包含：
  - asymmetric clipping
  - KL to reference model，但实际超参里 **KL penalty = 0.0**
  - advantage normalization：batch mean & std

论文中的目标形式为：

\[
J(\theta)=
\mathbb{E}\left[
\min\left(
\frac{\pi_\theta(a|h,s)}{\pi_{old}(a|h,s)}A,\ 
\text{clip}\left(\frac{\pi_\theta(a|h,s)}{\pi_{old}(a|h,s)},1-\epsilon_{low},1+\epsilon_{high}\right)A
\right)
\right]
-\beta D_{KL}(\pi_\theta \| \pi_{ref})
\]

实际配置中：

- PPO clip：`0.4`
- KL penalty：`0.0`
- Behavior importance weight cap：`5.0`
- Reward scaling：`10.0`
- Reward bias：`-0.5`
- Sampling temperature：`1.0`
- Trajectories per task：`n = 16`

### 3.2 Reward 设计

使用 **partial-progress reward**，即按 unit tests 通过比例给奖励：

\[
r = \frac{\#\text{tests passed}}{\#\text{total tests}}
+ \text{bonus} \cdot \mathbb{1}[\#\text{tests passed}=\#\text{total tests}]
\]

论文正文写的是：

- pass bonus = `0.2`

但 Table 8 里写的是：

- Pass bonus = `+0.5`

> 注意：论文内部存在 `0.2` vs `0.5` 的不一致。复现时建议以代码/配置为准；若严格按正文公式，则用 `0.2`，若按训练配置表则用 `0.5`。

论文说明 partial-progress reward 把有 reward 变化的任务比例从 `49%` 提升到 `86%`，显著增加有效学习信号。

---

## 4. 关键超参数

| 项目 | 配置 |
|---|---|
| Base model | Qwen3-8B |
| Optimizer | Adam |
| Learning rate | `1.70e-5` |
| LR schedule | Constant |
| Weight decay | `0.017` |
| Adam β1, β2 | `0.9, 0.999` |
| Adam ε | `1e-8` |
| Gradient clipping | `1.0` |
| Warmup proportion | `0.001` |
| PPO clip | `0.4` |
| KL penalty | `0.0` |
| Reward scaling | `10.0` |
| Reward bias | `-0.5` |
| Behavior importance weight cap | `5.0` |
| Advantage normalization | Batch mean & std |
| Sampling temperature | `1.0` |
| Batch size | `8` |
| Max tokens per microbatch | `32768` |
| Training epochs | `40` |
| Trajectories per task | `16` |
| Precision | `bfloat16` |

---

## 5. 分布式与基础设施配置

Qwen3-8B RL run 使用：

- **单节点 8× H200**
- AReal 集成
- Allocation：`sglang:d4p1t1 + fsdp:c2t2`
- Max concurrent rollouts：`8`
- Rollout queue size：`32`
- Max head off-policyness：`2`
- SGLang context length：`32768`
- SGLang memory fraction：`0.8`
- Checkpoint frequency：每 `50` steps 保存一次

---

## 6. 评估设置

- 主评估 benchmark：`Terminal-Bench 1.0 / 2.0`
- 指标：`pass@1`
- Qwen3-8B 结果来自重复 8 次运行，temperature = 1，std 反映温度带来的波动。
- 论文中 Qwen3-8B SETA-RL 结果：
  - Terminal-Bench 1.0：`17.8 ± 1.2`
  - Terminal-Bench 2.0：`10.7 ± 1.3`
- 摘要中提到 best run 达到 `12%` pass@1 on Terminal-Bench 2.0；Table 3 给的是 `10.7 ± 1.3`，说明不同 run / 汇总方式有差异。

---

## 7. 一句话复现 recipe

> 用 **Qwen3-8B non-thinking**，在 **SETA-Env** 中过滤对 base model 过难的任务后均匀采样 **560 个环境**，使用 **CAMEL ChatAgent + Terminal Toolkit + Docker** 做 rollout；用 **AReal** 跑 **GRPO**，采用 **DAPO dynamic sampling** 过滤零方差组，reward 为 **unit-test pass ratio + all-pass bonus**；每条任务采样 **16 条轨迹**，Adam，constant LR `1.7e-5`，PPO clip `0.4`，KL `0`，batch size `8`，训练 `40` epochs，bfloat16，单节点 8×H200。

---

## 8. 复现时最需要注意的点

1. **必须用 non-thinking mode**，否则轨迹容易超长失效。
2. **不是全量 4,567 环境训练**，而是过滤后采样 **560 个**。
3. **DAPO dynamic sampling 很关键**：零方差 reward group 要过滤。
4. **Reward bonus 有论文内部冲突**：正文 `0.2`，Table 8 `0.5`，需看代码确认。
5. **agent max iterations = 30**，不是 DeepSeek 评估时用的 200 iterations。
6. **KL penalty 实际为 0**，不要误以为用了 reference KL regularization。

代码与数据集：
- Code：`https://github.com/camel-ai/seta`
- Dataset：`https://huggingface.co/datasets/camel-ai/SETA-Env`
- Model：`https://huggingface.co/camel-ai/Qwen3-8B-SETA-Env-RL`