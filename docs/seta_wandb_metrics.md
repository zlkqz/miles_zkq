---
title: SETA 训练的 W&B 指标清单
description: 按组解释 Qwen3-8B SETA GRPO 运行的全部 83 个 W&B history 与 summary 指标，包括统计口径、阈值和实测区间。
---

本文针对 [W&B run kvylypk5](https://forge.coreweave.com/wandb/zlkqz-chinese-academy-of-sciences/miles_seta_p2o/runs/kvylypk5)，覆盖 2026-11-02 核对时 history 与 summary 中出现的 **83 个唯一指标名**，包括 W&B 的 4 个内部字段。按该 run 已记录的 step 0–9 给出实测范围，并结合仓库当前实现解释统计口径。其他 run 的参数与口径需重新核对。

**怎样读表：**

- “边界、阈值与解读”区分数学取值范围、当前配置约束、仓库 CI 硬检查和 Dashboard 经验告警。经验告警不等于训练失败判定，也不是 W&B 自动创建的报警规则。
- “本次记录范围”是各指标已有非空记录的最小值到最大值，**不是推荐区间或健康标准**；没有通用阈值的指标不编造数字。比例在 W&B 原始数据中是 0–1，表中显示为百分数。
- history 某行的 null 表示那次写入没有这个指标，不代表指标为 0；summary 通常是各字段最近写入值的集合，不是一批完整同步快照。
- 这里覆盖训练 history/summary，不包含 W&B System 页面单独采集的 GPU、网络等 telemetry；本 run 没有记录的 eval、参考模型 KL loss 等指标也不虚列。

本次相关参数为：每批 8 个任务 × 每任务 16 条轨迹、global batch 128、单次优化器更新、GRPO、目标 LR 1.7e-5、clip_grad=1、PPO eps=0.4、TIS 上限 5。其用途仅是解释下面的边界；配方与参数的权威来源是 [启动器](../scripts/run_qwen3_8b_seta.py)，部署和使用说明见 [SETA 实现说明](/seta_miles) 与 [Quick Start](/seta_quick_start)。

## 1. 步数与时间轴

W&B 由 rollout、trainer 等多个进程共同写入。一次训练更新会写多条 history，不能把 `_step` 当成优化器步数。推荐 reward、吞吐图使用 `rollout/step`，训练 loss、梯度和 LR 图使用 `train/step`。

| 指标 | 意义与统计口径 | 边界、阈值与解读 | 本次记录范围 |
|---|---|---|---|
| `_step` | W&B history 的内部记录序号，无单位。 | 记录增加不等于完成了一次训练更新；本次有 50 条 history，但只完成 10 次更新。 | 0–49 |
| `rollout/step` | rollout 批次编号，从 0 开始；本配置每批对应一次训练更新。 | 持续增长表示有新批次完成；长期不变要结合时间和日志检查是否卡在补采。 | 0–9 |
| `train/step` | 累计训练更新编号，从 0 开始。 | 本次 0–9 表示 10 次更新；若以后每 rollout 多次更新，它不再等于 rollout 编号。 | 0–9 |
| `_timestamp` | 该条记录的 Unix 时间戳，单位秒。 | 用于与原始日志对齐；转换时区后再比较。不是训练耗时。 | 10-29 10:00:03–10-30 19:06:08 UTC |
| `_runtime` | W&B 为该条 history 记录的运行经过时间，单位秒。 | 最后一条数据的 runtime 不能说明之后是否还在有效训练；卡在下一批时可能不再写新指标。 | 4,417.8–1.2355e+05 |
| `_wandb.runtime` | W&B summary 中的运行时长元数据，单位秒。 | 摘要口径可能与 history 的最后 runtime 略有差异；不是优化器计算时间。 | 123,549（summary） |

源码：[W&B 步数绑定](../miles/utils/tracking_utils/wandb_utils.py)；[训练步编号](../miles/backends/megatron_utils/model.py)。

## 2. 任务奖励与 GRPO 学习信号

本配方的原始奖励为 `passed / total + (0.2 if passed == total else 0)`。只有验证有效的轨迹才进入训练。本次每条 episode 对应一个训练 Sample，因此两种原始奖励均值一致；未来若启用分段/压缩，按 episode 平均与按训练行平均可能不同。

| 指标 | 意义与统计口径 | 边界、阈值与解读 | 本次记录范围 |
|---|---|---|---|
| `rollout/raw_reward` | 训练侧记录的原始环境奖励均值，尚未做 GRPO 中心化。 | 当前单条有效奖励在 [0, 1.2]；0 表示测试全未通过，1.2 表示全通过。批次均值不是全通过率，也不能直接当作百分比。 | 0.18657–0.5982 |
| `rollout/episode_raw_reward` | rollout 侧按原始 episode 等权计算的原始奖励均值。 | 当前应与 raw_reward 一致。它可避免未来一个 episode 拆多行后被重复加权。 | 0.18657–0.5982 |
| `rollout/raw_reward_unfiltered` | 动态筛选前、已被统计的非空奖励的样本均值；包括随后被丢弃的组。 | 无效验证可能先返回占位 0 并标为 ABORTED，该 0 也会进入这里。故障洪峰时不能把它解读为真实能力；也不是所有启动请求的完整均值。 | 0.18337–0.54721 |
| `rollout/rewards` | 训练实际使用的组内中心化/标准差归一化后的标量奖励均值。 | 本配置均值接近 0 是预期行为；不再受 [0, 1.2] 限制。均值为 0 不代表所有样本奖励或梯度为 0。 | -4.224e-08–2.266e-07 |
| `rollout/advantages` | 策略梯度使用的优势均值；这里是训练 token 上先按样本归约再汇总。 | 当前 GRPO、KL 系数为 0，每条轨迹的归一化奖励广播到其训练 token。正优势提高该输出概率，负优势降低；均值接近 0 正常。 | -3.539e-08–2.049e-07 |
| `rollout/returns` | 回报目标的均值，归约方式同 advantages。 | 当前无 critic 的 GRPO 路径与 advantages 一致；不能沿用 PPO 的 value-target 解读。 | -3.539e-08–2.049e-07 |

源码：[SETA 奖励](../examples/seta/progress.py)；[无效奖励处理](../examples/seta/agent.py)；[过滤前统计](../miles/rollout/filter_hub/base_types.py)；[奖励归一化](../miles/ray/rollout/train_data_conversion.py)；[优势计算](../miles/backends/training_utils/loss/hub/advantages.py)。

## 3. 有效样本与动态筛选

本次每批目标是 8 个有效任务组，每组 16 条轨迹，共 128 条训练样本。以下 drop 指标是本批次收集期间的计数，不是全 run 累计值；某原因未出现时可能没有这个 key，图上的缺失点不能直接当成缺少训练数据。

| 指标 | 意义与统计口径 | 边界、阈值与解读 | 本次记录范围 |
|---|---|---|---|
| `rollout/num_training_samples` | 最终进入本批训练的 Sample 行数。 | 本配置预期 128。减少、为 0 或与配置不符需要查过滤、动态 batch 或收集失败；不等于尝试启动的总 episode 数。 | 128 |
| `rollout/dynamic_filter/drop_group_has_aborted` | 组内至少一条轨迹为 ABORTED，导致整组被丢弃的组数。 | 0 最理想；持续高于每批保留的 8 组说明补采成本很高。不存在通用合格阈值；本次 1,386 是明确的故障洪峰。不要乘 16 当作失败轨迹数，一条失败就能丢整组。 | 2–1,386 |
| `rollout/dynamic_filter/drop_zero_std_0.0` | 因组内奖励无方差而被筛掉、奖励显示标签为 0.0 的组数。 | 表示该组缺少相对优劣信号；持续很多可能是任务太难或系统性失败。后缀由奖励四舍五入到一位小数生成，并不严格保证奖励恰好为 0。 | 1 |
| `rollout/zero_std/all_zero_percentage` | 筛选后任务组中，组内奖励全部严格等于 0 的组比例。 | 取值 [0, 1]。当前启用了非零方差过滤，预期为 0；它无法反映筛选前到底丢了多少全零组。 | 0% |
| `rollout/zero_std/all_one_percentage` | 筛选后任务组中，组内奖励全部严格等于 1 的组比例。 | 取值 [0, 1]。这里比较的是 1，不是 SETA 全通过奖励 1.2，不能用它统计全通过组。当前过滤后预期为 0。 | 0% |

源码：[过滤与原因标签](../miles/rollout/filter_hub/common_filters.py)；[计数器](../miles/rollout/filter_hub/base_types.py)；[组比例](../miles/ray/rollout/metrics.py)。

## 4. 多轮轨迹与 token 长度

先区分两个长度口径：**有效模型 token** 是 `loss_mask=1` 的输出，参与策略 loss；**response 区间 token** 还包含首轮输出之后插入的工具/终端观察，观察通常 `mask=0`。`total_lengths` 则还包含初始 prompt。本配置 4,096 是单轮生成上限，28,672 是线性训练序列总预算，不能拿 4,096 限制整条多轮 episode 的输出总和。

| 指标 | 意义与统计口径 | 边界、阈值与解读 | 本次记录范围 |
|---|---|---|---|
| `rollout/episode_response_length/mean` | 每条原始 episode 的有效模型输出 token 数均值；同 episode 若分段则先相加。 | 非负；变长可能是探索，也可能是重复或困难任务，需结合 reward、截断与耗时。当前长度受总序列预算约束。 | 4,780.4–8,841.2 |
| `rollout/episode_response_length/median` | 每条 episode 有效模型 token 数的中位数。 | mean 明显大于 median 表示少数很长轨迹拉高均值；无通用健康长度。 | 3,146–9,571 |
| `rollout/episode_response_length/max` | 本批最长 episode 的有效模型 token 数。 | 接近总序列预算时结合 total_lengths 判断；它不是最长一轮输出。 | 15,142–26,136 |
| `rollout/episode_response_length/min` | 本批最短 episode 的有效模型 token 数。 | 很小需要看是否提前完成；有效训练样本若为 0，应排查空输出或全零 mask。 | 371–1,373 |
| `rollout/episode_total_response_length/mean` | 每条 episode 的完整 response 区间长度均值，包含中间观察。 | 应不小于有效模型输出均值；差值较大说明条件/工具反馈占用较多上下文。 | 11,119–18,124 |
| `rollout/response_len/mean` | 每个训练 Sample 的有效模型 token 数均值。 | 当前一 episode 一 Sample，等于 episode_response_length/mean；未来拆分后未必相等。 | 4,780.4–8,841.2 |
| `rollout/response_len/median` | 每个 Sample 有效模型 token 数的中位数。 | 当前与 episode 对应值相等；用于看典型训练行长度。 | 3,146–9,571 |
| `rollout/response_len/max` | 最长训练 Sample 的有效模型 token 数。 | 当前与 episode 对应值相等；用于定位极长样本。 | 15,142–26,136 |
| `rollout/response_len/min` | 最短训练 Sample 的有效模型 token 数。 | 当前与 episode 对应值相等；0 需要排查空训练信号。 | 371–1,373 |
| `rollout/response_lengths` | 训练侧记录的完整 response 区间长度均值，包含中间观察。 | 注意复数名与 response_len/mean 口径不同；当前等于 episode_total_response_length/mean。 | 11,119–18,124 |
| `rollout/total_lengths` | 初始 prompt + 完整 response 区间的总 token 数均值。 | 本次每条样本不应超过 28,672；均值没有超限并不能保证个别样本没有超限。与 response_lengths 的差值约为初始 prompt 长度均值。 | 12,364–19,472 |

源码：[有效长度定义](../miles/utils/types.py)；[rollout 长度统计](../miles/ray/rollout/metrics.py)；[训练侧归约](../miles/backends/training_utils/metrics/log_utils.py)。

## 5. 截断与重复输出

截断不自动等于奖励无效：SETA 在已知长度预算耗尽后仍验证环境最终状态。达到 30 次 agent 调用也不一定被标成 Sample.TRUNCATED，因而下列截断比例不是“所有预算终止”的总比例。

| 指标 | 意义与统计口径 | 边界、阈值与解读 | 本次记录范围 |
|---|---|---|---|
| `rollout/truncated_ratio` | rollout 侧最终 Sample 状态为 TRUNCATED 的比例。 | 取值 [0, 1]。仓库 Dashboard 在 ≥25% 时给出截断告警，这是经验提示，不是训练拒绝阈值；低于 25% 也需看损失的任务质量。 | 3.125%–13.281% |
| `rollout/truncated` | 训练侧对截断标志求均值。 | 当前与 truncated_ratio 相等；不能再与它相加。解读和 ≥25% 的经验参考同上。 | 3.125%–13.281% |
| `rollout/repetition_frac` | 检测到高重复/高可压缩文本的 Sample 比例。 | 取值 [0, 1]，通常越低越好，但代码、表格、终端输出也可能被命中。当前检测器对文本窗口使用原始/压缩字节比 >10 的规则；这是检测阈值，不是该比例的合格线。 | 1.5625%–7.8125% |

源码：[截断后的验证](../examples/seta/terminus.py)；[指标统计](../miles/ray/rollout/metrics.py)；[重复检测器](../miles/utils/metric_utils.py)；[Dashboard 告警](../miles/dashboard/advisory.py)。

## 6. 采样权重版本

版本号表示采样时使用的已发布权重版本。统计先为每个样本取其 decode token 所覆盖的**最旧数字版本**，再对样本求均值/中位数/最大/最小；它不是每个 token 版本的加权均值。当前同步采样流程中，批次 k 的四个值都应为 k。

| 指标 | 意义与统计口径 | 边界、阈值与解读 | 本次记录范围 |
|---|---|---|---|
| `rollout/weight_version/mean` | 每个样本最旧采样版本号的均值。 | 本配置应与本批版本一致；出现小数通常意味着批内样本来自不同版本。数值本身没有固定上限。 | 0–9 |
| `rollout/weight_version/median` | 每个样本最旧采样版本号的中位数。 | 与 mean/min/max 一起看版本分布；不能只看 median 判断没有陈旧样本。 | 0–9 |
| `rollout/weight_version/max` | 所有样本“最旧版本”中的最大值。 | 不是所有 token 的全局最新版本；样本内部的混用要看 mixed_version_ratio。 | 0–9 |
| `rollout/weight_version/min` | 本批最旧的数字采样版本。 | 若落后于本批预期版本，检查权重同步与旧请求残留；异步训练另需按允许滞后预算解释。 | 0–9 |
| `rollout/weight_version/mixed_version_ratio` | 单条 Sample 内出现多个权重版本的样本比例。 | 取值 [0, 1]，本次同步配置预期 0。0 不保证所有样本都用同一个版本，还要检查 min=max。 | 0% |

源码：[版本统计](../miles/ray/rollout/metrics.py)；[版本发布](../miles/backends/training_utils/weight_update/updater.py)。

## 7. TITO：真实采样 token 与模板的一致性

TITO 保存真实模型采样 token，再与消息经 chat template 得到的 canonical 序列比较。分母是带有 mismatch 检查 metadata 的样本数。总比例按“是否有任何 mismatch”计数；分项可能重叠，不能相加求总比例。结构不匹配时比较器会提前返回，文本分项为 0 不保证结构失败样本中的文本已逐段比较成功。

| 指标 | 意义与统计口径 | 边界、阈值与解读 | 本次记录范围 |
|---|---|---|---|
| `rollout/tito_session_mismatch_rate/v1` | 至少存在一种 TITO mismatch 的样本比例。 | 目标 0；非零先看类型与原始样本。本次 16 条 mismatch 均为 truncated，其中 14 条仅缺尾部 im_end，另 2 条还存在最后文本段 token 数差 1。 | 0%–3.125% |
| `rollout/tito_session_mismatch_rate/v1/special_token_count` | 特殊 token 分段数量或结构不同的样本比例。 | CI 默认容忍阈值为 0，可由 ci_tito_special_token_count_threshold 显式放宽；该检查只在 CI 模式执行。截断缺尾部标记需单独判断，不能补造未采样 token。 | 0%–3.125% |
| `rollout/tito_session_mismatch_rate/v1/special_token_type` | 对齐位置上的特殊 token ID 不一致的样本比例。 | 目标 0；CI 硬阈值 0。非零要查模板、角色边界和会话拼接。 | 0% |
| `rollout/tito_session_mismatch_rate/v1/non_assistant_text` | 用户、系统、工具等非 assistant 区域 token 不一致的样本比例。 | 目标 0；CI 硬阈值 0。模型条件可能发生错位，应优先检查。 | 0% |
| `rollout/tito_session_mismatch_rate/v1/assistant_text` | assistant 内容 token 与 canonical 重新编码不一致的样本比例。 | 没有固定拒绝阈值：跨轮复用真实 token 可能与重新 tokenize 的结果不同。按比较器定义属于较轻类型；不能据此忽略任意文本变化。 | 0% |

源码：[TITO 指标与 CI 检查](../miles/ray/rollout/metrics.py)；[类型及提前返回行为](../miles/utils/chat_template_utils/token_seq_comparator.py)；[CI 默认参数](../miles/utils/arguments.py)。

## 8. Loss、梯度与学习率

本次使用 GRPO，`entropy_coef=0`、`observe_training_entropy=false`，也没有参考模型 KL loss。每批 128 条样本只做一次优化器更新。Loss 的数值与监督学习交叉熵不同，没有必须保持正数或单调下降的要求。

| 指标 | 意义与统计口径 | 边界、阈值与解读 | 本次记录范围 |
|---|---|---|---|
| `train/loss` | 本次更新的总训练目标；通常是 PG loss 减熵奖励，再加启用的 KL loss 等项。 | 当前等于 pg_loss，可正可负；接近 0 不等于没有梯度。NaN/Inf 或相对自身基线突然放大需要排查。 | -0.0001191–1.556e-05 |
| `train/pg_loss` | 带 PPO clipping、并乘 TIS 校正权重后的策略梯度损失。 | 组内优势中心化后均值可很小；应结合 grad_norm、reward 和独立评测判断，不能按 SFT loss 越低越好解释。 | -0.0001191–1.556e-05 |
| `train/entropy_loss` | 启用计算时是策略分布熵的归约值 H，loss 中贡献为 -entropy_coef × H。 | 当前未计算，0 是占位值，不代表策略熵为 0。启用后持续下降表示分布更集中，但是否过早塌缩需结合任务表现，没有通用下限。 | 0 |
| `train/grad_norm` | 优化器报告的裁剪前全局梯度 L2 范数。 | 本配置 clip_grad=1：>1 会触发梯度裁剪；这不是数值会被记录成 1 的上限。持续 >1、突然暴涨或长期为 0 要检查；有限且 <1 只代表未触发该裁剪。 | 0.15814–0.30145 |
| `train/lr-pg_0` | 优化器第 0 个参数组的调度学习率。 | 当前从 warmup 的小值升向目标 1.7e-5；持续为 0 才需检查调度/参数冻结。实际最优 LR 依模型与任务而定。 | 7.456e-07–7.456e-06 |
| `train/lr-pg_1` | 优化器第 1 个参数组的调度学习率。 | 不是第二个模型或第二张 GPU 的 LR；组号对应 optimizer.param_groups，本次与 pg_0 相同，不能单凭组号推断具体包含哪些参数。 | 7.456e-07–7.456e-06 |

源码：[训练目标](../miles/backends/training_utils/loss/hub/losses.py)；[梯度指标定义](../miles/backends/training_utils/metrics/log_utils.py)；[参数组 LR](../miles/backends/megatron_utils/model.py)；[当前配方](../scripts/run_qwen3_8b_seta.py)。

## 9. PPO 更新幅度与有效权重

设 `r = exp(log π_new - log π_old)`。当前 old policy 是训练侧更新前重新计算的 log-prob，因为 `use_rollout_logprobs=false`。它与下一组中用于 TIS 的“采样引擎策略”不是同一个比较对象。

| 指标 | 意义与统计口径 | 边界、阈值与解读 | 本次记录范围 |
|---|---|---|---|
| `train/ppo_kl` | 对训练 token 的 log π_old - log π_new 求均值，属于带符号的采样 log-ratio 统计。 | 并非严格非负的完整 KL；有限样本时可为负。当前每批只一次更新，loss 前向时 new 与 old 一致，0 正常；它不衡量更新完成后的新旧模型差异。 | 0 |
| `train/pg_clipfrac` | clipped 分支实际使 PG loss 受限的 token 比例，考虑优势的正负方向。 | 取值 [0, 1]。本次 eps_clip=eps_clip_high=0.4，对应 ratio 的 [0.6, 1.4] clip 区间；并非所有区间外 token 都计入 clipfrac。当前预期 0。多次更新时持续升高表示限制频繁生效，无统一合格线。 | 0% |
| `train/ess_ratio` | 按每条样本有效 token 的 r 计算 (Σr)² / (N × Σr²)，再归约的相对有效样本量。 | 有有效 token 时理论在 (0, 1]。1 表示该样本内权重一致；接近 0 表示权重集中。它是 token 权重的 ESS，不等于“有效轨迹数/128”，也不专门检查 train/rollout mismatch。 | 1 |

源码：[PPO loss 与 ESS](../miles/backends/training_utils/loss/hub/math_utils.py)；[old policy 选择](../miles/backends/training_utils/loss/hub/losses.py)。

## 10. 训练/采样概率对齐与 TIS

设 `p = π_train_old` 为训练侧独立打分策略，`q = π_rollout` 为 SGLang 实际采样策略。TIS 使用 `w = p(token|context) / q(token|context)` 修正两侧差异；当前 w 裁剪到 [0, 5]。以下概率差异只在有效模型 token 上归约。本次默认先对每条 Sample 求均值，再在 batch/并行 rank 间汇总，不能当成所有 token 直接拼接后的加权均值。

| 指标 | 意义与统计口径 | 边界、阈值与解读 | 本次记录范围 |
|---|---|---|---|
| `rollout/log_probs` | 训练侧为已采样 token 重新计算的 log p 的均值，使用自然对数。 | 有效 log 概率应 ≤0；越接近 0 表示对这些既定输出越确信，不保证答案越正确。与 rollout_log_probs 的两个均值相近也不能保证逐 token 相同。 | -0.44185–-0.18272 |
| `rollout/rollout_log_probs` | SGLang 实际采样并由 TITO 保存的 token log q 均值。 | 有效值应 ≤0；用于训练/采样对照，不是参考模型 log-prob。单个 token 的均值也不是整条序列的概率。 | -0.44144–-0.18227 |
| `train/train_rollout_logprob_abs_diff` | 有效 token 上 abs(log p - log q) 的归约均值。 | 理想接近 0，越大说明独立打分差异越大；没有通用硬阈值。本次约 0.00835–0.01145，后续应与相同精度/模板的自身基线比较，并检查尾部。 | 0.008349–0.011453 |
| `train/train_rollout_kl` | KL(q ∥ p) 的逐 token k3 近似：w - 1 - log w，再归约。 | 理想接近 0；与 ppo_kl 的比较对象和估计方法不同。当前实现对 log-ratio 和结果有数值截断，不能把它当成未经截断的精确 KL 或对参考模型的 KL。 | 0.0003994–0.0004978 |
| `train/ois` | exp(log π_new - log π_old) 的归约均值，也就是 PPO 的未截断概率比。 | 理想在 1 附近；本次为 1 是当前一次更新口径下的预期。它不是 TIS 的 p/q；均值为 1 不保证每个 token 的比例都为 1。 | 1 |
| `train/tis` | 校正前、尚未 clip 的 w=p/q 的归约均值。 | >0，通常希望接近 1；不能把 tis_clip=5 当成该指标上限，它记录的是 clip 前值。不同方向偏差可能在均值中抵消。 | 0.99994–1.0001 |
| `train/tis_abs` | abs(w - 1) 的归约均值，直接反映校正幅度。 | ≥0，越接近 0 表示越少需要校正；无通用阈值。本次约 0.00834–0.01144。与 abs(mean(w)-1) 不是一回事。 | 0.0083377–0.01144 |
| `train/tis_clipfrac` | w 被 TIS 上下界实际裁剪的有效 token 比例。 | 取值 [0, 1]，本配置 w>5 时触发上界 clip。0 表示没有命中边界，并不表示 p=q；持续增大要查精度、版本、模板与 log-prob 口径。 | 0%–0.00029548% |

源码：[TIS 公式](../miles/backends/training_utils/loss/hub/corrections.py)；[概率对齐指标](../miles/backends/training_utils/loss/hub/losses.py)；[KL 与 ESS 数学实现](../miles/backends/training_utils/loss/hub/math_utils.py)；[训练侧统计](../miles/backends/training_utils/metrics/log_utils.py)。

## 11. 前缀缓存

同一多轮任务会反复提交增长中的历史，缓存统计会跨模型调用累计。因此缓存 token 计数可以远大于单条轨迹最终长度；缓存命中率高也不能证明沙盒或整个训练流程健康。

| 指标 | 意义与统计口径 | 边界、阈值与解读 | 本次记录范围 |
|---|---|---|---|
| `rollout/prefix_cache_hit_rate` | 所有保留样本、所有已累计模型调用的 cached_tokens 总和 / prompt_tokens 总和。 | 取值 [0, 1]，通常越高越省 prefill；无 prompt 时实现返回 0。Dashboard 将 <10% 作为低缓存利用的经验提示，仅在无活跃故障且满足调优条件时使用，不是正确性阈值。 | 93.772%–97.103% |
| `rollout/avg_cached_tokens_per_sample` | 上述 cached_tokens 总和 / Sample 数，单位 token。 | ≥0，可超过 28,672；它是累计复用量，不是 KV cache 实际驻留 token 数或显存大小。 | 1.1282e+05–2.1074e+05 |

源码：[缓存累计定义](../miles/utils/types.py)；[批次缓存统计](../miles/ray/rollout/metrics.py)；[低缓存经验提示](../miles/dashboard/advisory.py)。

## 12. 整批耗时与等待占比

下列所有 time 原始单位均为秒。`step_time = train_wait_time + train_time`，`wait_time_ratio = train_wait_time / step_time`。这两个是训练 actor 的计时口径，不是把每个 perf 子阶段不重叠地加一遍。

| 指标 | 意义与统计口径 | 边界、阈值与解读 | 本次记录范围 |
|---|---|---|---|
| `perf/rollout_time` | 收齐本批有效样本的 rollout 墙钟耗时，含 agent 交互、环境等待和动态补采。 | 无通用阈值；与相同任务长度和并发的基线比较。本次常规约 1.2–1.5 小时，step 5 为 2.38 小时，step 1 为 18.44 小时。 | 4,296.7–66,369 |
| `perf/train_wait_time` | 训练 actor 不在 train 计时区间内的累计时间；同步 colocate 下主要等待 rollout，也可能包含同步/阶段切换等。 | 不是所有 GPU 的空闲时间；采样阶段 GPU 仍可能推理。远高于 train_time 时优先排查数据供应和沙盒。 | 4,358.8–66,413 |
| `perf/train_time` | actor 的 train 计时区间，包括 log-prob 前向、优势处理和 actor 更新等。 | 比 actor_train_time 口径宽；不等于纯反向时间，也不包含 rollout 的大部分等待。 | 847.21–1,360 |
| `perf/step_time` | train_wait_time + train_time，单位秒。 | 用于看每完成一批更新的节奏；不要再加 rollout_time，以免重复计数。不是严格覆盖启动/退出的端到端运行时长。 | 5,284.6–67,359 |
| `perf/wait_time_ratio` | 训练 actor 等待时间 / 上述 step_time。 | 取值 [0, 1]；接近 0 为 train 区间占主导，接近 1 为供应/等待占主导，没有通用目标。本次常规约 80%–86%，step 1 为 98.60%，但不能据此说 GPU 空闲了同样比例。 | 80.054%–98.596% |

源码：[rollout 耗时](../miles/ray/rollout/metrics.py)；[计时区间](../miles/backends/megatron_utils/actor.py)；[总耗时公式](../miles/backends/training_utils/metrics/perf.py)；[计时器](../miles/utils/timer.py)。

## 13. 训练与权重同步的子阶段耗时

这些指标来自同一统计窗口里的具名计时器。它们可能嵌套，或属于前后阶段切换；不要把所有子项直接相加当成 step_time。比较时还要控制 token 数和序列长度。

| 指标 | 意义与统计口径 | 边界、阈值与解读 | 本次记录范围 |
|---|---|---|---|
| `perf/sleep_time` | 训练 actor 卸载模型/梯度等状态、释放 GPU 资源所花时间。 | 不是人工 sleep 的等待时长；colocate 下属于训练与推理切换成本，突增时查 CPU/GPU 数据传输及卸载。 | 4.659–21.909 |
| `perf/wake_up_time` | 训练 actor 恢复驻留状态、清理/恢复相关资源和通信组的时间。 | 突增会增加切换成本；无固定合格秒数。 | 3.1357–11.336 |
| `perf/data_preprocess_time` | 取回 rollout 数据引用并整理训练输入的时间。 | 持续放大时查对象传输、CPU 处理和数据体积；并非数据集准备脚本的总耗时。 | 0.22526–0.46723 |
| `perf/log_probs_time` | 训练侧对固定轨迹重新计算 old-policy log-prob 的前向耗时。 | 本次启用该前向；不是 SGLang 生成耗时。变长样本会增加它，必要性由算法口径决定。 | 176.26–277.58 |
| `perf/actor_train_time` | actor 策略更新阶段耗时，包括训练前向、反向及优化器更新等。 | 不是纯反向；是 actor_train_tflops、actor_train_tok_per_s 和 MFU 的分母。 | 657.44–1,053.8 |
| `perf/update_weights_time` | 把训练后 actor 权重同步给推理引擎的外围总耗时。 | 无通用阈值；与模型大小、互联和同步方式有关。它不是 optimizer.step 的时间。 | 4.2935–4.3961 |
| `perf/update_weights_implementation_time` | 权重同步内部的分桶转换、传送及协议收尾/屏障阶段耗时。 | 是 update_weights 总过程中的子区间；不要再与外围总耗时相加。 | 3.5518–3.712 |
| `perf/finalize_and_resume_engines_time` | 完成权重同步协议、发布版本并恢复推理引擎等收尾阶段耗时。 | 也是同步内部子阶段；突然增加时查发布确认、屏障或引擎恢复。 | 0.047966–0.057544 |

源码：[actor 具名计时](../miles/backends/megatron_utils/actor.py)；[权重同步内部计时](../miles/backends/training_utils/weight_update/updater.py)；[计时汇总](../miles/backends/training_utils/metrics/perf.py)。

## 14. 吞吐、估算算力与 MFU

采样吞吐的分子只使用最终保留的样本，分母却包含本批补采耗时；失败越多，这个“交付有效 batch 的吞吐”就越低。FLOP 与 MFU 是按模型和序列长度估算的值，不是硬件 profiler 实测的 kernel FLOP。

| 指标 | 意义与统计口径 | 边界、阈值与解读 | 本次记录范围 |
|---|---|---|---|
| `perf/tokens_per_gpu_per_sec` | Σ完整 response 区间 token / rollout_time / rollout GPU 数。 | 单位 token/(GPU·秒)，≥0；这里包含中间观察 token，不是纯模型 decode 速度。没有跨模型通用阈值。 | 2.9818–53.088 |
| `perf/effective_tokens_per_gpu_per_sec` | Σ有效模型输出 token / rollout_time / rollout GPU 数。 | 单位 token/(GPU·秒)，通常更适合衡量当前 agent 训练的有效采样产出；同一批应不大于包含观察的吞吐。 | 1.6297–25.864 |
| `perf/longest_sample_tokens_per_sec` | max(完整 response 区间长度) / 整批 rollout_time。 | 单位 token/秒，不按 GPU 数除；不是最长轨迹自身用时，也不是引擎峰值速度。 | 0.41488–6.408 |
| `perf/longest_effective_sample_tokens_per_sec` | max(有效模型输出长度) / 整批 rollout_time。 | 单位 token/秒；它选出的轨迹未必与上一个指标是同一条，不是逐轨迹测速。 | 0.27727–5.814 |
| `perf/log_probs_tflops` | 估算的单 GPU 前向 TFLOP 总量 / log_probs_time。 | 单位 TFLOP/s/GPU，越高通常表示该前向阶段更高效；受序列、并行布局与计算估算口径影响。 | 22.752–26.042 |
| `perf/actor_train_tflops` | 3 × 估算单 GPU 前向 TFLOP 总量 / actor_train_time。 | 单位 TFLOP/s/GPU；系数 3 近似覆盖前向与反向，不是精确计入每次 recompute 的硬件 FLOP。 | 18.561–20.579 |
| `perf/actor_train_tok_per_s` | Σ本次训练计数的完整序列长度 / actor_train_time。 | 单位 token/秒；分子包含 prompt 和观察，不仅是 loss_mask=1 的 token。源码未再除 GPU 数，不能与前面的 per-GPU 采样吞吐直接比较。 | 2,350.9–2,412.5 |
| `perf/mfu_peak_tflops` | 计算 MFU 使用的单卡 BF16 理论峰值分母。 | 本次为 312 TFLOP/s。是设备峰值参考或显式配置值，不是实际测到的算力；设备型号或分母不对会让 MFU 失真。 | 312 |
| `perf/actor_train_mfu` | actor_train_tflops / mfu_peak_tflops。 | 本次约 5.95%–6.60%。Dashboard 以至少 3 个去掉首步后的稳态点、均值 <15% 给出低 MFU 经验告警；它衡量 actor_train 区间效率，rollout 等待不能直接拉低它。>100% 应先复核分母和估算口径。 | 5.9492%–6.5959% |

源码：[采样吞吐公式](../miles/ray/rollout/metrics.py)；[训练吞吐与 MFU](../miles/backends/training_utils/metrics/perf.py)；[FLOP 估算](../miles/utils/flops_utils.py)；[MFU 经验告警](../miles/dashboard/advisory.py)。

## 15. CPU 内存

该指标只由指定训练 rank 报告一次，但“由一个进程上报”不等于“只统计该进程”。

| 指标 | 意义与统计口径 | 边界、阈值与解读 | 本次记录范围 |
|---|---|---|---|
| `perf/cpu_memory_after_offload_train_gb` | 训练状态卸载后，psutil.virtual_memory().used / 1e9 返回的系统已用 CPU 内存。 | 单位十进制 GB，不是 GiB；不是训练进程 RSS，也不含远端 CPU 沙盒机器的内存。与同一系统容量、available 和容器限制一起判断，没有固定 GB 阈值；其他进程也会影响它。 | 317.4–339.05 |

源码：[系统内存统计](../miles/backends/training_utils/metrics/log_utils.py)。

## 16. 实际看图时的组合与误区

| 想判断什么 | 一起看哪些指标 | 需要避免的误读 |
|---|---|---|
| 任务有没有学好 | raw_reward、episode_raw_reward，加固定任务集独立 eval | 训练 batch 每次只有 8 个不同任务且经过筛选；reward 上涨不能独自证明泛化改善。本 run 未记录独立 eval。 |
| 有效数据是否供得上 | num_training_samples、drop_group_has_aborted、rollout_time、train/step 是否推进 | 收齐后的 128 并不说明此前没有失败；卡在未完成的一批时，新的汇总指标可能根本不会出现。 |
| 梯度和更新是否异常 | loss、pg_loss、grad_norm、LR、ppo_kl、pg_clipfrac | 当前一次更新下 loss 近 0、PPO KL=0、clipfrac=0 可以正常；grad_norm 非零也不能单独证明模型能力提升。 |
| 训练和采样是否对齐 | train_rollout_logprob_abs_diff、train_rollout_kl、tis_abs、tis_clipfrac、weight_version、TITO 分项 | tis 的均值约 1、ois=1 或 ESS=1，不能代替上述逐 token/版本/模板检查。 |
| 是否卡在长度或重复 | 截断率、repetition_frac、长度的 mean/median/max、reward | 不把 per-turn 的 4,096 当作整条 episode 的总输出上限；不把所有 30 轮终止都算成 TRUNCATED。 |
| 时间花在哪里 | rollout_time、train_wait_time、train_time、actor_train_time、wait_time_ratio | 等待占比高不等于 GPU 全程空闲，采样也在使用 GPU；子计时器有嵌套，不直接相加。 |
| 训练计算是否高效 | actor_train_tflops、actor_train_mfu、actor_train_tok_per_s | MFU 只看训练区间；actor_train_tok_per_s 不是纯模型生成 token/s。 |
| CPU 内存是否紧张 | cpu_memory_after_offload_train_gb，加系统 available 与容器限制 | 系统 used 不能全归因于本训练，也不能用来判断远端沙盒内存。 |

可以从以下已存在的数值规则开始排查，其他指标优先比较自身稳定基线：

- **grad_norm > 1：**本配方触发梯度裁剪；偶发触发与长期持续触发需要区别看待。
- **PPO ratio 超出 [0.6, 1.4]：**进入可能触发 clipping 的范围，是否实际限制 loss 还取决于优势符号。
- **TIS ratio > 5：**本配方触发上界校正裁剪；tis 指标本身仍记录裁剪前的值。
- **截断率 ≥25%：**仓库 Dashboard 的经验告警。需要确认是单轮输出预算、总上下文还是其他终止原因。
- **稳态平均 MFU <15%：**仓库 Dashboard 的经验告警，要求去掉首步后至少 3 个点；先处理管线故障，再分析重计算、microbatch 和并行布局。
- **TITO non_assistant_text / special_token_type >0：**CI 不容忍；special_token_count 的 CI 默认阈值也是 0，但截断边界需要读明细。正式运行不会因为这些 CI 规则自动停止。

逐步诊断及原始核对证据保存在仓库的 `seta_miles_runtime/diagnostics/training_review_kvylypk5_20261102/`；其中 `wandb_evidence.json` 提供完整指标键与历史数值，`steps.csv` 和 `report.md` 记录本次运行的分析。这里不重复部署故障处置流程。
