# P2O + SETA 训练与恢复

使用现有 [SETA 环境](seta_quick_start.md)。P2O 已接入同一个 Python 启动器 `scripts/run_qwen3_8b_seta.py`；算法与默认参数的权威定义分别在启动器、`examples/seta/p2o/scoring.py`、`search.py`、`mixing.py`、`training.py`。实施及实际验收证据见 [进度记录](p2o_implementation_progress.md)。

启动到参数更新、GEPA内部流程、轨迹陈旧度与共享沙盒的隔离边界见 [调用链与运行机制](p2o_call_chain.md)。

## 正式启动

在已有 GPU 训练容器的仓库目录运行：

```bash
SETA_ALGORITHM=p2o MODE=plan bash my_scripts/run_seta.sh
SETA_ALGORITHM=p2o bash my_scripts/run_seta.sh
```

共享脚本按算法生成带 `_p2o_` 的 EXP_NAME、日志目录和 SAVE_PATH，GRPO/DAPO 的独立入口继续可用。P2O 使用原生 fully-async，关闭同分组过滤，开启双上下文 old-policy 打分、CRM 和原生 IcePop。正常训练卡数由原有 train/infer 配置决定，多机仍按 SETA Quick Start 的原生 Ray 流程启动。

W&B 凭据不可用时可先离线记录（本次验收遇到 HTTP401，采用此方式）：

```bash
WANDB_MODE=offline SETA_ALGORITHM=p2o bash my_scripts/run_seta.sh
```

训练及GEPA指标仍写入本地W&B目录和日志；配置有效凭据后，使用 `wandb sync <offline-run目录>` 上传。离线记录不等于已经上传。

常用参数通过 Python 启动器的 CLI 传入，例如：

```bash
SETA_ALGORITHM=p2o bash my_scripts/run_seta.sh \
  --gepa-trigger-unique-hard-tasks 64 \
  --gepa-dev-task-count 16 --gepa-dev-repeats 4 \
  --gepa-concurrency 16 --gepa-budget 256 --gepa-patience 32 \
  --gepa-reflection-num-gpus 1 --gepa-reflection-tp 1 --gepa-policy-tp 1
```

反思卡数是整个 run 的总数，不是每节点卡数；只在 GEPA 期间划给初始 HF 模型，其余已分配 GPU 推理当前冻结 policy。训练暂停、全部在途任务收尾后才切换；GEPA 收尾后恢复原训练拓扑和权重。policy TP 默认1以利用奇数剩余卡；自定义TP必须能按每个节点的剩余GPU整除。

`--hard-reward-threshold` 默认由 K 自动计算，不含 full-pass bonus。其他训练和 GEPA 分数保留原奖励。`--gepa-minibatch`、`--gepa-reflection-max-turns`、`--gepa-max-template-tokens` 也可调整。混入和 CRM 参数由插件注册，可通过 `EXTRA_ARGS` 传递，例如 `--p2o-task-mix-ratio 0.25 --p2o-max-template-ratio 0.5 --p2o-crm-lower 0.01 --p2o-crm-upper 10`。

触发门槛必须不小于训练任务batch数，并为 dev 之外留下一个完整 mini-batch；batch至少2个任务。P2O 暂不支持同步/colocate、外部推理服务、critic或多policy训练。不可把普通GRPO checkpoint直接当作完整P2O断点。

## 保存与恢复

使用相同 SAVE_PATH 重启同一命令即可恢复。入口先读取 `SAVE_PATH/p2o_checkpoint.json`，将 Miles 的 load 根目录指向其中的完整配对快照；save 根目录仍为原 SAVE_PATH。无需手动挑选模型或rollout文件。

P2O入口启用原生 `--use-checkpoint-opt-param-scheduler`，恢复时沿用断点里的scheduler参数和进度。延长NUM_ROLLOUT只延长执行次数，不重新规划学习率曲线；不启用finetune或跳过optimizer/RNG加载。

- 原生模型、optimizer、scheduler和RNG：`SAVE_PATH/iter_XXXXXXX/`。
- 不可变配对快照：`SAVE_PATH/p2o_checkpoints/step_*_cycle_*_{periodic,pre_gepa,post_gepa}_*/`，包含模型目录链接与独立rollout状态。
- 难任务身份、收集器、模板历史/求解证据/分配、抽样RNG、自然数据游标、待提交窗口、在途重试与训练尾批：随配对的rollout快照保存。
- 搜索面板、预算预留、父模板/mini-batch/seed、候选/前沿、完成事件和停止锁存：`SAVE_PATH/gepa/cycle_XXXXXX/search_state.json`。已完成episode原始结果与session轨迹放在同目录 `episodes/`，恢复可复用；外部调用若在结果落盘前中断，可能重做，但不重复消费探索预算或提交candidate。

save_interval、GEPA之前和之后都会保存。前后没有参数更新时共享同一原生模型目录，分别保存rollout状态，最后原子发布配对指针。搜索或恢复资源失败时停止训练，保留上一完整指针，不把半完成状态当成功继续。不要手工删除仍被配对快照引用的模型目录。

## 日志与指标

每轮 `SAVE_PATH/gepa/cycle_XXXXXX/` 保存 panel、资源映射、模板、逐episode原始数据、搜索事务状态、metrics JSON/JSONL及PNG/SVG图。W&B 使用 `gepa/cycle_N/event` 独立横轴，记录候选数、Pareto数、已启动/完成探索、pending dev、best dev、patience、请求/turn/无效数等；同时保留关联训练step。终端日志打印阶段切换、预算预留、候选准入/拒绝、奖励、请求耗时、checkpoint配对路径及资源恢复状态。

无有效新模板时GEPA正常结束，历史难身份仍保留。整个历史模板池为空时训练使用原prompt；有历史模板后再按已知求解/旧分配/轮内归一化概率选择。短验收中未找到有效模板不能解读为实现失败，也不能作为方法效果结论。

真实短训练和恢复测试的最终状态、配置与限制在进度记录中维护。生产默认值不由诊断脚本覆盖；诊断使用独立数据/输出/SAVE目录。
