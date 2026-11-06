# P2O + SETA 实现 TODO 与验收记录

更新日期：2026-11-05。

算法设计以 [P2O_SETA_PLAN.md](../P2O_SETA_PLAN.md) 为准；本文只维护实施顺序、验收证据和问题记录。运行环境与部署方式沿用 [SETA 实现说明](seta_miles.md) 和 [Quick Start](seta_quick_start.md)。

## 工作约定

- 用户已授权正式实现，并允许使用本机 8 张 GPU 调试；沙盒使用 144 机器的既有 SETA 服务。
- 按下列 TODO 组逐组推进。每组编写或扩展功能测试，执行适合本组的检查，并记录命令、结果、日志位置、限制和改动文件。
- 最新用户授权：每组完成实现及测试后直接推进下一组，不再等待逐组确认；只在无法自行解决或需要用户决定的问题上暂停。最后须以独立调试参数覆盖GEPA并完成短训练，正式默认参数不因调试改变。
- CPU 测试通过不代替真实 GPU、Ray 或 Harbor 验收。单机通过不宣称多机已经实测。
- 每次调试使用独立实验名、运行身份和保存路径；只清理本次任务的进程、作业和沙盒。保留已有修改与训练产物。
- 算法约束变化更新主计划；实现和测试进展记录在本文，避免维护两份算法规格。

## TODO 总表

G1–G10已完成，各组测试及本机真实短训练验收通过。累计4个训练step、两轮加速GEPA，pre-GEPA与周期checkpoint实际恢复成功。最终证据见本文末尾G10交付记录及 [验收审计](../seta_miles_runtime/diagnostics/p2o_g10_20261105/acceptance_audit.json)。以下执行日志保留历史过程，以本段与最终交付记录为当前状态。

| 组 | 交付目标 | 主要测试与验收 | 状态 |
| --- | --- | --- | --- |
| G1 | 基线验收、测试环境与 IcePop 配置收尾 | GRPO/DAPO 参数和命令快照、IcePop/PPO 边界、现有 SETA CPU 回归、144 服务健康检查 | 已完成，用户已确认进入下一组 |
| G2 | 难任务与 GEPA 评分基础模块 | 无 bonus 判难、有效性、去重、H>64、固定 dev 划分、重复评价聚合、状态序列化 | 已完成（验收证据见下文） |
| G3 | 真实多轮模板注入与双上下文对齐验证 | 144 沙盒实际输入、两路 action token 对齐、工具 mask、长度边界、P2O 关闭回归 | 已完成（验收证据见下文） |
| G4 | Miles 原拓扑 → GEPA 推理 → 原拓扑资源验证 | 8 卡真实切换、初始反思权重、边界 policy 权重、optimizer 连续、失败回收、模拟多机映射 | 已完成（验收证据见下文） |
| G5 | 难任务收集接入、历史模板库与混入 | 自然/主动来源、模板优先级、轮内归一化、异步采样边界、缺池不等待 | 已完成（验收证据见下文） |
| G6 | Megatron 双上下文打分与 CRM + IcePop | 手算 loss/梯度、token 对齐、原分母、零优势/全 mask、实际 GPU 更新 | 已完成（验收证据见下文） |
| G7 | 异步暂停、收尾与 GEPA 前后 checkpoint | 触发幂等、慢任务、尾批保存、恢复、同 step 不同 phase、保存失败 | 已完成（验收证据见下文） |
| G8 | 独立轻量反思沙盒与证据文件 | 无效轨迹导出、只读约束、唯一模板写入、真实 Terminus-2 反思、并发和取消 | 已完成（验收证据见下文） |
| G9 | 全异步 GEPA 搜索与逐事件恢复 | C=1/16/17、Pareto、mini 准入、预算/早停、重复事件、真实小预算 cycle | 已完成（CPU与真实小预算搜索） |
| G10 | P2O 入口、W&B/本地记录与两轮端到端验收 | 两个 cycle、模板训练、断点恢复、指标/原始数据/绘图、退出清理、基线回归 | 已完成（本机短训练、两轮GEPA与实际恢复；W&B离线） |

## 分组实施清单

### G1：可执行基线与验证环境

- [x] 在既有 GPU 训练容器中确认 Python、pytest、Miles 导入与实际挂载，解决此前宿主机缺少依赖造成的测试阻塞。
- [x] 补齐上一轮 IcePop 启动命令快照；确认 GRPO/DAPO 只在同分过滤上不同，两者均继承用户指定的修正与 PPO clip。
- [x] 扩展现有启动器测试，验证默认配置、显式参数覆盖与最终命令；给 IcePop 数值边界编写针对性测试。
- [x] 执行启动器要求的测试套件与 SETA 奖励/远端客户端相关回归。失败保留日志并区分本次改动和已有问题，不把未运行报告成通过。
- [x] 按 Quick Start 执行 144 服务健康检查，并进行至多 1–2 steps 的独立短训练检查；不启动长时间实验。
- [x] 更新当前 SETA 文档中已经过时的基线名称、过滤开关和重要性修正描述。
- [x] 整理改动、测试和下一组范围并汇报；用户确认前不进入G2。

### G2：难任务与评分基础

- [x] 建立稳定任务身份、有效 episode 证据、自然/主动/搜索来源、难任务收集器和冻结 cycle 面板的数据结构。
- [x] 从现有 verifier 元数据读取 passed/total；仅判难用 progress，保留原训练与搜索 reward。
- [x] 实现完整有效组的阈值判定、自然任务去重、收尾后 H 个任务固定划分与可恢复随机状态。
- [x] 实现 dev 有效重复均值和全无效为零的搜索评分约定，保存有效数和错误信息。
- [x] 编写边界及状态往返测试：1/16 含等号、部分通过、bonus 不干扰判难、无效不冒充零分、H>64 不丢弃、dev/search 不交叠、3/2/1/0 次有效评价。
- [x] 先交付可测试模块；训练链路接通在 G5，不能把独立模块通过称为已启用 P2O。
- [x] 整理并汇报本组结果；用户确认前不进入G3。

### G3：实际输入与 TITO 双上下文

- [x] 确认模板进入远端实际 instruction/消息，而非仅修改 GPU JSONL prompt。
- [x] 明确模板片段在实际 token 历史中的位置和版本，保留原始动作/观察，不靠 decode 后重新 tokenize 重建动作。
- [x] 获取一条真实多轮轨迹，构造增强/原始上下文的评分输入及动作索引映射，记录脱敏 fixture。
- [x] 编写并执行多轮、stop token、截断、超长模板和工具 mask 测试；真实 144 沙盒验证输入确实含模板。
- [x] 如 TITO 无法支持可靠的双上下文对齐，停下来报告接口限制；不以近似对齐继续训练。
- [x] 完成验收，按最新授权直接进入G4。

### G4：GPU 资源与模型生命周期

- [x] 审计 Miles 现有 placement、offload、engine 生命周期与权重同步接口，确定可恢复切换的最小扩展。
- [x] 实现本 run 内的 phase 布局规划、配置化 reflection GPU 数/TP、node 与本地设备映射及失败回滚。
- [x] 在本机 8 卡执行最小真实切换：正常训练状态 → GEPA policy/初始 reflection 推理 → 原训练状态。
- [x] 检查两种模型权重来源、路由隔离、显存/Ray 资源释放、恢复后的 optimizer step、训练权重和 RNG；检查单卡反思的上下文显存容量。
- [x] 编写多机布局模拟与非法拓扑测试。多机实际 NCCL 验收未执行。
- [x] 若现有架构不能无损切换或模型容量不足，说明实测证据和替代方案后暂停，不静默减少 policy 算力或改变训练拓扑。
- [x] 记录本组实现与测试证据；验收通过后直接进入下一组。

### G5：收集、历史模板与训练混入

- [x] 将 G2 收集器接入真实组结果；难身份、最近 GEPA 轮次和模板来源分开保存。
- [x] 实现历史模板去重、每轮分数归一化及稳定概率抽样；求解证据/旧分配/全历史池按主计划决定优先级。
- [x] 按主计划第0.7节接入双阶段随机初选及条件补/替换；可用池不足时随机回退，不篡改已有轨迹的模板标记、不为比例等待。
- [x] 测试 0/1/2/3/4/5 个自然命中、最近轮次优先、空池、有限池、完成乱序、任务不丢失、同组不同模板、重复 hash 和零方差归一化。
- [x] 训练链路输出可审计的触发与混入指标；P2O 关闭时保持基线数据流。
- [x] 记录本组实现与测试证据；验收通过后直接进入下一组。

### G6：训练 loss 与双路 old 打分

- [x] 接通 rollout → train data → Megatron batch 的双上下文字段与动作索引。
- [x] 更新前同一 old actor 对两路上下文打分，训练 forward 使用去模板上下文；行为 log-prob 保留真实生成版本。
- [x] 接入 CRM，与 IcePop 的行为修正、PPO ratio 分开，保持原 mask 分母与 episode/group 优势语义。
- [x] 编写概率与梯度测试，覆盖边界、detach、全 mask、无模板退化、零优势、工具 token、多轮和异步版本差异。
- [x] 使用真实轨迹运行最小 GPU 更新，对比 P2O 关闭及无模板路径，记录 loss、梯度和有限值检查。
- [x] 记录本组实现与测试证据；验收通过后直接进入下一组。

### G7：异步边界与持久化

- [x] 实现 trigger_pending、停止新提交、在途收尾、训练/权重同步边界及 pending_train 恢复。
- [x] 在现有 save interval 外接入 pre_gepa/post_gepa 保存，使用原子 manifest 对齐模型和 P2O 状态。
- [x] 用模拟慢任务、取消、跨版本尾批、训练预算耗尽与保存失败测试控制流，避免停止 producer 后 get() 永远等不满 batch。
- [x] 验证同 step 两种 phase 不覆盖、游标与 RNG 连续、恢复不重复收集/注入/触发；执行一次真实保存恢复。
- [x] 记录本组实现与测试证据；验收通过后直接进入下一组。

### G8：受限 agentic reflection

- [x] 复用 Harbor 证据导出：实际 rollout 输入、轨迹、feedback、评估脚本和真实 result.json；无效轨迹尽量导出已有材料，明确错误来源。
- [x] 三个 mini 沙盒正常回收，单独创建轻量反思沙盒，使用 G4 的初始模型端点。
- [x] 实现只读分析与唯一模板文件写入约束、模板校验、独立预算及超时；不允许执行原评估或继续解题。
- [x] 反思与 policy episode 共用现有全局/endpoint 沙盒容量；不添加模型请求 semaphore。
- [x] 编写权限、缺文件、无效轨迹、越权命令、取消和资源释放测试；在 144 完成真实文件读取和模板交付。
- [x] 记录本组实现与测试证据；验收通过后直接进入下一组。

### G9：异步 GEPA 搜索

- [x] 实现单一状态拥有者、空模板 bootstrap、父代选择、ready/pending candidate、Pareto 与探索补位。
- [x] 实现 mini 前/反思/mini 后/dev 流程与 mini 准入条件，按探索身份维护预算、完成顺序、best、warmup 和早停。
- [x] 实现模板 hash 去重、事件持久化、停止锁存、在途收尾和重复响应幂等；bootstrap 成本单独记录。
- [x] 编写主计划第 11.5 节的调度测试，包括 C=1/16/17、mini 不涨计数、低 dev 仍入池、warmup 跨越、并发预算边界和迟到改善不重启。
- [x] 用真实 policy/reflection/144 沙盒跑一个缩小预算 cycle，验证并发受沙盒额度限制；缩小的验证参数明确留档，不更改正式默认值。
- [x] 记录本组实现与测试证据；验收通过后直接进入下一组。

### G10：完整入口与端到端交付

- [x] 添加 P2O 运行入口与独立实验名，复用 SETA 参数与 Miles 拓扑，暴露已经落到消费者的配置。
- [x] 将训练和 GEPA 指标接入 W&B 自定义横轴，并按 SAVE_PATH/gepa/cycle_* 保存原始数据、事件JSON/JSONL和图。
- [x] 验证 interval/pre/post checkpoint、搜索中恢复、模板提交后恢复，检查初始 reflection 权重身份与数据集身份。
- [x] 完成至少两个小规模 cycle：初始无模板训练、搜索、模板训练、第二次搜索、恢复继续；检查无资源和沙盒泄漏。
- [x] 执行最终相关回归，记录真实运行命令、耗时、配置差异、日志/模型位置和未验证范围。
- [x] 更新部署、启动和恢复文档，交付可直接运行的正式入口；不自动启动长时间实验。

## 问题记录

### Q1：提交与消费的比例边界（已确认）

用户确认两个阶段都先随机抽样，再在当前可用池能够满足时补/替换到25%–50%；无法满足则直接使用正常随机结果，不额外等待。完整规则由主计划第0.7节维护。G5测试增加提交端/消费端各自可满足、缺难任务、缺普通组、全模板缓冲和随机恢复情形。

### Q2：144 服务容量不一致（已解决）

G1健康检查发现部署配置max_concurrent=32，实际服务max_concurrent=16，协议正确、活动任务0。用户明确选择更新远端服务到32后再调试。沿用现有setup.py部署专用服务，不重启共享Docker。2026-11-04已部署完成，health容量32、活动任务0，复查通过。

### Q3：当前 fully-async 默认缓存与 SETA session v1 冲突（已解决）

真实短训练在参数校验阶段停止：`miles/utils/arguments.py:4033`在fully-async + in_place时默认启用namespaced_radix_cache；`:4044`要求使用session server v2，而SETA启动器裸`--use-session-server`选择v1。错误发生在模型初始化与训练更新之前，不能报告1–2步通过。

建议在G1内适配`--use-session-server v2`并保留版本隔离缓存，先核对真实多轮episode/group与behavior log-prob语义，再补全参数校验回归和两步短训练。现有agentic生成入口已有v2分支，但这不代表SETA全链路已验证。另一条配置路径是显式关闭namespaced cache并保持v1，代价是保留in_place更新后的共享前缀缓存；不自行选用该路径。

用户已明确选择切换v2，并要求把环境变化写入Quick Start。已适配启动器和SETA有效性后处理；完整参数校验和真实两步短训练通过。保留原生namespaced cache，不关闭缓存。G1已完成，等待用户确认；G2未开始。

## 执行日志

### 2026-11-04：授权后的准备检查

- 已阅读用户指定的两份 SETA 文档，核对主计划和未决问题。
- `docker ps`：专用训练容器 `miles_seta_zhangkaiqi2024` 运行中。
- `nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader`：本机 8 张 A100 80GB，查询时每卡约 10 MiB，利用率 0%。这是检查时的状态，不是持续资源预留。
- 尚未检查 144 服务当前健康，尚未启动 GPU、Ray 或沙盒任务。
- 上一轮遗留：IcePop 参数已修改；启动命令快照因宿主机文件权限未更新，pytest 因宿主机缺少 pybase64 未收集成功。G1 在文档指定训练容器中处理，不沿用失败的宿主机环境作验证结论。
- 本次产出为 TODO 文档及主计划交叉链接；未编写 P2O 功能代码、未完成任何 TODO 组、未执行功能测试。按用户“遇到问题先停下来”要求，等待 Q1 反馈。

后续每组追加：改动文件、测试命令、通过/失败/跳过计数、原始日志位置、真实环境证据、限制与用户确认结果。


### 2026-11-04：G1 首次执行，暂停于远端 sudo

- Q1 已按用户最新要求更新主计划第0.7节：提交/消费分别随机初选、可满足时补/替换、不可满足时随机回退，不新增配额等待；这次只修改设计，没有提前实现 G5。
- 容器 Python 为 `/opt/sglang/bin/python`，版本3.12.15；pytest 9.1.1，torch 2.14.1+cu130。容器中已成功收集和运行启动器测试，不再受宿主机缺少pybase64阻塞。
- 执行 `docker exec -w /ceph_home/zhangkaiqi2024/miles miles_seta_zhangkaiqi2024 python -m pytest tests/fast/launch_scripts tests/manual/launch_scripts -q --tb=short`：**616 passed、16 failed、10 errors，pytest报告24.35秒**。这是本轮新增测试/快照修改之前的基线。
- SETA execute快照仍对应旧clip/TIS参数，属于本组已计划修复项；另15项失败和10项错误位于其他启动器/模型/发现检查，本轮尚未逐项诊断，不能报告全仓启动器通过。
- 宿主机执行 `python3 seta_miles_runtime/seta-remote/setup.py --mode check` 失败：sandbox-144配置容量32，运行容量16；health协议正确、活动任务0。用户明确选择将远端更新为32。
- 随后执行 `python3 seta_miles_runtime/seta-remote/setup.py --mode apply --node sandbox-144`，在第一条远端`sudo install -d ...`失败，提示需要终端和密码；尚未进入代码同步、配置生成或服务重启。没有更改远端服务。
- 按用户遇到问题暂停的要求，停止后续实现。需要用户在交互终端执行上述apply命令并输入远端sudo密码；无需在对话中提供密码。完成后继续健康复查、G1测试补齐与至多1–2步短训练。
- 当前没有新增功能测试、没有P2O功能实现、没有GPU/Ray/Harbor任务。G1未完成，G2未开始。

原始证据目录：`seta_miles_runtime/diagnostics/p2o_g1_20261104/`（本地运行产物，Git忽略）：

| 文件 | 内容 |
| --- | --- |
| `launch_before.log` | 完整启动器基线测试输出与失败列表 |
| `remote_health.log` | 部署工具健康检查失败栈 |
| `remote_health_details.json` | 脱敏的部署容量与实际health对比 |
| `remote_apply.log` | sudo阻塞与部署失败栈 |


### 2026-11-04：G1继续，远端部署与CPU功能验收

- 使用用户提供的凭据在交互终端完成既有setup.py部署；凭据未写入文件、命令行参数或运行日志。远端专用服务容量更新为32，未重启共享Docker。部署后的独立check通过，原始结果见`remote_health_after.log`。
- SETA execute快照由现有快照测试生成，diff仅包含PPO epsilon 0.2/0.28以及用IcePop hook取代tis-clip=5；其他入口快照没有更新。
- 新增19个启动器用例（含参数化）：两个算法/两种训练模式的最终参数解析、IcePop函数真实加载、继承默认范围、两组仅过滤参数不同、rollout不筛同分、extra-args数值覆盖、非法算法/过滤覆盖拒绝，以及三个shell入口的实验名和拓扑验证。
- 新增4个数值用例：IcePop上下界含等号、界外置零与梯度、原mask与归一化分母保持、PPO正负优势下的非对称clip与梯度。
- 首次新启动器测试将Ray外层`--`也交给训练parser，导致4项测试失败；修正为仅解析train.py/train_async.py之后的参数，未因此修改生产实现。
- 相关功能与SETA回归最终结果：**93 passed，12.72秒**。命令为容器内`python -m pytest tests/fast/launch_scripts/test_run_seta.py tests/fast/backends/training_utils/test_importance_corrections.py tests/fast/test_seta_progress.py tests/fast/test_seta_remote.py tests/fast/test_seta_deploy.py tests/fast/test_seta_shutdown.py -q --tb=short --durations=5`。日志`focused_tests_final.log`。
- 已启动2步短训练：初始Qwen3-8B，GRPO过滤关闭，IcePop默认范围；1个已知任务、每组4条轨迹、4训练GPU+4推理GPU，agent最多8轮。缩短的序列/时间限制和所有覆盖保存在`smoke_manifest.json`；不改正式入口默认值。
- 短训练W&B关闭，仅最后保存weights-only checkpoint（不保存optimizer/RNG）；这不是断点恢复测试。独立SAVE_PATH为本证据目录下`smoke_checkpoint`，输出在`smoke_run`，控制台在`smoke_console.log`；最多900秒后发送TERM并按入口既有逻辑回收。
- 完整启动器套件复跑与短训练结果待补；尚未进入G2。


### 2026-11-04：G1短训练发现Q3，已清理并暂停

- 完整启动器最终结果：**636 passed、15 failed、10 errors，17.76秒**，日志`launch_final.log`。与改动前逐条比较失败身份：解决SETA快照1项，新增失败0项，剩余25项均已存在于本组修改前且不属于SETA。比较结果见`baseline_failure_comparison.json`。
- 剩余问题包括其他入口调用已不存在的command_utils API、缺失模型/启动器快照、快照差异及发现集合不一致；未修改无关入口或用重建全部快照掩盖问题。
- 短训练作业`g1-icepop-20261104a`在解析/校验阶段报Q3，进程退出码1；**实际训练steps=0，未创建checkpoint**。`smoke_console.log`包含错误及退出清理；`smoke_exit.json`记录退出码。不得把93项CPU通过或历史训练记录当成本次GPU训练成功。
- 启动器先停止专用Ray；shell清理再次查询已停止Ray时报告连接拒绝，随后仍完成按本run_id的远端flush，cancelled=0。复查raylet不存在，144服务健康、活动0，8卡每卡10MiB/0%利用率。
- Python compileall及`git diff --check`通过。容器未安装ruff，未报告ruff通过。
- 汇总保存在`g1_status.json`。暂停等待用户对Q3处理方向反馈；尚未修改session版本、缓存开关或P2O生产逻辑。


### 2026-11-04：G1 session v2 适配与验收

- 用户授权切换session server v2。所有SETA入口共享v2参数，并加载`examples.seta.agent.postprocess_session_samples`。v2默认后处理会提前赋reward，导致下游RM hook跳过；因此在原生后处理之后复用SETA有效性检查，保持无效结果ABORTED、有效零分可训练及1.2 full-pass reward。
- 后处理沿用Miles共享动作归属mask、leaf身份与rollout_id，不拆成逐step训练样本。新增8项测试覆盖有效性与共享动作；启动器新增同步/异步完整参数校验，快照只更新SETA入口。
- GPU容器现有Python/SGLang已支持v2及extra_key；本次未升级依赖、重建镜像或转换模型。144无需为v2更改环境或重部署，health协议名仍为harbor-agent-server-v1。本次没有环境配置变化；按用户后续要求，Quick Start不保留单独的v2切换说明。
- SETA相关回归：**103 passed，6.61秒**（`seta_tests.log`）；v2会话采集、合并、多轮token/logprob/mask等现有测试：**42 passed，1.39秒**（`session_collection_tests.log`）。首次离线运行后者因缺少Qwen/Qwen3-0.6B tokenizer缓存失败，下载该测试的config/tokenizer文件后离线复跑通过；未下载模型权重，也未安装依赖。
- 完整启动器套件：**638 passed、15 failed、10 errors，17.85秒**（`launch_suite.log`）。逐条对比上次失败身份：新增0、减少0，仍是原有25项非SETA问题；结果见`baseline_failure_comparison.json`。
- 两步真实训练作业为`g1-v2-20261104a`，GRPO过滤关闭、IcePop、4训练卡+4推理卡，具体缩小配置见`smoke_manifest.json`。仅做G1验收，不测试P2O、正式吞吐或断点恢复。最终完成step 0/1，退出码0，全程831.69秒（包含初始化、沙盒准备、保存和清理）；未触发900秒超时。

本轮证据目录：`seta_miles_runtime/diagnostics/p2o_g1_v2_20261104/`。G2尚未开始。

- 真实首批4条均为8轮线性episode、每episode一个Sample；奖励为0.1/0.2/0.2/0.3。`audit_smoke_samples.py`独立从weight_versions生成绝对动作区间，逐token比对loss_mask，并检查log-prob长度/有限性、reward与verifier一致、任务身份和v2 mismatch为空；首批通过。
- 后续G2/G3需注意原生身份契约：单叶输出的`rollout_id`可为None，本次实际用`lineage.source_sample_index`及`group_index`标识episode/任务组；只有多叶输出时agentic入口显式统一rollout_id。不能假定单叶也一定填充rollout_id，也不能把leaf数当环境重复次数。


#### G1 最终验收结果

- 完成两次真实参数更新，step 0/1 loss分别为`0.0003876686`、`-0.00004039705`，grad_norm分别为`1.0210266`、`1.1594404`，均为有限值。两次训练后均完成推理权重同步。第二批奖励为0.3/0.2/0.3/0.2。
- 最终审计**2批、8条episode全部通过**；均为8轮、每episode一个Sample。第二批3条轨迹含权重版本0和1，第4条仅版本0；v2 mismatch均为空，动作mask逐token匹配真实生成区间，行为log-prob长度及有限性正确。原始轨迹、训练张量及`sample_audit.json`保留在证据目录。
- weights-only checkpoint保存成功，耗时31.3秒。当前Miles按rollout索引保存到`iter_0000001`，tracker为1；这是第二次更新后的保存，并非只训练了一步。保留4个distcp分片及metadata；未执行加载恢复，不宣称optimizer/RNG或P2O状态恢复通过。
- 正常退出后专用Ray已停止。shell重复查询已停止的Ray dashboard产生ConnectionError，随后仍按run_id完成远端flush，清理退出码0；不影响训练成功。独立复查：8卡各10MiB/0%利用率、raylet不存在、144 health容量32/active=0、无seta-zkq任务容器。证据见`cleanup_check.json`、`gpu_after.txt`。
- 本轮环境变动仅有测试tokenizer/config缓存；生产GPU和沙盒均未升级依赖。Python compileall及`git diff --check`通过；容器没有ruff，未运行ruff。全启动器套件的原有25项非SETA问题仍保留，不能称全仓测试通过。
- 汇总：`g1_status.json`。G1完成；按用户逐组确认要求停在此处。下一组G2只实现难任务与评分基础模块及其测试，不提前接入GEPA调度、资源切换或训练loss。


### 2026-11-04：G1 启动器既有失败修复

用户要求无环境变更时不要为v2更新Quick Start，并询问既有失败的原因及修复。本轮移除Quick Start的v2专节和失效锚点；保留之前GRPO/DAPO入口与实验名的使用说明。没有安装包或修改GPU/144环境，也没有开始G2。

原有25项是失败测试数量，不是25个独立缺陷：

| 原因 | 失败/错误项数 | 修复 |
| --- | ---: | --- |
| AMD GPT-OSS LoRA仍调用已删除的command_utils模块接口 | 6 | 下载与训练改用args.create_backend()；快照仅新增原生backend的`--deploy-component all` |
| DeepSeek V4.1转换/训练测试未提供必需hf_checkpoint | 4 | 录制fixture显式提供沙盒内checkpoint路径；新增测试验证缺省拒绝及路径传递，不伪造生产默认checkpoint |
| Workplace示例导入未声明的tap依赖 | 1 | 单一`--config`参数改用标准库argparse，保留JSON配置接口；新增真实CLI解析测试 |
| 模型参数快照缺失 | 7 | 补齐DeepSeek V4.1、GLM5.3 Flash、Qwen3.5 0.8B、Qwen3.8 Flash Next及其切片快照 |
| 启动器快照缺失 | 4 | 补齐DeepSeek的prepare_data/prepare_download和Qwen3 Torchtitan的prepare/execute；DeepSeek另两个入口此前计入上面的4项错误 |
| GLM LoRA缓存环境变量快照过期 | 1 | 脚本只转发显式环境变量，测试会清除这两个变量；更新快照，不修改生产缓存策略 |
| 模型/启动器发现集合不一致 | 2 | 补齐缺失快照，并删除已不存在的Inkling full_train入口遗留快照 |

文件位置（仓库根目录起算）：

- [v2验收时的失败比较JSON](../seta_miles_runtime/diagnostics/p2o_g1_v2_20261104/baseline_failure_comparison.json)：保留当时25项名单，不覆盖历史结论。
- [最初G1改动前后的比较JSON](../seta_miles_runtime/diagnostics/p2o_g1_20261104/baseline_failure_comparison.json)：还记录了原SETA快照失败的修复。
- 本轮证据目录：`seta_miles_runtime/diagnostics/p2o_g1_launcher_fixes_20261104/`。这些是Git忽略的本地运行产物，`rg --files`默认不显示，可用`rg --files --hidden --no-ignore seta_miles_runtime/diagnostics`查找。

针对性测试4 passed；定向生成模型快照14 passed、启动器快照22 passed。首次完整复跑665 passed/1 failed，唯一剩余项明确定位到Inkling孤立快照并删除，最终完整复跑 **666 passed、0 failed、0 errors，16.36秒**，日志为`launch_suite_final.log`；[本轮比较JSON](../seta_miles_runtime/diagnostics/p2o_g1_launcher_fixes_20261104/baseline_failure_comparison.json)记录原25项全部解决。Python compileall和`git diff --check`通过。所有录制均拦截执行命令，不启动GPU/Ray、不下载模型；不等同于这些模型的真实训练验收。


### 2026-11-04：G2 难任务与评分基础模块

用户已明确开始下一组。本组模块位于`examples/seta/p2o/`，使用Miles的Sample/lineage及冻结Pydantic数据结构；SETA verifier语义留在示例侧，不复制或修改trainer。生产启动器尚未导入这些模块，G5才接通真实收集hook。

| 文件 | 本组交付 |
| --- | --- |
| [evidence.py](../examples/seta/p2o/evidence.py) | 数据集身份+instance_id任务键；自然/主动补入/搜索/反思/评估来源；合并一个episode的全部叶子，保留passed/total、原reward、错误及远端证据路径、policy版本和模板身份 |
| [scoring.py](../examples/seta/p2o/scoring.py) | 配置校验，阈值默认由Miles传入的K推导；完整有效K条判难；固定重复dev聚合与有序逐任务向量 |
| [hard_tasks.py](../examples/seta/p2o/hard_tasks.py) | 单拥有者自然难任务去重，阈值触发和收尾，固定H任务面板，dev随机划分/重复seed，历史参与身份及版本化JSON状态 |
| [测试目录](../tests/fast/p2o) | 评分边界、叶子完整性、去重/来源/数据集隔离、H>64、配置覆盖与状态恢复 |

实现约定与边界：

- `dataset_id`由调用方提供稳定版本/指纹；不使用瞬时row index。训练任务白名单在构造collector时显式传入，恢复时核对配置和完整catalog。G5负责从实际数据源构造白名单和fingerprint；最终评估任务不允许作为自然训练任务收集。
- `episode_from_samples`沿用Miles `rollout_id`、缺省回退`index`的episode身份；lineage检查全部保留叶子是否齐全且不重复。单叶可没有rollout_id。有效TRUNCATED仍可计分；ABORTED/FAILED/移除状态、无效验证、缺结果或冲突叶子不形成有效分数。
- reward直接保留Sample及verifier一致的值，不裁剪/二值化/重算bonus。progress只由passed/total派生，不保存第二份可变progress。无效证据保留错误和trial_dir等引用，数值字段为None；不会变成有效零奖励。实际文件导出及反思读取属于G8。
- 收集器只计自然来源，主动补入（Origin.REPLAY仅是来源名称，不新增数据迭代器）、GEPA搜索/反思/最终评估均忽略。每任务每cycle保留首次完整有效的难证据，后续容易结果不删除已经收集的任务。
- 达门槛后`trigger_pending`保持True，仍接受协调器收尾阶段的自然结果。`freeze()`保留全部H，dev固定数量，剩余全为search-train；重复freeze返回同一面板。停止提交/真实收尾/训练预算判断属于G5/G7，本模块不操作异步producer。
- 面板保存每dev任务的独立重复seed，所有候选复用；目前只验证seed生成和恢复，**尚未验证Harbor→模型请求的seed传递**，留待G9实际评价接线验收。
- dev必须收到预定次数的终态结果，允许其中部分或全部无效；只平均有效reward，全无效按搜索约定记0。attempted/valid/invalid数量、均值、std、all_invalid都从保留的原始attempts派生；单有效或全无效std为None。16维向量按固定dev任务次序组织，再均匀平均16个任务，不按有效episode数重新加权。
- 快照schema v1保存config/catalog、cycle、pending难证据、冻结面板/seed、历史last_participated_cycle和Python RNG状态；通过JSON恢复，不用pickle。未找到模板也能完成cycle，全部H任务（含dev）保留参与身份，再重新收集下一轮。配置/数据集不匹配、损坏面板或非难pending证据拒绝恢复。
- 这只是P2O状态的序列化契约；与模型/optimizer/RNG的原子checkpoint配对在G7，不能将本组JSON测试称为完整训练断点恢复。

测试与证据：

- 首次新增测试：45 passed，0.49秒。覆盖1/16含等号、一全通过/两全通过、连续部分通过、bonus保留、无效与零分区别、K变化、阈值0/1/非法数、4/3/2/1/0有效重复、缺终态拒绝、重复attempt/remote request、多叶去重与不完整、64→48/16及71→55/16、固定划分、空模板结果后历史保留、两轮RNG连续和恢复拒绝。
- 读取G1已保存的真实2批8条轨迹，用新adapter和评分器重新核对：8条全部有效，reward保持原值，progress均值0.20/0.25。该短跑K=4，所以默认判难阈值为0.25；这不是将正式K=16改成4。见[真实轨迹复核](../seta_miles_runtime/diagnostics/p2o_g2_20261104/real_g1_evidence_audit.json)。
- 本组没有启动新的GPU训练、Ray或144沙盒，没有新增依赖或环境修改；Quick Start无需更新。
- 最终回归命令：容器内`python -m pytest tests/fast/p2o tests/fast/test_seta_progress.py tests/fast/test_seta_session.py tests/fast/test_seta_remote.py -q --tb=short --durations=5`，**69 passed、0 failed、0 errors，1.20秒**（45项新增+24项既有回归）。日志[tests_final.log](../seta_miles_runtime/diagnostics/p2o_g2_20261104/tests_final.log)。Black、isort检查、compileall与`git diff --check`均通过；ruff未安装，未运行。

原始日志目录：`seta_miles_runtime/diagnostics/p2o_g2_20261104/`。G3尚未开始。

G2已完成独立模块验收，用户随后确认进入G3。G2没有提前实现真实模板注入与双上下文对齐。


### 2026-11-04：G3 模板输入与双上下文（暂停，尚未验收）

- 审计发现官方Harbor客户端的`prompt`参数不参与实际任务输入；模板通过`metadata.p2o_template`传递，在`SetaTerminus2.run()`中加入本次instruction，不改共享任务文件。
- 新增`examples/seta/p2o/templates.py`：模板文本hash与版本校验、唯一注入块、模型控制token检查及包含wrapper的token预算校验。预算函数可显式配置；生产入口参数在G10统一接入。当前后处理使用默认1024，未来模板入库时也必须先校验，不能只依靠rollout完成后的检查。
- 新增`examples/seta/p2o/contexts.py`：在原生TITO合并/截断/分支mask归属确定之后，验证首轮实际prompt并使用同一serving tokenizer渲染去模板前缀；完整响应后缀直接复制，包括动作、观察、分隔token。保留原`Sample.tokens/loss_mask/rollout_log_probs/reward`，双上下文通过可序列化metadata和`context_pair()`提供给后续G6。
- v2后处理函数可显式声明只读`sample_context`关键字，取得serving tokenizer和collection snapshot中的逐路径记录；旧两参数hook调用方式不变。通用session层不包含P2O判断。
- root已有assistant历史、历史改写成新root、实际模板缺失/重复、token前缀不符等情况明确422拒绝，不恢复被截断历史、不近似拼接。原root上的分支遵循Miles原picker和共享动作mask归属。
- 新增CPU测试覆盖非canonical动作token（decode再encode会改变ID）、多轮观察、停止/长度边界、分支、P2O关闭与native结果一致、模板回显不误删、wire/JSON往返和非法输入。相关回归204 passed、0 failed、0 errors（3.06秒），日志`seta_miles_runtime/diagnostics/p2o_g3_20261104/tests_final.log`。
- 144服务空闲后仅同步所需Python代码；每条episode在新子进程加载，无需重启服务、更新依赖或修改GPU/沙盒环境，Quick Start不作本组修改。
- 真实验收采用独立单卡Qwen3-8B推理服务+原生v2 session，带模板/无模板各一条SETA episode、每条最多3轮；不运行训练、不更改正式拓扑。执行脚本与完整命令在上述诊断目录的`real_rollout.py`和`manifest.json`。首次启动前检查修正了验收脚本的URL重复`/v1`问题，尚未提交沙盒任务即中断并回收；保留`preflight_cancelled_*`日志。

真实验收结果与暂停原因：

- 独立脚本成功启动单卡Qwen3-8B和v2 session，144已执行一次带模板trial。但首个模型请求被400拒绝：`bounded training-request sampling requires bounded rollout sampling`。没有生成任何token，`/samples`为空，脚本的单episode断言失败。因此不能声称真实多轮模板输入或双上下文已经验收；脱敏真实fixture也尚未生成。
- 原因已定位到**本次独立验收脚本**：session/request两处设置`top_p=0.95`，同时`use_sampling_support_replay=False`。Miles在`miles/rollout/generate_utils/sampling_mask.py`中明确禁止这一组合。G1已通过的真实基线使用`--rollout-top-p 1`，没有此问题。
- 待用户反馈后的修正方案：仅将独立验收脚本两处`top_p`改回1，保留训练请求校验，重跑带模板/无模板各一个最多3轮episode；不更改生产采样策略。尚未执行该修正或重跑。
- 已完成清理：8张卡均回落到10 MiB，无本组engine/session/scheduler存活；144健康、capacity=32、active=0。关闭时SGLang未在20秒内退出，按本组进程组升级终止，未操作其他服务。未启动训练或Ray。
- Black、isort、`git diff --check`通过。没有GPU/沙盒环境更新；未修改Quick Start。远端已同步所需功能代码，最后的本地格式化尚未同步（仅格式差异），恢复验收时同步最终版本。
- 状态与失败证据：[g3_status.json](../seta_miles_runtime/diagnostics/p2o_g3_20261104/g3_status.json)、[CPU测试日志](../seta_miles_runtime/diagnostics/p2o_g3_20261104/tests_final.log)、[真实验收日志](../seta_miles_runtime/diagnostics/p2o_g3_20261104/real_console.log)、[远端返回](../seta_miles_runtime/diagnostics/p2o_g3_20261104/templated_remote_result.json)。

按用户“有任何问题，也停下来向我汇报，我来给你反馈，完了再继续”的要求暂停G3，等待反馈；G4未开始。


### 2026-11-04：G3 修复并完成验收

- 用户授权不中断逐组推进，但每组必须先完成测试。独立验收脚本改回top_p=1，两条真实episode均完成3轮、reward=0.2；不是正式训练或效果评估。带模板：2082→1980 tokens，680动作token、158观察token，行为log-prob逐项一致；无模板：两路3061 tokens，933动作token、982观察token。两者TITO mismatch均为空。
- 完整原始记录及审计：`seta_miles_runtime/diagnostics/p2o_g3_20261104/retry_unbounded/real_audit.json`、`*_records.json`、`*_sample.json`、`*_contexts.json`。总耗时165.44秒（包括加载模型）。
- 新增脱敏真实fixture `tests/fast/p2o/fixtures/seta_multiturn.json`，去除端点、request/session身份、远端路径和时间等不相关信息；原始任务/动作token保留，确保真实对齐可复查。27项G3测试全部通过（fixture_tests.log），此前204项相关回归已通过；本次新增fixture后G3全部重跑通过。
- GPU和144沙盒任务均已回收，无环境/依赖变更，Quick Start不更新。按用户最新授权直接进入G4。


### 2026-11-04：G4 资源切换实现与 CPU 验证

- 新增 node-qualified GPU slot 规划器与临时推理 pool 生命周期，使用 Miles 当前 run 的 placement group、CommandActor、launch gate 和 session v2；不创建竞争 GPU 的第二个 placement group。
- 默认临时布局为每张 policy 卡一个 TP=1 engine，末尾 1 张卡加载初始 reflection 模型。正常训练 topology 原样恢复；多机只按各节点本地物理设备构造 engine。
- 原生 trainer offload 保留模型、optimizer 和 RNG；切换前集体导出当前 policy HF 权重。恢复时重建正常 engines 并同步 trainer 权重，然后才能恢复 producer。
- 首轮相关回归 **270 passed**；补充配置不变性、模型路径核验、跨节点 GPU 编号测试后，资源专项 **19 passed**。原始日志位于 `seta_miles_runtime/diagnostics/p2o_g4_20261104/`。
- 正在运行真实 8 卡验证。为单独审计资源生命周期，本组复用 G1 的真实 rollout 张量执行两次 optimizer 更新，前后保存 optimizer/RNG；不把它称为新 rollout 或最终 P2O 训练验收。G10 仍需新沙盒轨迹与 GEPA 搜索。
- 真实切换尚未通过，不进入 G5；本次未安装依赖或更改沙盒部署环境。

#### G4 真实验证中间记录（2026-11-05 本地时间）

- 第一轮调试因脚本误用不存在的 `/opt/Megatron-LM` 路径，在 trainer 初始化报 `ModuleNotFoundError: megatron.post_training`。已清理，改为容器现有 `/root/Megatron-LM`；没有安装依赖。失败原始日志保留为 `probe_console_attempt1.log`。
- 第二轮调试关闭 SGLang decode/prefill CUDA graph 加速重启，context length 使用32768；这是独立 debug 配置，正式参数不变。入口与完整 argv 保存在 G4 目录的 `launch_probe.py`、`launch_arguments.txt`。
- 第一次 optimizer step 完成，loss=0.0003876686、grad_norm=1.0210607。默认 warmup 初始 LR=0，抽查四层 q_proj 的导出权重与初始模型相同；不能将该边界声称为数值不同的 policy/reflection 权重。第二次更新及后续端到端验收继续检查非零 LR 更新。
- 首次完整 checkpoint（包含 optimizer/RNG）约107 GiB，保存含 HF 导出耗时224.6秒。临时 phase 进入144.05秒，8个引擎均成功生成，7 policy / 1 initial reflection 路径核验通过，均支持32768 context配置。原4+4恢复并同步权重成功。
- 恢复后再次保存同 step 的完整 checkpoint，耗时266.7秒；正在逐文件比较并执行恢复后训练。本组尚未完成，尚未进入G5。
- 辅助 canonical 审计第一次遇到标量 tensor 的 uint8 view 限制，已修正；第二次与同路径 checkpoint 重写重叠，报 EOF，因此该辅助结果不作通过证据。正式断点方案仍须使用不同 phase 的不可变快照/manifest。资源探针主程序的前后完整文件 SHA256 在各次 save 完成后串行计算，不受该辅助读取失败影响。
- 资源专项新增失败路径后 **21 passed**；此前相关大套件 **273 passed**（新增的两个失败路径随后单独验收）。


### 2026-11-05：G4 内容级审计发现并修复 wake-up 参数恢复缺口

- 第一轮非零学习率内容审计完成了八个临时推理引擎和原训练拓扑恢复，但四个 rank 的模型参数 SHA256 不一致；Adam 状态/FP32 master 参数、scheduler 和 RNG 全部一致。原始证据完整保留于 `seta_miles_runtime/diagnostics/p2o_g4_20261104/content_probe_attempt1/`，该次验证为失败，未进入 G5。
- 根因：原生 `--offload-train` 设置 `disable_param_buffers_cpu_backup=True`，memory-saver 唤醒只重建参数存储，原先依赖下一次 `_train_actor` 的 `_switch_model` 才重填数值。GEPA 恢复先 onload 再发布权重，期间 `_get_actor_weights` 把尚未恢复内容的存储当成 resident actor。
- 修复位于 `MegatronTrainRayActor.wake_up`：完成进程组恢复后、标记 resident 前，使用已有 TensorBackuper 恢复当前模型 tag；仍沿用原生 CPU backup/main-cast，不重载初始权重、不更换 optimizer。已 resident 的重复 wake 不重填，以免覆盖新参数。
- 新增 CPU 数值回归模拟被清空的参数存储，验证 wake 后恢复为备份值，二次 wake 不覆盖后续更新。生命周期及资源测试 **86 passed，26 warnings，1.00s**；日志 `p2o_g4_20261104/wake_restore_tests.log`。
- 在独立 `content_probe/` 重跑非零学习率八卡验证；结果待补。运行启动命令需使用模块方式或显式设置仓库 PYTHONPATH，直接执行诊断文件首次报 scripts 导入失败，未开始 GPU 工作。


#### G4 最终验收（2026-11-05）

- 修复后的真实八卡测试退出码0：完成非零学习率首步更新 → 7个 TP1 policy 引擎 + 1个 TP1 初始 reflection 引擎 → 原4训练/4推理拓扑 → 第二步更新。进入GEPA耗时130.19秒；8个引擎均验证实际 checkpoint 路径并成功生成。
- 四个训练rank在切换前后，全部模型参数、Adam状态/FP32 master参数、scheduler和Python/NumPy/CPU/CUDA/Megatron tracker RNG内容完全一致。第二次更新的模型/optimizer内容确实改变，scheduler样本步数为4→4→8。step1 loss=0.0011160374，grad_norm=0.90868646，均有限。
- 初始/训练后HF权重独立抽查4层q_proj，每层约418万–456万个BF16值发生变化，排除零学习率导致两种模型数值相同的弱验证。
- 核心证据：`seta_miles_runtime/diagnostics/p2o_g4_20261104/content_probe/acceptance.json`、`state_comparison.json`、`state_{0,1,2}_rank*.json`、`inference_probe.json`、`weight_identity_audit.json`和`console.log`。CPU相关回归 **364 passed，26 warnings，37.49s**（`resource_regression_after_fix.log`）。
- 退出清理停止本次Ray进程；GPU/进程复查见运行日志记录。
- 范围说明：本次仅资源生命周期验收，使用已录制的真实SETA rollout张量，未新跑沙盒；多机布局有单测但未实测跨机NCCL。内容摘要不是checkpoint重载；原生完整checkpoint写入已另测，断点恢复仍属于G7/G10。G4完成，继续G5。


### 2026-11-05：G5 收集、历史模板和双阶段混入验收

- 新增 `history.py`、`mixing.py`、`coordinator.py`、`sampling.py`、`rollout.py`。`P2ORolloutFn`继承原生fully-async，单个coordinator持有收集器、历史模板和抽样RNG；组结果在原生buffer过滤前观察。全失败的有效组正常保留，主动补入来源为replay，不增加自然难任务触发。
- 历史模板只提交完整cycle，空bootstrap不进入训练抽样；每轮全部非空模板做总体标准差标准化，再在已知求解/旧分配/全历史可选集合中稳定softmax。相同hash保留历次raw dev记录和求解证据，抽样只占一项、使用最近完整cycle分数。无模板cycle保留所有难身份和旧分配。
- rollout按原生随机数据顺序取逻辑窗口，天然命中抵扣25%目标、超50%按最新参与轮优先注入，其余保留原任务原prompt；缺可用难任务则原样回退。补入时调用新增 `RolloutDataSource.materialize_samples`分配原生group/sample身份，不推进自然游标；被替换任务保留FIFO。补入优先替换新取后缀，避免延后任务反复被替换造成饥饿。
- 训练消费复用原生有效性、staleness、背压和checkpoint，仅扩展 `_select_available`选择接口；从现有完整组随机初选，池可行时补/换，池不足直接返回原随机结果。模板分类使用生成时标记，任务优先级使用当前历史参与轮，均不修改既有轨迹。两处均有数量/回退原因/替换指标和日志。
- 原生fully-async只增加 `_next_prompt_groups`扩展点，默认路径逻辑保持；buffer默认仍FIFO。原生数据源、native rollout状态、P2O registry/RNG、未提交窗口和延后任务一起保存恢复。阶段事务和搜索状态仍在G7/G9实现。
- 首轮新增测试174 passed、1 failed：测试把K=2的随机旧分配误当成保证包含多个模板，修正fixture为K=16并实际验证同一个组可用不同模板。不是放宽模板选择语义。
- 验收：P2O+原生fully-async/buffer/checkpoint回归 **446 passed，26 warnings，3.84s**；补齐饥饿/恢复约束/无效轨迹测试后，P2O+原生data_source **278 passed，26 warnings，2.35s**。日志位于 `seta_miles_runtime/diagnostics/p2o_g5_20261105/{native_regression,final_tests}.log`，black/isort和diff检查通过。
- 已录制的两批真实SETA结果再次送入coordinator：8条多轮episode，progress均值0.2/0.25，K=4阈值0.25，首批新增难任务、第二批去重；奖励未改动。证据 `recorded_audit.json`和`recorded_audit.log`。这是已录制结果验证，不是本组新起GPU/沙盒或P2O训练。
- G5完成，继续G6双上下文评分与loss。现阶段P2O训练入口尚未交付，GEPA触发调度尚未启用，不能将这些独立回归视为端到端P2O已完成。


### 2026-11-05：G6 双上下文评分与 CRM/IcePop 接线

- 原生 Megatron actor 新增评分扩展点，在固定 old actor 后运行两次原生 forward-only；增强上下文提供 IcePop 的 old log-prob，去模板上下文提供 PPO 的 old/current log-prob。动作与观察 token 保持原值；CP 下按响应 token 重新切分增强分数与实际 behavior 分数。
- CRM 使用 detached old_plain/old_aug 的 log ratio，默认闭区间[0.01,10]，只乘 policy loss 分子；IcePop 默认[0,2]、PPO 默认[0.8,1.28]和原始动作分母不变。非模板样本CRM恒为1。
- 修复评分hook缺少模块级load_function导入；测试分母改为Miles实际的tensor类型。新增手算loss/梯度、无模板精确退化、全零mask/advantage/CRM以及CP=2/4的thd/bshd对齐测试。
- CPU回归：**359 passed，26 warnings，6.47s**；日志 `seta_miles_runtime/diagnostics/p2o_g6_20261105/cpu_final.log`。使用本地HF缓存离线运行，避免tokenizer联网检查延迟；没有安装依赖。
- 正在进行真实GPU验收，使用G3保存的带模板/无模板轨迹各两条组成batch，人工设置奖励仅用于产生非零优势、验证梯度，不用于效果结论。计划2次原生更新，尚未完成，不进入G7。


#### G6 最终验收与扩展回归

- 真实GPU首跑发现DP分片白名单漏传原生train_metadata（转换阶段已有metadata）；已修复通用分片接口，新增平衡/不平衡分片与任务对应测试。相关回归77 passed，日志`transport_final.log`。
- 第二轮真实GPU验收成功完成2次非零LR更新：loss=-0.0002039075/-0.0707364082，grad_norm=2.8616855/1.7679049，CRM动作保留率0.9985294/0.9977942，IcePop丢弃率0.0158900/0.0200839。两路old评分、原生优势、backward和optimizer均实际运行；回收本次Ray和GPU进程。证据`gpu_acceptance.json`、`gpu_console.log`，失败首跑保留`gpu_console_attempt1.log`。这是录制轨迹数值验收，奖励人工设置，不是新沙盒训练或方法效果结果。
- 扩展loss套件76 passed、6个历史快照失败。用HEAD原始loss替换测试入口后同6项仍失败（`baseline_snapshot.log`）；独立进程对12种配置逐项比较当前/HEAD的loss、梯度和metrics一致（`baseline_failure_comparison.json`）。未改快照或放宽容差，保留环境/测试执行差异证据；新增P2O测试均通过。
- G6完成，G7开始。后续完整验收仍需真实新轨迹、周期搜索与断点重载，不能用本次GPU更新代替。

### 2026-11-05：G7 异步边界与原子保存配对

- 触发后停止新提交、收尾所有已提交组。收尾期间仅为现有在途组预留有限buffer空间，避免buffer满与暂停训练互相等待；完整batch继续消费，不足batch发出显式phase boundary并保留尾批。恢复时不改已生成轨迹，未提交窗口重新按新模板池分配。
- 原生train_async增加可选phase_factory，默认路径保持；P2O仅关闭训练batch的前瞻get，后台producer仍异步。phase只能在训练更新、权重同步及eval收尾之后运行，不能与训练抢卡。训练预算结束不再拉取下一batch或启动GEPA。
- CheckpointLedger使用原生模型/optimizer/RNG保存，原生rollout快照分别保存到不可变pre/post目录，最后原子发布配对指针。GEPA不更新模型，因此同step的pre/post引用同一个已同步完成的模型目录，避免重写100GiB权重，也避免同step覆盖。每个配对目录本身可作为Miles的--load根目录；保存失败保持上一完整指针。
- 新增慢任务、H超门槛、尾批恢复、满buffer收尾、重复freeze、提交前拒绝resume、driver相位顺序、模型复用/保存失败和原生loader回读测试。相关回归459 passed；补充原生executor边界/快照加载测试后定向40 passed。日志`seta_miles_runtime/diagnostics/p2o_g7_20261105/{full,final}.log`。
- 本组实际将Sample/游标/待训练状态写盘并用原生loader重读；完整模型已在G4写盘，最终训练进程重启并恢复模型+P2O的组合验收仍在G10。当前phase接口与配对保存已具备，搜索调用和正式入口在G9/G10接通。G7完成，继续G8。

### 2026-11-05：G8 独立反思沙盒验收

- 新增独立轻量 Harbor task、ReflectionTerminus2、证据打包与封闭命令解析。原任务沙盒正常回收；反思证据包括实际初始消息/任务 instruction、轨迹、真实评估结果和隐藏 verifier 脚本，无效轨迹也可打包。模型只能读证据文件，唯一写入口为 `write_prompt` 写固定 template.md；原始命令不直接交给 shell。
- 沿用 SETA 客户端和远端服务现有并发槽，无新增模型请求 semaphore。首次反思自动构建 Ubuntu24.04+tmux 轻量镜像，没有更新 GPU Python/CUDA 或远端宿主服务依赖，没有重启服务。
- 首轮文件/命令/远端回归43 passed；完整P2O+远端回归见 `seta_miles_runtime/diagnostics/p2o_g8_20261105/final.log`。
- 实际初始 Qwen3-8B + session v2 + 144 Harbor 完成2轮反思，template.md为121 tokens，命令拒绝数0，端到端225.1秒（含首次构建）。证据 `real_audit.json`、`reflection_records.json`、`reflection_result.json`、`template.md`。这只验证链路，不代表模板有效；输出泛化且读取大raw trace导致上下文不足，已增加trajectory.md可读视图和更明确的先读后反思指令，最终GEPA联调继续检验。
- G8完成，进入G9搜索调度。G6快照差异进一步定位到公共测试fixture的导入环境：隔离运行12/12通过，正常公共fixture下原始HEAD和当前代码均同6项失败；继续定位，不改快照/容差。

#### G6 既有快照差异已定位并验证解决

- `ratio_probe.log` 定位到旧 TorchInductor 编译缓存：同样传入 eps_clip_high=0.2，已有compiled产物对一个token使用1.28上界，而eager得到正确1.2。公共fixture导入路径触发该缓存；不是P2O loss公式改变。
- 使用任务独立 `TORCHINDUCTOR_CACHE_DIR=/tmp/p2o_loss_snapshot_clean_20261105` 并关闭FX磁盘图缓存后，原有12项快照全部通过。进一步完整loss+重要性修正回归 **82 passed，34 warnings，319.68s**，日志 `p2o_g6_20261105/clean_loss_regression.log`。未删除共享缓存、修改快照、放宽容差或改loss公式。最终短训练将使用独立缓存路径。

### 2026-11-05：G9 异步 GEPA 搜索验收

- 实现持久化搜索调度、Pareto父代选择、C<=16一父一在途/C>16可重复前沿父代、探索预算预留、ready候选warmup、按全局best判断patience、停止锁存与收尾。minibatch严格改善才做dev；dev低于best仍准入；重复hash不增加池大小。固定面板/seed、阶段、提案、预算和完成事件均落盘，恢复复用episode缓存且不重复收费/提交。
- SearchRuntime直接走原生session v2和SETA的全局/端点沙盒限流，不添加模型信号量。分别路由policy/reflection，保留bonus、聚合有效dev重复、无效前置轨迹供反思。父/子模板在search-train的完整成功也写入求解证据，供后续任务特定分配。
- 增加各阶段/请求/时长/turn/无效数日志；W&B独立cycle事件横轴，SAVE/gepa下保存raw JSON、事件JSONL/JSON、PNG/SVG。CPU全P2O回归334 passed；扩充中断/求解证据/停止锁存后定向27 passed（`p2o_g9_20261105/solver_final.log`）。
- 真实初始模型、144沙盒小预算搜索完成，815.66秒（包含多个新任务镜像构建/超时）。1个bootstrap、3个前置、1个独立反思（4轮）、3个后置；1个任务apt安装超时，前/后无效结果按无效处理，前置无效证据正常参与反思。新模板未提高mini均分，未做候选dev，按budget=1正常结束，C=1。无伪造有效模板或成功分数。
- 证据：`seta_miles_runtime/diagnostics/p2o_g9_20261105/real_audit.json`、`cycle_result.json`、`cycle/search_state.json`、`cycle/episodes/**`。本组policy/reflection均使用同一初始checkpoint做接口验证；训练后policy与初始reflection的资源隔离已由G4验证，最终G10合并实测。G9完成，进入G10。

### 2026-11-05：G10 入口与整体验收进行中

- 已接入 `--algorithm p2o` 和 `SETA_ALGORITHM=p2o`，EXP_NAME自动区分；新增原生async phase adapter、配对恢复入口和每轮日志/图表。使用说明见 [P2O训练与恢复](p2o_seta_usage.md)。正常GPU拓扑沿用既有配置，P2O启动要求原生Ray、offload-train和fully-async。
- 入口回归发现原生 `--fully-async` 无条件覆盖显式插件path，导致P2O参数未注册。原生参数owner现允许FullyAsyncRolloutFn子类保留该模式，其他冲突函数仍拒绝；没有移除异步cache/driver/weight-update约束。参数+P2O相关回归888 passed，完整启动器+边界回归676 passed，日志 `p2o_g10_20261105/{plugin_regression,launch_final}.log`。
- 第一次正式入口调试在模型加载前退出：native validator将新的load目录回退成初始ref-load，被P2O误当作无配对的baseline断点。已修复：使用原生requested_load区分，并在parse_args的preprocess阶段选择配对load根目录，避免恢复时错误启用finetune/no-load-optim。定向5 passed，日志`fresh_load_fix.log`；失败日志保留`gpu_console_attempt1.log`。
- 正在重跑独立验收。诊断脚本 `p2o_g10_20261105/launch_smoke.py` 直接调用正式共享shell/家族启动器；3个原始真实任务、batch2×K2、hard阈值1、trigger3/dev1×4/mini2/budget1，仅用于加速覆盖GEPA。正式默认值保持不变。目标先3次更新，再从配对checkpoint重启到第4次更新并覆盖下一轮GEPA；当前尚未完成，不能报告端到端通过。

#### G10真实训练后的首次GEPA入口修复

- 正式入口已完成step0：loss=-0.0002608448，grad_norm=1.0076947，学习率1.7e-5；全失败组未被过滤，另一个组有非零优势。收集3个调试任务后drain保留1组尾批。pre_gepa完整checkpoint保存成功（197.8秒，约107GiB），配对manifest已发布。
- 随后export_hf断言失败：普通SETA配置没有提前创建SnapshotPublisher。修复原生按需HF导出：复用同一个publisher初始化方法，首次显式export时创建，不重置已有weight_updater，也不要求开启周期save-hf。补测首次/重复导出、updater身份不变和错误传播。下一次运行从此pre_gepa配对断点恢复，不重跑已成功step0。
- 最终CPU联合回归（本次导出修复前）1050 passed，167 warnings，58.21秒：P2O、全部启动器、SETA测试。使用已缓存tokenizer的HF离线模式，日志final_cpu_offline.log；未完成的联网版本保留final_cpu.log。反思指令移除固定“三个”表述以支持mini配置，31项反思回归通过。
- 当前W&B服务认证HTTP401，本次使用离线W&B并保留本地数据；未宣称在线上传成功。待有效凭据可同步。本次失败完整日志保留gpu_console_attempt3_export.log。

#### G10第一轮完整搜索、模板训练与长轨迹检查

- export_hf按需初始化修复后93项相关回归通过（export_fix_final.log）。从pre_gepa实际恢复成功；finetune=False，optimizer/RNG未跳过，保留同轮3个难任务和1个尾批。
- 真实资源切换131.76秒，7个policy实例加载训练后HF，1个reflection实例加载初始HF。抽查q_proj、down_proj、lm_head各64×64切片与初始权重均有差异，详情policy_initial_comparison.json。
- GEPA1：bootstrap dev四次0.2/0.2/0.7/0.2，均值0.325；mini前均值0.15，初始模型两轮反思产出360字符模板，mini后0.20；候选dev四次均0.20。候选按mini改善准入，即使dev低于best；C=2、Pareto=1、warmup不计patience、budget=1正常结束。反思输出偏泛化，本记录不作为方法有效性证据。
- 原拓扑恢复、post_gepa配对保存成功，step1实际消费4条轨迹，其中2条带模板；增强/原始token数12304/12030，双上下文评分和CRM运行。step1因组内同奖梯度0；step2有非零梯度1.3376487、loss=0.0005153716，正在保存最后checkpoint。
- 反思命令增加安全ls -la/-al/-l/-a，保持固定路径重建、禁止任意shell；37项通过。长轨迹证据仅剔除input_ids和choice.logprobs，保留完整消息/动作/采样/反馈；完整session原始记录仍存episode cache。新增3任务×30轮×30000token数组测试；反思+搜索51项通过（reflection_long_trace.log）。防止正式30轮轨迹被重复数值数组撑大，未更改训练token或loss数据。
- GEPA1的metrics.json/jsonl/png/svg已落盘并检查图像；W&B保持离线。下一步从最终配对checkpoint重启至总第4步，覆盖GEPA2及历史模板恢复。

#### G10延长运行步数时的scheduler恢复

- 第一次从最终step2断点重启至总4步，在Megatron scheduler加载时拒绝：新配置总样本数16与断点12不同。此前同一总步数的pre_gepa恢复已成功。错误日志resume_attempt1_scheduler.log，训练新step=0，原配对断点未改变。
- P2O入口现启用原生use-checkpoint-opt-param-scheduler，并在P2O preflight验证，沿用断点scheduler参数和计数；改变NUM_ROLLOUT不自动改写既有LR计划。未重置optimizer或跳过RNG。
- 新增入口/前置检查；修正原模型初始化测试mock缺失的get_checkpoint_name。完整启动器+P2O driver+模型初始化回归686 passed，167 warnings，42.35秒（scheduler_fix_final.log）。首次测试失败及原因保留scheduler_fix.log。


#### G10最终交付：短训练、第二轮GEPA与恢复全部通过

本节为最终验收状态；上面的“进行中”和失败记录是保留的历史过程。正式启动/配置/恢复操作统一见 [使用说明](p2o_seta_usage.md)。诊断根目录为 `seta_miles_runtime/diagnostics/p2o_g10_20261105/`。

- 正式共享入口累计完成step 0、1、2、3。实际从pre_gepa配对checkpoint恢复并完成GEPA1，再从step2周期checkpoint恢复，完成GEPA2及step3。两段完整运行均以启动器退出码0、远端清理退出码0结束，分别见 `gpu_console.log`、`resume_console.log`。第一次export失败前成功执行的step0证据保留在 `gpu_console_attempt3_export.log`。
- step3 loss=0.0001726896、grad_norm=0.8760603；消费4条episode，其中2条有模板，增强/原始上下文token数14609/14335。step1的零梯度来自同组同奖；step0、2、3均有有限非零梯度。真实双上下文评分、CRM与IcePop链路已运行，不只测试模板文本拼接。
- GEPA2实际切换耗时130.67秒，7个policy推理实例与1个初始reflection实例；bootstrap dev均值0.225，mini前0.15。反思8轮未交付模板，以no_template正常收尾，budget=1、C=1。历史模板及3个任务的旧分配保留，恢复原训练拓扑后成功更新step3。反思原任务目录访问被拒绝，诊断发现的合法目录尾斜线问题已修复并回归；不允许反思继续执行原任务。
- 最终原生模型保存于 `checkpoint/iter_0000003/`，完整配对路径见 [acceptance_audit.json](../seta_miles_runtime/diagnostics/p2o_g10_20261105/acceptance_audit.json)。scheduler num_steps由12恢复并推进至16，其余字段逐项相同，原LR计划保持不变。最终rollout保留2轮历史、1个有效模板、3个旧分配，window_id=6；cycle_id=3为下一轮待处理状态，训练预算已结束，未执行第三轮GEPA。
- 每轮 `checkpoint/gepa/cycle_000001/`、`cycle_000002/` 已保存原始episode、事务、模板、metrics JSON/JSONL及PNG/SVG。离线W&B文件中实际存在训练grad_norm和两轮GEPA的candidate指标，见 `wandb_offline_audit.json`。在线凭据返回HTTP401，未上传；更换有效凭据后可按使用说明同步。
- 最终清理审计通过：8张GPU各10MiB、利用率0%；训练容器没有Ray/训练/推理进程；144服务status=ok、max_concurrent=32、active=0。见 `final_cleanup_audit.json` 和 `final_sandbox_health.json`。未启动长期实验。

最终验证记录（各套件有重叠，不相加作独立测试总数）：

| 验证范围 | 结果 | 诊断日志 |
| --- | --- | --- |
| P2O + 全部启动器 + SETA联合回归，导出修复前 | 1050 passed | `final_cpu_offline.log` |
| 按需HF导出及生命周期相关回归 | 93 passed | `export_fix_final.log` |
| scheduler修复、完整启动器、P2O driver与模型初始化 | 686 passed | `scheduler_fix_final.log` |
| 长轨迹反思与搜索回归 | 51 passed | `reflection_long_trace.log` |
| 最终反思路径与权限回归 | 39 passed | `reflection_paths_final.log` |
| G6独立编译缓存loss回归 | 82 passed | `../p2o_g6_20261105/clean_loss_regression.log` |
| 最终真实产物审计 | passed | `audit.py` → `acceptance_audit.json` |

验收脚本 `launch_smoke.py` 调用正式入口，调试使用3个真实任务、batch2×K2、trigger3/dev1×4/mini2/budget1、policy最多3轮、reflection最多8轮。参数仅作用于独立诊断进程；正式默认值未改变。实际验证包括正常训练资源 → 全推理资源（7+1）→ 原训练资源恢复、初始reflection和训练后policy的权重差异、模板后续训练以及保存恢复。最后的资源、scheduler和历史分配审计由 `audit.py` 聚合，可重复执行。

验收边界：只实测单物理节点8卡；多节点GPU规划与非法布局经过单元测试，未实测跨机NCCL。搜索中途事务恢复经过单元测试，实际进程重启覆盖pre-GEPA和周期checkpoint。短预算反思质量有波动，两轮结果只证明功能路径可以运行，不是P2O效果结论；正式规模64任务、16维dev和并发16未做长时间效果实验。未因本轮验收更新CUDA、GPU或沙盒系统环境。


### 2026-11-05：统一训练参数、GEPA引擎继承与阶段计时

本次按用户追加要求完成；不占用GPU、不启动训练、不修改144服务或环境。正式用法统一维护在 [使用说明](p2o_seta_usage.md)，调用链及共享沙盒清理边界见 [调用链](p2o_call_chain.md)。

- GRPO/DAPO/P2O共享入口默认更新为staleness=4、LR=1e-6、constant且warmup=0、weight decay=0.01；显式覆盖仍可用。启动命令快照只出现上述预期差异。续训旧P2O断点仍遵循原生保存的optimizer/scheduler参数，新默认值不静默改写旧实验。
- GEPA取消独立reflection卡数/TP及policy TP参数。通过原生resolve_sglang_config读取训练同质engine配方（包含YAML overrides），phase_specs继承并调用原生engine spec/launcher，只覆盖模型路径、放置位置和显存生命周期。反思固定一个实例；其余本run GPU用相同TP组成policy实例。CPU覆盖单/双节点TP1/2/4/8、节点身份、非法布局、异质/PD拒绝、启动失败和资源恢复。当前不支持单个TP实例跨节点；未重新做GPU/NCCL验证。
- 新增GepaTimings持久化阶段墙钟时间。整体、前后checkpoint、切入/恢复资源、完整搜索、bootstrap dev、mini前/后、反思、候选dev均记录原始区间、累计值、成功/失败/中断状态，导出JSON/JSONL/PNG/SVG并写log和W&B独立timing事件轴。CPU验证异步完成乱序、重叠时长、失败、中断恢复和W&B字段/本地值一致；未验证线上上传，需有效W&B凭据。
- 新增跨轮checkpoint测试：两轮有模板、一轮无新模板，再收集下一轮难任务；保存/加载后原始历史、归一化概率、难任务优先级、旧分配、RNG、buffer、重试及预取/延后窗口一致，后续24次提交抽样一致。原有真实G10恢复证据仍保留，不以本次CPU测试替代真实GPU验收。
- 新增8组×16episode取样测试：按完整group保留，组内任一episode版本落后5次淘汰整组、落后4次保留。GRPO/DAPO原生FIFO、P2O随机完整组加混入调整，未更改基线取样顺序。

本次诊断目录：`seta_miles_runtime/diagnostics/p2o_updates_20261105/`。分组测试有重叠，不累加作为独立总数：

| 测试范围 | 结果 | 日志 |
| --- | --- | --- |
| 共享默认值与SETA启动器 | 68 passed | `defaults.log` |
| GEPA拓扑继承、phase driver和启动器 | 99 passed | `topology.log` |
| 阶段计时、driver与搜索 | 23 passed | `timing.log` |
| 跨轮恢复、完整组边界与计时 | 33 passed | `restore_groups.log` |
| SETA启动器快照更新并核对差异 | 4 passed | `snapshot_retry.log` |
| 全部启动器 + 全P2O + 原生异步buffer/rollout | **1206 passed，178 warnings，62.60s** | `final_cpu.log` |

快照首次执行使用无home的容器UID，PyTorch导入尝试创建`/.cache`而失败（`snapshot.log`）；改用该容器现有测试用户/缓存后通过，未修改系统环境。Black/isort格式化、shell语法及git diff空白检查通过。本次没有新的CRM耗时profiling，也没有在线W&B上传成功的结论。


### 2026-11-05：P2O训练消费对齐FIFO

用户进一步修订训练batch取样顺序：先执行原生陈旧度过滤，再以完成入池顺序FIFO取完整组。配额满足时直接使用前B组；模板不足时补最早入池的模板组、延后最晚入池的普通组；超上限时保留最早模板组并补最早普通组。可用池不能满足配额则直接返回FIFO前B组，不新增配额等待。延后组保留原队列相对位置，消费不再按GEPA轮次排序或消耗RNG。rollout提交侧的自然采样、最近GEPA优先级和模板抽样保持原逻辑。此修订取代上一节中“P2O随机完整组”的描述。

- 测试覆盖不同异步完成顺序、配额两端、FIFO回退、完整组、age=4/5边界、延后队列及checkpoint恢复、消费不推进模板抽样RNG。
- 独立枚举8组池的全部256种模板分布，与所有合法4组子集的最早FIFO解比较。
- 全P2O及原生异步buffer/rollout回归：**535 passed，28 warnings，9.95s**，日志 [fifo_cpu.log](../seta_miles_runtime/diagnostics/p2o_updates_20261105/fifo_cpu.log)。未运行GPU训练；格式和git diff空白检查通过。
