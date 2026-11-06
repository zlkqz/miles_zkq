# P2O + SETA：Miles 实现计划

状态：G1–G10实现与本机验收完成。正式入口累计完成4个训练step、两轮加速GEPA以及pre-GEPA和周期checkpoint恢复；各组测试与真实验收的范围、限制及证据统一见 [实现进度](docs/p2o_implementation_progress.md)。正式启动与恢复见 [使用说明](docs/p2o_seta_usage.md)。本文取代此前的epoch末触发方案。

## 实验组与已完成的启动配置

| 实验 | 同分过滤 | 入口 | 默认实验名中的标识 |
| --- | --- | --- | --- |
| GRPO | 关闭 | `my_scripts/run_seta_grpo.sh` | `_grpo_` |
| DAPO | 开启 | `my_scripts/run_seta_dapo.sh` | `_dapo_` |
| P2O | 关闭 | `SETA_ALGORITHM=p2o bash my_scripts/run_seta.sh` | `_p2o_` |

这里 DAPO 是用户指定的 filter-on 对照名称，不额外修改 clip、loss reduction 或其他训练参数。两份后缀脚本复用 `my_scripts/run_seta.sh`；旧入口保留默认 DAPO 行为。Python 启动器的 `--algorithm grpo|dapo|p2o` 是过滤配置的权威入口，纯 rollout 调试模式沿用不筛同分组的行为。

默认 EXP_NAME 根据实验组派生，SAVE_PATH、OUTPUT_DIR 和 WANDB_RUN_NAME 继续由 EXP_NAME 派生；显式用户环境覆盖仍保留。旧版本虽然名为 grpo，却开启过滤，属于本次定义的 DAPO 组；不移动、重命名或自动继续其旧 checkpoint。不同组正式比较应从相同初始权重独立启动。

共同基线保留 SETA 连续 reward、K=16、多轮轨迹、GRPO 优势和 fully-async，异步重要性修正采用 IcePop（见第 5 节）。P2O 中全零组仍参与正常数据流，其组内优势通常为零；GEPA 旨在通过模板让后续新轨迹获得可学习的奖励差异。

## 0. 当前设计：64 个难任务触发，异步 GEPA 与 agentic 反思

本节取代之前 32 个难任务、beam=2、episode 预算和直接 LLM 反思方案。以下为实现契约，已落地范围及测试证据以分组 TODO 为准。

### 0.1 配置与计分

| 配置 | 当前值 | 状态 |
| --- | --- | --- |
| `gepa_trigger_unique_hard_tasks` | 64 | 用户指定，可调 |
| `gepa_dev_task_count` | 16 | 用户指定，固定数量，不用划分比例 |
| `gepa_dev_repeats` | 4 | 用户指定，可调 |
| `gepa_max_inflight_explorations` | 16 | 用户最新指定，取代32 |
| GEPA reflection实例数 | 1 | GPU数/TP和其他引擎设置统一继承训练rollout engine，不另设参数 |
| `gepa_reflection_minibatch_size` | 3 | 用户已确认三个任务 |
| `p2o_task_mix_ratio` | 0.25 | 软目标；自然命中计入，允许超过25% |
| `p2o_max_template_ratio` | 0.50 | 提交窗口与训练 batch 各自按任务组控制；可用池无法满足 25%–50% 时按各阶段原始顺序回退、不等待 |
| `gepa_max_explorations` | 256 | 建议初值，可调；按探索启动次数预留预算 |
| `gepa_exploration_no_improve_patience` | 32 | 可调；C>16后按探索终态计数，包括minibatch未涨及dev未刷新最佳 |
| `hard_reward_threshold` | 默认 1/K，目前1/16 | 已确认，可调范围[0,1]，包含等号 |

每个难任务由K条有效episode的**无bonus测试通过比例**均值 `<= hard_reward_threshold` 判定；阈值范围[0,1]，默认1/K，当前K=16。P2O关闭同分过滤，有效全零组仍进入训练，沿用Miles的零优势行为。缺失奖励、ABORTED、基础设施错误不伪装成零分。

### 0.1.1 已确认的两种评分口径

用户明确：仅判别难样本时不引入full-pass bonus，训练和GEPA迭代仍保留bonus。

```
progress = passed / total                    # total > 0，范围[0,1]
reward = progress + full_pass_bonus * I(passed == total)
is_hard = mean(progress over K valid episodes) <= hard_reward_threshold
```

| 用途 | 使用分数 |
| --- | --- |
| 难样本判定、累计64个难任务 | 无bonus的progress均值 |
| GRPO/DAPO/P2O训练reward与优势 | 含bonus的reward |
| GEPA反思前/后minibatch比较 | 含bonus的reward |
| GEPA dev四次均值、Pareto、最佳分数和早停 | 含bonus的reward |
| 历史模板轮内标准化与抽样概率 | 基于含bonus的dev总分 |

默认bonus=0.2，训练/GEPA的reward最小0、最大1.2；判难progress始终在[0,1]。直接从同一有效verifier结果的passed/total得到progress，保留独立字段，不用reward除以1.2，也不对每条轨迹一律减0.2。全通过才有bonus，部分通过比例保持原值。

K=16时，一条全通过+15条零分的progress均值为1/16，包含边界，判难；两条全通过+14条零分为2/16，不判难。该判定仍使用连续进度，不等价于二值成功次数阈值；多条部分通过也可能使均值超出1/16。

日志分别记录hard/progress_mean、hard/reward_mean、hard/threshold及full_pass_bonus；反思文件同时保留progress与reward，明确其用途。新增判难字段从verifier结果贯穿远端响应、episode metadata和组统计；不改三个实验入口现有训练奖励公式。G2已实现独立评分与收集模块；训练接入在G5。

用户已确认：64是触发门槛，收尾后使用全部H>=64个难任务，固定抽16个作为dev，其余H-16个为search-train。典型为48/16，也允许更多search-train。划分在本次GEPA内固定，最终测试集隔离，不丢弃超过64的有效难任务。

候选z对dev任务i计划发起4次评价，令V为这些评价中有效结果的集合，任务分数为 `s[z,i] = mean(reward[z,i,j] for j in V)`；若V为空则分数记0。每次评价必须从独立初始环境运行。四次采用独立可复现采样 seed，同一面板 seed 列表跨候选复用以降低比较噪声；不把同一确定性 greedy 结果复制四次。

候选总体分数为16个任务均值的平均值。Pareto 比较使用16维向量 s[z,:]，不是单次结果，不是64维展开向量，也不是先平均整个 dev 后做标量 Pareto。A 支配 B 当且仅当所有任务均值不低于 B 且至少一个严格更高。等分去重/并列处理明确记录。

用户已确认dev按有效结果聚合：4次中有效3/2/1次就分别取3/2/1次均值，全无效记0，仍提交16维评分向量。不为凑满4次有效结果强制补采。保存attempted_count、valid_count、invalid_count、all_invalid标志和原始结果；单有效结果std记为不可估计，全无效0分是搜索评分约定，不伪装成4次真实零奖励。该约定只改变GEPA dev聚合，不把无效训练轨迹变成有效零分。父/子minibatch沿用相同任务和可重建初始条件；minibatch重复数与dev重复数是不同配置。

### 0.2 16 个探索流程与资源限流是不同层级

一次探索为：领取父模板 → 反思前 minibatch rollout → agentic reflection → 反思后 minibatch rollout → 若改进则完成dev评价 → 提交候选。整个生命周期占一个 exploration 槽位，包括等待资源、等待本流程子任务、反思及提交。

| 层级 | 限制对象 | 等待规则 |
| --- | --- | --- |
| 探索并发16 | 端到端探索流程 | 一个流程结束后立即尝试补位，不等待同批其他流程 |
| 全局SETА沙盒限额 | 所有阶段实际存活的沙盒 | 沿用训练配置；前/后minibatch、dev和反思共用额度 |
| endpoint限额 | 每台远端可容纳的沙盒 | 必须取得目标endpoint名额，不因探索还有空位而绕过 |
| 模型服务排队 | policy与初始反思模型的请求 | 直接请求对应Miles模型端点，由推理引擎调度；不新增应用层模型并发限制 |

如果16个流程同时做3任务minibatch，可能有48个待执行episode；如果都进入dev，则有16×16×4=1024个逻辑episode。后者只是排队工作量，绝不同时创建1024个沙盒。子任务逐个或有界分批领取资源；不能持有一个沙盒名额，再一次性等齐本流程所有名额才启动。

现有 `examples/seta/agent.py` 使用 `rollout_semaphore("seta-harbor", SETA_MAX_CONCURRENT_TRIALS)` 和 `endpoint_slot` 两层限制。两者底层是当前 rollout scope 下的文件锁，跨同一主机子进程共享；**不能假设新建一个Ray协调器或换主机后仍自动共用同一限额**。GEPA所有子任务应经现有拥有者提交，或由单一资源租约服务统一协调；远端自身限额作为最终约束。

每个前置minibatch任务在证据导出完成后正常回收，反思使用新建的轻量沙盒，不跨阶段保留原任务环境。反思沙盒同样取得全局与endpoint名额，执行和回收全程遵守训练侧已有容量约束。等待模型返回时沙盒仍存活，因此其名额仍占用。

不新增policy/reflection请求semaphore或人为QPS上限。这里继续使用Miles的推理后端（当前链路为SGLang），不因用户用vLLM描述排队而更换后端；请求由各自推理引擎内部排队。保留现有超时、失败处理和取消传播，不能因取消探索而遗留工具执行或沙盒。

探索目标并发由当前candidate数量决定，最多16，规则见下节。资源不足时等待中的探索仍占流程槽位；不通过增加探索槽位来绕过沙盒限制。

### 0.3 异步候选池与提交

coordinator 是candidate registry、Pareto front、父代分配状态和停止计数的唯一写入者。C表示已完成dev评价并提交的不同candidate数量，包括经bootstrap评价的空模板，不包含等待dev的临时proposal。

用户指定的并发分支：

- C<=16：目标并发=C，候选各占一个探索位置，不进行Pareto筛选，也不同时重复启动同一父模板。例如C=1时仅1个探索，C=8时为8个。
- C>16：目标并发16，只从当前Pareto前沿选择父模板，允许同一前沿父模板用于多个探索。建议优先不同父代，再以可复现随机抽样补足；重复父代使用不同minibatch/seed，且已在途的(parent,minibatch,seed)不得重复。
- 在领取时检查当前Pareto状态。已经启动的父代后来退出前沿，允许其探索收尾；补位只从新的前沿选择。C从16增长到17时，不取消已有探索，之后按新规则补位。

每次分配原子记录exploration_id、parent_id、C、front版本、minibatch、policy版本与seed。每个流程内部按依赖执行，不设全局代际屏障。新模板通过后置minibatch改进检查即满足candidate准入条件，先记为candidate_pending_dev；完成16×4的dev评价后可参与Pareto、父代选择和模板分配。dev即使比父代或历史最佳更低，也不因此拒绝candidate。调度用的C只计完成dev的ready candidate，pending数量单独记录；重复文本按hash去重，不增加candidate池大小；探索预算与早停按探索身份计数，不按是否入池计数，见0.4节。

探索终态和candidate提交统一按完成事件串行化；candidate提交获得单调commit_id并更新Pareto和best_dev_score，每个探索终态按0.4节更新no_improve_streak，不等exploration_id较小的慢任务。记录实际调度顺序与父代选择，固定seed不保证异步运行必然得到相同搜索路径。

用户已确认：C<=16时，同一父模板在上次探索结束后可以用新的minibatch/seed串行再次探索，每父模板最多一个在途流程。一次探索未产生新candidate时仍可继续探索，不等待候选数量增长。每次新探索独立消耗一次探索预算；它与同一次探索的基础设施重试分开计数。

### 0.4 探索预算与停止条件

探索预算默认256次，连续未改善上限默认32次，均可配置。用户最新要求：**这两个计数都覆盖dev之前的探索**，不能只计算成功进入candidate池的模板。

启动探索前原子预留一个预算token，确保started<=256；一次探索后续因minibatch未涨而结束，或者完成dev但未刷新best，都已经消耗一次预算。模板非法、基础设施失败同样保留已启动预算账目；同一exploration内部的有界重试不重复消耗探索预算，真实调用成本另计。初始空模板dev评价属于bootstrap，单独记录，不计为反思探索。

区分两个数量：ready candidate数C用于父代调度和warmup；explorations_started/completed用于256预算和32次早停。minibatch失败不增加C，但**warmup结束后会增加早停计数**。candidate准入仍只看minibatch改善，dev低分不拒绝入池。

协调器按探索完成事件串行处理：先保存结果；若有按有效次数聚合的完整dev向量则提交candidate并更新C、Pareto和best，再处理本次探索是否改善。比较使用提交前的本轮历史最佳**原始dev总体均分**，不使用历史模板抽样的归一化分数。

- C<=16：warmup只更新best，streak保持0。此前“candidate超过16才开始早停”继续生效，256探索预算始终生效。
- C>16且本次minibatch未涨：不做dev、不入池，streak加1。
- C>16且本次进入dev：dev严格超过本轮此前best则streak归零，否则加1。完整dev candidate无论是否刷新best都保留。
- 第17个ready candidate提交时开始适用计数；warmup的旧失败不回补。warmup期间启动但在C>16后才结束的探索按完成时状态处理。
- 一次探索最多贡献一次终态更新；不能minibatch未涨计一次、又以没有dev再计一次。运行中/重试中不计未改善；终态重复消息按exploration_id去重。
- 非法/重复模板或重试耗尽等终态未产生新最佳，也按一次无改善探索记账，另以reason区分；基础设施错误不写成零reward或有效dev分数。服务整体不可用按故障处理，不能靠重试无限消耗预算。

例如warmup结束best=0.60：一次探索minibatch未涨，streak=1；下次minibatch涨但dev=0.55，streak=2；再下次dev=0.63，best更新为0.63且streak归零。原始分数不变，candidate准入规则也不变。

达到预算或patience后停止新探索。预算在启动时耗尽，最多仍有16个在途流程；patience在某次探索终态事件触发，最多还有15个其他在途流程。允许这些流程收尾并记入最终结果；一旦触发停止，不因收尾出现更好candidate而重新启动探索。这样started不超预算，patience可能因在途结果超过阈值；日志保存trigger时和最终两个快照。不强制中断正在写模板的反思，也不留下未回收沙盒。

实际成本必须另外记录。若minibatch前/后各3个episode，一次进入dev的探索需要6+64=70个policy episode，另有1个多轮反思任务；256次全部进入dev的上界约17920个policy episode，加64个bootstrap episode，且未计失败重试。即使只有25%进入dev，也约有256×6+64×64+64=5696个policy episode。异步改变吞吐，**不消除这部分成本**；256应先作为可调上限，patience与minibatch筛选决定实际开销。

### 0.5 Agentic reflection：独立轻量沙盒、只读分析、唯一模板输出

用户已确认：前置minibatch的3个原任务沙盒正常回收，不复用其环境。先导出各自的input、实际轨迹及feedback到宿主侧，再创建一个独立轻量Harbor沙盒，由初始模型通过Terminus-2进行反思。每次探索的反思会话、模型路由和session均独立，不承接policy对话上下文。

建议文件布局：

```
/p2o_reflection/
  evidence/manifest.json
  evidence/parent_template.md
  evidence/tasks/<task_id>/rollout_input.json
  evidence/tasks/<task_id>/verifier/   # 实际评估脚本及必要只读依赖
  evidence/tasks/<task_id>/trajectory.jsonl
  evidence/tasks/<task_id>/feedback.json
  evidence/tasks/<task_id>/result.json
  output/template.md
```

用户指定的反思证据内容：rollout_input.json保存执行该任务时实际送入模型的初始messages/任务指令及相关输入引用，不能用一个数据集占位prompt代替真实输入；trajectory.jsonl保存后续动作与观察。verifier目录提供**本任务实际使用的评估脚本**及必要只读依赖，正常policy rollout不可见，反思模型可读取分析但不可执行。result.json就是该episode的评估结果，包含逐测试结果、passed/total、progress、含bonus reward和有效性；feedback.json提供错误/诊断与证据索引，不另造一套独立评分。无评估结果时result.json标记invalid及原因，不编造测试输出。

只导出本次3个minibatch任务的上述证据；不将评估脚本注入policy的前/后minibatch或训练prompt，不加入搜索dev/最终benchmark材料。原任务沙盒照常回收，反思仍只使用独立轻量沙盒。manifest记录文件路径、角色和hash，模型主动读取，宿主侧保留原件。

允许无效minibatch轨迹参与反思：已有部分轨迹、初始输入、报错、退出状态和可用verifier结果均可写入证据，明确valid=false和missing字段；没有轨迹时提供输入与失败原因即可。现有harbor_trial返回trial_dir、seta_valid_reward、seta_error/exception信息，基本导出成本预计较低；硬退出前未保存的终端记录不保证可恢复，不为补齐证据保留/重建原环境。无效证据被反思读取不代表获得有效训练reward；minibatch准入仍需可比较的前/后评分，完全无可比较评分时记录该探索未形成可评估改善，不凭基础设施错误冒充成功。

执行限制必须同时有prompt约束和实际权限约束：

1. 非root反思执行身份，evidence与根文件系统只读；预创建唯一可写的 `output/template.md`，其父目录不可由agent创建/删除/重命名文件。日志、终端记录和运行时必要文件由框架独立管理，不作为agent可写工作区。只读根目录与Terminus/tmux运行需求的兼容性需实测，不能把框架必须可写的目录顺便暴露为agent任意工作区。
2. 反思专用命令执行适配器只允许读取、列目录、检索及受控写模板。可支持cat、head、tail、wc、grep/rg等必要选项；禁用修改文件的选项，不允许通过shell展开、管道子进程、解释器、编辑器、exec/find动作或符号链接绕过范围。不能靠命令名正则就宣称实现只读。
3. 模板写入使用固定目标的受控写入操作，参数校验真实路径与文件类型，不能跟随符号链接写其他文件。若保留Terminus命令协议，写入需经过专用执行代理解析，不把未经检查的任意shell命令交给容器执行。
4. Agent不能继续运行原任务、测试新动作、安装依赖、联网获取额外信息或修改反馈；模型连接由框架提供，证据分析工具不获得任意联网能力。限制只作用于新反思任务，不改变正常SETA训练agent的工具能力。

建议启动prompt表达：你是提示优化分析agent；本次仅根据manifest中的3份任务证据阅读、检索和推理，不运行原任务、不修改证据、不开展额外实验；唯一交付物是指定绝对路径的template.md，写可复用的策略提示，保留原agent动作协议，不输出这3个任务的硬编码解答。完成后结束会话。

读取实际template.md、验证非空/长度/格式/hash后才进入后置minibatch；聊天回答不是模板交付物。无文件、越权操作或超时记为反思失败并按探索预算记账，不新增candidate。证据和最终模板落盘后正常回收反思沙盒；后置minibatch和dev均启动新的原任务环境。

当前SETA `harbor_trial.py` 专门加载真实任务并读取verifier结果，不能把反思任务伪装成普通SETA任务或让文件产出走测试通过率reward。计划新增独立reflection trial adapter，复用Harbor/Terminus调用、远端限流和清理，完成条件为合法文件交付。实施前验证受限执行代理能否接入Terminus命令调用链；如当前接口不支持，需扩展该接口并验证，不能只加prompt后称权限限制已完成。

### 0.6 W&B 与每轮本地记录

W&B 支持自定义横轴，不限于训练step。官方文档：<https://docs.wandb.ai/models/track/log/customize-logging-axes>。Miles 的 `miles/utils/tracking_utils/base.py` 已有 `define_step_key_metric_group`，`wandb_utils.py` 也分别定义train、rollout、eval轴。

计划由primary tracking进程注册 `gepa/*` 使用 `gepa/event_step`，这是跨cycle单调递增的提交/状态事件计数；每条日志同时写该轴、cycle_id、cycle内exploration/commit计数、训练step快照和耗时。不要把固定的训练step反复作为wandb内部step，也不要让16个探索worker竞争直接写主run。候选收敛曲线可以使用跨cycle单调candidate_commit_step并附cycle_id筛选；每轮本地图以本轮commit_id或elapsed_seconds为横轴。

建议metrics：candidate总数、Pareto数量、eligible父代数、每父代在途/累计探索次数；started/completed/inflight/failed探索数；minibatch通过率、candidate接纳/去重率；best/本次dev均值、每任务4次std、no_improve_streak；policy episode数、reflection轮数/tokens、各阶段耗时；沙盒active/waiting（按阶段）、endpoint占用、模型请求延迟（后端可提供时记录排队时间）；预算余量、停止原因、触发时与最终快照。

每次GEPA cycle持久化一个目录，目录下每个探索有子目录：

```
<SAVE_PATH>/gepa/cycle_000001/
  config.json  split.json  events.jsonl  candidates.jsonl
  scores.csv  summary.json  plots/
  explorations/000001/{manifest.json,feedback/,reflection/,template.md}
```

W&B与本地各保存一份。本地根目录使用最终解析的模型保存路径args.save（launcher的SAVE_PATH），不使用独立OUTPUT_DIR；每轮GEPA单独目录，保存原始episode/反思记录、dev重复结果、候选血缘、事件日志、CSV和plots。图由同一份原始数据生成；W&B仍使用独立GEPA横轴，离线或上传失败不影响本地记录。

2026-11-05已补充整轮及各阶段耗时的独立事件轴、持久化原始区间与图表；当前文件布局、阶段定义和并发计时边界统一见 [使用说明的日志与指标](docs/p2o_seta_usage.md#日志与指标)，上面的目录树为设计示意。

checkpoint除Miles既有save_interval外，在**GEPA前、GEPA后各额外保存一次**，与原P2O trainer的before_gepa_optimization/after_gepa_optimization意图一致。三类保存均覆盖P2O状态：

- 模型、optimizer、scheduler、训练step、所有必要RNG及Miles数据源/buffer恢复状态。
- 当前cycle/phase、触发难任务集合与progress依据、历史难任务身份和最近参与轮次、模板文本/hash/分配、各轮dev面板与原始/归一化分数。
- 自然数据游标、已预取未提交任务、主动补入随机状态与配额、已生成待训练组、behavior版本和template版本。
- GEPA中的candidate及pending状态、父代/探索身份、minibatch和dev任务/seed、每个重复结果、started预算、best、warmup与streak、停止锁存状态及W&B事件计数。

GEPA前先完成正在运行的更新与必要收尾，再保存pre_gepa checkpoint并确认落盘后切换GPU。GEPA后先完成探索收尾及模板/历史状态提交，再保存post_gepa checkpoint，确认落盘后才能继续训练。两次可能共享同一个训练step：存储使用不同phase/cycle状态目录与原子manifest，不能用相同step路径覆盖导致配对模型和P2O状态不一致。未变化的模型/optimizer可引用同一已完成快照，是否复用服从Miles checkpoint格式。

GEPA期间按探索终态持久化事件与进度快照，支持从pre_gepa基线加已提交搜索结果恢复，无须为每个事件重写大模型。崩溃恢复不重复扣预算/更新streak/提交模板；保存远端request身份用于清理与结果核对，不把进程对象或Ray句柄当可恢复状态。post_gepa已提交但训练布局尚未恢复时先恢复布局，不重复搜索。检查保存失败、保存中断、同step前后状态、恢复时world布局与初始反思权重身份。

### 0.7 训练混入：提交与消费双阶段条件配额

用户最新修订：rollout 提交窗口保留原采样逻辑；训练 batch 改为**先过滤陈旧组，再按完成入池顺序FIFO取组**。当前可用池能组成 25%–50% 的模板任务比例时，对初选结果作必要补/替换以满足区间。B=8 时为 2–4 个完整任务组。初始抽样已满足区间则保持不变。两个阶段独立判定，不能用提交比例推断消费比例。

若当前可用池无法组成该区间，则本次放弃比例控制，直接采用本阶段初选结果（提交侧原采样、训练侧FIFO），不等待难任务或普通任务，不为比例丢弃/重生成轨迹。这个回退可以低于25%，也可以超过50%；50%是可用池可满足条件时的上限，不是任何情况下都成立的绝对上限。正常训练等到足够完整任务组的既有逻辑不变，只是不新增为了配额的等待。

- **rollout 提交前**：从当前可用未消费任务随机采样，识别历史难身份、可用非空模板，再有限补/替换。自然命中计入比例；三个自然命中全部加模板；超过四个时按最近参与 GEPA 的优先级决定注入对象，其余保留原任务用原 prompt。剩余任务/历史池无法满足目标区间时正常随机回退。被替换但尚未提交的自然任务保留待取状态，不重复推进游标、不静默丢弃。
- **训练取 batch**：先取最早入池的B个有效完整任务组；比例合规直接使用。模板不足时，用后续最早入池的模板组替换初选中最晚入池的普通组；模板过多时，保留最早入池的模板组，用后续最早入池的普通组替换超额模板组。最终按入池顺序返回。消费侧不再按GEPA轮次排序或抽取RNG。分类使用实际生成时的模板标记，不改标、不拆组；被延后的组保留原队列位置。当前池无法满足区间时采用最初FIFO结果，无额外等待。

P2O在比例约束下尽可能保持原生FIFO，只有满足配额所必需的组会被延后；P2O关闭时保留原数据源/缓冲行为。沿用Miles的任务组、有效性、staleness和checkpoint接口，buffer顺序随原生快照恢复，提交与模板选择RNG仍由原采样拥有者保存。

每轮全部 H 个难任务（包括 dev）参与 GEPA 后都保留 ever_gepa 身份，无新模板也不删除；模板来源为所有历史 GEPA。记录 last_participated_cycle 与 template_source_cycle，保留已知求解模板/旧分配优先规则，跨轮模板抽样按第 10.5 节归一化。组内 K 条轨迹可以使用不同模板，但每条 episode 内固定，GRPO 组不按模板拆分。整个历史模板池为空时本次原 prompt 回退，不宣称完成了注入。

rollout提交侧主动补入任务优先最近参与 GEPA 的任务，同 cycle 随机；自然命中先抵扣主动补入数。任务优先级与跨轮模板抽样权重分别处理；主动补入结果不计入下一轮自然难任务触发。提交/消费分别记录初选比例、调整后比例、可用组数量、补/替换数量、回退次数与原因。

### 0.8 决策状态

已确认组内可抽取不同模板、全历史模板池、无新模板时保留旧分配或历史池抽样、预算对齐由用户后续自行调整，以及candidate准入只看minibatch改善。50%硬上限的超额处理、全历史空池回退及按GEPA轮次归一化均已确认，精确定义见第10.5节。用户另已确认仅C>16后开始按本轮最佳dev分数计早停；当前实施进度见分组 TODO。

## 2. 方法理解与参考实现差异

论文：[P2O，arXiv:2603.21877](https://arxiv.org/pdf/2603.21877)，主要依据第 3 节、公式 4–7。

论文原方法在一个 epoch 内固定模板集合。上一轮难任务再次被数据源取到时，使用分配的模板生成轨迹；梯度训练去掉模板，保留实际生成的回答与环境观察。训练过程中收集新的难任务。epoch 末固定当前 policy，GEPA 通过失败反馈生成、评价和选择模板，再将模板分配给下一轮难任务。

区分三个模型概念：当前可训练 policy；每次更新前用于 PPO/CRM 打分的冻结 old policy；始终来自初始 checkpoint 的 GEPA reflection model。第三者不能用于替代 CRM 的 old policy。

与 `P2O/verl/trainer/ppo/ray_trainer_baseline.py` 对比，`ray_trainer_gepa_wo_additional_grpo_in_distribution.py` 增加了：

| 增量 | 已定位的实现 | Miles 的处理方向 |
| --- | --- | --- |
| 难样本统计和去重 | `_collect_very_hard_samples`、epoch UID cache、prompt signature | 改为稳定任务 ID 和完整 episode 组统计 |
| 模板注入与还原 | `_apply_prompt_template_to_sample`、`_rebuild_batch_input` | 在真实 agent prompt 和 TITO token 流中记录精确变换 |
| 双上下文打分与 CRM | 训练循环中分别 `compute_log_prob`，生成 `template_ratio_mask` | 放入 Miles 通用 loss/data 管线，避免复制 trainer |
| GEPA 搜索 | 数据划分、反馈整理、批量反思、候选评价、Pareto 与模板分配 | 搜索算法与任务评价、模型资源拆分 |
| 反思后端 | vLLM/API 分支 | 替换为独立 Miles reflection pool |
| 生命周期和恢复 | `_finalize_epoch_prompt_optimization`、GEPA 状态保存/恢复 | 与异步 producer、checkpoint、资源切换统一协调 |

旧实现还包含 prompt replacement、importance-ratio 修复、hard/soft clip、多种反思后端等实验分支，不整体搬入 Miles。其默认 CRM 配置也不能直接当作论文参数：代码用 `[1-low, 1+high]` 表示范围，`my_scripts` 中的 0.01–10 配方实际设置 low=0.99、high=9；另有 0.8–1.2 配方。计划直接暴露 CRM 的正数上下界，论文主方法为 `[0.01, 10]` 的二值 mask。

`verl-agent/docs/p2o` 可借鉴任务身份、失败反馈、搜索预算和恢复状态的设计；其步级训练样本、K=8、≤1/8、20 epoch 间隔等不作为本实现默认值。这些文档是参考设计，不等于相应功能均已经实现或经过验证。

## 3. Miles 当前链路与接入位置

| 当前文件 | 已确认行为 | 计划接入 |
| --- | --- | --- |
| `my_scripts/run_seta.sh`、`scripts/run_qwen3_8b_seta.py` | 默认 fully-async；训练 TP=4；推理 engine 每个 2 卡；K=16；GRPO + TIS | 独立 P2O 配置入口，复用 SETA 基线配置 |
| `examples/seta/agent.py` | 请求远端 Harbor；无效验证标记 ABORTED | 传递任务 ID、模板、阶段和 epoch；区分无效结果与真实失败 |
| `examples/seta/harbor_trial.py`、`progress.py` | reward=测试通过比例+全通过 bonus；返回 passed/total/tests | 保留训练/GEPA奖励，仅判难提取无bonus的测试通过比例 |
| `examples/seta/terminus.py` | 长度预算耗尽后仍验证环境状态 | 保留该语义，不能一律视为基础设施错误 |
| `miles/rollout/session/samples/merge.py`、`generate_utils/sample_utils.py` | session 记录先转成中间 Sample，随后可合并；记录逐调用权重版本 | 两个上下文的动作 token 对齐、掩码和版本映射 |
| `miles/rollout/data_source.py` | 取样时可跨数据集边界并自动递增 epoch | P2O 模式下增加显式 cycle admission gate |
| `fully_async_rollout.py`、`fully_async_data_buffer.py` | 后台 producer、in-flight/retry/output buffer；独立消费 | 阶段收尾与恢复协议 |
| `miles/ray/rollout/train_data_conversion.py` | 奖励处理、序列字段打包、DP 分片均有字段白名单 | 新字段完整贯穿序列化、分片和训练读取 |
| `miles/backends/training_utils/loss/hub/` | PPO、优势、TIS 的通用实现 | 独立 CRM 项，与 TIS 明确组合 |
| `train_async.py` | 会提前调度下一批；fully-async producer 不依赖 get 持续运行 | cycle 屏障及 GEPA 阶段协调 |
| `miles/ray/placement_group.py`、`rollout/inference_controller.py` | placement、权重更新、offload、pause、cell 生命周期 | 临时推理资源租用与恢复 |

不能把 agent 每次 LLM 调用等同于一个独立 GRPO 样本。组内归一化按同一个任务的 episode 组进行；若 Miles 将一条轨迹转换成多个训练序列，应沿用 rollout 身份、权重与归一化规则，避免长 episode 被重复计奖或加权。

当前SETA基线已按用户要求切换session server v2，以兼容Miles纯异步默认的版本隔离KV cache。线性episode仍是一条叶路径Sample；分支情况下必须沿用Miles的rollout身份和共享动作mask，后续判难/CRM不能按叶子数替代独立episode数。本次切换无环境配置变更，验收见[实现进度](docs/p2o_implementation_progress.md)。

## 5. 多轮 context distillation、CRM 与异步重要性修正（IcePop）

### 5.1 三个时刻与两种ratio

以一个实际生成的assistant token为例，轨迹生成时可能使用policy版本v10；它进入训练时actor已经到了v12。先冻结v12作为本batch的old policy，再做梯度更新得到当前可训练policy。初始反思模型完全不参与这些概率。

| 字段 | 来源和时刻 | 用途 |
| --- | --- | --- |
| l_behavior_aug | rollout引擎在实际生成token时返回的log-prob，增强上下文h+；例v10 | 记录样本真实行为来源；对应Miles rollout_log_probs |
| l_old_aug | 更新前冻结actor对同一token、同一增强历史重新打分；例v12 | TIS分子、CRM分母 |
| l_old_plain | 同一冻结actor对去模板历史h打分；例v12 | CRM分子、PPO分母 |
| l_current_plain | 训练forward在去模板历史下计算，带梯度；随优化变化 | PPO分子 |

`tis_ratio=exp(l_old_aug-l_behavior_aug)`修正**生成→训练前**的policy版本及后端差异。SETA 启动器通过 `--use-tis` 和 `--custom-tis-function-path miles.backends.training_utils.loss.hub.corrections.icepop_function` 选择内置 IcePop，范围继承 `miles/utils/arguments.py` 的默认值 `[0,2]`。范围内用原比率加权，范围外将该 token 的 policy-gradient loss 权重置零，不把超上界比率截到 2。该权重与 old 评分均 detach，不通过行为模型反向传播。

`ppo_ratio=exp(l_current_plain-l_old_plain)`衡量**训练前→当前梯度更新**对原始上下文概率的改变，进入PPO clipped surrogate，限制这次更新幅度。它随训练forward变化，不是样本生成时保存的概率。第一次更新前两者同权重时PPO ratio数值可为1，但对当前概率仍有梯度。

SETA 启动器显式设置 `--eps-clip 0.2 --eps-clip-high 0.28`，使 PPO surrogate 的 clip 区间为 `[0.8,1.28]`；这不是将真实 ratio 强制限制在该区间。核心参数默认上界为 `1.2`，因此这里必须显式设置上侧 epsilon。

CRM另比较同一个old policy的两种上下文，过滤过度依赖模板的token。三个量职责分别是行为差异修正、更新幅度控制、上下文差异筛选；不是三次生成。对模板数据需要两路old打分，对无模板数据上下文相同，可复用评分且CRM=1。多轮行为版本可能逐调用变化，behavior log-prob按实际生成token保留。

例如某token生成时概率0.10，训练前增强概率0.15、去模板概率0.12，当前训练概率0.132：TIS=1.5，PPO=1.1，CRM检查0.12/0.15=0.8是否落入允许区间。该组合仍是P2O的上下文蒸馏适配，不声称CRM校正了全部环境状态分布差异。

### 5.2 公式与实现契约

令 h 为保留实际动作和环境反馈、移除辅助模板后的历史，h+ 为增强历史；所有概率对齐到相同的实际 assistant token。

在一次训练更新之前冻结 old policy，计算：

```
l_old_plain = log pi_old(action_token | h)
l_old_aug   = log pi_old(action_token | h+)
crm = 1[log(alpha) <= l_old_plain - l_old_aug <= log(beta)]
ppo_ratio = exp(l_current_plain - l_old_plain)
```

CRM 两侧必须来自同一权重版本，均不参与梯度。训练 forward 使用 h；动作和工具反馈保持真实轨迹，不重新运行环境生成所谓“无模板轨迹”。工具输出及其他非动作 token 的原 loss mask 保持为零。

不要 decode 整条生成文本后重新 tokenize 回答：TITO 的 stop token、轮次分隔和截断规则可能改变 token 身份。先在真实两轮/三轮轨迹上验证可逆的提示变换，保留动作 token IDs；prefix 不一致、上下文压缩或重试导致分支变化时，不允许静默拼接。两路序列长度、position、mask 和动作索引分别重建。

异步适配（用户已确认）：保留实际生成的 `l_behavior_aug`，将 TIS 的版本/后端差异与 CRM 的上下文差异拆开：

```
tis_ratio = exp(l_old_aug - l_behavior_aug)
icepop_weight = tis_ratio * 1[tis_clip_low <= tis_ratio <= tis_clip]
loss_token = original_action_mask * crm * icepop_weight * PPO(ppo_ratio, advantage)
```

无模板样本两路上下文相同，CRM 恒为 1，并退化为基线 IcePop。增强样本不能继续用 `l_old_plain - l_behavior_aug` 当纯异步修正，否则会再次引入上下文比例。这是对 fully-async SETA 的明确算法适配，不声称是论文原有异步实现。

旧代码在训练前重算两路概率，因此不需要为了 CRM 常驻保存每个行为版本的模型。多轮 behavior log-prob 仍需逐调用保留实际版本；该适配的 TIS 正确性和方差须单独测试。

CRM 应作为 policy loss 的乘法 mask，并保留论文按原动作长度归一化的语义；直接修改基础 loss mask 可能同时改变分母、优势、KL 或指标统计，须避免这种隐式变更。全 mask 情况安全地产生零 policy loss。保留 episode/group 优势语义，不对每个 turn 再做 GRPO。

## 7. 按Miles原训练拓扑切换GEPA推理资源

### 7.1 保留正常训练布局

正常训练GPU分配完全采用Miles已有逻辑，不能写死4+4，也不重写其单/多机策略。4+4和双机8+8仅为配置示例。阶段切换前保存已解析的trainer/rollout placement、node身份、每节点物理GPU和Ray bundle映射；GEPA后按该记录恢复原布局及权重同步。

GEPA使用本次run可支配且已安全释放的GPU集合，按训练rollout engine解析后的同一配置分配policy/reflection。reflection恰好一个实例，加载初始模型；其余实例加载当前policy。2026-11-05用户修订取消独立反思卡数/TP：8卡、TP2对应3个policy实例和1个reflection实例；16卡、TP2对应7+1个实例。原训练拓扑不变。

多节点按(node_id, physical_gpu_id)定位，临时实例使用Miles已有spec/placement/launch逻辑。继承SGLang CLI/YAML设置，只有模型身份及GEPA常驻显存生命周期按phase覆盖。当前支持同质、节点内完整实例的布局；异质推理池、PD分离、单实例跨节点TP或无法整除的布局在分配前拒绝。详细操作与采样参数边界统一见 [使用说明](docs/p2o_seta_usage.md)。

### 7.2 生命周期、权重和回收

训练阶段不改变world size。GEPA前保存、暂停训练并offload参数/optimizer/buffers；显存释放与Ray资源可复用分别确认。复用Miles的资源提供者与offload机制，必要时扩展phase协调接口，不临时全局改CUDA_VISIBLE_DEVICES来抢卡。

当前policy加载本次边界快照，reflection加载训练开始前的固定模型；独立model_id/路由，反思池不参与policy权重广播。启动临时engine前停用旧路由，验证端口、跨节点通信、模型版本、GPU映射及健康状态后再开放任务。

GEPA全部收尾并保存post_gepa后，关闭临时policy/reflection engines、清理路由/通信组/沙盒，确认资源释放，再恢复原训练和rollout布局、optimizer/scheduler/RNG及权重同步。任一步失败停在持久化phase；不能部分恢复后继续训练，也不能重复搜索已提交的cycle。

验证至少覆盖单机默认配置、双机8+8示例、继承TP2/4/8与恰好一个反思实例、节点异构/不足的拒绝路径、部分初始化失败和重启恢复。此前单机7+1实例GPU验收见实现记录；本次继承TP的新布局仅做CPU规划/生命周期测试，未重新启动GPU，也未实测多机NCCL。

## 8. 建议的实现拆分与顺序

以下是设计阶段的工作拆分；正式实施顺序、分组验收和进度以 [实现 TODO](docs/p2o_implementation_progress.md) 为准。具体新文件名实施时按已有模块归属确定，避免复制一个大型 trainer。

1. **数据与算法契约**：审阅当前调度参数；定义任务身份、无bonus测试通过比例均值阈值、两路上下文、CRM/TIS 和阶段尾批策略。提取脱敏的真实 SETA 多轮样本作为 fixture。
2. **资源可行性 spike**：只验证暂停、保存/offload、配置化初始反思模型、当前policy临时推理、关闭和原Miles布局恢复；比较切换前后权重、optimizer step。此步先于大规模 GEPA 实现。
3. **任务统计与模板注入**：新增 P2O 状态模块，接入 SETA metadata、实际 instruction 和 session；关闭 P2O 时保持原链路。
4. **训练适配**：先验证双上下文 token 对齐，再接通打分与 CRM；明确修改 train-data 字段白名单、后端 batch 读取和 loss 的必要位置。首个后端以 SETA 当前 Megatron 配置为范围。
5. **异步 cycle 屏障**：实现 producer admission、drain、尾批、retry 和 checkpoint；用模拟慢 episode 验证不会串 cycle。
6. **GEPA 模块**：将纯搜索状态、反思客户端、SETA 评价适配和资源管理分离；通过真实环境反馈跑通一个 cycle 边界搜索。
7. **恢复与启动配置**：新增 Python P2O launcher，复用基线参数；保存模板、难集、数据游标、随机状态、预算、搜索候选和阶段事务。区分 pre-GEPA、search-in-progress、templates-committed、training-restored。
8. **小规模端到端验证**：至少两个 cycle，覆盖首次空模板训练、GEPA、次轮模板训练与恢复，以及16探索流程受沙盒限额约束的情形；通过后再开展正式比较。

配置归属：运行级算法开关/阈值/预算/资源布局在启动配置；模板 ID 和 cycle 版本在任务请求；状态只由一个 P2O coordinator 持有，不让 trainer、data source、远端服务分别维护另一份难集。

## 9. 验证与交付标准

- CPU：稳定任务身份、难度阈值边界、无效奖励排除、Pareto/预算、恢复幂等性。
- 轨迹：两轮以上工具交互、stop token、截断、重试；两路动作 token 完全对应；工具 mask 为零；不改变环境历史。
- 数值：相同上下文 CRM 恒为 1；手工构造概率验证阈值和 PPO/TIS 三种比率；全 mask 不 NaN；关闭 P2O 与基线一致。
- 异步：同任务不同完成顺序、跨权重版本、慢任务、尾批、buffer/retry、故障恢复；每轮仅一次模板提交，无跨 cycle 污染。
- GPU：reflection 永远加载初始 checkpoint；GEPA policy 为 cycle 边界权重且不更新；恢复后 optimizer 状态连续、无资源或路由泄漏。
- 端到端日志：raw reward、全通过率、有效 episode 数、难任务数、模板命中率、CRM 保留率、TIS 分布、GEPA 候选/预算/收益、资源切换耗时。
- 实验后续建议：GRPO（filter 关）、DAPO（filter 开）、P2O（filter 关）；无 CRM 和无 context distillation 仅作后续可选消融；统一数据、训练预算和独立评测。最终评测默认不带优化模板，检验能力是否内化；额外记录推理与反思成本。

以上是未来 P2O 验收要求，本轮未执行 P2O 测试或 GPU 试验；基线启动配置验证单独记录。


## 10. 最新确认与仍需定义的边界

### 10.1 全部难任务的模板分配

本轮64个任务都标记为历史难任务，dev/search-train只是搜索中的角色划分，不决定以后是否可注入模板。用户确认沿用原P2O的分配思路：dev任务优先从具有该任务较好评分证据的模板分配，其他任务按dev总体表现抽样；同任务K条轨迹可以使用不同模板，轨迹内部固定，GRPO组不按模板拆分。

模板池扩展到所有历史GEPA。每条模板保存文本hash、来源cycle、该cycle的dev面板、4次评价状态、原始reward与有效均值向量、归一化权重和分配证据。不能把模板在不含当前任务的dev面板得分称为该任务实测效果。历史candidate不因用于训练抽样而自动加入本轮搜索Pareto；若用于新一轮搜索，必须按本轮policy/dev重评。

本轮Pareto计算仍用16维每任务至多四次评价的有效均值；跨轮归一化服务于**训练模板抽样**，不把不同任务集合的历史向量拼到同一个Pareto比较中。同文本模板保留各次评价记录，抽样前按hash合并，避免仅因重复收录而获得额外概率。

### 10.2 整轮没有有效模板

GEPA正常结束，提交本轮历史难任务标记，清空本轮收集器，继续收集下一轮难任务，不因无模板阻塞训练。

任务已有旧分配则保留；无旧分配时从所有历史可用非空模板中按归一化dev权重抽取。无模板产出的本轮不能虚构新评分覆盖旧证据；last_participated_cycle与template_source_cycle分别保存。

若**整个运行到目前为止都没有非空模板**，历史池也无法抽样。用户已确认：暂用原prompt并保留难任务身份，之后有历史模板即可正常抽样；空池阶段不伪造非空模板或宣称完成模板注入。

### 10.3 预算比较

用户明确暂不讨论预算对齐，由后续实验调整。本实现只保证准确记录训练更新、自然任务覆盖、主动增强任务数、搜索探索数、所有policy episode、反思tokens与时间，不强制三组采用新的停止或算力匹配策略。

源码中的NUM_EPOCH通过 `len(dataset)//rollout_batch_size` 换算训练批次；过滤/混入后不能据此声称自然遍历相同。延后的未消费任务在预算结束时落盘并记录，不隐式延长训练。

### 10.4 准入、dev评价、早停分开

- candidate准入：反思后minibatch平均reward相对反思前提高。dev分数不构成第二个准入门槛，不要求超过父模板、空模板或历史最佳才保留candidate。
- dev评价：准入后完成16任务×4次评价，用于Pareto和模板选择；低dev candidate仍在registry，可能不属于Pareto。
- 搜索结束：探索预算耗尽，或warmup结束后连续若干次探索没有刷新本轮最佳dev。minibatch未涨同样计入streak，不要求先进入dev或candidate池；这是用户最新明确的计数口径。

用户已确认：以本轮历史最佳原始dev均分为比较基准，且ready candidate数C>16才开始累计streak。完整更新次序、warmup和预算语义统一见0.4节。归一化分数仅用于训练模板抽样，不参与本轮早停。

### 10.5 按GEPA轮次归一化，再从全历史模板抽样

用户已明确归一化的组是**GEPA轮次**：例如3轮就有3组模板，每组用本轮所有模板的dev总分进行归一化，再合并历史模板抽样。不在单个任务维度上分别标准化，也不先均匀选择轮次。

采用以下概率转换规则（用户授权确定具体转概率方式）：

1. 令d[c,z]为第c轮模板z的原始dev分数，即16个任务各至多4次有效reward均值（全无效记0）再求平均。原始分数与16维向量始终保存不变。
2. 对本轮完成dev评价的所有非空模板计算mu[c]和总体标准差sigma[c]，`q[c,z]=(d[c,z]-mu[c])/sigma[c]`。只有一个模板或零方差时，本轮q全部置0；近零方差使用数值容差保护。
3. 本轮搜索完全收尾后冻结mu、sigma和q。空模板只作为搜索baseline，不参与非空训练模板组的归一化。记录统计所用模板集合，不能随某个任务的筛选结果重算归一化。
4. 汇集所有历史轮次的非空模板，先按task的已知求解模板/旧分配等规则确定可选集合E，再在E上计算 `p(z|task)=softmax(q[z]/T)`，T默认1、可调；实现采用减最大值的稳定softmax。
5. 若task有已知可解决模板，优先从对应历史模板子集中抽取；没有该证据则按已保存分配或全历史池回退。实际SETА完整求解证据必须来自全部测试通过，部分reward>0不自动等于已解决。证据携带task、policy、cycle和实例身份，不宣称旧policy成功能保证当前policy成功。
6. 同一模板hash若在多个cycle重复出现，保留全部原始评价及任务求解证据，抽样时只占一个模板条目；使用其最近一次完整评价所属cycle的q作为权重，避免重复存储人为增加概率。

这是直接在所有可选历史模板上抽样；模板更多的cycle可能拥有更大的合计概率，不额外强制各轮等概率。归一化消除轮间绝对评分尺度差异，但不等于重评了旧模板在当前policy上的效果。每条训练轨迹记录最终模板hash、分数来源cycle、可选集合类型和抽样RNG。

### 10.6 已确认的配额与空池边界

B=8且有5个以上历史难任务时，本批原8个任务均保留；按最近参与GEPA优先给最多4个加模板，同cycle随机打破并列，其余本次用原prompt，历史难标记不变。这是可用池满足配额条件时对自然命中的处理；当前池无法满足时遵循第0.7节的随机回退。

模板选择在rollout前完成，不能把已经带模板生成的轨迹在训练前改标为“未注入”来满足比例。提交和训练消费按第0.7节分别采用原采样/FIFO，再条件补/替换，池不足时不等待；训练消费只选择实际生成的完整组。当历史池为空时全部用原prompt，保留历史难任务身份，后续恢复正常模板抽样。

## 11. 可直接完善的工程契约与实施前验证

以下补齐实现约束，不需要用户重新选择已确认的算法。

### 11.1 触发、切换和尾批

- 触发集合只计本cycle自然取样完整有效组的去重任务。记录观察时policy/template版本；主动补入、GEPA评价、反思不增加触发数。自然再次命中的模板辅助难度不等于裸policy难度，指标名称明确区分。
- 第64个难任务到达后设置一次trigger_pending，停止新逻辑窗口，已提交任务收尾；将收尾结果归属本轮冻结快照H，固定dev16，其余为search-train。训练结束无后续更新时不额外启动GEPA。
- 结束正在运行的训练更新，等待权重同步、eval和异步保存。已完成可训练组尽量按整batch消费；不足整batch的组原样持久化为pending_train，GEPA后与新数据合批，保留旧template_version/behavior log-prob及staleness处理。不能在恢复时给既有轨迹重新套新模板。
- 补齐尾批前若当前训练预算已耗尽，记录未消费组；不通过新建隐式训练step超出预算。任何必要的部分batch支持都须沿用Miles后端约束，不能擅自改变batch归一化。
- `get()`可等待满组；trigger必须能让协调流程取得控制权，不能只停止producer后阻塞等尾批。checkpoint记录训练消费、数据预取、待提交普通任务和已完成待训练组，不能只保存dataset offset。

### 11.2 搜索状态与并发幂等性

- 本轮初始化从空模板bootstrap开始；上一轮registry保留供训练，不自动把旧分数带入本轮candidate或Pareto。以后若加warm-start，必须用新policy和本轮dev重评并单独计bootstrap成本。
- candidate总数C为累计已提交去重模板数，不随Pareto淘汰减少；否则C可能在16阈值上下反复切换。Pareto只是candidate的派生子集。
- 相同文本的proposal在候选提交时按hash去重，不增加candidate个数；已在途的重复dev评价可完成，其实际episode成本仍记录。每个已启动探索的终态独立记账，同一探索重复响应不得双重更新patience。语义相近但文本不同不凭主观判断合并。
- 保留seed列表和任务初始状态身份；核实Harbor→Miles请求链是否真正传递seed，不能仅在JSON中保存却未让模型消费。GPU调度不同仍不保证位级确定性。
- 预算账本区分started/completed/failed，所有实际episode另计成本；恢复复用已落盘episode并按探索ID幂等提交。正常异常退出沿用SETA取消及run回收；进程被强制终止、结果尚未落盘时，远端调用可能重做，不能宣称跨进程物理请求恰好一次。未回收请求仍受原生沙盒容量和超时约束，不重复消费逻辑探索预算。
- GEPA全部收尾并关闭沙盒后再恢复训练GPU布局；模板版本与registry更新原子提交。恢复读取版本化schema，检查policy、初始反思模型、tokenizer、数据集和模板hash。

### 11.3 历史任务再采样与饥饿

第0.7节已确认提交与消费双阶段条件配额。B=8时提交侧沿用原采样、消费侧先FIFO，再在可用池允许时调整到2–4个模板任务组；不可满足则采用本阶段初选结果，不分队列等待。B<2无法容纳一个模板组，P2O启动配置应明确拒绝。自然命中先抵扣主动补入，超过软目标但未到硬上限时不作跨批补偿。

自然任务被替换后按FIFO继续处理，不能重复推进游标。可用历史池不足或在途冲突使区间不可满足时直接随机回退，不做配额等待。当前“主动补入优先最近cycle”的策略可能长期不主动访问旧cycle，记录各cycle暴露次数/等待时间；自然旧任务命中进入模板选择，但受50%硬上限及第10.6节规则约束。若要主动补入对全历史均匀抽样，作为单独采样策略更改处理，不静默改变现有提案。

### 11.4 模板长度、反思预算与CRM数值口径

模板有显式token上限，反思有独立轮数、输入/输出token和工具超时预算。建议模板初值1024个policy tokens、反思最多20轮，均可配置；这些是工程起步值，不是模型性能保证。主任务现有长度预算继续作为约束，模板占用空间必须计入实际增强prompt，不能在移除模板时把被截掉的历史补回。若需要额外扩长增强上下文，应作为显式配置并先验证当前TP下GEPA推理内存。

双上下文打分保留同一动作token和真实观察；没有重新渲染另一份环境历史。验证SGLang行为log-prob与训练端temperature、sampling-mask及选中token口径一致。当前top-p=1的配置较简单，但不能只靠字段名判等。CRM的alpha/beta直接配置为0.01/10，log域比较；非模板数据恒为1，mask不改变原loss归一化分母。

反思只读权限验收覆盖命令参数写文件、重定向、符号链接、解释器、超长命令与多命令组合；正常读取/grep和唯一模板文件写入应可用。越权请求拒绝并记录，不得让模型借失败工具绕过权限。容器运行时/日志权限与agent权限分开。

### 11.5 首批验证顺序

先做两项小型可行性验证：真实多轮TITO轨迹的双上下文对齐；原Miles布局→GEPA反思/策略推理布局→原布局恢复且optimizer连续（包含多机）。随后验证受限Terminus反思能完成文件交付。三项通过后才组合完整搜索，避免完成大部分算法后才发现基础接口不支持。

纯CPU调度测试覆盖C=1/16/17、warmup不累计patience且保留best、第17个candidate首次计数、warmup后minibatch失败计数、dev低分计数、一次探索只计一次、warmup启动且之后完成的探索、warmup预算耗尽、前沿少于16、全相同dev向量、全候选不改善、重复proposal、预算最后一个token、patience触发后在途改善、沙盒限额小于探索并发、三自然命中超过25%、五自然命中受50%限制、全历史任务场景、模板池为空、归一化零方差及负分转概率、暂停恢复。真实GPU测试只验证这些接口及两轮小规模闭环，不将无GPU的模拟通过当作资源可行性结论。

## 当前决策状态与后续步骤

已确认按GEPA轮次用各模板dev总分做组内归一化、合并全历史模板转概率抽样，并保留task已有求解模板优先规则；超过50%的自然命中按最近GEPA优先注入，其余本次原prompt；历史模板池为空时原prompt回退。早停比较本轮历史最佳原始dev均分，只有C>16后累计，minibatch未涨与dev未刷新最佳都按探索终态计入32次，不影响candidate准入。探索上限256从启动即计，不要求进入dev；两个上限均可调。已获连续实施授权，各组先完成测试后推进；当前实际进度见实现 TODO。

Reward口径已确认：仅判难使用无bonus的passed/total，训练及GEPA全部评分保留full-pass bonus，详见0.1.1节。

最新修订：H可超过64、dev固定16；dev按有效重复均值且全无效为0；无效minibatch证据可供反思，实际rollout输入与评估脚本一并提供；W&B和SAVE_PATH/gepa双份记录，save_interval及GEPA前后均保存可恢复状态；正常训练GPU拓扑完全沿用Miles，反思固定一个实例、卡数/TP及引擎配置继承训练rollout；详见第7节。采样不为比例等待，提交/消费双阶段条件配额及池不足回退规则见0.7节（训练消费已改FIFO）。
