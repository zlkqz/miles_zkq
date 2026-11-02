# Harbor agent-server 的本地扩展

上游来源仍由 harbor-source.json 记录；其中 commit/tree 是补丁前的上游版本，不代表当前源码完全未修改。harbor-local.patch 记录这份快照上的本地差异，更新上游时应重新检查和生成补丁。

本方案直接启动上游 miles_agent_server.py，复用 /run、/flush、并发信号量、心跳、每轨迹子进程及 dashboard。没有另一套 /runs 状态轮询服务。

本地扩展限于以下内容，SETA 奖励/代理/镜像逻辑在 examples/seta，不写入通用服务：

- HARBOR_TRIAL_RUNNER：服务持有的可配置 runner 导入路径，由每轨迹子进程加载；不接受客户端指定代码路径。
- RunResponse.metadata：回传实验奖励有效性与远端轨迹位置。
- run_id 与 /flush 的 run_id 选择器：仅取消指定训练运行，避免共享服务上的全量取消。
- 可选 HARBOR_RUN_SECRET / HARBOR_ADMIN_SECRET 及其 _FILE 版本：运行与管理请求认证。默认未设置时保留原行为。
- HARBOR_DOCKER_PRUNE_ENABLED、HARBOR_REQUIRE_DOCKER_LOGIN：默认 true；SETA 在共享 Docker / 镜像站部署中明确设 false，不执行宿主机范围 prune，不强制 Docker Hub 登录。
- 子进程取消先 SIGTERM 让 Harbor 清理，超出 HARBOR_WORKER_CANCEL_TIMEOUT_SEC 才终止进程组；该超时默认 120 秒。
- /health 增加容量、活动数、路径和磁盘余量，供部署检查使用；保持 status=ok。
- 会话心跳的 HTTP 客户端使用 trust_env=false，直接访问 Miles 会话服务，不继承用于下载的外网代理。

这些扩展不改变默认 Harbor 任务标量奖励算法；启用 SETA runner 后才采用逐测试通过率与 bonus，以及无效轨迹丢弃语义。
