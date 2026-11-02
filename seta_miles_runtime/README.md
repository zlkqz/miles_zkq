# SETA 依赖与运行目录

本目录同时容纳可版本管理的依赖源码，以及不提交的本地运行内容。部署命令统一见 [Quick Start](../docs/seta_quick_start.md)。

| 随 Miles 提交 | 保持本地、由 Git 忽略 |
| --- | --- |
| `harbor/` 的上游源码、`pyproject.toml`、`uv.lock` 和许可证 | Harbor `.venv`、缓存和运行产物；GPU 训练不需要本地 Harbor 虚拟环境 |
| `harbor-source.json`：来源提交与许可证 | 本机模型转换与测试产物 |
| `harbor-local.patch`、`harbor-local-patches.md`：官方 agent-server 的本地通用扩展 | 测试与部署时产生的临时文件 |
| `seta-remote/config.json`、`setup.py`、README：完整远端部署入口 | 生成的 endpoint JSON、实际 token 与远端运行产物 |
| 本文件 | `models/`、`runs/`、`diagnostics/`、`remote/`、`secrets/` 等运行内容 |

Harbor 以源码快照交付，克隆 Miles 后即有源码，无需另行 clone 或初始化 submodule。快照已包含 `harbor-local.patch` 的改动，不要重复应用；来源 commit/tree 指补丁前的上游版本。远端沙盒节点复制这份源码后，在自己的本地磁盘安装独立 `.venv` 和 Python；不要复用 GPU 节点或共享盘的虚拟环境。GPU 侧训练不需要安装 Harbor。

GPU 的 `remote/endpoints.json` 引用 GPU 本地 token 文件；远端的 `server.json`、`service.env`、`docker_config/`、token、任务和 state 由远端服务自己管理，不通过 GPU 路径共享。`remote/`、真实配置和凭据均保持 Git 忽略。

**GPU 仓库内原有的 `docker_config/` 已废弃并清理。** 代理的部署输入是 `seta-remote/config.json` 中各节点的 proxy/no_proxy；部署后在对应远端的 `nodes[].root/docker_config/config.json` 生效。远端这份实际配置仍然需要，修改方式统一见 Quick Start；不能因为清理了 GPU 的旧目录就删除远端配置。

GPU 本地保留的运行内容：

- `models/`：转换后的 Megatron 权重。
- `remote/endpoints.json`：部署工具生成的服务地址与 GPU 本地 token 路径；`active_run.json` 在训练启动时生成。
- `secrets/remote-token`：远端请求认证；训练还会按 RUN_ID 生成 W&B NETRC。
- `runs/`：保留已有的训练轨迹数组和轨迹 JSONL；新训练按实验名继续生成日志等产物。
- `diagnostics/`：GPU 交接证据和必要验证记录。旧完整 Ray 日志已压缩为 `remote_migration_20261028/ray_logs.tar.gz`。

旧本地模式的代理目录、工具下载、镜像导出、任务检查目录、过期服务配置副本、日志和 GPU 本地 Harbor `.venv` 已清理；源码和锁文件保留。清理没有改动远端服务或远端自己的虚拟环境。

根 `.gitignore` 先忽略本目录的运行内容，再白名单放行 Harbor 源码、来源/补丁文件、部署配置/脚本和本 README。白名单内的源码仍适用缓存、虚拟环境等通用忽略规则；模型、实际 endpoint/token、训练输出和诊断不随 clone 交付。

更新上游时以 `harbor-source.json` 为来源记录，检查本地适配兼容性后更新源码和锁文件。不要把另一个 `.git` 目录放回 `harbor/`，否则父仓库可能将它作为嵌套仓库处理，普通 clone 无法带上源码。
