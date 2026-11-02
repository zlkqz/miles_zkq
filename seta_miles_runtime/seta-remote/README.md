# 远端沙盒部署配置

`config.json` 是部署输入，已经填好当前 GPU 节点和 144 沙盒的实际配置；不含密钥。`setup.py` 负责创建目录、复制代码和任务、安装独立 Harbor 依赖、生成 systemd/代理配置、同步 token 并生成 GPU endpoint JSON。

操作步骤及需要修改的字段统一见 [Quick Start 第 3 节](../../../docs/seta_quick_start.md)。

本目录是随仓库提交的部署入口，不是让远端服务读取 GPU 的共享盘。实际远端任务、虚拟环境、Docker 配置和 state 仍位于每个节点 `root` 指定的本地磁盘目录中。

`endpoints.json`、token、生成的 service.env/server.json 和运行日志不在这里手工维护，由部署工具生成；真实凭据、缓存和其他运行产物保持 Git 忽略。
