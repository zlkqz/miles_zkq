"""读取部署配置并 exec 官方 Harbor agent-server；此文件不定义 HTTP 接口。

保留稳定的 systemd 启动入口，更新服务实现不需要对共享 GPU 主机反复
执行 systemctl daemon-reload。实际服务进程是 Harbor/miles_agent_server.py。
"""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    harbor = Path(__file__).resolve().parents[2] / "seta_miles_runtime/harbor/miles_agent_server.py"
    command = [sys.executable, str(harbor), "--host", config["host"], "--port", str(config["port"]), "--max-concurrent", str(config["max_concurrent"]), "--trials-dir", str(Path(config["state_dir"]) / "trials"), "--dashboard-port", "0"]
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
