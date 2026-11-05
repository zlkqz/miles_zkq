#!/usr/bin/env bash
# SETA DAPO 实验入口；共享配置与清理逻辑由 run_seta.sh 管理。
set -euo pipefail
export SETA_ALGORITHM=dapo
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_seta.sh" "$@"
