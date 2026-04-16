#!/usr/bin/env bash
# Piper 真机 Pi0.5（OpenPI）推理：单进程经 Ray 把 Env 调度到带机械臂的节点。
# 默认配置：realworld_piper_pi05_infer（内含与 realworld_piper_pi05_5090 一致的 cluster）。
#
# 用法：
#   bash examples/embodiment/infer_realworld_piper_pi05.sh
#   bash examples/embodiment/infer_realworld_piper_pi05.sh realworld_piper_pi05_infer actor.model.model_path=/your/sft

set -euo pipefail

export EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH="$(dirname "$(dirname "${EMBODIED_PATH}")")"
export SRC_FILE="${EMBODIED_PATH}/collect_real_data.py"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"

export HYDRA_FULL_ERROR=1

CONFIG_NAME="${1:-realworld_piper_pi05_infer}"
shift || true

echo "Using Python at $(command -v python)"
LOG_DIR="${RLINF_COLLECT_LOG_DIR:-${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')}-piper-infer"
MEGA_LOG_FILE="${LOG_DIR}/infer_piper_pi05.log"
mkdir -p "${LOG_DIR}"

CMD=(python "${SRC_FILE}" --config-path "${EMBODIED_PATH}/config/" --config-name "${CONFIG_NAME}"
  runner.logger.log_path="${LOG_DIR}" "$@")
printf "%s\n" "${CMD[*]}" >"${MEGA_LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"
