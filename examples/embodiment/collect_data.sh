#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/collect_real_data.py"

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

export HYDRA_FULL_ERROR=1


if [ -z "$1" ]; then
    CONFIG_NAME="realworld_collect_data"
else
    CONFIG_NAME=$1
fi

echo "Using Python at $(which python)"
# 多机 Ray：Env/DataCollector 在 worker 节点写盘；勿用仅 head 存在的路径（如 REPO_PATH=/data/...），
# 否则 worker 上 mkdir 会 PermissionError: '/data'。默认写到当前用户 HOME；共享盘可设 RLINF_COLLECT_LOG_DIR。
LOG_DIR="${RLINF_COLLECT_LOG_DIR:-${HOME}/RLinf/logs/$(date +'%Y%m%d-%H:%M:%S')}"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}