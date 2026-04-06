#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_realworld_offline_async.py"

if [ -z "$1" ]; then
  CONFIG_NAME="realworld_piper_resnet_5090"
else
  CONFIG_NAME=$1
fi

if [ -z "$2" ]; then
  echo "Usage: bash $0 <config_name> <replay_buffer_dir>"
  echo "  replay_buffer_dir: directory containing metadata.json and trajectory_index.json (e.g. logs/.../demos)"
  exit 1
fi

REPLAY_DIR=$2

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
export HYDRA_FULL_ERROR=1

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/offline/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
MEGA_LOG_FILE="${LOG_DIR}/run_offline_realworld.log"
mkdir -p "${LOG_DIR}"

# If Hydra says load_path is not in struct, either add load_path: null under algorithm.replay_buffer
# in that YAML, or use: +algorithm.replay_buffer.load_path=${REPLAY_DIR}
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} \
  runner.logger.log_path=${LOG_DIR} \
  algorithm.replay_buffer.load_path=${REPLAY_DIR} \
  algorithm.replay_buffer.auto_save=False \
  algorithm.demo_buffer=null"

echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}

