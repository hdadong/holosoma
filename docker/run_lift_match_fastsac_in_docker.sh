#!/bin/bash
# Launch the LIFT3-matched holosoma FastSAC baseline (Part C) on a single GPU.
# Mirrors docker/run_fly_kick_train_in_docker.sh but uses the dedicated
# exp:g1-29dof-wbt-fast-sac-lift-match config:
#   * reward matched to the GPU0-5 brax SAC (no undesired_contacts, weak weights),
#   * NO domain randomization (g1_29dof_wbt_randomization_empty),
#   * adaptive motion-frame sampling, fight1 motion,
#   * FastSAC native (1 env-step/iter), num_envs=1000, save_interval=50,
#   * windowed Metrics/avg_total_reward + Metrics/avg_episode_length per 1000 steps.
set -euo pipefail

HOLOSOMA_DIR=/home/weidong/holosoma_wt_fastsac_match
HOST_GPU=${HOST_GPU:-5}
SEED=${SEED:-1}
IMAGE=${IMAGE:-holosoma:wbt}
GPU_MEM_FRACTION=${GPU_MEM_FRACTION:-0.55}
NUM_ENVS=${NUM_ENVS:-1000}
SAVE_INTERVAL=${SAVE_INTERVAL:-50}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_ENTITY=${WANDB_ENTITY:-bigeasthuang}
EXP_TAG=${EXP_TAG:-fight1_fastsac_lift_match}
CONTAINER_NAME=holosoma-fastsac-${EXP_TAG}

TS=$(date +%Y%m%d-%H%M%S)
LOGS_DIR=/home/weidong/holosoma_fastsac_match_logs/${TS}-${EXP_TAG}
mkdir -p "${LOGS_DIR}"
chmod -R 777 "${LOGS_DIR}" || true

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

echo "[fastsac] EXP_TAG=${EXP_TAG} GPU=${HOST_GPU} seed=${SEED} num_envs=${NUM_ENVS}"
echo "[fastsac] LOGS_DIR=${LOGS_DIR}"

docker run -d \
    --name "${CONTAINER_NAME}" \
    --runtime=nvidia \
    --gpus "\"device=${HOST_GPU}\"" \
    --shm-size=16g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -e OMNI_KIT_ACCEPT_EULA=1 \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e PYTHONUNBUFFERED=1 \
    -e HOLOSOMA_GPU_MEM_FRACTION="${GPU_MEM_FRACTION}" \
    -v "${HOLOSOMA_DIR}":/workspace/holosoma \
    -v "${LOGS_DIR}":/workspace/holosoma/logs \
    -v "${HOME}/.netrc":/root/.netrc:ro \
    --entrypoint bash \
    "${IMAGE}" \
    -c "source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
        conda activate hssim && \
        cd /workspace/holosoma && \
        python src/holosoma/holosoma/train_agent.py \
            exp:g1-29dof-wbt-fast-sac-lift-match \
            logger:wandb \
            --logger.mode=${WANDB_MODE} \
            --logger.entity=${WANDB_ENTITY} \
            --logger.video.enabled=False \
            --training.num-envs=${NUM_ENVS} \
            --training.seed=${SEED} \
            --training.headless=True \
            --algo.config.save-interval=${SAVE_INTERVAL} \
            2>&1 | tee /workspace/holosoma/logs/train.log"

echo "[fastsac] container ${CONTAINER_NAME} started"
echo "[fastsac] tail -f ${LOGS_DIR}/train.log"
echo "${LOGS_DIR}" > /home/weidong/holosoma_fastsac_match_logs/LATEST_${EXP_TAG}.txt
