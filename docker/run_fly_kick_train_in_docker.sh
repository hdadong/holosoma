#!/bin/bash
# Train a G1-29dof whole-body-tracking FastSAC policy from scratch on the
# fly-kick motion, inside the holosoma:wbt docker image. IsaacSim (data
# collection / simulation) and the PyTorch trainer run in the SAME process on a
# SINGLE GPU; the torch allocator is capped to GPU_MEM_FRACTION (default 0.55)
# via HOLOSOMA_GPU_MEM_FRACTION so IsaacSim's renderer + PhysX get the rest.
#
# Reward is the IsaacSim env reward, aligned with the world-model rollout reward
# (the ``undesired_contacts`` contact-force penalty is dropped because the WM
# cannot reproduce it). wandb logs per-episode average total reward
# (Train/mean_reward) and average episode length (Train/mean_episode_length)
# every 1000 env steps, and a third-person mp4 of env 0 is saved per episode.
#
# Usage (full run, GPU 0)::
#   HOST_GPU=0 bash docker/run_fly_kick_train_in_docker.sh
#
# Background it (recommended)::
#   HOST_GPU=0 nohup bash docker/run_fly_kick_train_in_docker.sh > fly_kick_train.log 2>&1 &
#
# Quick smoke test (small + short + offline wandb)::
#   HOST_GPU=0 NUM_ENVS=64 NUM_TIMESTEPS=2000 WANDB_MODE=offline \
#     CONTAINER_NAME=fly-kick-smoke bash docker/run_fly_kick_train_in_docker.sh
set -e

HOLOSOMA_DIR=${HOLOSOMA_DIR:-/home/weidong/holosoma}
HOLOSOMA_LOGS_DIR=${HOLOSOMA_LOGS_DIR:-/home/weidong/holosoma_fly_kick_logs}

HOST_GPU=${HOST_GPU:-0}
IMAGE=${IMAGE:-holosoma:wbt}
GPU_MEM_FRACTION=${GPU_MEM_FRACTION:-0.55}   # torch allocator cap; IsaacSim uses the rest
# The replay buffer is num_envs * buffer_size transitions and lives in the torch
# allocator, so num_envs * BUFFER_SIZE must fit under GPU_MEM_FRACTION. Defaults
# below (2048 * 512 ~= 1M transitions, ~4 GB) fit the 55% cap on a 24 GB GPU; the
# preset's 8192 * 1024 would need a full GPU. Raise both if you have more headroom.
NUM_ENVS=${NUM_ENVS:-2048}
BUFFER_SIZE=${BUFFER_SIZE:-512}              # replay buffer size PER ENV
SEED=${SEED:-1}
VIDEO_INTERVAL=${VIDEO_INTERVAL:-1}          # save a third-person mp4 every N episodes
WANDB_MODE=${WANDB_MODE:-online}             # online | offline
WANDB_ENTITY=${WANDB_ENTITY:-bigeasthuang}
NUM_TIMESTEPS=${NUM_TIMESTEPS:-}             # override algo.config.num_learning_iterations (blank = preset)
EXTRA_ARGS=${EXTRA_ARGS:-}
CONTAINER_NAME=${CONTAINER_NAME:-fly-kick-train-gpu${HOST_GPU}}
TRAIN_LOG=${TRAIN_LOG:-logs/fly_kick_train_seed${SEED}.log}

mkdir -p "${HOLOSOMA_LOGS_DIR}"

extra_iter_arg=""
if [ -n "${NUM_TIMESTEPS}" ]; then
    extra_iter_arg="--algo.config.num-learning-iterations=${NUM_TIMESTEPS}"
fi

echo "[fly-kick-train] HOST_GPU=${HOST_GPU}  IMAGE=${IMAGE}  CONTAINER=${CONTAINER_NAME}"
echo "[fly-kick-train] NUM_ENVS=${NUM_ENVS}  BUFFER_SIZE=${BUFFER_SIZE}  SEED=${SEED}  GPU_MEM_FRACTION=${GPU_MEM_FRACTION}"
echo "[fly-kick-train] WANDB_MODE=${WANDB_MODE}  WANDB_ENTITY=${WANDB_ENTITY}  VIDEO_INTERVAL=${VIDEO_INTERVAL}"
echo "[fly-kick-train] LOGS -> ${HOLOSOMA_LOGS_DIR} (container logs/)"
echo "[fly-kick-train] num_learning_iterations override: '${NUM_TIMESTEPS:-<preset default>}'"

# IsaacSim + torch share one GPU: --gpus device=${HOST_GPU} pins the whole
# container to that card; HOLOSOMA_GPU_MEM_FRACTION caps only the torch trainer.
docker run --rm \
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
    -v "${HOLOSOMA_LOGS_DIR}":/workspace/holosoma/logs \
    -v "${HOME}/.netrc":/root/.netrc:ro \
    --entrypoint bash \
    "${IMAGE}" \
    -c "source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
        conda activate hssim && \
        cd /workspace/holosoma && \
        python src/holosoma/holosoma/train_agent.py \
            exp:g1-29dof-wbt-fast-sac-fly-kick \
            logger:wandb \
            --logger.mode=${WANDB_MODE} \
            --logger.entity=${WANDB_ENTITY} \
            --logger.video.enabled=True \
            --logger.video.interval=${VIDEO_INTERVAL} \
            --training.num-envs=${NUM_ENVS} \
            --training.seed=${SEED} \
            --training.headless=True \
            --algo.config.buffer-size=${BUFFER_SIZE} \
            ${extra_iter_arg} \
            ${EXTRA_ARGS} \
            2>&1 | tee ${TRAIN_LOG}"
