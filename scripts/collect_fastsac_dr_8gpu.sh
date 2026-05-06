#!/usr/bin/env bash
# Collect FastSAC checkpoint rollouts in IsaacSim with domain randomization,
# 100 envs and up to 1000 control steps per checkpoint, distributed
# round-robin across GPUs 0-7. Each GPU gets its own docker container.
#
# Inputs (host paths)
#   CKPT_DIR  : 224 model_*.pt files for the FastSAC training run
#   LIFT3_DIR : the LIFT3 worktree on branch h4_exp27 that owns the
#               (modified) eval_pytorch_policy_in_isaacsim_box.py collector
#   LIFT_POLICY_PT : the .pt file with env_constants the collector needs
#   MOTION_FILE    : the box-tracking motion npz
#   OBJECT_URDF    : the 4kg box urdf used during training
#
# Output (host paths)
#   $OUTPUT_BASE/train/ckpt_<name>/transitions.npz  — per-checkpoint rollout
#   $OUTPUT_BASE/train/ckpt_<name>/collector.log    — stdout/stderr
#
# Usage:  bash scripts/collect_fastsac_dr_8gpu.sh
#         (no args; everything is configured below)

set -euo pipefail

# --- Configuration --------------------------------------------------------
HOLOSOMA_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIFT3_DIR="/home/weidong/LIFT3_wt_exp27"
CKPT_DIR="/home/weidong/LIFT3/holosoma_runs/logs/holosoma-isaacsim-wbt/20260325_152422-g1_wbt_fastsac_box_seed1-fastsac-wbt-box"
LIFT_POLICY_PT="/home/weidong/LIFT3_perstep/logs/G1BoxTrackingFlatTerrain-20260415-132023-box16cp_isaacsim/policies_torch/policy_pytorch_epoch_-1.pt"
MOTION_FILE="/home/weidong/LIFT3/annotated_contact/sub3_largebox_003_mj_w_obj_tracking.npz"
# Note: we deliberately do NOT pass --object_urdf to the collector. The
# eval config's default URDF (objects_largebox.urdf, 0.1 kg) lives inside
# the holosoma data package next to its referenced mesh
# (largebox.obj). Bind-mounting the URDF to a standalone path like
# /object.urdf breaks the IsaacSim URDF importer's mesh resolution and
# falls back to instanced prims, which makes the rigid-body material
# backend unable to randomize friction/mass/inertia. Letting the config
# default through resolve_data_file_path keeps full DR working.

OUTPUT_BASE="${HOLOSOMA_ROOT}/fastsac_dr_data"
DOCKER_IMAGE="holosoma:wbt"

NUM_ENVS=100
MAX_STEPS=1000
MAX_RETRIES=3
RETRY_DELAY=60
NUM_GPUS=8
STAGGER_SECONDS=30

# --- Sanity checks --------------------------------------------------------
for f in "$LIFT_POLICY_PT" "$MOTION_FILE"; do
    [ -f "$f" ] || { echo "Missing required file: $f" >&2; exit 1; }
done
[ -d "$CKPT_DIR" ] || { echo "Missing checkpoint dir: $CKPT_DIR" >&2; exit 1; }
[ -d "$LIFT3_DIR/scripts" ] || { echo "Missing LIFT3 scripts dir: $LIFT3_DIR/scripts" >&2; exit 1; }
docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1 \
    || { echo "Docker image not found: $DOCKER_IMAGE" >&2; exit 1; }

mkdir -p "${OUTPUT_BASE}/train"

# --- Discover checkpoints -------------------------------------------------
mapfile -t ALL_CKPTS < <(ls "${CKPT_DIR}"/model_*.pt | sort -t_ -k2 -n)
TOTAL=${#ALL_CKPTS[@]}
[ "$TOTAL" -gt 0 ] || { echo "No checkpoints found under $CKPT_DIR" >&2; exit 1; }
echo "=== Found ${TOTAL} checkpoints; distributing across ${NUM_GPUS} GPUs ==="
echo "=== Output: ${OUTPUT_BASE} ==="

# --- Distribute checkpoints (round-robin) --------------------------------
for g in $(seq 0 $((NUM_GPUS - 1))); do
    eval "GPU${g}_CKPTS=()"
done
for i in $(seq 0 $((TOTAL - 1))); do
    g=$((i % NUM_GPUS))
    eval "GPU${g}_CKPTS+=(\"${ALL_CKPTS[$i]}\")"
done
for g in $(seq 0 $((NUM_GPUS - 1))); do
    eval "n=\${#GPU${g}_CKPTS[@]}"
    echo "GPU${g}: ${n} ckpts"
done

# --- Single-checkpoint runner --------------------------------------------
collect_one() {
    local gpu="$1"
    local ckpt_path="$2"

    local ckpt_name
    ckpt_name=$(basename "$ckpt_path" .pt)
    local out_dir="${OUTPUT_BASE}/train/ckpt_${ckpt_name}"

    if [ -f "${out_dir}/transitions.npz" ]; then
        echo "[GPU${gpu}] SKIP ${ckpt_name} (already done)"
        return 0
    fi
    mkdir -p "$out_dir"

    local container_name="holosoma-collect-gpu${gpu}-${ckpt_name}"
    # Per-container portable root so concurrent containers don't fight over
    # /root/.cache/isaacsim_portable.
    local portable_root="/tmp/isaacsim_portable_gpu${gpu}"

    local attempt=0
    while [ "$attempt" -lt "$MAX_RETRIES" ]; do
        attempt=$((attempt + 1))
        echo "[GPU${gpu}] ${ckpt_name} attempt ${attempt}/${MAX_RETRIES}"

        # Each invocation gets its own container so a crash on one ckpt does
        # not bleed into the next. --rm cleans up on exit.
        local exit_code=0
        docker run --rm \
            --name "$container_name" \
            --runtime=nvidia \
            --gpus "\"device=${gpu}\"" \
            --shm-size=16g \
            -e OMNI_KIT_ACCEPT_EULA=1 \
            -e ISAACSIM_PORTABLE_ROOT="$portable_root" \
            -v "${HOLOSOMA_ROOT}:/workspace/holosoma" \
            -v "${LIFT3_DIR}:/lift3" \
            -v "${CKPT_DIR}:/checkpoints:ro" \
            -v "${LIFT_POLICY_PT}:/lift_policy.pt:ro" \
            -v "${MOTION_FILE}:/motion.npz:ro" \
            "$DOCKER_IMAGE" \
            bash -c "
                set -e
                source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh
                conda activate hssim
                cd /lift3
                python /lift3/scripts/eval_pytorch_policy_in_isaacsim_box.py \
                    --policy_path /lift_policy.pt \
                    --fastsac_checkpoint /checkpoints/${ckpt_name}.pt \
                    --motion_file /motion.npz \
                    --num_envs ${NUM_ENVS} \
                    --max_steps ${MAX_STEPS} \
                    --headless \
                    --stochastic \
                    --randomize_initial_pose \
                    --save_transition_npz \
                    --transition_npz_path /workspace/holosoma/fastsac_dr_data/train/ckpt_${ckpt_name}/transitions.npz
            " > "${out_dir}/collector.log" 2>&1 || exit_code=$?

        if [ "$exit_code" -eq 0 ] && [ -f "${out_dir}/transitions.npz" ]; then
            echo "[GPU${gpu}] OK  ${ckpt_name}"
            return 0
        fi

        echo "[GPU${gpu}] FAIL ${ckpt_name} (exit=${exit_code}); waiting ${RETRY_DELAY}s before retry"
        rm -f "${out_dir}/transitions.npz"
        sleep "$RETRY_DELAY"
    done

    echo "[GPU${gpu}] GAVE UP on ${ckpt_name} after ${MAX_RETRIES} attempts"
    return 1
}

# --- Per-GPU worker -------------------------------------------------------
gpu_worker() {
    local gpu="$1"
    shift
    local ckpts=("$@")
    local n=${#ckpts[@]}
    local ok=0 fail=0
    echo "[GPU${gpu}] starting (${n} checkpoints)"
    for ckpt in "${ckpts[@]}"; do
        if collect_one "$gpu" "$ckpt"; then
            ok=$((ok + 1))
        else
            fail=$((fail + 1))
        fi
    done
    echo "[GPU${gpu}] done: ${ok}/${n} ok, ${fail} failed"
}

# --- Launch workers, staggered to ease IsaacSim cold-start contention ----
PIDS=()
for g in $(seq 0 $((NUM_GPUS - 1))); do
    eval "ckpts=(\"\${GPU${g}_CKPTS[@]}\")"
    gpu_worker "$g" "${ckpts[@]}" &
    PIDS+=($!)
    if [ "$g" -lt $((NUM_GPUS - 1)) ]; then
        sleep "$STAGGER_SECONDS"
    fi
done

echo "=== Worker PIDs: ${PIDS[*]} ==="
wait "${PIDS[@]}"
echo "=== All workers finished ==="

DONE=$(find "${OUTPUT_BASE}/train" -name "transitions.npz" | wc -l)
echo ""
echo "=== Summary ==="
echo "  Completed: ${DONE}/${TOTAL} checkpoints"
echo "  Output:    ${OUTPUT_BASE}/train"
