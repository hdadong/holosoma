# Holosoma

Holosoma (Greek: "whole-body") is a comprehensive humanoid robotics framework for training and deploying reinforcement learning policies on humanoid robots, as well as motion retargeting. Supports locomotion (velocity tracking) and whole-body tracking tasks across multiple simulators (IsaacGym, IsaacSim, MJWarp, MuJoCo) with algorithms like PPO and FastSAC.

## Features

- **Multi-simulator support**: IsaacGym, IsaacSim, MuJoCo Warp (MJWarp), and MuJoCo (inference only)
- **Multiple RL algorithms**: PPO and FastSAC
- **Robot support**: Unitree G1 and Booster T1 humanoids
- **Task types**: Locomotion (velocity tracking) and whole-body tracking
- **Sim-to-sim and sim-to-real deployment**: Shared inference pipeline across simulation and real robot control
- **Motion retargeting**: Convert human motion capture data to robot motions while preserving interactions with objects and terrain
- **Wandb integration**: Video logging, automatic ONNX checkpoint uploads, and direct checkpoint loading from Wandb

## Repository Structure

```
src/
├── holosoma/              # Core training framework (locomotion & whole-body tracking)
├── holosoma_inference/    # Inference and deployment pipeline
└── holosoma_retargeting/  # Motion retargeting from human motion data to robots
```

## Documentation

- **[Training Guide](src/holosoma/README.md)** - Train locomotion and whole-body tracking policies in IsaacGym/IsaacSim
- **[Inference & Deployment Guide](src/holosoma_inference/README.md)** - Deploy policies to real robots or evaluate in MuJoCo simulation
- **[Retargeting Guide](src/holosoma_retargeting/README.md)** - Convert human motion capture data to robot motions

## Quick Start

### Setup

Choose the appropriate setup script based on your use case:

```bash
# For IsaacGym training
bash scripts/setup_isaacgym.sh

# For IsaacSim training
# Requires Ubuntu 22.04 or later due to IsaacSim dependencies
bash scripts/setup_isaacsim.sh

# For MJWarp training and MuJoCo simulation (inference)
bash scripts/setup_mujoco.sh

# For inference/deployment
bash scripts/setup_inference.sh

# For motion retargeting
bash scripts/setup_retargeting.sh
```

### Docker Setup (recommended for hosts without Ubuntu 22.04+)

If the host does not satisfy IsaacSim's native requirements (e.g. Ubuntu 20.04 /
glibc 2.31), use Docker. The container base is
[`nvcr.io/nvidia/isaac-sim:5.1.0`](docker/Dockerfile) (Ubuntu 22.04, glibc 2.35), so
the host only needs an NVIDIA driver compatible with CUDA 12.8 and
`nvidia-container-toolkit`. GPU compute goes through the NVIDIA runtime with
essentially no overhead compared to a native install.

Prerequisites on the host:
- Docker with BuildKit (v20.10+)
- `nvidia-container-toolkit` (verify with `docker info | grep -i nvidia`)
- NVIDIA driver ≥ 525 (560+ recommended for Blackwell)

Two Dockerfiles are provided:

| Dockerfile | Installs | Use case |
|---|---|---|
| [`docker/Dockerfile`](docker/Dockerfile) | IsaacSim + IsaacGym + MJWarp + inference + retargeting | Full stack, all simulators |
| [`docker/Dockerfile.wbt`](docker/Dockerfile.wbt) | IsaacSim only | WBT training (e.g. box-carrying) — fastest build, smallest image |

#### Build the WBT (IsaacSim-only) image

```bash
# From the repository root.
# BuildKit is required (heredoc syntax).
DOCKER_BUILDKIT=1 docker build -t holosoma:wbt -f docker/Dockerfile.wbt .
```

Full image build (all environments, using `docker/build.sh`):

```bash
ECR_REPO=local bash docker/build.sh
```

#### Run WBT FastSAC training (box-carrying example)

The repo already ships the pre-retargeted motion file
`sub3_largebox_003_mj_w_obj.npz` in
[`src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/`](src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/),
so retargeting is **not** required for this task — training can start
immediately.

Run on specific GPUs (e.g. 4–7) with the repo and logs bind-mounted for
persistence:

```bash
# Create a host directory for logs/checkpoints so they survive container removal.
mkdir -p ~/holosoma_logs

docker run --rm -d \
    --name holosoma-wbt-boxcarry \
    --runtime=nvidia \
    --gpus '"device=4,5,6,7"' \
    --shm-size=16g \
    -e OMNI_KIT_ACCEPT_EULA=1 \
    -v "$(pwd)":/workspace/holosoma \
    -v ~/holosoma_logs:/workspace/holosoma/logs \
    -v ~/.netrc:/root/.netrc:ro \
    holosoma:wbt \
    bash -c '
      source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh &&
      conda activate hssim &&
      cd /workspace/holosoma &&
      python src/holosoma/holosoma/train_agent.py \
          exp:g1-29dof-wbt-fast-sac-w-object \
          logger:wandb-offline \
          --logger.video.enabled=True \
          --training.num-envs=2048 \
          2>&1 | tee logs/boxcarry_train.log
    '

# Follow training progress:
docker logs -f holosoma-wbt-boxcarry
# or
tail -f ~/holosoma_logs/boxcarry_train.log
```

Flag notes:
- `--gpus '"device=4,5,6,7"'` — pin the container to specific GPUs. Use
  `--gpus all` to expose every GPU, or `--gpus '"device=4"'` for a single GPU.
- `--shm-size=16g` — IsaacSim / data loaders put tensors in `/dev/shm`; the
  default 64 MB is too small.
- `--training.num-envs=2048` — the `g1_29dof_wbt_fast_sac_w_object` preset
  defaults to `num_envs=8192`, which OOMs on a single 24 GB GPU (RTX 4090).
  Drop it to `2048` for 24 GB cards; scale up proportionally for larger GPUs
  or when using multi-GPU training.
- `-v ~/.netrc:/root/.netrc:ro` — mount host Wandb credentials into the
  container. Omit this (and pass `logger:disabled` instead of
  `logger:wandb`) if you do not use Wandb.
- `logger:wandb-offline` — run Wandb in offline mode (avoid network calls to
  `api.wandb.ai` during training). Sync later with `wandb sync`. Swap to
  `logger:wandb` for online logging, or `logger:disabled` to turn Wandb off
  entirely (TensorBoard still runs).
- `--logger.video.enabled=True` — record training-rollout videos and upload
  them to Wandb. Set `False` if your host cannot do offscreen rendering.
  Video recording auto-enables `--enable_cameras`.

#### Train the pre100_app100 variant (1 kg box, no obj-mass/Ixx DR)

This preset uses the OMOMO box-carrying motion with 100 frames lerp-prepended
from the WBT default pose and 100 frames lerp-appended back to it, runs on a
fixed 1 kg `objects_largebox_1kg.urdf`, disables the box mass/Ixx DR, and turns
off the runtime default-pose prepend/append (the transitions are now baked
into the npz). Save_interval is tightened to every 100 steps so that
distillation/data-collection downstream has dense checkpoints.

```bash
mkdir -p ~/holosoma_logs

docker run --rm -d \
    --name holosoma-wbt-boxcarry-pre100app100 \
    --runtime=nvidia \
    --gpus '"device=2"' \
    --shm-size=16g \
    -e OMNI_KIT_ACCEPT_EULA=1 \
    -v "$(pwd)":/workspace/holosoma \
    -v ~/holosoma_logs:/workspace/holosoma/logs \
    -v ~/.netrc:/root/.netrc:ro \
    --entrypoint bash \
    holosoma:wbt -c '
      source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh &&
      conda activate hssim &&
      cd /workspace/holosoma &&
      python src/holosoma/holosoma/train_agent.py \
          exp:g1-29dof-wbt-fast-sac-w-object-1kg-pre100-app100 \
          logger:wandb \
          --logger.video.enabled=True \
          --training.num-envs=3072 \
          --algo.config.save-interval=100 \
          --algo.config.num-learning-iterations=400000 \
          2>&1 | tee logs/pre100app100_400k_train.log
    '
```

Notes specific to this preset:
- `--training.num-envs=3072` — 4096 envs OOMs PhysX on a 24 GB GPU (RTX 4090)
  partway through training; 3072 is the largest stable setting for this
  config. Fall back to 2048 if 3072 also OOMs.
- `--algo.config.save-interval=100` and `--algo.config.num-learning-iterations=400000`
  → 4000 light checkpoints, ~52 GB on disk. Drop the interval (e.g. 1000) if
  you don't need that density.
- `logger:wandb` (online) requires the host `~/.netrc` mount. Swap to
  `logger:wandb-offline` if the box has no internet.
- The reference motion `sub3_largebox_003_mj_w_obj_pre100_app100.npz` ships
  with the repo. To regenerate from the base clip:
  `python scripts/add_default_pose_transitions.py --prepend-seconds 2.0 --append-seconds 2.0`.

#### Evaluate a trained checkpoint and render mp4

```bash
RUN_DIR=WholeBodyTracking/20260518_143923-g1_29dof_wbt_fast_sac_manager-locomotion
CKPT=model_0249400.pt

docker run --rm -d \
    --name holosoma-wbt-boxcarry-eval \
    --runtime=nvidia \
    --gpus '"device=0"' \
    --shm-size=16g \
    -e OMNI_KIT_ACCEPT_EULA=1 \
    -v "$(pwd)":/workspace/holosoma \
    -v ~/holosoma_logs:/workspace/holosoma/logs \
    --entrypoint bash \
    holosoma:wbt -c "
      source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh &&
      conda activate hssim &&
      cd /workspace/holosoma &&
      mkdir -p logs/eval_videos &&
      python src/holosoma/holosoma/eval_agent.py \
          --checkpoint=/workspace/holosoma/logs/${RUN_DIR}/${CKPT} \
          --training.headless=True \
          --training.export-onnx=False \
          --training.max-eval-steps=1000 \
          --eval-overrides.headless=True \
          --eval-overrides.num-envs=1 \
          --eval-overrides.disable-logger=False \
          --logger.video.enabled=True \
          --logger.video.interval=1 \
          --logger.video.output-format=mp4 \
          --logger.video.upload-to-wandb=False \
          --logger.video.save-dir=/workspace/holosoma/logs/eval_videos \
          2>&1 | tee logs/eval_videos/eval.log
    "
```

`eval_overrides.num-envs=1` keeps the rendered scene clean (training rollouts
look "ghosted" because `env_spacing=0.0` stacks all envs on top of each other
visually). One mp4 is written per episode.

To **disable all domain randomization** (no pushes, no friction/CoM/joint-bias
noise) for a clean playback test, add `randomization:g1-29dof-wbt-empty`
right after `--checkpoint=...`:

```bash
        --checkpoint=/workspace/holosoma/logs/${RUN_DIR}/${CKPT} \
        randomization:g1-29dof-wbt-empty \
        --training.headless=True \
        ...
```

#### Multi-policy IsaacSim data collection (N ckpts in parallel)

[`scripts/collect_wbt_box_multipolicy.py`](scripts/collect_wbt_box_multipolicy.py)
runs **N different FastSAC checkpoints** in N parallel IsaacSim envs
(one ckpt per env) for downstream world-model pretraining. It forks
the single-policy [`scripts/collect_wbt_box_multienv.py`](scripts/collect_wbt_box_multienv.py)
collector and swaps `algo.actor` for a `BatchedActorEnsemble` —
N actors' weights are stacked along a leading dim and one `einsum`
per `Linear` layer drives all envs in a single forward pass.

Helpers added alongside (lightweight, holosoma-free except where
noted):
- [`scripts/_batched_actor_ensemble.py`](scripts/_batched_actor_ensemble.py)
  — stacks N `Actor` state_dicts + `EmpiricalNormalization` mean/std
  into one `nn.Module`. Layout assumes the WBT box-carry preset
  (3 × `(Linear → LayerNorm → SiLU)` blocks; `obs_keys=['actor_obs']`).
  Mean / log_std / deterministic action / stochastic action all match
  the reference `Actor.explore` to within `1e-5` absolute (see
  [`scripts/_test_batched_actor_ensemble.py`](scripts/_test_batched_actor_ensemble.py)).
- [`scripts/_multipolicy_utils.py`](scripts/_multipolicy_utils.py)
  — `select_ckpts(...)` deterministically partitions a sorted ckpt
  list into uniform train/test subsets (no RNG); `MemmapSaver` writes
  per-field `.npy` memmaps and flushes every chunk so a mid-run crash
  leaves the prefix on disk.

The collector is driven through LIFT3's docker launcher (lives in the
LIFT3 worktree at `docker/run_collect_wbt_box_multipolicy_in_docker.sh`);
a typical 2000-train + 200-test collection looks like:

```bash
SPLIT=both NUM_TRAIN=2000 NUM_TEST=200 NUM_STEPS=500 CHUNK_STEPS=50 \
  HOST_GPU=2 SEED=0 OUT_SUBDIR=prod_$(date +%Y%m%d_%H%M%S) \
  bash <LIFT3_WORKTREE>/docker/run_collect_wbt_box_multipolicy_in_docker.sh
```

Output structure (under `<LIFT3>/data/box_pretrain_eval_multipolicy/${OUT_SUBDIR}/`):

```
train/
  robot_root_pos_w.npy            # (num_steps, num_envs, 3) float32 memmap
  robot_root_quat_wxyz.npy        # wxyz convention (IsaacLab xyzw -> wxyz)
  robot_joint_pos.npy / _vel.npy  # (..., 29)
  robot_track_body_*.npy          # 14 tracked bodies, world + local
  box_*.npy                       # box pos/quat/lin_vel/ang_vel
  motion_*.npy / ref_*.npy        # full motion reference at the per-env phase
  action.npy / done.npy / terminated.npy / truncated.npy
  ckpt_step.npy + ckpt_paths.json # per-env ckpt manifest
  metadata.json                   # dof names, motion file, schema
test/   (same layout, num_envs=NUM_TEST)
```

`SPLIT=train|test|both` controls which split(s) to collect; `SEED`
seeds IsaacSim physics / DR / reset so train and test runs share the
exact same simulation context (only the policies and env count
differ). `--collector.save-actor-obs=False` /
`--collector.save-critic-obs=False` are the new defaults — those two
fields are policy-specific permutations that aren't meaningful when
each env runs a different policy, and downstream world-model training
reconstructs the WM-state directly from the raw saved fields anyway.

#### Interactive shell

```bash
docker run --rm -it \
    --runtime=nvidia --gpus all --shm-size=16g \
    -e OMNI_KIT_ACCEPT_EULA=1 \
    -v "$(pwd)":/workspace/holosoma \
    holosoma:wbt bash
# inside container:
#   conda activate hssim
#   python src/holosoma/holosoma/train_agent.py --help
```

### Training

Train a G1 robot with FastSAC on IsaacGym:

```bash
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    logger:wandb \
    --training.seed 1
```

> **Note:** For headless servers, see the [training guide](src/holosoma/README.md#video-recording) for video recording configuration.

See the [Training Guide](src/holosoma/README.md) for more examples and configuration options.

### Quick Demo

We provide scripts to run the complete pipeline: (data downloading and processing for LAFAN), retargeting, data conversion, and whole-body tracking policy training.

```bash
# Run retargeting and whole-body tracking policy training using OMOMO data
bash demo_scripts/demo_omomo_wb_tracking.sh

# Run retargeting and whole-body tracking policy training using LAFAN data
bash demo_scripts/demo_lafan_wb_tracking.sh
```

### Deployment & Evaluation

After training, deploy your policies:

- **Real Robot**: See [Real Robot Locomotion](src/holosoma_inference/docs/workflows/real-robot-locomotion.md) or [Real Robot WBT](src/holosoma_inference/docs/workflows/real-robot-wbt.md)
- **MuJoCo Simulation**: See [Sim-to-Sim Locomotion](src/holosoma_inference/docs/workflows/sim-to-sim-locomotion.md) or [Sim-to-Sim WBT](src/holosoma_inference/docs/workflows/sim-to-sim-wbt.md)

Or browse all deployment options in the [Inference & Deployment Guide](src/holosoma_inference/README.md).

### Demo Videos

Watch real-world deployments of Holosoma policies *(click thumbnails to play)*

<table>
  <tr>
    <th>G1 Locomotion</th>
    <th>T1 Locomotion</th>
    <th>G1 Dancing</th>
  </tr>
  <tr>
    <td width="33%">
      <a href="https://youtu.be/YYMgj5BDIMI">
        <img src="https://img.youtube.com/vi/YYMgj5BDIMI/hqdefault.jpg" width="100%" alt="▶ G1 Locomotion">
      </a>
    </td>
    <td width="33%">
      <a href="https://youtu.be/Q6rNHJZ2a6Y">
        <img src="https://img.youtube.com/vi/Q6rNHJZ2a6Y/hqdefault.jpg" width="100%" alt="▶ T1 Locomotion">
      </a>
    </td>
    <td width="33%">
      <a href="https://youtu.be/ouPk69_eFfE">
        <img src="https://img.youtube.com/vi/ouPk69_eFfE/hqdefault.jpg" width="100%" alt="▶ G1 Dancing">
      </a>
    </td>
  </tr>
</table>


## Issue Reporting

We welcome feedback and issue reports to help improve holosoma. Please use issues to:

- Report bugs and technical issues
- Request new features

## Support

If you need help with anything aside from issues feel free to join our [discord server](https://discord.gg/TPupMvpqHc).

Use the discord to discuss larger plans and other more involved problems.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## Citation

If you use Holosoma in your research, please cite it according to the "Cite this repository" panel on the right sidebar of the Github repo.

## License

This project is licensed under the Apache-2.0 License.
