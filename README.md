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
          --logger.video.enabled=False \
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
- `--logger.video.enabled=False` — disable rendering on headless hosts.
  Remove this flag if a display / xvfb is available.

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
