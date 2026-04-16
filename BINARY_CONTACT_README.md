# Binary Contact Reward Training for Box Holding

This branch replaces object trajectory tracking rewards (box position/orientation) with a
**binary contact reward** for training the G1 robot to hold a box. The robot is rewarded
for matching reference contact patterns (which body parts should be in contact with the box)
rather than tracking the box's exact position and orientation.

## What Changed

- **Reward**: Removed `object_global_ref_position_error_exp` and `object_global_ref_orientation_error_exp`.
  Added `BinaryContactReward` (weight=3.0, force_threshold=10.0 N) that computes
  `matched_contacts / reference_contacts` using IsaacSim contact sensor forces.
- **Motion file**: Uses `sub3_largebox_003_mj_w_obj_contact.npz` which includes a
  `contact_mask` field (shape: 325x15, binary) annotating which of 15 upper-body contact
  points should be touching the box at each timestep.
- **Config**: New experiment `exp:g1-29dof-wbt-fast-sac-binary-contact`.

## Environment Setup

```bash
conda activate env_isaaclab
source scripts/source_isaacsim_setup.sh
```

## Training Command (Single GPU)

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt-fast-sac-binary-contact \
    simulator:isaacsim \
    logger:wandb \
    --training.seed 1 \
    --training.headless=True \
    --training.num-envs=4096 \
    --training.project=holosoma-isaacsim-wbt \
    --training.name=g1_wbt_fastsac_binary_contact_seed1 \
    --logger.mode=online \
    --logger.entity=bigeasthuang \
    --logger.project=holosoma-isaacsim-wbt \
    --logger.group=fastsac-wbt-binary-contact \
    --logger.name=g1-wbt-fastsac-binary-contact-seed1 \
    --logger.base-dir=/home/weidong/LIFT3/holosoma_runs/logs \
    --logger.video.enabled=False
```

## Training Command (Multi-GPU, e.g. 2x 4090)

```bash
CUDA_VISIBLE_DEVICES=<GPU0>,<GPU1> torchrun --nproc_per_node=2 \
    src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt-fast-sac-binary-contact \
    simulator:isaacsim \
    logger:wandb \
    --training.seed 1 \
    --training.headless=True \
    --training.num-envs=4096 \
    --training.project=holosoma-isaacsim-wbt \
    --training.name=g1_wbt_fastsac_binary_contact_seed1 \
    --logger.mode=online \
    --logger.entity=bigeasthuang \
    --logger.project=holosoma-isaacsim-wbt \
    --logger.group=fastsac-wbt-binary-contact \
    --logger.name=g1-wbt-fastsac-binary-contact-seed1 \
    --logger.base-dir=/home/weidong/LIFT3/holosoma_runs/logs \
    --logger.video.enabled=False
```

## Reference Motion File

The annotated motion file is at:
```
src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj_contact.npz
```

It was generated from the original `sub3_largebox_003_mj_w_obj.npz` using the
`annotate_contact.py` script in the LIFT3 repo. The contact mask covers 15 body
contact points; active contacts are on indices 11-14 (left_elbow, right_elbow,
left_hand, right_hand) for frames 85-324.

## Contact Points (15 upper body)

| Index | Label | Link Name |
|-------|-------|-----------|
| 0 | pelvis | pelvis |
| 1 | left_thigh | left_hip_roll_link |
| 2 | right_thigh | right_hip_roll_link |
| 3 | left_shin | left_knee_link |
| 4 | right_shin | right_knee_link |
| 5 | torso_collision1 | torso_link |
| 6 | torso_collision2 | torso_link |
| 7 | torso_collision3 | torso_link |
| 8 | head_collision | torso_link |
| 9 | left_shoulder | left_shoulder_yaw_link |
| 10 | right_shoulder | right_shoulder_yaw_link |
| 11 | left_elbow | left_elbow_link |
| 12 | right_elbow | right_elbow_link |
| 13 | left_hand | left_wrist_yaw_link |
| 14 | right_hand | right_wrist_yaw_link |
