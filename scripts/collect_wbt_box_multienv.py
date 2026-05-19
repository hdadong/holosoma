"""Multi-env data collector for the holosoma WBT box-carry policy.

This is the multi-env counterpart of holosoma's single-env ``eval_agent.py``
loop. It is intended to be invoked **exactly like** ``eval_agent.py`` (same
``--checkpoint`` flag, same tyro override syntax) but instead of running
``algo.evaluate_policy()`` it rolls out ``--training.num-envs=N`` parallel
envs for ``--training.max-eval-steps=T`` steps and dumps a single ``.npz``
file with every quantity needed downstream (robot state, box state, raw
policy action, env signals, motion-reference data).

Recommended invocation (from the README's preset; matches the run that
produced ``20260518_143923-g1_29dof_wbt_fast_sac_manager-locomotion``)::

    python scripts/collect_wbt_box_multienv.py \\
        exp:g1-29dof-wbt-fast-sac-w-object-1kg-pre100-app100 \\
        --checkpoint=/workspace/holosoma/logs/<RUN_DIR>/<CKPT>.pt \\
        --training.num-envs=3072 \\
        --training.headless=True \\
        --training.max-eval-steps=400 \\
        --rollout-output=/path/to/rollout.npz

Coordinate-frame conventions saved in the npz (so downstream consumers know
what they're getting):

* ``robot_root_pos_w`` is ``env.simulator.robot_root_states[:, :3]``
  (= IsaacLab ``Articulation.data.root_state_w[:, :3]``), expressed in
  the **simulation world** (not per-env). The companion
  ``robot_root_pos_local`` subtracts ``scene.env_origins`` so it's the
  pelvis position relative to each env's origin. This is the **pelvis
  link reference-frame origin** (the URDF link anchor), NOT the COM and
  NOT a geometric center. (For the box this distinction collapses: the
  box's inertial/visual/collision origins are all ``0 0 0`` in the URDF,
  so link origin == COM == geometric center.)
* Linear velocity is read from ``root_link_state_w`` (link velocity) to
  match the brax / MuJoCo convention. Reading it from
  ``robot_root_states[:, 7:10]`` would give the **COM** velocity which
  differs when the pelvis COM is offset from the link origin.
* All quaternions in the saved file are converted to **wxyz** for
  consistency with brax. IsaacLab natively stores them as xyzw.
* ``box_pos_w``, ``box_quat_wxyz``, etc. are read from the simulator,
  not the motion file -- so they reflect what the policy actually
  produced, not the reference.

Each row of the saved arrays carries shape ``(num_steps, num_envs, ...)``.
"""
from __future__ import annotations

import functools
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from loguru import logger
from pydantic.dataclasses import dataclass as pyd_dataclass

# Force line-buffered stdout/stderr so progress prints reach the docker log
# even when tee/pipe block-buffers Python's stdout.
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover
    pass
print = functools.partial(print, flush=True)  # noqa: A001

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import (
    close_simulation_app,
    setup_simulation_environment,
)
from holosoma.utils.tyro_utils import TYRO_CONIFG


# ---------------------------------------------------------------------------
# CLI: extra config on top of eval_agent's checkpoint flag.
# ---------------------------------------------------------------------------


@pyd_dataclass(frozen=True)
class CollectorConfig:
    """Flat first-pass CLI: ``--checkpoint`` is kept top-level (same as
    holosoma's ``eval_agent.py``) plus a few collector-specific flags."""

    checkpoint: str | None = None
    """Path to a local checkpoint file, or W&B URI."""

    rollout_output: str = "rollout.npz"
    """Destination .npz path for the collected rollout."""

    stochastic: bool = True
    """If True, sample actions from the actor's tanh-Gaussian (matches training).
    If False, use the deterministic tanh(mean) action."""

    save_actor_obs: bool = True
    """Save the full per-step actor observation tensor (handy for retraining)."""

    save_critic_obs: bool = True
    """Save the full per-step critic observation tensor."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_bool(s: str) -> bool:
    """argparse-friendly bool parser accepting 'True'/'False'/'1'/'0'/etc."""
    if isinstance(s, bool):
        return s
    s_lower = str(s).strip().lower()
    if s_lower in ("true", "1", "yes", "y", "t"):
        return True
    if s_lower in ("false", "0", "no", "n", "f"):
        return False
    raise ValueError(f"Cannot parse bool from {s!r}")


def _xyzw_to_wxyz(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Convert (..., 4) quat from xyzw -> wxyz."""
    return torch.stack(
        [quat_xyzw[..., 3], quat_xyzw[..., 0], quat_xyzw[..., 1], quat_xyzw[..., 2]],
        dim=-1,
    )


def _quat_normalize_wxyz(q: torch.Tensor) -> torch.Tensor:
    return q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-8)


def _quat_conjugate_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)


def _quat_mul_wxyz(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = lhs.unbind(-1)
    rw, rx, ry, rz = rhs.unbind(-1)
    return torch.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dim=-1,
    )


def _quat_rotate_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q (both wxyz)."""
    qw = q[..., 0:1]
    qv = q[..., 1:4]
    uv = torch.cross(qv, v, dim=-1)
    uuv = torch.cross(qv, uv, dim=-1)
    return v + 2.0 * (qw * uv + uuv)


def _quat_to_rotmat_wxyz(q: torch.Tensor) -> torch.Tensor:
    """(... ,4) wxyz -> (..., 3, 3) rotation matrix."""
    qw, qx, qy, qz = q.unbind(-1)
    twx, twy, twz = 2 * qw * qx, 2 * qw * qy, 2 * qw * qz
    txx, txy, txz = 2 * qx * qx, 2 * qx * qy, 2 * qx * qz
    tyy, tyz, tzz = 2 * qy * qy, 2 * qy * qz, 2 * qz * qz
    col0 = torch.stack([1.0 - (tyy + tzz), txy + twz, txz - twy], dim=-1)
    col1 = torch.stack([txy - twz, 1.0 - (txx + tzz), tyz + twx], dim=-1)
    col2 = torch.stack([txz + twy, tyz - twx, 1.0 - (txx + tyy)], dim=-1)
    return torch.stack([col0, col1, col2], dim=-1)


def _subtract_frame_transforms_wxyz(
    p1_pos: torch.Tensor, p1_quat: torch.Tensor,
    p2_pos: torch.Tensor, p2_quat: torch.Tensor,
):
    """Express frame 2 in frame 1. Returns (rel_pos, rel_quat). All wxyz."""
    q1_conj = _quat_conjugate_wxyz(p1_quat)
    rel_pos = _quat_rotate_wxyz(q1_conj, p2_pos - p1_pos)
    rel_quat = _quat_normalize_wxyz(_quat_mul_wxyz(q1_conj, p2_quat))
    return rel_pos, rel_quat


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Per-step transition capture.
# ---------------------------------------------------------------------------


def _capture_transition(env, motion_command, actor_obs, critic_obs):
    """Snapshot every per-env quantity we want to save for the current step."""
    sim = env.simulator
    env_origins = sim.scene.env_origins  # (num_envs, 3)
    num_envs = env_origins.shape[0]

    # ---- Robot raw ----
    # robot_root_states is IsaacLab's root_state_w (link pos, xyzw quat,
    # COM vel, COM ang vel). Use root_link_state_w for the link velocity
    # (matches brax convention).
    robot_root_states = sim.robot_root_states[:]
    root_link_state_w = sim._robot.data.root_link_state_w  # (N, 13)

    robot_root_pos_w = robot_root_states[:, 0:3]              # link frame origin (world)
    robot_root_pos_local = robot_root_pos_w - env_origins
    robot_root_quat_wxyz = _xyzw_to_wxyz(robot_root_states[:, 3:7])
    robot_root_lin_vel_w = root_link_state_w[:, 7:10]
    robot_root_ang_vel_w = root_link_state_w[:, 10:13]
    robot_joint_pos = sim.dof_pos[:]
    robot_joint_vel = sim.dof_vel[:]

    # ---- Robot body (14 tracked bodies) in world ----
    track_pos_w = motion_command.robot_body_pos_w  # (N, 14, 3) link frame world
    track_quat_wxyz = _xyzw_to_wxyz(motion_command.robot_body_quat_w)
    track_lin_vel_w = motion_command.robot_body_lin_vel_w
    track_ang_vel_w = motion_command.robot_body_ang_vel_w
    track_pos_local = track_pos_w - env_origins[:, None, :]

    # ---- Robot body in anchor frame (= robot_body_pos_b / ori_b) ----
    # Anchor = the simulator's anchor body (robot_ref_*_w).
    anchor_pos_w = motion_command.robot_ref_pos_w           # (N, 3)
    anchor_quat_wxyz = _xyzw_to_wxyz(motion_command.robot_ref_quat_w)  # (N, 4)
    anc_pos_b = anchor_pos_w[:, None, :].expand(-1, track_pos_w.shape[1], -1)
    anc_quat_b = anchor_quat_wxyz[:, None, :].expand(-1, track_quat_wxyz.shape[1], -1)
    rb_pos_b, rb_quat_b = _subtract_frame_transforms_wxyz(
        anc_pos_b, anc_quat_b, track_pos_w, track_quat_wxyz
    )
    robot_body_ori_6d = _quat_to_rotmat_wxyz(rb_quat_b)[..., :2].reshape(num_envs, -1)
    robot_body_pos_b = rb_pos_b.reshape(num_envs, -1)

    # ---- Box raw (simulator) ----
    obj_idx = motion_command.object_indices_in_simulator
    obj_states = sim.all_root_states[obj_idx]      # (N, 13) IsaacLab format
    box_pos_w = motion_command.simulator_object_pos_w  # (N, 3)
    box_pos_local = box_pos_w - env_origins
    box_quat_wxyz = _xyzw_to_wxyz(motion_command.simulator_object_quat_w)
    box_lin_vel_w = motion_command.simulator_object_lin_vel_w
    box_ang_vel_w = obj_states[:, 10:13]

    # ---- Reference (from motion file at current time_steps) ----
    ref_step = motion_command.time_steps  # (N,) long
    motion_cmd = motion_command.command   # (N, 58) = [joint_pos, joint_vel]
    ref_anchor_pos_w = motion_command.ref_pos_w   # (N, 3) world (includes env_origin)
    ref_anchor_quat_wxyz = _xyzw_to_wxyz(motion_command.ref_quat_w)
    ref_body_pos_w = motion_command.body_pos_w      # (N, 14, 3) world (inc env_origin)
    ref_body_quat_wxyz = _xyzw_to_wxyz(motion_command.body_quat_w)
    ref_body_lin_vel_w = motion_command.body_lin_vel_w
    ref_body_ang_vel_w = motion_command.body_ang_vel_w
    # Object reference (full ref info: pos, quat, lin_vel; ang_vel not provided
    # by motion).
    ref_obj_pos_w = motion_command.object_pos_w
    ref_obj_quat_wxyz = _xyzw_to_wxyz(motion_command.object_quat_w)
    ref_obj_lin_vel_w = motion_command.object_lin_vel_w

    # motion_ref_pos_b / motion_ref_ori_b: ref anchor expressed in current
    # sim anchor frame (matches what the obs term uses).
    motion_ref_pos_b, motion_ref_quat_b = _subtract_frame_transforms_wxyz(
        anchor_pos_w, anchor_quat_wxyz,
        ref_anchor_pos_w, ref_anchor_quat_wxyz,
    )
    motion_ref_ori_6d = _quat_to_rotmat_wxyz(motion_ref_quat_b)[..., :2].reshape(num_envs, -1)

    out = {
        # robot raw
        "robot_root_pos_w": robot_root_pos_w,
        "robot_root_pos_local": robot_root_pos_local,
        "robot_root_quat_wxyz": robot_root_quat_wxyz,
        "robot_root_lin_vel_w": robot_root_lin_vel_w,
        "robot_root_ang_vel_w": robot_root_ang_vel_w,
        "robot_joint_pos": robot_joint_pos,
        "robot_joint_vel": robot_joint_vel,
        "robot_track_body_pos_w": track_pos_w,
        "robot_track_body_pos_local": track_pos_local,
        "robot_track_body_quat_wxyz": track_quat_wxyz,
        "robot_track_body_lin_vel_w": track_lin_vel_w,
        "robot_track_body_ang_vel_w": track_ang_vel_w,
        # box raw
        "box_pos_w": box_pos_w,
        "box_pos_local": box_pos_local,
        "box_quat_wxyz": box_quat_wxyz,
        "box_lin_vel_w": box_lin_vel_w,
        "box_ang_vel_w": box_ang_vel_w,
        # reference
        "ref_step": ref_step,
        "motion_command": motion_cmd,
        "motion_ref_pos_b": motion_ref_pos_b,
        "motion_ref_ori_b": motion_ref_ori_6d,
        "motion_ref_anchor_pos_w": ref_anchor_pos_w,
        "motion_ref_anchor_quat_wxyz": ref_anchor_quat_wxyz,
        "ref_body_pos_w": ref_body_pos_w,
        "ref_body_quat_wxyz": ref_body_quat_wxyz,
        "ref_body_lin_vel_w": ref_body_lin_vel_w,
        "ref_body_ang_vel_w": ref_body_ang_vel_w,
        # 14-body anchor-frame (current sim)
        "robot_body_pos_b": robot_body_pos_b,
        "robot_body_ori_b": robot_body_ori_6d,
        # box reference (full info)
        "ref_box_pos_w": ref_obj_pos_w,
        "ref_box_quat_wxyz": ref_obj_quat_wxyz,
        "ref_box_lin_vel_w": ref_obj_lin_vel_w,
        # obs vectors (optional, written based on collector cfg)
        "actor_obs": actor_obs,
        "critic_obs": critic_obs,
    }
    return out


# ---------------------------------------------------------------------------
# Multi-env collect loop (replaces algo.evaluate_policy).
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_multienv_collect(algo, num_steps: int, output_path: Path,
                         stochastic: bool, save_actor_obs: bool,
                         save_critic_obs: bool) -> None:
    # ``algo.env`` is the FastSACEnv wrapper; ``algo.unwrapped_env`` is the
    # raw holosoma BaseTask that owns the simulator / motion_command.
    fastsac_env = algo.env
    base_env = algo.unwrapped_env
    sim = base_env.simulator
    motion_command = base_env.command_manager.get_state("motion_command")

    # We deliberately do NOT call ``base_env.set_is_evaluating()`` -- that
    # would zero out the per-env phase so all 3072 envs reset to motion
    # frame 0 with identical states. Training-style reset (random phases,
    # init-pose noise, all DR except the disabled box mass/Ixx) gives the
    # diversity we want for downstream data.
    print("[collect] resetting envs (training-style: random phase + noise) ...")
    actor_obs = fastsac_env.reset()
    num_envs = actor_obs.shape[0]
    print(f"[collect] num_envs={num_envs} actor_obs_dim={actor_obs.shape[1]}")

    # Prepare per-step storage on host (numpy). We pull each tensor to CPU
    # immediately each step to keep GPU memory bounded.
    storage: dict[str, list[np.ndarray]] = {}

    # joint names + body names for metadata
    dof_names = list(sim.dof_names)
    track_body_names = [
        sim._body_list[i] for i in motion_command.tracked_body_indexes.tolist()
    ]
    anchor_body_name = sim._body_list[int(motion_command.ref_body_index)]

    print(f"[collect] dof_names={dof_names}")
    print(f"[collect] track_body_names={track_body_names}")
    print(f"[collect] anchor_body_name={anchor_body_name}")

    # Critic obs computation (matches FastSACEnv.step)
    def _compute_critic_obs():
        d = base_env.obs_buf_dict
        return torch.cat([d[k] for k in fastsac_env._critic_obs_keys], dim=1)

    critic_obs = _compute_critic_obs()
    print(f"[collect] critic_obs_dim={critic_obs.shape[1]}")
    print(f"[collect] obs_normalization={algo.obs_normalization}")

    # ---------- Main loop ----------
    for step in range(num_steps):
        if algo.obs_normalization:
            normalized_actor_obs = algo.obs_normalizer(actor_obs, update=False)
        else:
            normalized_actor_obs = actor_obs

        action_raw = algo.actor.explore(
            normalized_actor_obs, deterministic=not stochastic
        )

        # Snapshot of pre-step state (current obs/action correspond to this).
        snap = _capture_transition(base_env, motion_command, actor_obs, critic_obs)
        snap["action"] = action_raw

        # Step env.
        next_actor_obs, rew, reset_buf, extras = fastsac_env.step(action_raw)
        time_outs = extras["time_outs"].to(torch.bool)
        reset_bool = reset_buf.to(torch.bool)

        # Re-classify holosoma's ``motion_ends`` termination as ``truncated``.
        # The g1 wbt termination cfg marks ``motion_ends`` with the default
        # ``is_timeout=False`` (so the manager lumps it into reset_flags), but
        # semantically it's a truncation: the motion ran out, the policy did
        # not fail. We detect it by checking whether the pre-step ref_step
        # was within 1 frame of ``time_step_total - 2`` (the threshold the
        # ``motion_ends`` term uses) -- after env.step bumps time_steps by 1,
        # such an env satisfies the threshold and gets reset.
        motion_end_threshold = motion_command.motion.time_step_total - 3
        motion_end_triggered = reset_bool & (
            snap["ref_step"] >= motion_end_threshold
        )
        truncated = time_outs | motion_end_triggered
        terminated = reset_bool & (~truncated)
        done = reset_bool

        snap["reward"] = rew
        snap["done"] = done.to(torch.float32)
        snap["terminated"] = terminated.to(torch.float32)
        snap["truncated"] = truncated.to(torch.float32)
        # Separate breakdown bits so downstream can tell the two truncation
        # sources apart and recover the original holosoma classification.
        snap["motion_ended"] = motion_end_triggered.to(torch.float32)
        snap["episode_length_timeout"] = time_outs.to(torch.float32)

        # Persist to host.
        for k, v in snap.items():
            if (k == "actor_obs" and not save_actor_obs) or (
                k == "critic_obs" and not save_critic_obs
            ):
                continue
            storage.setdefault(k, []).append(_np(v))

        actor_obs = next_actor_obs
        critic_obs = _compute_critic_obs()

        if step < 5 or step % 50 == 0 or step == num_steps - 1:
            print(
                f"[collect] step {step + 1:4d}/{num_steps}  "
                f"reward[mean]={float(rew.mean()):.4f} "
                f"done={int(done.sum().item())}/{num_envs} "
                f"term={int(terminated.sum().item())}/{num_envs} "
                f"trunc={int(truncated.sum().item())}/{num_envs}"
            )

    # ---------- Save .npz ----------
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stacked = {k: np.stack(v, axis=0) for k, v in storage.items()}
    stacked.update({
        "num_envs": np.asarray(num_envs, dtype=np.int32),
        "num_steps": np.asarray(num_steps, dtype=np.int32),
        "ctrl_dt": np.asarray(float(base_env.dt), dtype=np.float32),
        "fps": np.asarray(float(1.0 / base_env.dt), dtype=np.float32),
        "dof_names": np.asarray(dof_names),
        "track_body_names": np.asarray(track_body_names),
        "anchor_body_name": np.asarray(anchor_body_name),
        "motion_file": np.asarray(motion_command.motion_cfg.motion_file),
        "motion_time_step_total": np.asarray(
            int(motion_command.motion.time_step_total), dtype=np.int32
        ),
        "stochastic": np.asarray(bool(stochastic)),
    })

    print(f"[collect] writing {output_path} ...")
    tmp_path = str(output_path) + f".tmp.{os.getpid()}"
    np.savez(tmp_path, **stacked)
    if not os.path.exists(tmp_path) and os.path.exists(tmp_path + ".npz"):
        tmp_path = tmp_path + ".npz"
    os.replace(tmp_path, output_path)
    print(f"[collect] saved: {output_path} "
          f"({output_path.stat().st_size / (1024 ** 2):.1f} MB)")

    # Drop a sidecar with per-field shapes so downstream tools can sanity-check.
    import json
    report = {
        k: list(v.shape) for k, v in stacked.items()
        if isinstance(v, np.ndarray) and v.ndim > 0
    }
    shapes_path = output_path.with_suffix(".shapes.json")
    with open(shapes_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[collect] shape report: {shapes_path}")


# ---------------------------------------------------------------------------
# Tyro CLI wrapper (mirrors eval_agent.main).
# ---------------------------------------------------------------------------


def main() -> None:
    init_eval_logging()

    # First pass: flat CLI (so --checkpoint=... works directly, matching
    # eval_agent.py). Unknown args pass through to the second tyro pass.
    collector_cfg, remaining = tyro.cli(
        CollectorConfig, return_unknown_args=True, add_help=False
    )
    if collector_cfg.checkpoint is None:
        raise SystemExit(
            "No --checkpoint provided. Pass either a local .pt path or a "
            "wandb:// URI."
        )
    checkpoint_cfg = CheckpointConfig(checkpoint=collector_cfg.checkpoint)

    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    eval_cfg = saved_cfg.get_eval_config()

    # NOTE: We deliberately do NOT re-invoke ``tyro.cli(ExperimentConfig, ...)``
    # for a second pass like eval_agent.py does. Some struct-typed list fields
    # in the saved config (e.g. ``list[SceneFileConfig]``) can't have their
    # CLI parser regenerated without defaults, which crashes the second
    # tyro pass. For data collection we only need to flip a handful of
    # training/logger flags, so parse those manually here.
    import argparse
    import dataclasses

    override_parser = argparse.ArgumentParser(add_help=False)
    override_parser.add_argument(
        "--training.num-envs", dest="num_envs", type=int, default=None,
    )
    override_parser.add_argument(
        "--training.max-eval-steps", dest="max_eval_steps", type=int,
        default=None,
    )
    override_parser.add_argument(
        "--training.headless", dest="headless", type=_parse_bool, default=None,
    )
    override_parser.add_argument(
        "--training.export-onnx", dest="export_onnx", type=_parse_bool,
        default=None,
    )
    override_parser.add_argument(
        "--logger.video.enabled", dest="logger_video_enabled",
        type=_parse_bool, default=None,
    )
    override_args, leftover = override_parser.parse_known_args(remaining)
    if leftover:
        logger.warning(f"Ignoring unparsed CLI tokens: {leftover}")

    training_updates = {"headless": True, "export_onnx": False}
    if override_args.num_envs is not None:
        training_updates["num_envs"] = override_args.num_envs
    if override_args.max_eval_steps is not None:
        training_updates["max_eval_steps"] = override_args.max_eval_steps
    if override_args.headless is not None:
        training_updates["headless"] = override_args.headless
    if override_args.export_onnx is not None:
        training_updates["export_onnx"] = override_args.export_onnx

    tyro_config = dataclasses.replace(
        eval_cfg,
        training=dataclasses.replace(eval_cfg.training, **training_updates),
    )
    if override_args.logger_video_enabled is not None:
        tyro_config = dataclasses.replace(
            tyro_config,
            logger=dataclasses.replace(
                tyro_config.logger,
                video=dataclasses.replace(
                    tyro_config.logger.video,
                    enabled=override_args.logger_video_enabled,
                ),
            ),
        )
    print(
        f"[collect] num_envs={tyro_config.training.num_envs} "
        f"max_eval_steps={tyro_config.training.max_eval_steps} "
        f"headless={tyro_config.training.headless}"
    )
    print(f"[collect] checkpoint={checkpoint_cfg.checkpoint}")
    print(f"[collect] rollout_output={collector_cfg.rollout_output}")
    print(f"[collect] stochastic={collector_cfg.stochastic}")

    # ----- Setup IsaacSim, build env+algo, load checkpoint -----
    env, device, simulation_app = setup_simulation_environment(tyro_config)

    eval_log_dir = get_experiment_dir(
        tyro_config.logger, tyro_config.training, get_timestamp(),
        task_name="collect",
    )
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving collector logs to {eval_log_dir}")
    tyro_config.save_config(str(eval_log_dir / CONFIG_NAME))

    checkpoint = load_checkpoint(checkpoint_cfg.checkpoint, str(eval_log_dir))
    checkpoint_path = str(checkpoint)

    algo_class = get_class(tyro_config.algo._target_)
    algo: BaseAlgo = algo_class(
        device=device, env=env, config=tyro_config.algo.config,
        log_dir=str(eval_log_dir), multi_gpu_cfg=None,
    )
    algo.setup()
    algo.attach_checkpoint_metadata(saved_cfg, saved_wandb_path)
    algo.load(checkpoint_path)

    # ----- Run collection -----
    try:
        run_multienv_collect(
            algo,
            num_steps=int(tyro_config.training.max_eval_steps),
            output_path=Path(collector_cfg.rollout_output),
            stochastic=bool(collector_cfg.stochastic),
            save_actor_obs=bool(collector_cfg.save_actor_obs),
            save_critic_obs=bool(collector_cfg.save_critic_obs),
        )
    finally:
        if simulation_app:
            close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
