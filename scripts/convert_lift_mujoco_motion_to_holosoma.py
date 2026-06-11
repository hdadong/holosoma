"""Convert a LIFT3 "mujoco" tracking motion into the Holosoma WBT motion format.

The LIFT3 tracking_motion ``*_mujoco.npz`` files store only the 14 tracked
bodies (``TRACK_BODY_NAMES``), a 29-dim actuated ``joint_pos``/``joint_vel`` (no
floating base) and per-body world poses/velocities.  Holosoma's ``MotionLoader``
(``holosoma.managers.command.terms.wbt``) instead expects:

* ``body_names`` / ``joint_names`` string arrays,
* every robot body in the IsaacSim asset (``simulator._body_list``, 51 links for
  the G1) to be present in ``body_names`` -- it maps robot bodies to motion
  bodies *by name* and asserts each one exists,
* ``joint_pos`` of shape ``(T, 7 + 29)`` (the leading 7 floating-base dims are
  sliced off and ignored: ``data["joint_pos"][:, 7:]``),
* ``joint_vel`` of shape ``(T, 6 + 29)`` (leading 6 base dims ignored),
* ``body_quat_w`` in **wxyz** convention.

Only the *tracked* bodies (``body_names_to_track``, which include the pelvis root
and the torso reference) are ever read from the reference motion for reward /
observation / reset / termination -- the remaining robot links are stored but
never consumed.  We therefore copy the 14 tracked-body trajectories verbatim and
fill every other robot body with a placeholder (the pelvis pose, zero velocity)
purely so the name->index mapping and array shapes are valid.  The canonical 51
``body_names`` and 29 ``joint_names`` (matching the IsaacSim USD asset) are read
from an existing holosoma motion (``--ref-holosoma-motion``).

Example::

    python scripts/convert_lift_mujoco_motion_to_holosoma.py \
        --src /workspace/lift3/tracking_motion/fly_kick_mujoco.npz \
        --out src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/fly_kick.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# The 14 tracked bodies, in the order the LIFT3 *_mujoco.npz files store them.
# Body 0 must be the pelvis (used as the floating-base root on reset).
TRACK_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)

DEFAULT_REF_HOLOSOMA_MOTION = (
    Path(__file__).resolve().parents[1]
    / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz"
)


def _as_str_list(values: np.ndarray) -> list[str]:
    return [str(v) for v in values.tolist()]


def convert(src: Path, out: Path, ref_holosoma_motion: Path) -> None:
    src_data = np.load(src, allow_pickle=True)
    ref_data = np.load(ref_holosoma_motion, allow_pickle=True)

    # Canonical names matching the IsaacSim G1 asset (read from a real holosoma motion).
    body_names = _as_str_list(ref_data["body_names"])  # e.g. 51 links (incl. "world")
    joint_names = _as_str_list(ref_data["joint_names"])  # 29 actuated G1 joints
    if len(joint_names) != 29:
        raise ValueError(f"Expected 29 joint names in {ref_holosoma_motion}, got {len(joint_names)}.")

    joint_pos = np.asarray(src_data["joint_pos"], dtype=np.float64)  # (T, 29)
    joint_vel = np.asarray(src_data["joint_vel"], dtype=np.float64)  # (T, 29)
    body_pos_w = np.asarray(src_data["body_pos_w"], dtype=np.float64)  # (T, 14, 3)
    body_quat_w = np.asarray(src_data["body_quat_w"], dtype=np.float64)  # (T, 14, 4) wxyz
    body_lin_vel_w = np.asarray(src_data["body_lin_vel_w"], dtype=np.float64)  # (T, 14, 3)
    body_ang_vel_w = np.asarray(src_data["body_ang_vel_w"], dtype=np.float64)  # (T, 14, 3)

    num_frames = joint_pos.shape[0]
    if joint_pos.shape[1] != 29 or joint_vel.shape[1] != 29:
        raise ValueError(f"Expected 29 actuated joints, got {joint_pos.shape}/{joint_vel.shape}.")
    if body_pos_w.shape[1] != len(TRACK_BODY_NAMES):
        raise ValueError(
            f"Expected {len(TRACK_BODY_NAMES)} tracked bodies in {src}, got {body_pos_w.shape[1]}."
        )

    # The LIFT3 mujoco joint order is identical to the holosoma/Unitree XML order
    # (convert_holosoma_motion.py round-trips with a plain [:, 7:] slice, no
    # reordering), so the 29 actuated joints map 1:1 onto holosoma joint_names.
    pelvis_pos = body_pos_w[:, 0]  # (T, 3)
    pelvis_quat = body_quat_w[:, 0]  # (T, 4) wxyz
    pelvis_lin_vel = body_lin_vel_w[:, 0]  # (T, 3)
    pelvis_ang_vel = body_ang_vel_w[:, 0]  # (T, 3)

    # joint_pos = [pelvis xyz, pelvis wxyz, 29 joints]; first 7 dims are sliced
    # off and ignored by the loader but we fill them for a self-consistent file.
    out_joint_pos = np.concatenate([pelvis_pos, pelvis_quat, joint_pos], axis=1)  # (T, 36)
    # joint_vel = [pelvis lin vel, pelvis ang vel, 29 joint vels]; first 6 ignored.
    out_joint_vel = np.concatenate([pelvis_lin_vel, pelvis_ang_vel, joint_vel], axis=1)  # (T, 35)

    # Build full per-body arrays. Placeholder = pelvis pose / identity / zero for
    # every body that is not one of the 14 tracked bodies (never read downstream).
    num_bodies = len(body_names)
    full_pos = np.broadcast_to(pelvis_pos[:, None, :], (num_frames, num_bodies, 3)).copy()
    full_quat = np.zeros((num_frames, num_bodies, 4), dtype=np.float64)
    full_quat[..., 0] = 1.0  # identity wxyz placeholder
    full_lin_vel = np.zeros((num_frames, num_bodies, 3), dtype=np.float64)
    full_ang_vel = np.zeros((num_frames, num_bodies, 3), dtype=np.float64)

    name_to_idx = {name: i for i, name in enumerate(body_names)}
    for src_idx, name in enumerate(TRACK_BODY_NAMES):
        if name not in name_to_idx:
            raise ValueError(f"Tracked body '{name}' missing from reference body_names {body_names}.")
        dst = name_to_idx[name]
        full_pos[:, dst] = body_pos_w[:, src_idx]
        full_quat[:, dst] = body_quat_w[:, src_idx]
        full_lin_vel[:, dst] = body_lin_vel_w[:, src_idx]
        full_ang_vel[:, dst] = body_ang_vel_w[:, src_idx]

    fps = np.asarray(src_data["fps"]).reshape(-1).astype(np.int64)  # (1,) to match holosoma

    converted = {
        "fps": fps,
        "joint_names": np.asarray(joint_names),
        "body_names": np.asarray(body_names),
        "joint_pos": out_joint_pos.astype(np.float64),
        "joint_vel": out_joint_vel.astype(np.float64),
        "body_pos_w": full_pos.astype(np.float64),
        "body_quat_w": full_quat.astype(np.float64),
        "body_lin_vel_w": full_lin_vel.astype(np.float64),
        "body_ang_vel_w": full_ang_vel.astype(np.float64),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **converted)

    print(f"src={src}")
    print(f"ref_holosoma_motion={ref_holosoma_motion}")
    print(f"out={out}")
    print(f"frames={num_frames}  fps={int(fps[0])}  duration_s={num_frames / float(fps[0]):.2f}")
    print(f"num_bodies={num_bodies}  num_joints={len(joint_names)}")
    print(f"joint_pos={converted['joint_pos'].shape}  joint_vel={converted['joint_vel'].shape}")
    print(f"tracked bodies copied verbatim: {list(TRACK_BODY_NAMES)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="LIFT3 *_mujoco.npz motion to convert.")
    parser.add_argument("--out", type=Path, required=True, help="Output holosoma-format motion npz.")
    parser.add_argument(
        "--ref-holosoma-motion",
        type=Path,
        default=DEFAULT_REF_HOLOSOMA_MOTION,
        help="Existing holosoma motion used only for canonical body_names/joint_names.",
    )
    args = parser.parse_args()

    src = args.src.expanduser().resolve()
    out = args.out.expanduser().resolve()
    ref = args.ref_holosoma_motion.expanduser().resolve()
    for path in (src, ref):
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
    convert(src, out, ref)


if __name__ == "__main__":
    main()
