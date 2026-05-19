"""Apply the holosoma WBT framework's "default-pose prepend/append" to a motion
.npz offline, and render the resulting trajectory in MuJoCo to .mp4.

Reproduces the runtime behavior of
[holosoma.managers.command.terms.wbt.MotionCommand._maybe_add_default_pose_transition]:
  * prepend N_pre frames interpolating default-pose -> motion[0]
  * append  N_app frames interpolating motion[-1] -> default-pose
where the "default pose" is:
  * joint angles = robot.g1_29dof.init_state.default_joint_angles
  * root xyz: x,y anchored to the motion-edge pelvis, z = 0.76 (WBT init height)
  * root quat: roll/pitch = 0 (init), yaw from the motion-edge pelvis
  * all velocities = 0
  * object pose/vel = motion-edge values (so the box stays put during the
    transition window, while the robot lerps toward / away from standing)

joint_pos column layout in the on-disk npz: 3 (pelvis xyz) + 4 (pelvis wxyz) + 29 (joints).
Interpolation: lerp on positions/velocities, slerp on quaternions (root quat,
body quats, object quat).

Usage:
  python scripts/add_default_pose_transitions.py
  python scripts/add_default_pose_transitions.py --prepend-seconds 1.5 --append-seconds 3.0
  python scripts/add_default_pose_transitions.py --src .../sub3_largebox_003_mj_w_obj_short_0_85.npz
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")  # headless-safe; must be set before mujoco import

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation, Slerp  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SRC = (
    REPO_ROOT
    / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz"
)
DEFAULT_XML = (
    REPO_ROOT
    / "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof_w_largebox.xml"
)

# WBT G1 default standing pose, copied verbatim from
# src/holosoma/holosoma/config_values/robot.py (g1_29dof init_state) and
# src/holosoma/holosoma/config_values/wbt/g1/experiment.py (pos override).
INIT_STANDING_Z = 0.76
DEFAULT_JOINT_ANGLES: dict[str, float] = {
    "left_hip_pitch_joint": -0.312,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.312,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.669,
    "right_ankle_pitch_joint": -0.363,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.6,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.6,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

# Position-like (lerp) vs quaternion-like (slerp) vs velocity-like (lerp, target=0) fields.
QUAT_FIELDS = {"body_quat_w", "object_quat_w"}


def _yaw_only_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Project a quaternion onto a yaw-only (roll=pitch=0) rotation, returning wxyz."""
    # scipy expects xyzw
    q_xyzw = quat_wxyz[[1, 2, 3, 0]]
    yaw = Rotation.from_quat(q_xyzw).as_euler("xyz")[2]
    out = Rotation.from_euler("z", yaw).as_quat()  # xyzw
    return np.array([out[3], out[0], out[1], out[2]], dtype=np.float64)


def _slerp_wxyz(q1: np.ndarray, q2: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Slerp a single quaternion pair (q1, q2) at the given alphas in [0, 1].

    Inputs in wxyz, shape (4,). alphas shape (N,). Output (N, 4) wxyz.
    """
    q1_xyzw = q1[[1, 2, 3, 0]]
    q2_xyzw = q2[[1, 2, 3, 0]]
    # Pick shortest path
    if np.dot(q1_xyzw, q2_xyzw) < 0.0:
        q2_xyzw = -q2_xyzw
    rots = Rotation.from_quat(np.stack([q1_xyzw, q2_xyzw], axis=0))
    slerp = Slerp([0.0, 1.0], rots)
    out_xyzw = slerp(alphas).as_quat()
    return out_xyzw[:, [3, 0, 1, 2]]


def _slerp_wxyz_batch(q1: np.ndarray, q2: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Slerp per-body. q1, q2 shape (B, 4) wxyz, alphas (N,). Returns (N, B, 4) wxyz."""
    n = alphas.shape[0]
    b = q1.shape[0]
    out = np.zeros((n, b, 4), dtype=np.float64)
    for j in range(b):
        out[:, j, :] = _slerp_wxyz(q1[j], q2[j], alphas)
    return out


def _build_alphas(num_steps: int, drop_first: bool, drop_last: bool) -> np.ndarray:
    """Replicate holosoma's alpha schedule from
    [_build_and_apply_transition](src/holosoma/.../wbt.py#L1056)."""
    alphas = np.linspace(0.0, 1.0, num_steps + 1)
    if drop_first:
        alphas = alphas[1:]
    if drop_last:
        alphas = alphas[:-1]
    return alphas


def _compute_default_pose_npz_state(
    npz: dict[str, np.ndarray],
    anchor_idx: int,
    xml_path: Path,
) -> dict[str, np.ndarray]:
    """Build the npz-format state dict for the default standing pose, anchored
    to npz[anchor_idx]. Mirrors
    [MotionCommand._build_default_pose_state](src/holosoma/.../wbt.py#L788)
    but uses MuJoCo FK in place of IsaacSim's articulation buffer.

    Returns a dict keyed by the SAME field names as the npz, with each value
    shape == the per-frame slice of that npz field.
    """
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    if model.nq != 43:
        raise AssertionError(
            f"Expected nq=43; got {model.nq}. Use the combined G1+largebox XML."
        )

    npz_body_names: list[str] = list(npz["body_names"])
    npz_joint_names: list[str] = list(npz["joint_names"])
    pelvis_npz_idx = npz_body_names.index("pelvis")

    # Anchor xy/yaw from motion-edge pelvis pose. The on-disk npz stores
    # body_quat_w in wxyz (we saw this in MotionLoader._load_data_from_motion_npz).
    anchor_pelvis_xyz = npz["body_pos_w"][anchor_idx, pelvis_npz_idx].astype(np.float64)
    anchor_pelvis_quat_wxyz = npz["body_quat_w"][anchor_idx, pelvis_npz_idx].astype(np.float64)

    default_root_xyz = np.array(
        [anchor_pelvis_xyz[0], anchor_pelvis_xyz[1], INIT_STANDING_Z],
        dtype=np.float64,
    )
    default_root_quat_wxyz = _yaw_only_quat_wxyz(anchor_pelvis_quat_wxyz)

    # Set the MuJoCo state to the default pose anchored at the motion-edge xy/yaw,
    # then run FK to get every body's world pose.
    data.qpos[0:3] = default_root_xyz
    data.qpos[3:7] = default_root_quat_wxyz
    for joint_name, angle in DEFAULT_JOINT_ANGLES.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid == -1:
            raise KeyError(f"Joint {joint_name} not found in XML")
        data.qpos[model.jnt_qposadr[jid]] = angle
    # Box pose: take the motion-edge values (so the box stays put during the transition).
    data.qpos[36:39] = npz["object_pos_w"][anchor_idx]
    data.qpos[39:43] = npz["object_quat_w"][anchor_idx]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # joint_pos for the on-disk format: 3 (pelvis xyz) + 4 (pelvis wxyz) + 29 joints (in NPZ order).
    joint_pos_field = np.zeros(npz["joint_pos"].shape[1], dtype=npz["joint_pos"].dtype)
    joint_pos_field[0:3] = default_root_xyz
    joint_pos_field[3:7] = default_root_quat_wxyz
    for joint_name, angle in DEFAULT_JOINT_ANGLES.items():
        npz_jidx = npz_joint_names.index(joint_name)
        joint_pos_field[7 + npz_jidx] = angle

    joint_vel_field = np.zeros(npz["joint_vel"].shape[1], dtype=npz["joint_vel"].dtype)

    # body_*_w via FK, mapped from MJCF body order -> npz body_names order.
    n_bodies = len(npz_body_names)
    body_pos = np.zeros((n_bodies, 3), dtype=npz["body_pos_w"].dtype)
    body_quat = np.zeros((n_bodies, 4), dtype=npz["body_quat_w"].dtype)
    body_quat[:, 0] = 1.0  # identity fallback for any unmatched body
    for npz_idx, name in enumerate(npz_body_names):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid == -1:
            continue
        body_pos[npz_idx] = data.xpos[bid]
        body_quat[npz_idx] = data.xquat[bid]  # MuJoCo stores wxyz

    state = {
        "joint_pos": joint_pos_field,
        "joint_vel": joint_vel_field,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": np.zeros((n_bodies, 3), dtype=npz["body_lin_vel_w"].dtype),
        "body_ang_vel_w": np.zeros((n_bodies, 3), dtype=npz["body_ang_vel_w"].dtype),
    }
    # Carry object_* alongside (motion-edge values, constant through the window).
    for k in ("object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w"):
        if k in npz:
            state[k] = npz[k][anchor_idx].astype(npz[k].dtype)
    return state


def _state_at(npz: dict[str, np.ndarray], idx: int) -> dict[str, np.ndarray]:
    """Slice every time-major field at a single frame index, return as a dict."""
    non_time = {"fps", "joint_names", "body_names"}
    return {k: npz[k][idx].copy() for k in npz if k not in non_time}


def _interpolate_states(
    start: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    alphas: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-field lerp/slerp between two per-frame state dicts.

    Special handling for joint_pos: columns [:3] lerp, [3:7] slerp, [7:] lerp,
    so the pelvis quaternion stays valid (a naive lerp on a unit quat creates
    intermediate non-unit quats which both rendering and downstream FK dislike).
    """
    out: dict[str, np.ndarray] = {}
    a1 = alphas.reshape(-1, 1)
    a2 = alphas.reshape(-1, 1, 1)

    for k, v_target in target.items():
        v_start = start[k]
        if k == "joint_pos":
            xyz = v_start[:3] + a1 * (v_target[:3] - v_start[:3])  # (n, 3)
            quat = _slerp_wxyz(v_start[3:7], v_target[3:7], alphas)  # (n, 4)
            joints = v_start[7:] + a1 * (v_target[7:] - v_start[7:])  # (n, J)
            out[k] = np.concatenate([xyz, quat, joints], axis=1)
        elif k == "body_quat_w":
            out[k] = _slerp_wxyz_batch(v_start, v_target, alphas)  # (n, B, 4)
        elif k == "object_quat_w":
            out[k] = _slerp_wxyz(v_start, v_target, alphas)  # (n, 4)
        elif v_target.ndim == 1:
            out[k] = v_start + a1 * (v_target - v_start)  # (n, D)
        elif v_target.ndim == 2:
            out[k] = v_start + a2 * (v_target - v_start)  # (n, B, D)
        else:
            raise NotImplementedError(f"Unhandled field shape for {k}: {v_target.shape}")
    return out


def add_default_pose_transitions(
    src: Path,
    dst: Path,
    xml_path: Path,
    prepend_seconds: float,
    append_seconds: float,
) -> tuple[int, float, int, int]:
    """Prepend / append default-pose interpolated frames to src and save to dst.

    Returns (total_frames, fps, n_prepend, n_append).
    """
    with np.load(src) as d_npz:
        npz = {k: d_npz[k] for k in d_npz.files}
    fps = float(np.asarray(npz["fps"]).reshape(-1)[0])

    n_pre = int(round(prepend_seconds * fps)) if prepend_seconds > 0 else 0
    n_app = int(round(append_seconds * fps)) if append_seconds > 0 else 0

    parts: dict[str, list[np.ndarray]] = {}
    time_keys = [k for k in npz if k not in {"fps", "joint_names", "body_names"}]
    for k in time_keys:
        parts[k] = []

    if n_pre > 0:
        default_pre = _compute_default_pose_npz_state(npz, anchor_idx=0, xml_path=xml_path)
        motion_first = _state_at(npz, 0)
        # holosoma's prepend uses drop_last=True so that motion[0] isn't duplicated when concatenated.
        alphas_pre = _build_alphas(n_pre, drop_first=False, drop_last=True)
        pre_seg = _interpolate_states(default_pre, motion_first, alphas_pre)
        for k in time_keys:
            parts[k].append(pre_seg[k])

    for k in time_keys:
        parts[k].append(npz[k])

    if n_app > 0:
        default_post = _compute_default_pose_npz_state(npz, anchor_idx=-1, xml_path=xml_path)
        motion_last = _state_at(npz, -1)
        # holosoma's append uses drop_first=True (motion[-1] not duplicated).
        alphas_app = _build_alphas(n_app, drop_first=True, drop_last=False)
        app_seg = _interpolate_states(motion_last, default_post, alphas_app)
        for k in time_keys:
            parts[k].append(app_seg[k])

    out: dict[str, np.ndarray] = {}
    for k in npz:
        if k in {"fps", "joint_names", "body_names"}:
            out[k] = npz[k]
        else:
            out[k] = np.concatenate(parts[k], axis=0)

    np.savez(dst, **out)
    total = out["joint_pos"].shape[0]
    return total, fps, n_pre, n_app


def render_motion(
    npz_path: Path,
    xml_path: Path,
    out_mp4: Path,
    width: int,
    height: int,
    track_pelvis: bool,
) -> None:
    """Render the kinematic playback (robot + box) of npz_path to out_mp4."""
    with np.load(npz_path) as d:
        joint_pos = d["joint_pos"]
        object_pos_w = d["object_pos_w"]
        object_quat_w = d["object_quat_w"]
        fps = int(np.asarray(d["fps"]).reshape(-1)[0])
        total = joint_pos.shape[0]

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    if model.nq != 43:
        raise AssertionError(f"Expected nq=43; got {model.nq}. Use combined G1+largebox XML.")

    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    renderer = mujoco.Renderer(model, height=height, width=width)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 3.2
    cam.elevation = -15.0
    cam.azimuth = -135.0
    pelvis_mean_xy = joint_pos[:, :2].mean(axis=0)
    cam.lookat = np.array([pelvis_mean_xy[0], pelvis_mean_xy[1], 0.8], dtype=np.float64)

    with imageio.get_writer(out_mp4, fps=fps, codec="libx264", quality=8) as writer:
        for t in range(total):
            data.qpos[0:3] = joint_pos[t, 0:3]
            data.qpos[3:7] = joint_pos[t, 3:7]
            data.qpos[7:36] = joint_pos[t, 7:36]
            data.qpos[36:39] = object_pos_w[t]
            data.qpos[39:43] = object_quat_w[t]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            if track_pelvis:
                cam.lookat[0] = data.qpos[0]
                cam.lookat[1] = data.qpos[1]
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())

    renderer.close()


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--dst", type=Path, default=None,
                   help="Output .npz (default: <src>_pre<N>_app<N>.npz next to --src)")
    p.add_argument("--xml", type=Path, default=DEFAULT_XML)
    p.add_argument("--mp4", type=Path, default=None,
                   help="Output .mp4 (default: same stem as --dst)")
    p.add_argument("--prepend-seconds", type=float, default=2.0)
    p.add_argument("--append-seconds", type=float, default=2.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--track-pelvis", action="store_true",
                   help="Camera follows pelvis xy each frame")
    p.add_argument("--no-render", action="store_true", help="Only build the .npz")
    args = p.parse_args()

    fps_est = 50  # for filename only; real fps is read from npz
    n_pre = int(round(args.prepend_seconds * fps_est))
    n_app = int(round(args.append_seconds * fps_est))
    if args.dst is None:
        args.dst = args.src.with_name(f"{args.src.stem}_pre{n_pre}_app{n_app}.npz")
    if args.mp4 is None:
        args.mp4 = args.dst.with_suffix(".mp4")

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    total, fps, n_pre_actual, n_app_actual = add_default_pose_transitions(
        src=args.src, dst=args.dst, xml_path=args.xml,
        prepend_seconds=args.prepend_seconds, append_seconds=args.append_seconds,
    )
    print(
        f"[npz]  {args.dst}  ({total} frames @ {fps:g} fps -> {total / fps:.2f}s; "
        f"prepend={n_pre_actual}, append={n_app_actual})"
    )

    if args.no_render:
        return

    render_motion(
        npz_path=args.dst, xml_path=args.xml, out_mp4=args.mp4,
        width=args.width, height=args.height, track_pelvis=args.track_pelvis,
    )
    print(f"[mp4]  {args.mp4}")


if __name__ == "__main__":
    main()
