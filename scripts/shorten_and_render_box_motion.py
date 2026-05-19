"""Shorten a WBT box-carrying motion .npz and render the truncated trajectory
in MuJoCo to an .mp4 video.

Defaults:
  --src   src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz
  --xml   src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof_w_largebox.xml
  --start 0
  --end   100   (exclusive; @50 fps -> 2.0 s)

Outputs (next to --src by default):
  sub3_largebox_003_mj_w_obj_short_<start>_<end>.npz
  sub3_largebox_003_mj_w_obj_short_<start>_<end>.mp4

Usage:
  conda activate lift3       # or any env with mujoco>=3.0 and imageio[ffmpeg]
  python scripts/shorten_and_render_box_motion.py
  python scripts/shorten_and_render_box_motion.py --start 50 --end 200
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")  # headless-safe; must be set before mujoco import

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SRC = (
    REPO_ROOT
    / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz"
)
DEFAULT_XML = (
    REPO_ROOT
    / "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof_w_largebox.xml"
)


def shorten_npz(
    src: Path,
    dst: Path,
    start: int,
    end: int,
    freeze_tail_seconds: float = 0.0,
) -> tuple[int, float]:
    """Slice every time-major array of src[start:end] and save to dst.

    All non-time-major scalars/arrays (fps, joint_names, body_names) are
    copied verbatim.

    If freeze_tail_seconds > 0, pad N = round(freeze_tail_seconds * fps) frames
    onto the end where:
      - position-like fields (everything except *_vel*) are repeated from the
        sliced motion's last frame, and
      - velocity-like fields (any name containing "vel") are zeroed.
    The reference for those padded frames is therefore "stand still in the
    final pose, with zero velocity" -- consistent for reward/termination.

    Returns (new_frame_count, fps).
    """
    non_time_keys = {"fps", "joint_names", "body_names"}

    with np.load(src) as d:
        T = d["joint_pos"].shape[0]
        if not (0 <= start < end <= T):
            raise ValueError(f"bad range [{start}:{end}) for T={T}")

        fps = float(np.asarray(d["fps"]).reshape(-1)[0])
        pad = int(round(freeze_tail_seconds * fps)) if freeze_tail_seconds > 0 else 0

        out: dict[str, np.ndarray] = {}
        for k in d.files:
            v = d[k]
            if k in non_time_keys:
                out[k] = v
                continue
            if v.shape[0] != T:
                raise AssertionError(
                    f"unexpected T for {k}: {v.shape} (expected first dim {T})"
                )
            sliced = v[start:end]
            if pad > 0:
                if "vel" in k:  # joint_vel, body_lin_vel_w, body_ang_vel_w, object_*_vel_w
                    tail = np.zeros((pad, *sliced.shape[1:]), dtype=sliced.dtype)
                else:
                    tail = np.broadcast_to(sliced[-1:], (pad, *sliced.shape[1:])).copy()
                out[k] = np.concatenate([sliced, tail], axis=0)
            else:
                out[k] = sliced

    np.savez(dst, **out)
    return (end - start) + pad, fps


def render_motion(
    npz_path: Path,
    xml_path: Path,
    out_mp4: Path,
    width: int,
    height: int,
    track_pelvis: bool,
) -> None:
    """Render the kinematic playback of npz_path using xml_path.

    The XML must contain (in this body order):
      1. G1 pelvis as a free joint (qpos[0:7])  -> 29 robot joints (qpos[7:36])
      2. largebox as a free joint              (qpos[36:43])
    Total nq == 43.
    """
    with np.load(npz_path) as d:
        joint_pos = d["joint_pos"]            # (T, 36): pelvis xyz(3) + pelvis wxyz(4) + 29 joints
        object_pos_w = d["object_pos_w"]      # (T, 3) world
        object_quat_w = d["object_quat_w"]    # (T, 4) wxyz
        fps = int(np.asarray(d["fps"]).reshape(-1)[0])
        T = joint_pos.shape[0]

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    if model.nq != 43:
        raise AssertionError(
            f"Expected nq=43 (pelvis free + 29 joints + box free); got nq={model.nq}. "
            "Use the combined G1 + largebox XML."
        )

    # MuJoCo defaults the offscreen framebuffer to 640x480; bump it so the
    # renderer can produce the requested resolution without editing the XML.
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
        for t in range(T):
            # robot pelvis free joint
            data.qpos[0:3] = joint_pos[t, 0:3]
            data.qpos[3:7] = joint_pos[t, 3:7]          # wxyz, matches MuJoCo
            # robot 29 joints
            data.qpos[7:36] = joint_pos[t, 7:36]
            # box free joint
            data.qpos[36:39] = object_pos_w[t]
            data.qpos[39:43] = object_quat_w[t]         # wxyz

            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)              # kinematics only

            if track_pelvis:
                cam.lookat[0] = data.qpos[0]
                cam.lookat[1] = data.qpos[1]

            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())

    renderer.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--dst", type=Path, default=None,
                   help="Output shortened .npz (default: alongside --src)")
    p.add_argument("--start", type=int, default=0, help="Start frame (inclusive)")
    p.add_argument("--end", type=int, default=100, help="End frame (exclusive)")
    p.add_argument("--xml", type=Path, default=DEFAULT_XML)
    p.add_argument("--mp4", type=Path, default=None,
                   help="Output .mp4 (default: same stem as --dst)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--track-pelvis", action="store_true",
                   help="Camera follows the pelvis xy each frame")
    p.add_argument("--no-render", action="store_true",
                   help="Only slice the npz, skip the .mp4")
    p.add_argument("--freeze-tail-seconds", type=float, default=0.0,
                   help="Pad N=round(seconds*fps) frames at the tail where positions "
                        "= sliced last frame and velocities = 0 (default: 0, no pad)")
    args = p.parse_args()

    if args.dst is None:
        suffix = f"_short_{args.start}_{args.end}"
        if args.freeze_tail_seconds > 0:
            tail_frames = int(round(args.freeze_tail_seconds * 50))  # placeholder fps; recompute after load if needed
            suffix += f"_freeze{tail_frames}"
        args.dst = args.src.with_name(f"{args.src.stem}{suffix}.npz")
    if args.mp4 is None:
        args.mp4 = args.dst.with_suffix(".mp4")

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    n, fps = shorten_npz(args.src, args.dst, args.start, args.end, args.freeze_tail_seconds)
    print(f"[npz]  {args.dst}  ({n} frames @ {fps:g} fps -> {n / fps:.2f}s)")

    if args.no_render:
        return

    render_motion(
        npz_path=args.dst,
        xml_path=args.xml,
        out_mp4=args.mp4,
        width=args.width,
        height=args.height,
        track_pelvis=args.track_pelvis,
    )
    print(f"[mp4]  {args.mp4}")


if __name__ == "__main__":
    main()
