#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import logging
import os
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.retargeter import RetargeterConfig
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.config_types.task import TaskConfig
from holosoma_retargeting.examples.robot_retarget import (
    build_retargeter_kwargs_from_config,
    create_task_constants,
    save_retarget_video,
    setup_object_data,
)
from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter
from holosoma_retargeting.src.utils import extract_foot_sticking_sequence_velocity

REPO_ROOT = Path(__file__).resolve().parents[1]
RETARGET_ROOT = REPO_ROOT / "src/holosoma_retargeting/holosoma_retargeting"


def _retarget_single_npz(src_npz: Path, out_npz: Path) -> tuple[bool, str]:
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            # Resolve model/object relative paths used by setup_object_data().
            os.chdir(RETARGET_ROOT)
            logging.getLogger().setLevel(logging.WARNING)

            data = np.load(src_npz)
            human_joints = data["human_joints"]
            qpos = data["qpos"]
            object_poses = qpos[:, -7:].copy()  # MuJoCo order: [x, y, z, qw, qx, qy, qz]

            robot_cfg = RobotConfig(robot_type="g1")
            motion_cfg = MotionDataConfig(data_format="skillmimic", robot_type="g1")
            task_cfg = TaskConfig(object_name="basketball")
            constants = create_task_constants(robot_cfg, motion_cfg, task_cfg, task_type="object_interaction")

            smpl_scale = constants.ROBOT_HEIGHT / (motion_cfg.default_human_height or 1.94)
            object_local_pts, object_local_pts_demo, object_urdf_path = setup_object_data(
                "object_interaction", constants, None, smpl_scale, task_cfg, augmentation=False
            )

            ret_cfg = RetargeterConfig(visualize=False, debug=False)
            retargeter_kwargs = build_retargeter_kwargs_from_config(
                ret_cfg, constants, object_urdf_path, "object_interaction"
            )
            retargeter = InteractionMeshRetargeter(**retargeter_kwargs)

            foot_sticking_sequences = extract_foot_sticking_sequence_velocity(
                human_joints, retargeter.demo_joints, motion_cfg.toe_names
            )
            foot_sticking_sequences[0][motion_cfg.toe_names[0]] = False
            foot_sticking_sequences[0][motion_cfg.toe_names[1]] = False

            q_init = qpos[0, :36].copy()

            retargeter.retarget_motion(
                human_joint_motions=human_joints,
                object_poses=object_poses,
                object_poses_augmented=object_poses,
                object_points_local_demo=object_local_pts_demo,
                object_points_local=object_local_pts,
                foot_sticking_sequences=foot_sticking_sequences,
                q_a_init=q_init,
                q_nominal_list=None,
                original=True,
                dest_res_path=str(out_npz),
            )

        return True, src_npz.name
    except Exception:
        return False, f"{src_npz.name}\n{traceback.format_exc()}"


def _process_retarget_batch(src_files: list[Path], out_npz_dir: Path, workers: int, skip_existing: bool) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    jobs: list[tuple[Path, Path]] = []
    for src in src_files:
        out_npz = out_npz_dir / src.name
        if skip_existing and out_npz.exists():
            continue
        jobs.append((src, out_npz))

    if not jobs:
        print("No retarget jobs to run (all outputs exist).")
        return

    print(f"Retarget jobs: {len(jobs)}, workers: {workers}")
    ok = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_retarget_single_npz, src, out): (src, out) for src, out in jobs}
        with tqdm(total=len(futures), desc="Retarget", unit="file") as pbar:
            for fut in as_completed(futures):
                success, msg = fut.result()
                if success:
                    ok += 1
                else:
                    failed += 1
                    print(f"\nFAILED:\n{msg}")
                pbar.update(1)

    print(f"Retarget done. success={ok}, failed={failed}")
    if failed > 0:
        raise RuntimeError(f"{failed} retarget jobs failed")


def _render_video_batch(out_npz_dir: Path, out_video_dir: Path, workers_hint: int, skip_existing: bool) -> None:
    # Rendering is done sequentially for stability across headless environments.
    robot_cfg = RobotConfig(robot_type="g1")
    motion_cfg = MotionDataConfig(data_format="skillmimic", robot_type="g1")
    task_cfg = TaskConfig(object_name="basketball")
    constants = create_task_constants(robot_cfg, motion_cfg, task_cfg, task_type="object_interaction")

    npz_files = sorted(out_npz_dir.glob("*.npz"))
    jobs: list[tuple[Path, Path]] = []
    for npz in npz_files:
        mp4 = out_video_dir / (npz.stem + ".mp4")
        if skip_existing and mp4.exists():
            continue
        jobs.append((npz, mp4))

    if not jobs:
        print("No render jobs to run (all videos exist).")
        return

    print(f"Render jobs: {len(jobs)} (workers hint: {workers_hint}, running sequentially)")
    for npz, mp4 in tqdm(jobs, desc="Render", unit="file"):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            save_retarget_video(str(npz), constants, str(mp4), fps=30)

    print("Render done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch retarget from existing SkillMimic NPZ files and render videos.")
    parser.add_argument(
        "--src-npz-dir",
        type=Path,
        default=RETARGET_ROOT / "demo_results/g1/object_interaction/skillmimic_ballplaym_ballfix",
    )
    parser.add_argument("--out-npz-dir", type=Path, default=Path("/tmp/skillmimic_ballplaym_ballfix_tuned_npz"))
    parser.add_argument("--out-video-dir", type=Path, default=Path("/tmp/skillmimic_ballplaym_ballfix_tuned_videos"))
    parser.add_argument("--workers", type=int, default=max(1, min(6, os.cpu_count() or 1)))
    parser.add_argument("--skip-existing", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(RETARGET_ROOT)

    src_npy_dir = args.src_npz_dir
    out_npz_dir = args.out_npz_dir
    out_video_dir = args.out_video_dir

    out_npz_dir.mkdir(parents=True, exist_ok=True)
    out_video_dir.mkdir(parents=True, exist_ok=True)

    src_files = sorted(src_npy_dir.glob("*.npz"))
    if not src_files:
        raise FileNotFoundError(f"No npz files found in {src_npy_dir}")

    print(f"Source files: {len(src_files)}")
    _process_retarget_batch(src_files, out_npz_dir, args.workers, args.skip_existing)
    _render_video_batch(out_npz_dir, out_video_dir, args.workers, args.skip_existing)

    print("\nOutputs:")
    print(f"  npz:   {out_npz_dir}")
    print(f"  video: {out_video_dir}")


if __name__ == "__main__":
    main()
