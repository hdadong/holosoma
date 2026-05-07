"""IsaacSim first-person-view (FPV) capture for evaluation.

The FPV ``TiledCamera`` sensors are registered in
``IsaacSim._setup_fpv_tiled_camera()`` (gated by ``HOLOSOMA_FPV_ENABLE=1``)
and live in ``simulator.scene.sensors["fpv_camera_<name>"]``. By default
three cameras are registered: ``head``, ``left_wrist``, ``right_wrist``.

This module wraps that registry into a tiny multi-camera recorder: each
step it slices the env-0 RGB tile from every camera, buffers frames, and
at the end dumps one PNG dir + mp4 per camera under
``<output_dir>/<camera_name>/``.

Mirrors NVlabs/GR00T-VisualSim2Real's ``ego_camera`` pattern (TiledCamera
registered under the robot articulation; per-step ``data.output["rgb"]``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from holosoma.simulator.isaacsim.isaacsim import IsaacSim

FPV_SENSOR_PREFIX = "fpv_camera_"


class IsaacSimFPVRecorder:
    """Buffer per-step RGB frames from all registered FPV ``TiledCamera`` sensors.

    Parameters
    ----------
    simulator : IsaacSim
        Live simulator instance. Must have been built with
        ``HOLOSOMA_FPV_ENABLE=1`` so that one or more
        ``simulator.scene.sensors["fpv_camera_<name>"]`` entries exist.
    record_env_id : int
        Env index whose tile is captured (default 0).
    """

    def __init__(
        self,
        simulator: "IsaacSim",
        record_env_id: int = 0,
        # Legacy kwargs accepted (and ignored) for backward compatibility
        # with the original single-camera recorder. All camera intrinsics
        # are now driven by HOLOSOMA_FPV_* env vars at scene-init time.
        width: int | None = None,
        height: int | None = None,
        vertical_fov_deg: float | None = None,
        head_body_name: str | None = None,
        pos_offset: tuple[float, float, float] | None = None,
    ) -> None:
        self.simulator = simulator
        self.record_env_id = int(record_env_id)
        self._sensors: dict[str, "object"] = {}  # name -> TiledCamera
        self._frames: dict[str, list[np.ndarray]] = {}
        self._dims: dict[str, tuple[int, int]] = {}  # name -> (W, H)
        # Aux per-step buffers for trajectory dump (action/reward/done).
        self._actions: list[np.ndarray] = []
        self._rewards: list[float] = []
        self._dones: list[bool] = []

    def setup(self) -> None:
        """Resolve all registered FPV cameras from the live scene."""
        sensors = getattr(self.simulator.scene, "sensors", None)
        if sensors is None:
            raise RuntimeError("Simulator scene has no sensors attribute.")

        for key, sensor in list(sensors.items()):
            if not key.startswith(FPV_SENSOR_PREFIX):
                continue
            name = key[len(FPV_SENSOR_PREFIX):]
            self._sensors[name] = sensor
            self._frames[name] = []
            cfg = sensor.cfg
            self._dims[name] = (int(cfg.width), int(cfg.height))
            logger.info(
                f"[fpv:{name}] Resolved scene.sensors['{key}']: "
                f"prim_path={cfg.prim_path!r}, resolution={self._dims[name]}, "
                f"record_env_id={self.record_env_id}"
            )

        if not self._sensors:
            raise RuntimeError(
                "No FPV TiledCamera found. Did the IsaacSim scene init run with "
                "HOLOSOMA_FPV_ENABLE=1 and --enable_cameras passed to AppLauncher?"
            )

    def capture(
        self,
        action: np.ndarray | None = None,
        reward: float | None = None,
        done: bool | None = None,
    ) -> dict[str, np.ndarray]:
        """Read the env_0 RGB tile from every registered FPV camera.

        Optionally accepts the per-step ``action`` (1-D float array),
        scalar ``reward``, and ``done`` flag, which are buffered for
        downstream ``save_npz`` calls. ``TiledCamera`` updates each
        ``scene.update()`` (called from
        :meth:`IsaacSim.simulate_at_each_physics_step`) because IsaacLab
        marks it as an RTX sensor → ``has_rtx_sensors()=True`` triggers
        the renderer tick. No manual ``sim.render()`` call needed.
        """
        out: dict[str, np.ndarray] = {}
        for name, sensor in self._sensors.items():
            w, h = self._dims[name]
            rgb = sensor.data.output.get("rgb")
            if rgb is None or rgb.numel() == 0:
                frame = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                frame_t = rgb[self.record_env_id, :, :, :3].detach().to("cpu").contiguous()
                frame = frame_t.numpy().astype(np.uint8, copy=False)
            self._frames[name].append(frame)
            out[name] = frame
            if len(self._frames[name]) == 1:
                logger.info(
                    f"[fpv:{name}] First frame: shape={frame.shape}, "
                    f"dtype={frame.dtype}, non_zero={bool(frame.any())}"
                )

        if action is not None:
            self._actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
        if reward is not None:
            self._rewards.append(float(reward))
        if done is not None:
            self._dones.append(bool(done))
        return out

    def num_captured(self) -> int:
        if not self._frames:
            return 0
        return min(len(v) for v in self._frames.values())

    def save(self, output_dir: str, fps: int = 50) -> Path:
        """Write captured frames per-camera to ``<output_dir>/<name>/``.

        Returns the root output dir. For each camera dumps PNGs
        ``fpv_<step>.png`` and an mp4 ``fpv_eval.mp4`` (imageio +
        ffmpeg fallback).
        """
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        if not self._frames:
            logger.warning("[fpv] No frames captured; nothing to save")
            return root

        for name, frames in self._frames.items():
            if not frames:
                logger.warning(f"[fpv:{name}] No frames captured")
                continue
            sub = root / name
            sub.mkdir(parents=True, exist_ok=True)
            self._save_one(name, frames, sub, fps=fps)
        return root

    @staticmethod
    def _save_one(name: str, frames: list[np.ndarray], out_dir: Path, fps: int) -> Path:
        from PIL import Image

        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(out_dir / f"fpv_{i:06d}.png")
        logger.info(f"[fpv:{name}] Saved {len(frames)} PNG frames to {out_dir}")

        mp4_path = out_dir / "fpv_eval.mp4"
        try:
            import imageio.v2 as imageio
            with imageio.get_writer(
                str(mp4_path), fps=int(fps), codec="libx264",
                pixelformat="yuv420p", quality=8,
            ) as writer:
                for frame in frames:
                    writer.append_data(frame)
            logger.info(
                f"[fpv:{name}] Wrote video {mp4_path} via imageio "
                f"({len(frames)} frames @ {fps} fps)"
            )
            return mp4_path
        except Exception as exc:
            logger.warning(
                f"[fpv:{name}] imageio video write failed ({exc}); trying system ffmpeg"
            )

        try:
            import subprocess
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(out_dir / "fpv_%06d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                str(mp4_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"[fpv:{name}] Wrote video {mp4_path} via ffmpeg")
        except Exception as exc:
            logger.warning(f"[fpv:{name}] ffmpeg failed ({exc}); PNG frames are still on disk")
        return mp4_path

    def save_npz(self, path: str) -> Path:
        """Dump frames + per-step action/reward/done to a single .npz.

        Output keys (numpy arrays):
          ``obs_<camera>_rgb`` : (T, H, W, 3) uint8
          ``actions``          : (T, action_dim) float32
          ``rewards``          : (T,) float32
          ``dones``            : (T,) bool
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if not self._frames:
            logger.warning(f"[fpv] save_npz: no frames buffered, skipping {out}")
            return out

        save_dict: dict[str, np.ndarray] = {}
        T = min(len(v) for v in self._frames.values())
        for name, frames in self._frames.items():
            arr = np.stack(frames[:T], axis=0)  # (T, H, W, 3) uint8
            save_dict[f"obs_{name}_rgb"] = arr

        if self._actions:
            T_a = min(T, len(self._actions))
            save_dict["actions"] = np.stack(self._actions[:T_a], axis=0)
        if self._rewards:
            T_r = min(T, len(self._rewards))
            save_dict["rewards"] = np.asarray(self._rewards[:T_r], dtype=np.float32)
        if self._dones:
            T_d = min(T, len(self._dones))
            save_dict["dones"] = np.asarray(self._dones[:T_d], dtype=bool)

        np.savez_compressed(out, **save_dict)
        sizes = ", ".join(f"{k}={v.shape}" for k, v in save_dict.items())
        logger.info(f"[fpv] save_npz wrote {out} ({sizes})")
        return out

    def cleanup(self) -> None:
        # Sensor lifecycle is owned by the scene; nothing to detach here.
        self._sensors.clear()
