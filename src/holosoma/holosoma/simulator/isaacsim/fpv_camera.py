"""IsaacSim first-person-view (FPV) capture for evaluation.

The FPV ``TiledCamera`` is registered in
``IsaacSim._setup_fpv_tiled_camera()`` (gated by ``HOLOSOMA_FPV_ENABLE=1``)
and lives in ``simulator.scene.sensors["fpv_camera"]``. This module wraps
that sensor into a tiny recorder: each step it slices the env-0 RGB tile,
buffers frames, then dumps PNG + mp4 at the end.

This mirrors NVlabs/GR00T-VisualSim2Real's ``ego_camera`` pattern (see
``gr00t/rl/simulator/isaacsim/isaacsim.py``: TiledCameraCfg under the
robot articulation; per-step ``data.output["rgb"]``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from holosoma.simulator.isaacsim.isaacsim import IsaacSim

FPV_SENSOR_NAME = "fpv_camera"


class IsaacSimFPVRecorder:
    """Buffer per-step RGB frames from the FPV ``TiledCamera`` sensor.

    Parameters
    ----------
    simulator : IsaacSim
        Live simulator instance. Must have been built with
        ``HOLOSOMA_FPV_ENABLE=1`` so that ``simulator.scene.sensors[
        "fpv_camera"]`` is present.
    record_env_id : int
        Env index whose tile is captured (default 0).
    """

    def __init__(
        self,
        simulator: "IsaacSim",
        record_env_id: int = 0,
        # Legacy kwargs accepted (and ignored) for backward compatibility with
        # the replicator-based recorder; the camera intrinsics are now driven
        # entirely by HOLOSOMA_FPV_* env vars at IsaacSim scene-init time.
        width: int | None = None,
        height: int | None = None,
        vertical_fov_deg: float | None = None,
        head_body_name: str | None = None,
        pos_offset: tuple[float, float, float] | None = None,
    ) -> None:
        self.simulator = simulator
        self.record_env_id = int(record_env_id)
        self._frames: list[np.ndarray] = []
        self._sensor = None
        self._width: int | None = None
        self._height: int | None = None
        # Stash legacy kwargs only so callers don't break; the actual config
        # was applied at scene-construction time via env vars.
        self._legacy_kwargs = {
            "width": width,
            "height": height,
            "vertical_fov_deg": vertical_fov_deg,
            "head_body_name": head_body_name,
            "pos_offset": pos_offset,
        }

    def setup(self) -> None:
        """Resolve the registered sensor and log basic config."""
        sensors = getattr(self.simulator.scene, "sensors", None)
        if sensors is None or FPV_SENSOR_NAME not in sensors:
            raise RuntimeError(
                "FPV TiledCamera not found. Did the IsaacSim scene init run with "
                "HOLOSOMA_FPV_ENABLE=1 and --enable_cameras passed to AppLauncher?"
            )
        self._sensor = sensors[FPV_SENSOR_NAME]
        cfg = self._sensor.cfg
        self._width = int(cfg.width)
        self._height = int(cfg.height)
        logger.info(
            f"[fpv] Resolved scene.sensors['{FPV_SENSOR_NAME}']: "
            f"prim_path={cfg.prim_path!r}, "
            f"resolution=({self._width}x{self._height}), "
            f"record_env_id={self.record_env_id}"
        )

    def capture(self) -> np.ndarray:
        """Read the env_0 RGB tile from the registered TiledCamera sensor.

        ``TiledCamera`` is automatically updated each ``scene.update()``
        (called once per ``IsaacSim.simulate_at_each_physics_step``) because
        IsaacLab marks it as an RTX sensor and the existing render gating in
        that method (``has_rtx_sensors()`` returns True) causes the renderer
        to be ticked. So no manual ``sim.render()`` call is needed.
        """
        if self._sensor is None:
            placeholder = np.zeros((self._height or 480, self._width or 640, 3), dtype=np.uint8)
            self._frames.append(placeholder)
            return placeholder

        rgb = self._sensor.data.output["rgb"]  # (N, H, W, 3) torch.uint8
        if rgb is None or rgb.numel() == 0:
            frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        else:
            frame_t = rgb[self.record_env_id, :, :, :3].detach().to("cpu").contiguous()
            frame = frame_t.numpy().astype(np.uint8, copy=False)

        self._frames.append(frame)
        if len(self._frames) == 1:
            non_zero = bool(frame.any())
            logger.info(
                f"[fpv] First frame: shape={frame.shape}, dtype={frame.dtype}, "
                f"non_zero={non_zero}"
            )
        return frame

    def num_captured(self) -> int:
        return len(self._frames)

    def save(self, output_dir: str, fps: int = 50) -> Path:
        """Write captured frames as PNGs and an mp4. Returns the mp4 path."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not self._frames:
            logger.warning("[fpv] No frames captured; nothing to save")
            return out_dir

        from PIL import Image

        for i, frame in enumerate(self._frames):
            Image.fromarray(frame).save(out_dir / f"fpv_{i:06d}.png")
        logger.info(f"[fpv] Saved {len(self._frames)} PNG frames to {out_dir}")

        mp4_path = out_dir / "fpv_eval.mp4"
        try:
            import imageio.v2 as imageio
            with imageio.get_writer(
                str(mp4_path), fps=int(fps), codec="libx264",
                pixelformat="yuv420p", quality=8,
            ) as writer:
                for frame in self._frames:
                    writer.append_data(frame)
            logger.info(
                f"[fpv] Wrote video {mp4_path} via imageio "
                f"({len(self._frames)} frames @ {fps} fps)"
            )
            return mp4_path
        except Exception as exc:
            logger.warning(f"[fpv] imageio video write failed ({exc}); trying system ffmpeg")

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
            logger.info(f"[fpv] Wrote video {mp4_path} via ffmpeg")
        except Exception as exc:
            logger.warning(f"[fpv] ffmpeg failed ({exc}); PNG frames are still on disk")
        return mp4_path

    def cleanup(self) -> None:
        # Sensor lifecycle is owned by the scene; nothing to detach here.
        self._sensor = None
