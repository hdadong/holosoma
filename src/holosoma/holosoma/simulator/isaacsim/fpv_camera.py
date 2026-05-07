"""IsaacSim first-person-view (FPV) camera capture for evaluation.

Spawns a USD camera as a child of the robot's head_link prim so that it follows
the head pose, then uses omni.replicator.core to read RGB frames each step.

Mirrors the approach used by NVlabs/GR00T-VisualSim2Real (TiledCameraCfg parented
under the robot articulation), but stays within holosoma's lighter replicator-based
recording style and only operates on env 0.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from holosoma.simulator.isaacsim.isaacsim import IsaacSim


def _focal_length_from_fov(vertical_fov_degrees: float, vertical_aperture_mm: float) -> float:
    vertical_fov_rad = math.radians(vertical_fov_degrees)
    return vertical_aperture_mm / (2.0 * math.tan(vertical_fov_rad / 2.0))


class IsaacSimFPVRecorder:
    """Head-mounted FPV camera attached under /World/envs/env_0/Robot/<head_link>.

    Parameters
    ----------
    simulator : IsaacSim
        Live simulator instance (env 0 is captured).
    width, height : int
        Image resolution in pixels.
    vertical_fov_deg : float
        Vertical field of view in degrees.
    head_body_name : str
        Robot body name to attach the camera to (default ``head_link``).
    pos_offset : tuple[float, float, float]
        Camera position offset relative to the parent body, in meters.
        Convention: x=forward, y=left, z=up (matches USD's local frame for the link).
    record_env_id : int
        Environment index to record (default 0).
    """

    def __init__(
        self,
        simulator: "IsaacSim",
        width: int = 640,
        height: int = 480,
        vertical_fov_deg: float = 60.0,
        head_body_name: str = "head_link",
        pos_offset: tuple[float, float, float] = (0.15, 0.0, 0.10),
        record_env_id: int = 0,
    ) -> None:
        self.simulator = simulator
        self.width = int(width)
        self.height = int(height)
        self.vertical_fov_deg = float(vertical_fov_deg)
        self.head_body_name = head_body_name
        self.pos_offset = tuple(pos_offset)
        self.record_env_id = int(record_env_id)

        self._render_product = None
        self._rgb_annotator = None
        self._camera_prim_path: str | None = None
        self._frames: list[np.ndarray] = []

    def setup(self) -> None:
        """Create the USD camera as a child of the head body and wire up replicator.

        Must be called after the IsaacLab scene has spawned the robot articulation,
        and after the simulation app has been started with ``--enable_cameras``.
        """
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()

        # Resolve the head body prim and verify it exists. If the requested body
        # is not present (e.g. older USD), fall back to torso_link.
        candidate_bodies = [self.head_body_name, "torso_link", "pelvis"]
        parent_prim_path = None
        for body in candidate_bodies:
            candidate = f"/World/envs/env_{self.record_env_id}/Robot/{body}"
            if stage.GetPrimAtPath(candidate).IsValid():
                parent_prim_path = candidate
                if body != self.head_body_name:
                    logger.warning(
                        f"[fpv] {self.head_body_name} prim not found, attaching camera to {body} instead"
                    )
                self.head_body_name = body
                break
        if parent_prim_path is None:
            raise RuntimeError(
                f"Cannot find any of {candidate_bodies} under /World/envs/env_{self.record_env_id}/Robot"
            )

        self._camera_prim_path = f"{parent_prim_path}/FPVCamera"

        camera_prim = UsdGeom.Camera.Define(stage, self._camera_prim_path)

        # 35mm-equivalent aperture; pick aspect-matched horizontal aperture so the
        # vertical FOV matches the requested value.
        aspect_ratio = self.width / max(1, self.height)
        vertical_aperture = 24.0
        horizontal_aperture = vertical_aperture * aspect_ratio
        focal_length = _focal_length_from_fov(self.vertical_fov_deg, vertical_aperture)
        camera_prim.GetFocalLengthAttr().Set(focal_length)
        camera_prim.GetClippingRangeAttr().Set((0.05, 1000.0))
        camera_prim.GetHorizontalApertureAttr().Set(horizontal_aperture)
        camera_prim.GetVerticalApertureAttr().Set(vertical_aperture)

        # USD camera convention: -Z forward, +Y up, +X right.
        # Parent body convention (G1 link, ROS REP-103): +X forward, +Y left, +Z up.
        # Rotation R that maps camera basis → parent basis:
        #   camera +x (right) → parent -y (robot's right side)
        #   camera +y (up)    → parent +z
        #   camera +z (back)  → parent -x
        # ⇒ R = [[0,0,-1],[-1,0,0],[0,1,0]], trace=0,
        #   q (w,x,y,z) = (0.5, 0.5, -0.5, -0.5).
        xform = UsdGeom.Xformable(camera_prim)
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(*self.pos_offset))
        rotate_op = xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat)
        rotate_op.Set(Gf.Quatf(0.5, Gf.Vec3f(0.5, -0.5, -0.5)))

        resolution = (self.width, self.height)
        self._render_product = rep.create.render_product(self._camera_prim_path, resolution)
        self._rgb_annotator = rep.AnnotatorRegistry.get_annotator(
            "rgb", device=self.simulator.device, do_array_copy=False
        )
        self._rgb_annotator.attach([self._render_product])

        logger.info(
            f"[fpv] Attached IsaacSim FPV camera at {self._camera_prim_path} "
            f"(resolution={resolution}, vfov={self.vertical_fov_deg}deg, offset={self.pos_offset})"
        )

    def capture(self) -> np.ndarray:
        """Read a single RGB frame.

        Forces ``sim.render()`` first because the holosoma IsaacSim wrapper only
        triggers rendering when there's a GUI / RTX sensor / active video
        recorder — none of which are true in our minimal eval. Without an
        explicit render call the replicator annotator returns empty data.

        Always appends a frame to the buffer (a zero-filled placeholder if the
        renderer hasn't warmed up yet) so the caller can rely on
        ``num_captured() == n_steps``.
        """
        if self._rgb_annotator is None:
            placeholder = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self._frames.append(placeholder)
            return placeholder

        # Force the IsaacLab sim to render so replicator has fresh data.
        try:
            self.simulator.sim.render()
        except Exception as exc:
            logger.warning(f"[fpv] sim.render() failed: {exc}")

        rgb_data = self._rgb_annotator.get_data()
        arr: np.ndarray | None = None
        if isinstance(rgb_data, np.ndarray):
            arr = rgb_data
        elif hasattr(rgb_data, "numpy"):
            try:
                arr = rgb_data.numpy()
            except Exception:
                arr = None
        elif hasattr(rgb_data, "shape") and hasattr(rgb_data, "tobytes"):
            arr = np.frombuffer(rgb_data.tobytes(), dtype=np.uint8).reshape(*rgb_data.shape)

        if arr is None or arr.size == 0:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            if arr.ndim == 3 and arr.shape[-1] >= 3:
                arr = arr[:, :, :3]
            frame = np.ascontiguousarray(arr, dtype=np.uint8)

        self._frames.append(frame)
        if len(self._frames) == 1:
            non_zero = frame.any()
            logger.info(
                f"[fpv] First frame: shape={frame.shape}, dtype={frame.dtype}, "
                f"non_zero={bool(non_zero)}"
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

        # Prefer imageio (bundles its own ffmpeg via imageio-ffmpeg) since the
        # IsaacSim container does not always ship a system ffmpeg.
        try:
            import imageio.v2 as imageio
            with imageio.get_writer(
                str(mp4_path), fps=int(fps), codec="libx264",
                pixelformat="yuv420p", quality=8,
            ) as writer:
                for frame in self._frames:
                    writer.append_data(frame)
            logger.info(f"[fpv] Wrote video {mp4_path} via imageio "
                        f"({len(self._frames)} frames @ {fps} fps)")
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
            logger.info(f"[fpv] Wrote video {mp4_path} via ffmpeg "
                        f"({len(self._frames)} frames @ {fps} fps)")
        except Exception as exc:
            logger.warning(f"[fpv] ffmpeg failed ({exc}); PNG frames are still on disk")
        return mp4_path

    def cleanup(self) -> None:
        if self._rgb_annotator is not None and self._render_product is not None:
            try:
                self._rgb_annotator.detach([self._render_product])
            except Exception as exc:
                logger.warning(f"[fpv] annotator detach failed: {exc}")
        self._rgb_annotator = None
        self._render_product = None
