"""MuJoCo simulation of the SO-101 arm, driven by joint position targets."""

from __future__ import annotations

import mujoco
import numpy as np

from webcam_teleop.ik import GRIPPER_JOINT_INDEX, N_ARM_JOINTS
from webcam_teleop.paths import SO101_SCENE_XML

N_JOINTS = N_ARM_JOINTS + 1  # arm joints plus the gripper

_DEFAULT_CAMERA = dict(lookat=(0.2, 0.0, 0.08), distance=0.7, azimuth=140.0, elevation=-20.0)


class SO101Sim:
    """Thin wrapper around the SO-101 scene: set targets, step, read state."""

    def __init__(self, render_height: int = 480, render_width: int = 480) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(SO101_SCENE_XML))
        self.data = mujoco.MjData(self.model)
        home_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(self.model, self.data, home_id)
        mujoco.mj_forward(self.model, self.data)
        self._site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self._marker_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_marker")
        self.substeps = max(1, round((1.0 / 30.0) / self.model.opt.timestep))
        # Offscreen renderer rather than mujoco.viewer's own window: on macOS
        # that window requires the mjpython launcher, under which no other
        # GUI toolkit (OpenCV, Dear PyGui) can also open a window -- so this
        # is rendered to a plain array and shown as a texture in our own UI.
        self._renderer = mujoco.Renderer(self.model, height=render_height, width=render_width)

        # An orbiting free camera, mouse-driven from the UI, rather than a
        # fixed camera baked into the XML -- see orbit()/zoom() below.
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.reset_camera()

    @property
    def joint_positions(self) -> np.ndarray:
        return self.data.qpos[:N_JOINTS].copy()

    @property
    def site_position(self) -> np.ndarray:
        return self.data.site_xpos[self._site_id].copy()

    @property
    def site_rotation(self) -> np.ndarray:
        return self.data.site_xmat[self._site_id].reshape(3, 3).copy()

    def set_target_marker(self, position: np.ndarray) -> None:
        if self._marker_id >= 0:
            self.data.mocap_pos[self.model.body_mocapid[self._marker_id]] = position

    def set_joint_targets(self, q: np.ndarray) -> None:
        """Set actuator position targets (arm joints + gripper), radians."""
        self.data.ctrl[:N_JOINTS] = np.asarray(q, dtype=float)

    def gripper_target_from_gap(self, gap_fraction: float) -> float:
        low, high = self.model.actuator_ctrlrange[GRIPPER_JOINT_INDEX]
        return float(low + np.clip(gap_fraction, 0.0, 1.0) * (high - low))

    def step(self) -> None:
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

    def orbit(self, d_azimuth: float, d_elevation: float) -> None:
        """Rotate the free camera around its look-at point, in degrees."""
        self.camera.azimuth = (self.camera.azimuth + d_azimuth) % 360.0
        self.camera.elevation = float(np.clip(self.camera.elevation + d_elevation, -89.0, 89.0))

    def zoom(self, factor: float) -> None:
        """Scale the free camera's distance from its look-at point."""
        self.camera.distance = float(np.clip(self.camera.distance * factor, 0.15, 3.0))

    def reset_camera(self) -> None:
        """Return the free camera to its default angle and distance."""
        self.camera.lookat = np.array(_DEFAULT_CAMERA["lookat"])
        self.camera.distance = _DEFAULT_CAMERA["distance"]
        self.camera.azimuth = _DEFAULT_CAMERA["azimuth"]
        self.camera.elevation = _DEFAULT_CAMERA["elevation"]

    def render(self) -> np.ndarray:
        """RGB render of the scene from the interactive free camera."""
        self._renderer.update_scene(self.data, camera=self.camera)
        return self._renderer.render()
