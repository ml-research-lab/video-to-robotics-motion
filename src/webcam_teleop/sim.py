"""Assembles and simulates a scene: floor, a physics ball, and one robot.

The robot is attached into a small base scene via ``mujoco.MjSpec.attach``
rather than baked into a fixed XML, so any :class:`~webcam_teleop.robots.RobotSpec`
can be dropped in -- an arm-with-gripper or a dexterous hand, vendored or
fetched from MuJoCo Menagerie. Every attached name gets a ``"robot_"`` prefix
(MjSpec's own doing), so all lookups here go through that prefix once, and
callers (ik.py, retarget.py, hand_retarget.py) never see it.
"""

from __future__ import annotations

import mujoco
import numpy as np

from webcam_teleop.robots import ROBOT_PREFIX, RobotSpec, attach_robot, load_and_prepare

_BALL_RADIUS = 0.0125


class RobotSim:
    """A floor, a graspable ball, and one attached robot."""

    def __init__(self, spec: RobotSpec, render_height: int = 480, render_width: int = 480) -> None:
        self.spec = spec
        self._build(spec)
        self.data = mujoco.MjData(self.model)
        home_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(self.model, self.data, home_id)
        mujoco.mj_forward(self.model, self.data)

        self.substeps = max(1, round((1.0 / 30.0) / self.model.opt.timestep))
        # Offscreen renderer rather than mujoco.viewer's own window: on macOS
        # that window requires the mjpython launcher, under which no other
        # GUI toolkit (OpenCV, Dear PyGui) can also open a window -- so this
        # is rendered to a plain array and shown as a texture in our own UI.
        self._renderer = mujoco.Renderer(self.model, height=render_height, width=render_width)

        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.reset_camera()

    def _build(self, spec: RobotSpec) -> None:
        robot = load_and_prepare(spec)
        n_joint = sum(j.type != mujoco.mjtJoint.mjJNT_FREE for j in robot.joints)
        n_actuator = len(robot.actuators)

        base = mujoco.MjSpec()
        # Attached first, so the robot's own qpos/qvel occupy the first
        # slots and the ball's free joint comes after: MjSpec assigns qpos
        # in the order bodies are added to worldbody, not the order written
        # here otherwise, and joint_positions()/the home keyframe below both
        # assume the robot comes first.
        attach_robot(base, robot, spec)

        base.worldbody.add_light(pos=[0.3, 0.0, 1.2], dir=[0.0, 0.0, -1.0], diffuse=[0.7, 0.7, 0.7])
        base.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01], rgba=[0.25, 0.28, 0.33, 1.0]
        )

        ball_pos = self._ball_position(spec)
        ball = base.worldbody.add_body(name="ball", pos=ball_pos)
        ball.add_freejoint(name="ball_free")
        ball.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[_BALL_RADIUS, 0, 0], rgba=[0.85, 0.2, 0.16, 1],
            friction=[1.1, 0.02, 0.002], condim=4, mass=0.02,
        )

        home_qpos, home_ctrl = self._home_pose(spec, n_joint, n_actuator)
        base.add_key(name="home", qpos=np.concatenate([home_qpos, ball_pos, [1.0, 0.0, 0.0, 0.0]]), ctrl=home_ctrl)

        self.model = base.compile()

    @staticmethod
    def _ball_position(spec: RobotSpec) -> list[float]:
        if spec.kind == "arm" and spec.arm is not None:
            x = spec.arm.workspace.x[0] + 0.02
            return [x, 0.0, _BALL_RADIUS]
        return [0.15, 0.15, _BALL_RADIUS]

    @staticmethod
    def _home_pose(spec: RobotSpec, n_joint: int, n_actuator: int) -> tuple[np.ndarray, np.ndarray]:
        """Arm/hand qpos and ctrl for the "home" keyframe.

        Any joint beyond the ones a controller commands directly (a gripper's
        secondary linkage joints, coupled to the driven one by an equality
        constraint) is left at zero: those constraints settle it correctly
        within the first few physics steps regardless of its starting value.
        """
        if spec.kind == "hand":
            return np.zeros(n_joint), np.zeros(n_actuator)

        assert spec.arm is not None
        arm_q = np.array(spec.arm.home_qpos)
        # Left at zero regardless of the gripper's actual open/closed pose:
        # the position actuator's ctrl (set every frame in the main loop,
        # engaged or not) drives it to the right place within a few physics
        # steps, and any secondary linkage joint is pulled along by its own
        # equality constraint -- so the exact starting qpos here is only ever
        # visible for a handful of the very first rendered frames.
        extra = np.zeros(n_joint - len(arm_q))
        qpos = np.concatenate([arm_q, extra])
        ctrl = np.concatenate([arm_q, [spec.arm.gripper_open_ctrl]])
        return qpos, ctrl

    # -- generic frame lookups, prefix applied once here -----------------

    def site_position(self, name: str) -> np.ndarray:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ROBOT_PREFIX + name)
        return self.data.site_xpos[site_id].copy()

    def site_rotation(self, name: str) -> np.ndarray:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ROBOT_PREFIX + name)
        return self.data.site_xmat[site_id].reshape(3, 3).copy()

    def body_position(self, name: str) -> np.ndarray:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ROBOT_PREFIX + name)
        return self.data.xpos[body_id].copy()

    @property
    def n_actuators(self) -> int:
        return self.model.nu

    @property
    def joint_positions(self) -> np.ndarray:
        """The robot's own actuated-joint qpos (arm+gripper, or all hand joints)."""
        return self.data.qpos[: self.model.nu].copy()

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        self.data.ctrl[: self.model.nu] = np.asarray(ctrl, dtype=float)

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
        """Return the free camera to the current robot's default view."""
        cam = self.spec.camera
        self.camera.lookat = np.array(cam["lookat"])
        self.camera.distance = cam["distance"]
        self.camera.azimuth = cam["azimuth"]
        self.camera.elevation = cam["elevation"]

    def render(self) -> np.ndarray:
        """RGB render of the scene from the interactive free camera."""
        self._renderer.update_scene(self.data, camera=self.camera)
        return self._renderer.render()
