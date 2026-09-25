"""Differential inverse kinematics for the SO-101 arm.

Runs on the arm-only MJCF (so101.xml), not the full scene, so there are no
extra free joints for the solver to "solve" by moving something else. The
SO-101 has five joints before the gripper, so an arbitrary 6-DoF pose is
generally unreachable; the orientation cost is kept below the position cost
so when the two conflict the gripper goes where it was asked and tilts as
close as it can.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import mink
import mujoco
import numpy as np

from webcam_teleop.config import IKConfig
from webcam_teleop.paths import SO101_ARM_XML

TCP_SITE = "gripperframe"
N_ARM_JOINTS = 5  # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
GRIPPER_JOINT_INDEX = 5


def _orthonormalize(rot: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(rot, dtype=float))
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def top_down_frame(jaw_azimuth: float) -> np.ndarray:
    """Gripper orientation that approaches straight down, jaws at ``jaw_azimuth``."""
    approach = np.array([0.0, 0.0, -1.0])
    jaw = np.array([np.cos(jaw_azimuth), np.sin(jaw_azimuth), 0.0])
    z = jaw - np.dot(jaw, approach) * approach
    z = z / np.linalg.norm(z)
    y = np.cross(z, approach)
    return np.column_stack([approach, y, z])


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray  # 5 arm joint angles, radians
    position_error: float
    orientation_error: float
    ok: bool


class ArmIK:
    """Warm-started differential IK for the SO-101 gripper site."""

    def __init__(self, config: IKConfig | None = None) -> None:
        self.config = config or IKConfig()
        self.model = mujoco.MjModel.from_xml_path(str(SO101_ARM_XML))
        self.data = mujoco.MjData(self.model)
        self._site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
        if self._site_id < 0:
            raise RuntimeError(f"site {TCP_SITE!r} not found in {SO101_ARM_XML}")

        self._configuration = mink.Configuration(self.model)
        self._frame_task = mink.FrameTask(
            TCP_SITE, "site",
            position_cost=self.config.position_cost,
            orientation_cost=self.config.orientation_cost,
            lm_damping=self.config.lm_damping,
        )
        self._posture_task = mink.PostureTask(self.model, cost=self.config.posture_cost)
        # Freeze the gripper finger DOF: the caller commands it directly, and
        # leaving it in the optimization lets the solver "reach" by opening
        # the hand instead of moving the arm.
        self._freeze_task = mink.DofFreezingTask(self.model, list(range(N_ARM_JOINTS, self.model.nv)), gain=1.0)
        self._tasks = [self._frame_task, self._posture_task, self._freeze_task]
        self._limits = [mink.ConfigurationLimit(self.model)]

        joint_range = self.model.jnt_range[:N_ARM_JOINTS]
        control_range = self.model.actuator_ctrlrange[:N_ARM_JOINTS]
        self.joint_low = np.maximum(joint_range[:, 0], control_range[:, 0]).copy()
        self.joint_high = np.minimum(joint_range[:, 1], control_range[:, 1]).copy()

    def solve(self, target_position: np.ndarray, target_rotation: np.ndarray, q_init: np.ndarray) -> IKResult:
        cfg = self.config
        q_init = np.asarray(q_init, dtype=float)
        arm = np.clip(q_init[:N_ARM_JOINTS], self.joint_low, self.joint_high)

        full = np.zeros(self.model.nq)
        full[:N_ARM_JOINTS] = arm
        self._configuration.update(full.copy())
        self._posture_task.set_target(full.copy())

        target = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(_orthonormalize(target_rotation)),
            np.asarray(target_position, dtype=float),
        )
        self._frame_task.set_target(target)

        for _ in range(cfg.iterations):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                velocity = mink.solve_ik(
                    self._configuration, self._tasks, cfg.integration_dt, cfg.solver, 1e-10, limits=self._limits
                )
            if not np.all(np.isfinite(velocity)):
                break
            self._configuration.integrate_inplace(velocity, cfg.integration_dt)

        error = self._frame_task.compute_error(self._configuration)
        position_error = float(np.linalg.norm(error[:3]))
        orientation_error = float(np.linalg.norm(error[3:]))

        solved = np.clip(self._configuration.q[:N_ARM_JOINTS], self.joint_low, self.joint_high)
        step = np.clip(solved - arm, -cfg.max_joint_step, cfg.max_joint_step)
        q = np.clip(arm + step, self.joint_low, self.joint_high)

        return IKResult(
            q=q, position_error=position_error, orientation_error=orientation_error,
            ok=position_error <= 0.02 and orientation_error <= 0.5,
        )
