"""Differential inverse kinematics over one or more frame targets.

Runs on a bare robot-only model (no scene, no ball), so there are no extra
free joints for the solver to "solve" by moving something else. Used two
ways:

* an arm's gripper site: one frame task, position weighted above orientation
  so when the two conflict the gripper goes where it was asked and tilts as
  close as it can; the gripper's own joint(s) are frozen out since the
  caller commands those directly.
* a hand's fingertips: one frame task per tracked finger, position only
  (orientation_cost=0), nothing frozen -- every joint is solved for.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import mink
import mujoco
import numpy as np

from webcam_teleop.config import IKConfig


def _orthonormalize(rot: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(rot, dtype=float))
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def ctrl_from_qpos(model: mujoco.MjModel, qpos: np.ndarray, tendon_actuators: dict[str, tuple[str, ...]]) -> np.ndarray:
    """Actuator ctrl vector that would hold the joints at ``qpos``.

    For an ordinary position actuator (one joint each) this is just that
    joint's angle. Some hands drive a *tendon* spanning several joints from
    one actuator (Shadow Hand's coupled distal joints) -- ``tendon_actuators``
    names, for each such actuator, the joints summing into it, and the ctrl
    is that sum. Everything else in ``qpos`` beyond what an actuator reaches
    (an unactuated coupled joint, pulled along by its own equality
    constraint) needs no ctrl at all.
    """
    ctrl = np.zeros(model.nu)
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
            joint_id = model.actuator_trnid[i][0]
            ctrl[i] = qpos[model.jnt_qposadr[joint_id]]
        elif name in tendon_actuators:
            ctrl[i] = sum(qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]] for j in tendon_actuators[name])
    return ctrl


def top_down_frame(jaw_azimuth: float) -> np.ndarray:
    """Gripper orientation that approaches straight down, jaws at ``jaw_azimuth``."""
    approach = np.array([0.0, 0.0, -1.0])
    jaw = np.array([np.cos(jaw_azimuth), np.sin(jaw_azimuth), 0.0])
    z = jaw - np.dot(jaw, approach) * approach
    z = z / np.linalg.norm(z)
    y = np.cross(z, approach)
    return np.column_stack([approach, y, z])


@dataclass(frozen=True)
class FrameTarget:
    name: str
    frame_type: str  # "site" | "body"
    position_cost: float = 1.0
    orientation_cost: float = 0.0


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray  # solved angles for the controlled joints, radians
    position_error: float  # of the first frame target
    orientation_error: float
    ok: bool


class MultiFrameIK:
    """Warm-started differential IK over a fixed list of frame targets."""

    def __init__(self, model: mujoco.MjModel, frames: list[FrameTarget], n_controlled: int, config: IKConfig | None = None) -> None:
        self.config = config or IKConfig()
        self.model = model
        self.n_controlled = n_controlled

        self._configuration = mink.Configuration(model)
        self._frame_tasks = [
            mink.FrameTask(
                f.name, f.frame_type,
                position_cost=f.position_cost, orientation_cost=f.orientation_cost,
                lm_damping=self.config.lm_damping,
            )
            for f in frames
        ]
        self._posture_task = mink.PostureTask(model, cost=self.config.posture_cost)
        self._tasks = list(self._frame_tasks) + [self._posture_task]
        if n_controlled < model.nv:
            # Freeze whatever the caller commands directly (an arm's gripper
            # finger DOF): leaving it in the optimization lets the solver
            # "reach" a target by opening the hand instead of moving the arm.
            self._tasks.append(mink.DofFreezingTask(model, list(range(n_controlled, model.nv)), gain=1.0))
        self._limits = [mink.ConfigurationLimit(model)]

        joint_range = model.jnt_range[:n_controlled]
        control_range = model.actuator_ctrlrange[:n_controlled] if model.nu >= n_controlled else joint_range
        self.joint_low = np.maximum(joint_range[:, 0], control_range[:, 0]).copy()
        self.joint_high = np.minimum(joint_range[:, 1], control_range[:, 1]).copy()

    def solve(self, targets: list[tuple[np.ndarray, np.ndarray | None]], q_init: np.ndarray) -> IKResult:
        """One warm-started solve. ``targets`` matches the ``frames`` list order.

        A target's rotation may be ``None`` when its frame task has zero
        orientation cost (fingertips): an identity rotation is used as a
        placeholder since it contributes nothing to the optimization.
        """
        cfg = self.config
        q_init = np.asarray(q_init, dtype=float)
        controlled = np.clip(q_init[: self.n_controlled], self.joint_low, self.joint_high)

        full = np.zeros(self.model.nq)
        full[: self.n_controlled] = controlled
        self._configuration.update(full.copy())
        self._posture_task.set_target(full.copy())

        for task, (position, rotation) in zip(self._frame_tasks, targets):
            rot = np.eye(3) if rotation is None else _orthonormalize(rotation)
            target = mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(rot), np.asarray(position, dtype=float))
            task.set_target(target)

        for _ in range(cfg.iterations):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                velocity = mink.solve_ik(
                    self._configuration, self._tasks, cfg.integration_dt, cfg.solver, 1e-10, limits=self._limits
                )
            if not np.all(np.isfinite(velocity)):
                break
            self._configuration.integrate_inplace(velocity, cfg.integration_dt)

        error = self._frame_tasks[0].compute_error(self._configuration)
        position_error = float(np.linalg.norm(error[:3]))
        orientation_error = float(np.linalg.norm(error[3:]))

        solved = np.clip(self._configuration.q[: self.n_controlled], self.joint_low, self.joint_high)
        step = np.clip(solved - controlled, -cfg.max_joint_step, cfg.max_joint_step)
        q = np.clip(controlled + step, self.joint_low, self.joint_high)

        return IKResult(
            q=q, position_error=position_error, orientation_error=orientation_error,
            ok=position_error <= 0.02 and orientation_error <= 0.5,
        )
