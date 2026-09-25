"""Per-robot-kind controllers: turn tracked hand data into robot ctrl.

teleop.py picks one of these based on the chosen RobotSpec's ``kind`` and
otherwise treats them identically: ``engaged``, ``gain``/``set_gain``,
``toggle_clutch(tracked, sim)``, ``step(tracked, sim) -> ctrl vector``.
"""

from __future__ import annotations

import numpy as np

from webcam_teleop.config import HandConfig
from webcam_teleop.hand_pose import HandPose, Landmarks
from webcam_teleop.hand_retarget import HandToFingers
from webcam_teleop.ik import FrameTarget, MultiFrameIK, ctrl_from_qpos, top_down_frame
from webcam_teleop.retarget import HandToGripper
from webcam_teleop.robots import ROBOT_PREFIX, RobotSpec, mounted_bare_model
from webcam_teleop.sim import RobotSim


class ArmController:
    """Drives an arm+gripper RobotSpec from a tracked hand pose."""

    def __init__(self, spec: RobotSpec, initial_gain: float | None = None) -> None:
        assert spec.kind == "arm" and spec.arm is not None
        self.spec = spec
        model = mounted_bare_model(spec)
        frame = FrameTarget(ROBOT_PREFIX + spec.arm.tcp_site, "site", position_cost=1.0, orientation_cost=0.3)
        self.ik = MultiFrameIK(model, [frame], n_controlled=spec.arm.n_arm_joints)
        self.retargeter = HandToGripper(HandConfig(), workspace=spec.arm.workspace)
        if initial_gain is not None:
            self.retargeter.set_gain(initial_gain)
        self.q = np.array(spec.arm.home_qpos, dtype=float)

    @property
    def engaged(self) -> bool:
        return self.retargeter.engaged

    @property
    def gain(self) -> float:
        return self.retargeter.position_gain

    def set_gain(self, value: float) -> None:
        self.retargeter.set_gain(value)

    def toggle_clutch(self, pose: HandPose | None, sim: RobotSim) -> None:
        if self.retargeter.engaged:
            self.retargeter.disengage()
        elif pose is not None:
            self.retargeter.engage(pose, sim.site_position(self.spec.arm.tcp_site))

    def step(self, pose: HandPose | None, sim: RobotSim) -> np.ndarray:
        site_pos = sim.site_position(self.spec.arm.tcp_site)
        command = self.retargeter(pose, site_pos)
        if command.engaged:
            rotation = top_down_frame(command.jaw_azimuth)
            result = self.ik.solve([(command.position, rotation)], self.q)
            self.q = result.q

        arm = self.spec.arm
        gap = float(np.clip(command.jaw_gap, 0.0, 1.0))
        gripper_ctrl = arm.gripper_closed_ctrl + gap * (arm.gripper_open_ctrl - arm.gripper_closed_ctrl)
        return np.concatenate([self.q, [gripper_ctrl]])


class HandController:
    """Drives a dexterous-hand RobotSpec from tracked MediaPipe landmarks."""

    def __init__(self, spec: RobotSpec, initial_gain: float | None = None) -> None:
        assert spec.kind == "hand" and spec.hand is not None
        self.spec = spec
        self.model = mounted_bare_model(spec)
        self._fingers = list(spec.hand.fingertip_bodies)
        frames = [
            FrameTarget(ROBOT_PREFIX + spec.hand.fingertip_bodies[f], "body", position_cost=1.0, orientation_cost=0.0)
            for f in self._fingers
        ]
        self.ik = MultiFrameIK(self.model, frames, n_controlled=self.model.nv)
        self.retargeter = HandToFingers(spec.hand, position_gain=initial_gain if initial_gain is not None else 1.0)
        self.q = np.zeros(self.model.nv)

    @property
    def engaged(self) -> bool:
        return self.retargeter.engaged

    @property
    def gain(self) -> float:
        return self.retargeter.position_gain

    def set_gain(self, value: float) -> None:
        self.retargeter.set_gain(value)

    def toggle_clutch(self, landmarks: Landmarks | None, sim: RobotSim) -> None:
        if self.retargeter.engaged:
            self.retargeter.disengage()
        elif landmarks is not None:
            self.retargeter.engage()

    def step(self, landmarks: Landmarks | None, sim: RobotSim, dt: float = 1.0 / 30.0) -> np.ndarray:
        command = self.retargeter(landmarks, dt)
        if command.engaged and command.targets:
            targets = [(command.targets[f], None) for f in self._fingers]
            result = self.ik.solve(targets, self.q)
            self.q = result.q
        return ctrl_from_qpos(self.model, self.q, self.spec.hand.tendon_actuators)


def build_controller(spec: RobotSpec, initial_gain: float | None = None):
    if spec.kind == "arm":
        return ArmController(spec, initial_gain)
    return HandController(spec, initial_gain)
