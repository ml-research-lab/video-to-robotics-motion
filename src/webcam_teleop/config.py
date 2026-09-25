"""Tunable defaults for tracking, retargeting and IK."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HandConfig:
    num_hands: int = 1
    min_detection_confidence: float = 0.6
    min_presence_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    assumed_hfov_deg: float = 65.0
    depth_min: float = 0.15
    depth_max: float = 1.2

    # Position mapping: metres of gripper travel per metre of hand travel.
    position_gain: float = 1.4
    # One-euro filter tuning; the depth axis is far noisier than the other two.
    position_min_cutoff: float = 1.2
    position_beta: float = 0.4
    depth_min_cutoff: float = 0.5
    depth_beta: float = 0.15
    derivative_cutoff: float = 1.0
    orientation_min_cutoff: float = 1.0
    orientation_beta: float = 0.3
    gripper_min_cutoff: float = 1.5
    gripper_beta: float = 0.5

    deadband_radius: float = 0.003  # metres
    max_hand_speed: float = 3.0  # m/s, rejects glitchy detections
    max_command_speed: float = 0.6  # m/s, gripper target speed cap
    nominal_dt: float = 1.0 / 30.0

    pinch_closed_m: float = 0.02
    pinch_open_m: float = 0.12


@dataclass
class WorkspaceBox:
    x: tuple[float, float] = (0.14, 0.30)
    y: tuple[float, float] = (-0.16, 0.16)
    z: tuple[float, float] = (0.02, 0.20)


@dataclass
class GripperConfig:
    min_gap_rad: float = -0.174533
    max_gap_rad: float = 1.745329


@dataclass
class IKConfig:
    position_cost: float = 1.0
    orientation_cost: float = 0.3
    posture_cost: float = 1e-2
    lm_damping: float = 1e-2
    integration_dt: float = 0.05
    iterations: int = 6
    max_joint_step: float = 0.15  # radians per solve, warm-start safety net
    solver: str = "daqp"
