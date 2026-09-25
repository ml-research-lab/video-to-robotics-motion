"""Map a tracked hand onto a gripper command: position, jaw azimuth, jaw gap.

Two design choices make this usable rather than merely plausible:

* **Position is relative, anchored at the clutch.** Monocular depth is the
  weakest signal available, so hand translation is measured from an anchor
  captured when the operator engages the clutch, scaled by a gain, and added
  to wherever the gripper already was. Releasing and re-engaging re-anchors,
  like lifting a mouse. It is taken from the rigid palm, never the
  fingertips, so pinching to grasp does not also drag the arm.
* **The camera-to-robot map is a mirror, not a rotation.** The camera faces
  the operator (reversing rotation sense once) and the preview is mirrored
  (reversing it again). The two cancel, so hand-right maps to robot-right,
  hand-down to robot-down, and hand-towards-the-camera to robot-away, all at
  once, only under a determinant -1 map.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from webcam_teleop.config import HandConfig, WorkspaceBox
from webcam_teleop.filters import AngleFilter, OneEuroFilter
from webcam_teleop.hand_pose import HandPose


def wrap_to_half_turn(angle: float) -> float:
    """Wrap to (-pi/2, pi/2]: a parallel jaw looks the same after a half turn."""
    wrapped = float((angle + np.pi / 2) % np.pi - np.pi / 2)
    return np.pi / 2 if wrapped == -np.pi / 2 else wrapped


@dataclass(frozen=True)
class GripperCommand:
    position: np.ndarray  # metres, robot base frame
    jaw_azimuth: float  # radians, direction the jaws open in the horizontal plane
    jaw_gap: float  # 0..1, fraction of the pinch range
    engaged: bool


class HandToGripper:
    """Stateful hand-to-gripper retargeter with a clutch and smoothing."""

    # Camera axes are (right, down, forward). See module docstring for why
    # this is a mirror rather than a rotation.
    CAMERA_TO_ROBOT = np.array(
        [
            [0.0, 0.0, 1.0],   # camera forward (towards operator) -> robot +x
            [1.0, 0.0, 0.0],   # camera right                      -> robot +y
            [0.0, -1.0, 0.0],  # camera down                       -> robot -z
        ]
    )

    def __init__(self, hand_config: HandConfig | None = None, workspace: WorkspaceBox | None = None) -> None:
        self.hand = hand_config or HandConfig()
        self.workspace = workspace or WorkspaceBox()
        self.position_gain = float(self.hand.position_gain)

        h = self.hand
        self._hand_filter = OneEuroFilter(
            min_cutoff=[h.position_min_cutoff, h.position_min_cutoff, h.depth_min_cutoff],
            beta=[h.position_beta, h.position_beta, h.depth_beta],
            d_cutoff=h.derivative_cutoff,
        )
        self._azimuth_filter = AngleFilter(h.orientation_min_cutoff, h.orientation_beta, h.derivative_cutoff)
        self._gap_filter = OneEuroFilter(h.gripper_min_cutoff, h.gripper_beta, h.derivative_cutoff)

        self._hand_anchor: np.ndarray | None = None
        self._robot_anchor: np.ndarray | None = None
        self._azimuth_hand_anchor: float | None = None
        self._azimuth_robot_anchor: float | None = None
        self._filtered_position: np.ndarray | None = None
        self._filtered_azimuth: float | None = None
        self._filtered_gap: float | None = None
        self._last_hand: np.ndarray | None = None
        self._park_point: np.ndarray | None = None
        self._raw_history: deque[np.ndarray] = deque(maxlen=15)
        self._last_timestamp: float | None = None
        self._last_dt: float = self.hand.nominal_dt
        self._engaged = False
        self.saturated = False

    @property
    def engaged(self) -> bool:
        return self._engaged

    @property
    def tracked_hand(self) -> np.ndarray | None:
        return None if self._last_hand is None else self._last_hand.copy()

    def jaw_azimuth_from_hand(self, pose: HandPose) -> float:
        jaw_camera = pose.rotation[:, 2]
        jaw_robot = self.CAMERA_TO_ROBOT @ jaw_camera
        planar = jaw_robot[:2]
        if np.linalg.norm(planar) < 1e-6:
            return self._filtered_azimuth if self._filtered_azimuth is not None else 0.0
        return wrap_to_half_turn(float(np.arctan2(planar[1], planar[0])))

    def jaw_gap_from_pinch(self, pinch_distance: float) -> float:
        span = self.hand.pinch_open_m - self.hand.pinch_closed_m
        return float(np.clip((pinch_distance - self.hand.pinch_closed_m) / span, 0.0, 1.0))

    def engage(self, pose: HandPose, current_position: np.ndarray) -> None:
        """Anchor the mapping to the operator's current hand and gripper pose."""
        seed = (
            np.mean(self._raw_history, axis=0)
            if len(self._raw_history) >= 3
            else np.asarray(pose.palm_position, dtype=float)
        )
        self._hand_filter.reset()
        self._hand_filter(seed, self.hand.nominal_dt)
        self._last_hand = None
        smoothed = self.track(pose)
        self._hand_anchor = smoothed.copy()

        anchor = self._filtered_position if self._filtered_position is not None else np.asarray(current_position, dtype=float)
        self._robot_anchor = self._clip_workspace(np.asarray(anchor, dtype=float))
        self._filtered_position = self._robot_anchor.copy()

        self._azimuth_hand_anchor = self.jaw_azimuth_from_hand(pose)
        if self._filtered_azimuth is None:
            self._filtered_azimuth = self._azimuth_hand_anchor
        self._azimuth_robot_anchor = float(self._filtered_azimuth)
        self._park_point = smoothed.copy()
        self._engaged = True

    def disengage(self) -> None:
        self._hand_anchor = None
        self._robot_anchor = None
        self._engaged = False

    def adjust_gain(self, delta: float) -> float:
        self.position_gain = float(np.clip(self.position_gain + delta, 0.3, 4.0))
        if self._last_hand is not None and self._filtered_position is not None:
            anchor = self._park_point if self._park_point is not None else self._last_hand
            self._hand_anchor = anchor.copy()
            self._robot_anchor = self._filtered_position.copy()
        return self.position_gain

    def _clip_workspace(self, p: np.ndarray) -> np.ndarray:
        w = self.workspace
        return np.clip(p, [w.x[0], w.y[0], w.z[0]], [w.x[1], w.y[1], w.z[1]])

    def timestep(self, pose: HandPose) -> float:
        dt = self.hand.nominal_dt
        if self._last_timestamp is not None:
            elapsed = pose.timestamp - self._last_timestamp
            if 1e-4 < elapsed < 0.5:
                dt = elapsed
        self._last_timestamp = pose.timestamp
        return dt

    def _plausible(self, position: np.ndarray, dt: float) -> np.ndarray:
        """Clamp a sample to a physically possible hand speed, dropping glitches."""
        if self._last_hand is None:
            return np.asarray(position, dtype=float)
        step = np.asarray(position, dtype=float) - self._last_hand
        distance = float(np.linalg.norm(step))
        limit = self.hand.max_hand_speed * dt
        if distance > limit > 0:
            return self._last_hand + step * (limit / distance)
        return np.asarray(position, dtype=float)

    def track(self, pose: HandPose) -> np.ndarray:
        """Feed one hand pose through the smoothing filter, clutch or not."""
        dt = self.timestep(pose)
        self._last_dt = dt
        self._raw_history.append(np.asarray(pose.palm_position, dtype=float))
        smoothed = self._hand_filter(self._plausible(pose.palm_position, dt), dt)
        self._last_hand = smoothed.copy()
        return smoothed

    def position_from_hand(self, effective_hand: np.ndarray) -> np.ndarray:
        delta = (effective_hand - self._hand_anchor) * self.position_gain
        raw = self._robot_anchor + self.CAMERA_TO_ROBOT @ delta
        clipped = self._clip_workspace(raw)
        self.saturated = bool(np.linalg.norm(clipped - raw) > 1e-9)
        return clipped

    def hold(self, current_position: np.ndarray) -> GripperCommand:
        return GripperCommand(
            position=(
                self._filtered_position.copy()
                if self._filtered_position is not None
                else self._clip_workspace(np.asarray(current_position, dtype=float))
            ),
            jaw_azimuth=self._filtered_azimuth if self._filtered_azimuth is not None else 0.0,
            jaw_gap=self._filtered_gap if self._filtered_gap is not None else 1.0,
            engaged=False,
        )

    def __call__(self, pose: HandPose | None, current_position: np.ndarray) -> GripperCommand:
        """Produce the next command; holds the last one when not engaged."""
        if pose is None:
            return self.hold(current_position)

        smoothed_hand = self.track(pose)
        if not self._engaged or self._hand_anchor is None:
            return self.hold(current_position)

        # Deadband: the commanded point is dragged on a short rope behind the
        # smoothed hand, so residual jitter under the radius moves nothing.
        if self._park_point is None:
            self._park_point = smoothed_hand.copy()
        offset = smoothed_hand - self._park_point
        distance = float(np.linalg.norm(offset))
        radius = self.hand.deadband_radius
        if distance > radius > 0:
            self._park_point = smoothed_hand - offset * (radius / distance)
        effective_hand = self._park_point

        previous = self._filtered_position.copy() if self._filtered_position is not None else None
        target = self.position_from_hand(effective_hand)
        if previous is not None:
            step = target - previous
            step_len = float(np.linalg.norm(step))
            limit = self.hand.max_command_speed * self._last_dt
            if step_len > limit > 0:
                target = previous + step * (limit / step_len)
        self._filtered_position = target

        raw_azimuth = self.jaw_azimuth_from_hand(pose)
        if self._azimuth_hand_anchor is not None and self._azimuth_robot_anchor is not None:
            turned = wrap_to_half_turn(raw_azimuth - self._azimuth_hand_anchor)
            raw_azimuth = wrap_to_half_turn(self._azimuth_robot_anchor + turned)
        self._filtered_azimuth = self._azimuth_filter(raw_azimuth, self._last_dt)
        self._filtered_gap = float(self._gap_filter(self.jaw_gap_from_pinch(pose.pinch_distance), self._last_dt)[0])

        return GripperCommand(
            position=self._filtered_position.copy(),
            jaw_azimuth=float(self._filtered_azimuth),
            jaw_gap=float(self._filtered_gap),
            engaged=True,
        )
