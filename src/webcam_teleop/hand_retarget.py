"""Map a tracked hand's fingertips onto a dexterous hand's fingertip targets.

Unlike the arm's gripper mapping (retarget.py), this needs no clutch-anchor
step: MediaPipe's "world" landmarks share the camera's axis convention with
an unknown absolute *translation* (origin at the hand's centroid), but a
fingertip's position relative to the wrist cancels that unknown out, leaving
a real, metric offset. So a fingertip's target is simply the robot's own
fixed mount position plus that offset (mirrored and scaled) -- live, with no
anchoring, the moment the clutch is engaged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from webcam_teleop.filters import OneEuroFilter
from webcam_teleop.hand_pose import CAMERA_TO_ROBOT, INDEX_TIP, Landmarks, MIDDLE_TIP, PINKY_TIP, RING_TIP, THUMB_TIP, WRIST
from webcam_teleop.robots import HandSpec

_FINGERTIP_LANDMARKS = {"thumb": THUMB_TIP, "index": INDEX_TIP, "middle": MIDDLE_TIP, "ring": RING_TIP, "pinky": PINKY_TIP}


@dataclass(frozen=True)
class FingerCommand:
    #: Finger name -> target position in world coordinates, one per finger
    #: the robot has (a subset of "thumb"/"index"/"middle"/"ring"/"pinky").
    targets: dict[str, np.ndarray]
    engaged: bool


class HandToFingers:
    """Stateful per-fingertip retargeter with a clutch and smoothing."""

    def __init__(
        self, hand_spec: HandSpec, position_gain: float = 1.0,
        min_cutoff: float = 1.2, beta: float = 0.4, d_cutoff: float = 1.0,
    ) -> None:
        self.hand_spec = hand_spec
        self.mount_pos = np.array(hand_spec.mount_pos, dtype=float)
        self.position_gain = float(position_gain)
        self._fingers = [f for f in _FINGERTIP_LANDMARKS if f in hand_spec.fingertip_bodies]
        self._filters = {f: OneEuroFilter(min_cutoff, beta, d_cutoff) for f in self._fingers}
        self._engaged = False
        self._last_targets: dict[str, np.ndarray] | None = None

    @property
    def engaged(self) -> bool:
        return self._engaged

    def engage(self) -> None:
        self._engaged = True

    def disengage(self) -> None:
        self._engaged = False

    def set_gain(self, value: float) -> float:
        self.position_gain = float(np.clip(value, 0.3, 4.0))
        return self.position_gain

    def hold(self) -> FingerCommand:
        return FingerCommand(targets=self._last_targets or {}, engaged=False)

    def __call__(self, landmarks: Landmarks | None, dt: float) -> FingerCommand:
        """Produce the next command; holds the last one when not engaged."""
        if landmarks is None or not self._engaged:
            return self.hold()

        wrist = landmarks.world[WRIST]
        targets = {}
        for finger in self._fingers:
            rel_camera = landmarks.world[_FINGERTIP_LANDMARKS[finger]] - wrist
            rel_robot = (CAMERA_TO_ROBOT @ rel_camera) * self.hand_spec.scale * self.position_gain
            targets[finger] = self.mount_pos + self._filters[finger](rel_robot, dt)
        self._last_targets = targets
        return FingerCommand(targets=targets, engaged=True)
