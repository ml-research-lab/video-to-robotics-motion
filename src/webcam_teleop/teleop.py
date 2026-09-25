"""Main loop: webcam -> hand tracking -> retargeting -> IK -> MuJoCo sim.

Run with ``python -m webcam_teleop.teleop``. One window shows the webcam feed
(with hand landmarks) side by side with the simulated arm. Controls (focus
that window):

  c              toggle the clutch (engage/disengage hand tracking)
  =  / -         increase / decrease the position gain
  drag with mouse (right/sim panel) -- left button: orbit the camera
                                        right button, or scroll: zoom
  q / Esc        quit

If the webcam panel shows a static placeholder rather than a real picture of
you, device index 0 is not your built-in camera (macOS sometimes puts
Continuity Camera or another virtual device first). Run
``scripts/list_cameras.py`` to find the right index, then set
``WEBCAM_TELEOP_DEVICE=<index>`` before running this.
"""

from __future__ import annotations

import os
import time

import cv2
import numpy as np

from webcam_teleop.config import HandConfig
from webcam_teleop.hand_pose import Landmarks
from webcam_teleop.ik import ArmIK, N_ARM_JOINTS, top_down_frame
from webcam_teleop.retarget import HandToGripper
from webcam_teleop.sim import SO101Sim
from webcam_teleop.tracker import HandTracker, Webcam

WINDOW = "webcam teleop (c: clutch, +/-: gain, q: quit)"


def draw_overlay(frame_bgr: np.ndarray, landmarks: Landmarks | None, engaged: bool, gain: float, rejection: str | None) -> np.ndarray:
    frame = frame_bgr.copy()
    h, w = frame.shape[:2]
    if landmarks is not None:
        pts = (landmarks.image[:, :2] * np.array([w, h])).astype(int)
        for x, y in pts:
            cv2.circle(frame, (int(x), int(y)), 3, (60, 220, 60), -1)

    status = "ENGAGED" if engaged else "clutch released"
    color = (60, 220, 60) if engaged else (60, 160, 220)
    cv2.putText(frame, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"gain: {gain:.2f}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    if landmarks is None and rejection:
        cv2.putText(frame, rejection, (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 220), 1)
    return frame


def side_by_side(webcam_bgr: np.ndarray, sim_rgb: np.ndarray) -> np.ndarray:
    """Stack the webcam preview and the sim render at matching heights."""
    sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)
    h = webcam_bgr.shape[0]
    scale = h / sim_bgr.shape[0]
    sim_resized = cv2.resize(sim_bgr, (int(sim_bgr.shape[1] * scale), h))
    return np.hstack([webcam_bgr, sim_resized])


class OrbitMouseHandler:
    """Mouse-driven orbit/zoom for the sim panel of the combined window.

    Left-drag rotates the camera around its look-at point; right-drag or the
    scroll wheel zooms. Only active over the sim panel (to the right of the
    webcam panel), so dragging over the webcam preview does nothing.
    """

    def __init__(self, sim: SO101Sim, panel_boundary_x: int) -> None:
        self.sim = sim
        self.panel_boundary_x = panel_boundary_x
        self._drag_button: int | None = None
        self._last: tuple[int, int] | None = None

    def __call__(self, event: int, x: int, y: int, flags: int, userdata) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and x >= self.panel_boundary_x:
            self._drag_button, self._last = cv2.EVENT_LBUTTONDOWN, (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN and x >= self.panel_boundary_x:
            self._drag_button, self._last = cv2.EVENT_RBUTTONDOWN, (x, y)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self._drag_button, self._last = None, None
        elif event == cv2.EVENT_MOUSEMOVE and self._drag_button is not None and self._last is not None:
            dx, dy = x - self._last[0], y - self._last[1]
            self._last = (x, y)
            if self._drag_button == cv2.EVENT_LBUTTONDOWN:
                self.sim.orbit(d_azimuth=-dx * 0.3, d_elevation=-dy * 0.3)
            else:
                self.sim.zoom(1.0 + dy * 0.005)
        elif event == cv2.EVENT_MOUSEWHEEL and x >= self.panel_boundary_x:
            delta = cv2.getMouseWheelDelta(flags)
            self.sim.zoom(1.0 - np.sign(delta) * 0.1)


def run() -> None:
    sim = SO101Sim()
    ik = ArmIK()
    retargeter = HandToGripper(HandConfig())

    q = sim.joint_positions[:N_ARM_JOINTS].copy()
    device = int(os.environ.get("WEBCAM_TELEOP_DEVICE", "0"))

    with Webcam(device=device) as cam, HandTracker(cam.width, cam.height) as tracker:
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, OrbitMouseHandler(sim, panel_boundary_x=cam.width))
        print(f"Webcam teleop running (camera device {device}). Press 'c' in the window to engage the clutch.")

        frame_period = 1.0 / 30.0
        while True:
            frame_start = time.perf_counter()
            frame_rgb = cam.read()
            if frame_rgb is None:
                break
            pose, landmarks = tracker.detect(frame_rgb)

            site_pos = sim.site_position
            command = retargeter(pose, site_pos)

            if command.engaged:
                target_rotation = top_down_frame(command.jaw_azimuth)
                result = ik.solve(command.position, target_rotation, q)
                q = result.q
                sim.set_target_marker(command.position)

            gripper_target = sim.gripper_target_from_gap(command.jaw_gap)
            sim.set_joint_targets(np.concatenate([q, [gripper_target]]))
            sim.step()

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            overlay = draw_overlay(frame_bgr, landmarks, retargeter.engaged, retargeter.position_gain, tracker.last_rejection)
            combined = side_by_side(overlay, sim.render())
            cv2.imshow(WINDOW, combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                if retargeter.engaged:
                    retargeter.disengage()
                elif pose is not None:
                    retargeter.engage(pose, site_pos)
            elif key == ord("="):
                retargeter.adjust_gain(+0.1)
            elif key == ord("-"):
                retargeter.adjust_gain(-0.1)

            elapsed = time.perf_counter() - frame_start
            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
