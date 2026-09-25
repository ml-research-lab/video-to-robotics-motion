"""Main loop: webcam -> hand tracking -> retargeting -> IK -> MuJoCo sim.

Run with ``python -m webcam_teleop.teleop [device]``, where ``device`` is an
optional camera index (e.g. ``1``) -- the same index the UI's own camera
dropdown lists, useful when the default camera is a placeholder. A window
opens with the webcam feed (hand landmarks overlaid) and the simulated arm
side by side, plus controls: a camera picker, a clutch button, a sensitivity
slider, and reset-view/quit buttons. The clutch can also be toggled with the
**c** key, and quitting with **q** / **Esc**. Left-drag the sim panel to
orbit the camera; right-drag or scroll to zoom.
"""

from __future__ import annotations

import argparse
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
from webcam_teleop.ui import TeleopUI, list_camera_devices


def draw_landmarks(frame_rgb: np.ndarray, landmarks: Landmarks | None) -> np.ndarray:
    if landmarks is None:
        return frame_rgb
    frame = frame_rgb.copy()
    h, w = frame.shape[:2]
    pts = (landmarks.image[:, :2] * np.array([w, h])).astype(int)
    for x, y in pts:
        cv2.circle(frame, (int(x), int(y)), 3, (60, 220, 60), -1)
    return frame


def _open_camera(device: int) -> tuple[Webcam, HandTracker]:
    cam = Webcam(device=device)
    tracker = HandTracker(cam.width, cam.height)
    return cam, tracker


def _parse_device_arg() -> int | None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "device", type=int, nargs="?", default=None,
        help="camera device index (e.g. 0, 1, 2); overrides WEBCAM_TELEOP_DEVICE",
    )
    return parser.parse_args().device


def run(device: int | None = None) -> None:
    sim = SO101Sim()
    ik = ArmIK()
    retargeter = HandToGripper(HandConfig())
    q = sim.joint_positions[:N_ARM_JOINTS].copy()

    devices = list_camera_devices() or [0]
    if device is None:
        device = int(os.environ.get("WEBCAM_TELEOP_DEVICE", devices[0]))
    initial_device = device
    cam, tracker = _open_camera(initial_device)

    ui = TeleopUI(
        sim=sim,
        webcam_size=(cam.width, cam.height),
        sim_size=(sim._renderer.width, sim._renderer.height),
        camera_devices=devices,
        initial_device=initial_device,
        initial_gain=retargeter.position_gain,
    )
    print(f"Webcam teleop running (camera device {initial_device}).")

    frame_period = 1.0 / 30.0
    try:
        while ui.is_running():
            frame_start = time.perf_counter()

            if ui.state.camera_device_changed:
                cam.close()
                tracker.close()
                cam, tracker = _open_camera(ui.state.camera_device)
                ui.state.camera_device_changed = False

            frame_rgb = cam.read()
            if frame_rgb is None:
                break
            pose, landmarks = tracker.detect(frame_rgb)

            if retargeter.position_gain != ui.state.gain:
                retargeter.set_gain(ui.state.gain)

            if ui.state.clutch_toggle_requested:
                ui.state.clutch_toggle_requested = False
                if retargeter.engaged:
                    retargeter.disengage()
                elif pose is not None:
                    retargeter.engage(pose, sim.site_position)

            command = retargeter(pose, sim.site_position)
            if command.engaged:
                result = ik.solve(command.position, top_down_frame(command.jaw_azimuth), q)
                q = result.q
                sim.set_target_marker(command.position)

            gripper_target = sim.gripper_target_from_gap(command.jaw_gap)
            sim.set_joint_targets(np.concatenate([q, [gripper_target]]))
            sim.step()

            if ui.state.reset_view_requested:
                sim.reset_camera()
                ui.state.reset_view_requested = False

            ui.update_webcam_frame(draw_landmarks(frame_rgb, landmarks))
            ui.update_sim_frame(sim.render())
            ui.set_status(retargeter.engaged, None if landmarks else tracker.last_rejection)
            ui.poll()

            elapsed = time.perf_counter() - frame_start
            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)
    finally:
        ui.destroy()
        cam.close()
        tracker.close()


if __name__ == "__main__":
    run(device=_parse_device_arg())
