"""Main loop: webcam -> hand tracking -> retargeting -> IK -> MuJoCo sim.

Run with ``python -m webcam_teleop.teleop [device] [--robot NAME]``, where
``device`` is an optional camera index (e.g. ``1``) and ``--robot`` picks
which robot to drive (``so101`` by default; see ``webcam_teleop.robots.ROBOTS``
for the full list, also shown in the UI's own Robot dropdown -- arms are
driven by hand position + pinch, dexterous hands finger by finger). A window
opens with the webcam feed (hand landmarks overlaid) and the simulated robot
side by side, plus controls: robot and camera pickers, a clutch button, a
sensitivity slider, and reset-view/quit buttons. The clutch can also be
toggled with the **c** key, and quitting with **q** / **Esc**. Left-drag the
sim panel to orbit the camera; right-drag or scroll to zoom.
"""

from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

from webcam_teleop.controllers import build_controller
from webcam_teleop.hand_pose import Landmarks
from webcam_teleop.robots import ROBOTS, get_robot
from webcam_teleop.sim import RobotSim
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "device", type=int, nargs="?", default=None,
        help="camera device index (e.g. 0, 1, 2); overrides WEBCAM_TELEOP_DEVICE",
    )
    parser.add_argument(
        "--robot", type=str, default=None, choices=sorted(ROBOTS),
        help="which robot to drive (default: so101)",
    )
    return parser.parse_args()


def run(device: int | None = None, robot: str | None = None) -> None:
    spec = get_robot(robot)
    sim = RobotSim(spec)
    controller = build_controller(spec)

    devices = list_camera_devices() or [0]
    if device is None:
        device = int(os.environ.get("WEBCAM_TELEOP_DEVICE", devices[0]))
    cam, tracker = _open_camera(device)

    ui = TeleopUI(
        sim=sim,
        webcam_size=(cam.width, cam.height),
        sim_size=(sim._renderer.width, sim._renderer.height),
        camera_devices=devices,
        initial_device=device,
        initial_gain=controller.gain,
        robot_names=sorted(ROBOTS),
        initial_robot=spec.name,
    )
    print(f"Webcam teleop running: robot={spec.name}, camera device {device}.")

    frame_period = 1.0 / 30.0
    try:
        while ui.is_running():
            frame_start = time.perf_counter()

            if ui.state.camera_device_changed:
                cam.close()
                tracker.close()
                cam, tracker = _open_camera(ui.state.camera_device)
                ui.state.camera_device_changed = False

            if ui.state.robot_changed:
                spec = get_robot(ui.state.robot_name)
                sim = RobotSim(spec)
                controller = build_controller(spec, initial_gain=ui.state.gain)
                ui.set_sim(sim)
                ui.state.robot_changed = False

            frame_rgb = cam.read()
            if frame_rgb is None:
                break
            pose, landmarks = tracker.detect(frame_rgb)
            tracked = pose if spec.kind == "arm" else landmarks

            if controller.gain != ui.state.gain:
                controller.set_gain(ui.state.gain)

            if ui.state.clutch_toggle_requested:
                ui.state.clutch_toggle_requested = False
                controller.toggle_clutch(tracked, sim)

            ctrl = controller.step(tracked, sim)
            sim.set_ctrl(ctrl)
            sim.step()

            if ui.state.reset_view_requested:
                sim.reset_camera()
                ui.state.reset_view_requested = False

            ui.update_webcam_frame(draw_landmarks(frame_rgb, landmarks))
            ui.update_sim_frame(sim.render())
            ui.set_status(controller.engaged, None if tracked is not None else tracker.last_rejection)
            ui.poll()

            elapsed = time.perf_counter() - frame_start
            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)
    finally:
        ui.destroy()
        cam.close()
        tracker.close()


if __name__ == "__main__":
    args = _parse_args()
    run(device=args.device, robot=args.robot)
