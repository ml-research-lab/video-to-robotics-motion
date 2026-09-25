"""Dear PyGui frontend: video panels, camera/clutch/gain controls, orbit camera.

Owns the window and all widgets. The main loop in teleop.py never touches
dearpygui directly -- it feeds frames in via update_*_frame(), reads operator
intent back out via .state, and calls poll() once per iteration to apply
camera-drag input and render one frame.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

from webcam_teleop.sim import SO101Sim


def list_camera_devices(max_index: int = 6) -> list[int]:
    """Indices of camera devices that open and produce at least one frame."""
    backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY] if sys.platform == "darwin" else [cv2.CAP_ANY]
    found = []
    for index in range(max_index):
        for backend in backends:
            cap = cv2.VideoCapture(index, backend)
            ok = cap.isOpened() and cap.read()[0]
            cap.release()
            if ok:
                found.append(index)
                break
    return found


def _to_texture_data(frame_rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """uint8 HxWx3 RGB -> flattened float32 RGBA in [0, 1], resized if needed."""
    if frame_rgb.shape[1] != width or frame_rgb.shape[0] != height:
        frame_rgb = cv2.resize(frame_rgb, (width, height))
    rgba = np.empty((height, width, 4), dtype=np.float32)
    rgba[:, :, :3] = frame_rgb.astype(np.float32) / 255.0
    rgba[:, :, 3] = 1.0
    return rgba.ravel()


@dataclass
class UIState:
    """Operator intent, written only by the UI and read only by the main loop."""

    clutch_toggle_requested: bool = False
    quit_requested: bool = False
    reset_view_requested: bool = False
    gain: float = 1.4
    camera_device: int = 0
    camera_device_changed: bool = False


class TeleopUI:
    def __init__(
        self,
        sim: SO101Sim,
        webcam_size: tuple[int, int],
        sim_size: tuple[int, int],
        camera_devices: list[int],
        initial_device: int,
        initial_gain: float,
    ) -> None:
        self.sim = sim
        self.state = UIState(gain=initial_gain, camera_device=initial_device)
        self._webcam_w, self._webcam_h = webcam_size
        self._sim_w, self._sim_h = sim_size
        self._last_mouse: tuple[float, float] | None = None

        dpg.create_context()
        dpg.create_viewport(
            title="webcam-teleop",
            width=self._webcam_w + self._sim_w + 40,
            height=self._webcam_h + 200,
        )

        blank_webcam = np.zeros((self._webcam_h, self._webcam_w, 4), dtype=np.float32).ravel()
        blank_sim = np.zeros((self._sim_h, self._sim_w, 4), dtype=np.float32).ravel()
        with dpg.texture_registry():
            self._webcam_tex = dpg.add_raw_texture(
                self._webcam_w, self._webcam_h, blank_webcam, format=dpg.mvFormat_Float_rgba
            )
            self._sim_tex = dpg.add_raw_texture(
                self._sim_w, self._sim_h, blank_sim, format=dpg.mvFormat_Float_rgba
            )

        with dpg.window(label="webcam-teleop", tag="main_window", no_close=True):
            with dpg.group(horizontal=True):
                dpg.add_image(self._webcam_tex)
                self._sim_image = dpg.add_image(self._sim_tex)

            with dpg.group(horizontal=True):
                self._status_text = dpg.add_text("clutch released", color=(90, 170, 230))
                dpg.add_spacer(width=20)
                self._tracking_text = dpg.add_text("", color=(220, 90, 90))

            with dpg.group(horizontal=True):
                dpg.add_text("Camera:")
                dpg.add_combo(
                    items=[str(d) for d in camera_devices],
                    default_value=str(initial_device),
                    width=90,
                    callback=self._on_camera_changed,
                )
                dpg.add_spacer(width=20)
                self._clutch_button = dpg.add_button(
                    label="Engage Clutch (C)", callback=self._on_clutch_clicked, width=180
                )
                dpg.add_spacer(width=20)
                dpg.add_button(label="Reset View", callback=self._on_reset_view)
                dpg.add_spacer(width=20)
                dpg.add_button(label="Quit (Q)", callback=self._on_quit)

            with dpg.group(horizontal=True):
                dpg.add_text("Sensitivity:")
                dpg.add_slider_float(
                    default_value=initial_gain, min_value=0.3, max_value=4.0, width=260,
                    callback=self._on_gain_changed,
                )

        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=self._on_wheel)
            dpg.add_key_press_handler(key=dpg.mvKey_C, callback=self._on_clutch_clicked)
            dpg.add_key_press_handler(key=dpg.mvKey_Q, callback=self._on_quit)
            dpg.add_key_press_handler(key=dpg.mvKey_Escape, callback=self._on_quit)

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)

    # -- widget callbacks -----------------------------------------------

    def _on_clutch_clicked(self, *_args) -> None:
        self.state.clutch_toggle_requested = True

    def _on_quit(self, *_args) -> None:
        self.state.quit_requested = True

    def _on_reset_view(self, *_args) -> None:
        self.state.reset_view_requested = True

    def _on_gain_changed(self, _sender, value: float) -> None:
        self.state.gain = float(value)

    def _on_camera_changed(self, _sender, value: str) -> None:
        device = int(value)
        if device != self.state.camera_device:
            self.state.camera_device = device
            self.state.camera_device_changed = True

    def _on_wheel(self, _sender, delta: int) -> None:
        if dpg.is_item_hovered(self._sim_image):
            self.sim.zoom(1.0 - float(np.sign(delta)) * 0.1)

    # -- called once per frame by the main loop --------------------------

    def set_status(self, engaged: bool, rejection: str | None) -> None:
        dpg.set_value(self._status_text, "ENGAGED" if engaged else "clutch released")
        dpg.configure_item(self._status_text, color=(90, 220, 90) if engaged else (90, 170, 230))
        dpg.configure_item(self._clutch_button, label="Disengage Clutch (C)" if engaged else "Engage Clutch (C)")
        dpg.set_value(self._tracking_text, rejection or "")

    def update_webcam_frame(self, frame_rgb: np.ndarray) -> None:
        dpg.set_value(self._webcam_tex, _to_texture_data(frame_rgb, self._webcam_w, self._webcam_h))

    def update_sim_frame(self, frame_rgb: np.ndarray) -> None:
        dpg.set_value(self._sim_tex, _to_texture_data(frame_rgb, self._sim_w, self._sim_h))

    def _poll_camera_drag(self) -> None:
        """Left-drag orbits, right-drag zooms, scoped to the sim panel."""
        x, y = dpg.get_mouse_pos(local=False)
        if self._last_mouse is None:
            self._last_mouse = (x, y)
        dx, dy = x - self._last_mouse[0], y - self._last_mouse[1]
        self._last_mouse = (x, y)

        if not dpg.is_item_hovered(self._sim_image):
            return
        if dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
            self.sim.orbit(d_azimuth=-dx * 0.3, d_elevation=-dy * 0.3)
        elif dpg.is_mouse_button_down(dpg.mvMouseButton_Right):
            self.sim.zoom(1.0 + dy * 0.005)

    def is_running(self) -> bool:
        return dpg.is_dearpygui_running() and not self.state.quit_requested

    def poll(self) -> None:
        """Apply mouse-drag camera input and render one UI frame."""
        self._poll_camera_drag()
        dpg.render_dearpygui_frame()

    def destroy(self) -> None:
        dpg.destroy_context()
