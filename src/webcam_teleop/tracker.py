"""Webcam capture and MediaPipe hand tracking."""

from __future__ import annotations

import sys
import time
from typing import Iterator

import numpy as np

from webcam_teleop.config import HandConfig
from webcam_teleop.hand_pose import CameraIntrinsics, HandPose, Landmarks, resolve_hand_pose
from webcam_teleop.paths import HAND_LANDMARKER_TASK


class Webcam:
    """Minimal OpenCV capture that yields mirrored RGB frames.

    Mirrored so that moving your hand right moves it right on screen, and
    landmarks are computed on that mirrored image so the whole pipeline stays
    in one consistent, intuitive frame.

    Opening the camera is retried: on macOS the first attempt is what
    triggers the permission prompt and fails immediately while it's still on
    screen; it succeeds a moment after the user clicks Allow.
    """

    def __init__(
        self, device: int = 0, width: int = 640, height: int = 480, fps: int = 30,
        attempts: int = 4, retry_delay: float = 1.5,
    ) -> None:
        import cv2

        self._cv2 = cv2
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY] if sys.platform == "darwin" else [cv2.CAP_ANY]

        self.capture = None
        for attempt in range(attempts):
            for backend in backends:
                capture = cv2.VideoCapture(device, backend)
                if capture.isOpened():
                    ok, _ = capture.read()
                    if ok:
                        self.capture = capture
                        break
                capture.release()
            if self.capture is not None:
                break
            if attempt < attempts - 1:
                print(f"waiting for camera access (attempt {attempt + 1} of {attempts})...")
                time.sleep(retry_delay)

        if self.capture is None:
            raise RuntimeError(
                f"could not open camera {device}.\n"
                "  On macOS: System Settings > Privacy & Security > Camera, enable\n"
                "  your terminal, then fully quit (Cmd+Q) and reopen it -- macOS only\n"
                "  applies camera permission on relaunch."
            )

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)

        frame = self.read()
        if frame is None:
            raise RuntimeError(f"camera {device} opened but returned no frames")
        self.height, self.width = frame.shape[:2]

    def read(self) -> np.ndarray | None:
        ok, frame_bgr = self.capture.read()
        if not ok:
            return None
        frame_bgr = self._cv2.flip(frame_bgr, 1)
        return self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame

    def __enter__(self) -> "Webcam":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()


class HandTracker:
    """Detects one hand per frame and turns it into a metric :class:`HandPose`."""

    def __init__(self, width: int, height: int, config: HandConfig | None = None) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self.config = config or HandConfig()
        self.hands_seen = 0
        self.last_rejection: str | None = None
        self.intrinsics = CameraIntrinsics.from_hfov(width, height, self.config.assumed_hfov_deg)

        if not HAND_LANDMARKER_TASK.exists():
            raise FileNotFoundError(
                f"hand landmarker model missing at {HAND_LANDMARKER_TASK}. "
                "Run: scripts/fetch_models.sh"
            )

        # CPU delegate: MediaPipe's macOS wheels can abort inside their Metal
        # graph helper when no Metal graph service is available, as is the
        # case in a plain Python process.
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(HAND_LANDMARKER_TASK),
                delegate=mp_python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.config.num_hands,
            min_hand_detection_confidence=self.config.min_detection_confidence,
            min_hand_presence_confidence=self.config.min_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._start = time.perf_counter()
        self._last_timestamp_ms = -1

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def detect(self, frame_rgb: np.ndarray) -> tuple[HandPose | None, Landmarks | None]:
        height, width = frame_rgb.shape[:2]
        if (width, height) != (self.intrinsics.width, self.intrinsics.height):
            self.intrinsics = CameraIntrinsics.from_hfov(width, height, self.config.assumed_hfov_deg)

        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        now = time.perf_counter() - self._start
        timestamp_ms = int(now * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks:
            self.hands_seen = 0
            self.last_rejection = "no hand in frame"
            return None, None

        self.hands_seen = len(result.hand_landmarks)
        # One hand only: whichever the detector ranks first.
        image_lm = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=float)
        world_lm = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[0]], dtype=float)
        handedness = result.handedness[0][0].category_name if result.handedness else "Unknown"
        score = float(result.handedness[0][0].score) if result.handedness else 0.0
        landmarks = Landmarks(image=image_lm, world=world_lm, handedness=handedness, score=score)

        pose, self.last_rejection = resolve_hand_pose(
            landmarks, self.intrinsics, timestamp=now, depth_range=(self.config.depth_min, self.config.depth_max)
        )
        return pose, landmarks
