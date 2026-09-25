"""Recover a metric 3D hand pose from a single uncalibrated webcam.

MediaPipe reports two landmark sets per hand: normalized image-plane
coordinates (no absolute scale) and "world" landmarks in metres (no absolute
translation, origin at the hand's centre). Combining them recovers an
absolute position: a known rigid segment of the hand (the palm) has a known
metric size, so the ratio between its pixel extent and its metric extent
gives the pinhole scale factor, from which depth follows.

Everything here is built from the rigid palm landmarks (wrist + four
knuckles), which barely move relative to each other regardless of what the
fingers do. That keeps closing the hand to pinch from also dragging the
tracked position sideways. Only the gripper open/close signal reads the
fingertips.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_TIP = 5, 8
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17
N_LANDMARKS = 21

PALM_LANDMARKS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
KNUCKLE_AXIS = (INDEX_MCP, PINKY_MCP)  # rolls with the wrist; drives jaw azimuth
POINTING_AXIS = (WRIST, MIDDLE_MCP)  # direction the hand points


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros(3)


def frame_from_axes(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """Right-handed frame: ``primary`` is exactly x, ``secondary`` informs z."""
    x = unit(primary)
    if np.linalg.norm(x) < 1e-9:
        raise ValueError("primary axis is degenerate")
    s = np.asarray(secondary, dtype=float)
    z = s - np.dot(s, x) * x
    if np.linalg.norm(z) < 1e-6:
        raise ValueError("secondary axis is parallel to primary")
    z = unit(z)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float) -> "CameraIntrinsics":
        f = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
        return cls(width=width, height=height, fx=f, fy=f, cx=width / 2.0, cy=height / 2.0)

    def unproject(self, u: float, v: float, depth: float) -> np.ndarray:
        return np.array([(u - self.cx) * depth / self.fx, (v - self.cy) * depth / self.fy, depth])


@dataclass(frozen=True)
class Landmarks:
    image: np.ndarray  # (21, 3) normalized [0,1] image-space, z relative
    world: np.ndarray  # (21, 3) metres, origin at hand centre
    handedness: str
    score: float


@dataclass(frozen=True)
class HandPose:
    palm_position: np.ndarray  # metres, camera frame (x right, y down, z forward)
    rotation: np.ndarray  # columns: (pointing, up, knuckles), camera frame
    pinch_distance: float  # metres, thumb tip to index tip
    depth: float
    timestamp: float


def pinch_distance(landmarks: Landmarks) -> float:
    return float(np.linalg.norm(landmarks.world[THUMB_TIP] - landmarks.world[INDEX_TIP]))


def estimate_hand_depth(
    landmarks: Landmarks, intrinsics: CameraIntrinsics, min_pixels: float = 12.0
) -> float | None:
    """Fit one weak-perspective scale across the rigid palm landmarks.

    ``pixels ~= scale * metres``, solved in a least-squares sense over all five
    palm points so a rotation that foreshortens one of them is averaged
    against the others instead of moving the whole depth estimate.
    """
    indices = list(PALM_LANDMARKS)
    scale = np.array([intrinsics.width, intrinsics.height])

    pixels = landmarks.image[indices, :2] * scale
    pixels = pixels - pixels.mean(axis=0)
    metres = landmarks.world[indices, :2]
    metres = metres - metres.mean(axis=0)

    if float(np.linalg.norm(pixels, axis=1).max()) < min_pixels:
        return None
    denom = float(np.sum(metres * metres))
    if denom < 1e-9:
        return None
    s = float(np.sum(pixels * metres) / denom)
    if s <= 1e-6:
        return None
    return float(intrinsics.fx / s)


def palm_frame(world: np.ndarray) -> np.ndarray:
    pointing = world[POINTING_AXIS[1]] - world[POINTING_AXIS[0]]
    knuckles = world[KNUCKLE_AXIS[1]] - world[KNUCKLE_AXIS[0]]
    return frame_from_axes(pointing, knuckles)


def palm_centre(world: np.ndarray) -> np.ndarray:
    return world[list(PALM_LANDMARKS)].mean(axis=0)


def resolve_hand_pose(
    landmarks: Landmarks,
    intrinsics: CameraIntrinsics,
    timestamp: float,
    depth_range: tuple[float, float] = (0.15, 1.2),
) -> tuple[HandPose | None, str | None]:
    """Turn one detection into a metric pose, or say why it could not be resolved."""
    depth = estimate_hand_depth(landmarks, intrinsics)
    if depth is None:
        return None, "hand too small in frame"
    if depth < depth_range[0]:
        return None, "hand too close to the camera"
    if depth > depth_range[1]:
        return None, "hand too far from the camera"

    world = landmarks.world
    try:
        rotation = palm_frame(world)
    except ValueError:
        return None, "palm edge-on to the camera"

    centre_uv = landmarks.image[:, :2].mean(axis=0) * np.array([intrinsics.width, intrinsics.height])
    centre = intrinsics.unproject(centre_uv[0], centre_uv[1], depth)
    return (
        HandPose(
            palm_position=centre + palm_centre(world),
            rotation=rotation,
            pinch_distance=pinch_distance(landmarks),
            depth=depth,
            timestamp=timestamp,
        ),
        None,
    )
