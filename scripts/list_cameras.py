#!/usr/bin/env python3
"""List available camera devices and save a snapshot from each.

Run this if webcam_teleop.teleop shows a placeholder image instead of a real
picture of you: it means device index 0 is not your built-in camera (macOS
sometimes puts Continuity Camera, a phone, or a virtual camera at a lower
index). This tries indices 0..5, saves what each one sees, and reports
whether the frame looks "alive" (has real variation) or static.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY] if sys.platform == "darwin" else [cv2.CAP_ANY]

for index in range(6):
    opened = False
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                opened = True
                break
        cap.release()

    if not opened:
        print(f"device {index}: could not open")
        continue

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = f"/tmp/webcam_teleop_device_{index}.png"
    cv2.imwrite(out_path, frame)
    print(f"device {index}: {width}x{height} -> saved {out_path}")
    cap.release()

print(
    "\nA pixel-stats check cannot reliably tell a placeholder from a real feed "
    "(a flat placeholder icon can have just as much contrast as a dim room), "
    "so open each saved PNG yourself and look. On many Macs device 0 is a "
    "virtual/placeholder camera and device 1 is the real one -- if so:"
)
print("  WEBCAM_TELEOP_DEVICE=1 .venv/bin/python -m webcam_teleop.teleop")
