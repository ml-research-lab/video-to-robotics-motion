"""Filesystem locations for vendored assets."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"

SO101_ARM_XML = ASSETS_DIR / "so101" / "so101.xml"

HAND_LANDMARKER_TASK = ASSETS_DIR / "models" / "hand_landmarker.task"
