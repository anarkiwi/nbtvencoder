"""Scan-geometry transforms shared by the encoder and the emulator.

NBTV scans vertically, so the encoder flips and rotates an image before
sampling it; the emulator applies the exact inverse to put a decoded frame back
into its original orientation.
"""

from __future__ import annotations

import cv2
import numpy as np


def apply_scan_geometry(image: np.ndarray) -> np.ndarray:
    """Flip then rotate, matching how the televisor scans (no resizing)."""
    return cv2.rotate(cv2.flip(image, 0), cv2.ROTATE_90_COUNTERCLOCKWISE)


def restore_scan_geometry(image: np.ndarray) -> np.ndarray:
    """Inverse of :func:`apply_scan_geometry`."""
    return cv2.flip(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), 0)
