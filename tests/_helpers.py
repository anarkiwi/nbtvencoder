"""Shared test helpers (kept out of conftest so they can be imported directly)."""

import cv2
import numpy as np


def frame_regions(config, frame):
    """Split one encoded frame into (framesync, [(pixels, linesync), ...])."""
    framesync = frame[: config.framesync_samples]
    offset = config.framesync_samples
    lines = []
    for _ in range(config.lines):
        pixels = frame[offset : offset + config.dots]
        offset += config.dots
        linesync = frame[offset : offset + config.linesync_samples]
        offset += config.linesync_samples
        lines.append((pixels, linesync))
    return framesync, lines


def write_test_video(path, frames=20, fps=30, size=(64, 48)):
    """Write a short test video; return False if no codec is available."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        writer.release()
        return False
    for i in range(frames):
        value = (i * 12) % 256
        frame = np.full((size[1], size[0], 3), value, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return True
