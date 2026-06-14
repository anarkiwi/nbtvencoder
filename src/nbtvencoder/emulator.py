"""NBTV emulator: decode an NBTV signal back into frames.

This is the inverse of :mod:`nbtvencoder.encoder`.  It recovers the picture an
NBTV televisor would display from the WAV signal by separating sync pulses,
slicing each scan line, mapping sample levels back to intensity, and undoing the
scan geometry.  Decoded frames can be rendered to PNG stills or video.
"""

from __future__ import annotations

import cv2
import numpy as np

from .encoder import NBTVConfig
from .geometry import restore_scan_geometry

DEFAULT_SCALE = 8
"""Default integer upscaling applied when rendering the low-resolution frames."""


def find_line_syncs(samples: np.ndarray, threshold: float) -> np.ndarray:
    """Return the start index of every line-sync pulse.

    A line sync is a run of samples below ``threshold`` (the strongly negative
    end-of-line pulses); only the first sample of each run is reported.
    """
    mask = samples < threshold
    if not mask.any():
        return np.empty(0, dtype=np.int64)
    previous = np.empty_like(mask)
    previous[0] = False
    previous[1:] = mask[:-1]
    return np.flatnonzero(mask & ~previous).astype(np.int64)


def group_frame_syncs(sync_starts: np.ndarray, config: NBTVConfig) -> list:
    """Group line-sync indices into frames of ``config.lines`` lines.

    Frame boundaries are found from the wider gap left by the frame-sync samples.
    Configurations without a frame sync (so every gap is identical) fall back to
    fixed chunking.
    """
    count = len(sync_starts)
    if count < config.lines:
        return []

    gaps = np.diff(sync_starts)
    boundary = config.samples_per_line + max(1, config.framesync_samples // 2)
    frame_starts = [0]
    for index in range(1, count):
        if gaps[index - 1] >= boundary:
            frame_starts.append(index)

    groups = []
    for position, start in enumerate(frame_starts):
        end = frame_starts[position + 1] if position + 1 < len(frame_starts) else count
        groups.append(sync_starts[start:end])

    # Gap detection failed to delimit whole frames -> chunk by line count.
    if any(len(group) != config.lines for group in groups[:-1]) or len(groups[-1]) < config.lines:
        groups = [
            sync_starts[start : start + config.lines]
            for start in range(0, count - config.lines + 1, config.lines)
        ]
    return groups


class NBTVDecoder:
    """Decode and render NBTV signals (the televisor-emulating half)."""

    def __init__(self, config: NBTVConfig | None = None, threshold: float | None = None):
        self.config = config or NBTVConfig()
        self._threshold = threshold

    @property
    def frame_rate(self) -> float:
        return self.config.frame_rate

    @property
    def threshold(self) -> float:
        if self._threshold is not None:
            return self._threshold
        # Halfway between pixel levels (>= 0) and the line-sync level.
        return -self.config.linesync_level / 2.0

    def decode_grids(self, samples: np.ndarray) -> list:
        """Decode samples into a list of ``(lines, dots)`` intensity grids in [0, 1]."""
        cfg = self.config
        samples = np.asarray(samples).astype(np.int64).ravel()
        sync_starts = find_line_syncs(samples, self.threshold)
        grids = []
        for group in group_frame_syncs(sync_starts, cfg):
            group = group[: cfg.lines]
            if len(group) < cfg.lines or group[0] - cfg.dots < 0:
                continue
            # Each line's pixels are the `dots` samples immediately before its sync.
            line_pixels = np.stack([samples[start - cfg.dots : start] for start in group])
            intensity = 1.0 - np.clip(line_pixels, 0, cfg.pixel_scale) / cfg.pixel_scale
            grids.append(intensity)
        return grids

    def grid_to_gray(self, grid: np.ndarray) -> np.ndarray:
        """Convert one intensity grid to an upright 8-bit grayscale image."""
        gray = np.clip(grid * 255.0, 0, 255).astype(np.uint8)
        return restore_scan_geometry(gray)

    def render_frame(self, grid: np.ndarray, scale: int = DEFAULT_SCALE) -> np.ndarray:
        """Render one intensity grid to an upscaled BGR image."""
        gray = self.grid_to_gray(grid)
        if scale and scale != 1:
            gray = cv2.resize(
                gray,
                (gray.shape[1] * scale, gray.shape[0] * scale),
                interpolation=cv2.INTER_NEAREST,
            )
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def decode_images(self, samples: np.ndarray, scale: int = DEFAULT_SCALE) -> list:
        """Decode samples straight into a list of rendered BGR frames."""
        return [self.render_frame(grid, scale) for grid in self.decode_grids(samples)]
