"""Core NBTV sample generation.

This module is the faithful port of the original ``make-nbtv-still.pl`` proof of
concept (see ``legacy/``).  It turns 8-bit images into the 16-bit PCM sample
stream understood by an NBTV televisor, using OpenCV (``cv2``) for the image
geometry that ImageMagick used to handle.

The signal for a single frame is laid out as::

    [ framesync ][ line 0 pixels ][ line 0 sync ] ... [ line N pixels ][ line N sync ]

with the same magic numbers as the Perl original so existing televisor kits keep
working.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Union

import cv2
import numpy as np

DEFAULT_STILL_DURATION = 10.0
"""Default number of seconds a still image is held in the output signal."""

RngLike = Union[None, int, np.random.Generator]


def make_rng(rng: RngLike = None, default_seed: int | None = None) -> np.random.Generator:
    """Coerce ``rng`` into a :class:`numpy.random.Generator`.

    ``None`` builds a generator from ``default_seed`` (which itself may be
    ``None`` for a non-deterministic generator), an existing ``Generator`` is
    returned untouched, and anything else is treated as a seed.
    """
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None:
        return np.random.default_rng(default_seed)
    return np.random.default_rng(rng)


def _noise(rng: np.random.Generator, high: int, shape) -> np.ndarray:
    """Uniform integer noise in ``[0, high)``; an all-zero array when disabled."""
    if high <= 0:
        return np.zeros(shape, dtype=np.int64)
    return rng.integers(0, high, size=shape)


def _to_int16(samples: np.ndarray) -> np.ndarray:
    return np.clip(samples, -32768, 32767).astype(np.int16)


@dataclass(frozen=True)
class NBTVConfig:
    """Timing and signal-level parameters for the NBTV format.

    The defaults reproduce the original 32-line Baird-style signal at 44.1 kHz,
    which yields roughly 12.7 frames per second.
    """

    lines: int = 32
    dots: int = 102
    sample_rate: int = 44100
    channels: int = 1
    # Sync / blanking structure.
    framesync_samples: int = 16
    linesync_samples: int = 6
    # Signal levels (matching the legacy encoder).
    pixel_scale: int = 6000
    pixel_noise: int = 200
    framesync_center: int = 200
    framesync_range: int = 400
    linesync_level: int = 4000
    linesync_jitter: int = 1000

    @property
    def samples_per_line(self) -> int:
        """Pixel samples plus the trailing line-sync samples."""
        return self.dots + self.linesync_samples

    @property
    def samples_per_frame(self) -> int:
        """Total samples emitted for one complete frame."""
        return self.framesync_samples + self.lines * self.samples_per_line

    @property
    def frame_rate(self) -> float:
        """Frames per second implied by the sample rate and frame size."""
        return self.sample_rate / self.samples_per_frame


class NBTVEncoder:
    """Encode images and image sequences into NBTV PCM samples."""

    def __init__(self, config: NBTVConfig | None = None, seed: int | None = None):
        self.config = config or NBTVConfig()
        self._default_seed = seed

    # -- convenience pass-throughs -------------------------------------------------
    @property
    def samples_per_frame(self) -> int:
        return self.config.samples_per_frame

    @property
    def frame_rate(self) -> float:
        return self.config.frame_rate

    def make_rng(self, rng: RngLike = None) -> np.random.Generator:
        return make_rng(rng, self._default_seed)

    # -- image geometry ------------------------------------------------------------
    def _to_nbtv_gray(self, image: np.ndarray) -> np.ndarray:
        """Reproduce the legacy gray/flip/rotate/resize pipeline.

        Returns a ``(lines, dots)`` float array of normalised intensity in
        ``[0, 1]``.  NBTV scans vertically, hence the rotation: each row of the
        returned array becomes one vertical scan line of the televisor image.
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 2:
            gray = image
        else:
            raise ValueError("image must be a 2D gray or 3D BGR array")

        gray = cv2.flip(gray, 0)  # ImageMagick Flip()
        gray = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Rotate(-90)
        gray = cv2.resize(gray, (self.config.dots, self.config.lines), interpolation=cv2.INTER_AREA)
        return gray.astype(np.float64) / 255.0

    # -- sample generation ---------------------------------------------------------
    def encode_image_array(self, image: np.ndarray, rng: RngLike = None) -> np.ndarray:
        """Encode a single image array into one frame of ``int16`` samples."""
        cfg = self.config
        rng = self.make_rng(rng)

        intensity = self._to_nbtv_gray(image)
        inverted = 1.0 - intensity  # black -> bright sample, as in the original

        pixels = (inverted * cfg.pixel_scale).astype(np.int64)
        pixels = pixels + _noise(rng, cfg.pixel_noise, pixels.shape)

        linesync = -cfg.linesync_level - _noise(
            rng, cfg.linesync_jitter, (cfg.lines, cfg.linesync_samples)
        )

        line_block = np.concatenate([pixels, linesync], axis=1)
        body = line_block.reshape(-1)

        framesync = cfg.framesync_center - _noise(rng, cfg.framesync_range, cfg.framesync_samples)

        frame = np.concatenate([framesync, body])
        return _to_int16(frame)

    def encode_still(
        self,
        image: np.ndarray,
        duration: float = DEFAULT_STILL_DURATION,
        rng: RngLike = None,
        frames: int | None = None,
    ) -> np.ndarray:
        """Encode a still image, repeating the frame to fill ``duration`` seconds.

        Pass ``frames`` to specify an explicit frame count instead of a duration.
        Like the legacy encoder, the identical frame is repeated.
        """
        frame = self.encode_image_array(image, rng)
        if frames is None:
            frames = max(1, round(duration * self.frame_rate))
        return np.tile(frame, frames)

    def encode_frames(self, frames: Iterable[np.ndarray], rng: RngLike = None) -> np.ndarray:
        """Encode an iterable of image arrays into a continuous sample stream."""
        rng = self.make_rng(rng)
        chunks = [self.encode_image_array(frame, rng) for frame in frames]
        if not chunks:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(chunks)
