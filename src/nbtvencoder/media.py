"""File and frame I/O: reading images/video and writing WAV output."""

from __future__ import annotations

import wave
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Union

import cv2
import numpy as np

PathLike = Union[str, Path]

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
    ".ppm",
    ".pgm",
}
VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".wmv",
    ".flv",
    ".m2v",
    ".ogv",
    ".3gp",
}


def is_video_path(path: PathLike) -> bool:
    """True when ``path`` has a recognised video extension."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def is_image_path(path: PathLike) -> bool:
    """True when ``path`` has a recognised image extension."""
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def read_image(path: PathLike) -> np.ndarray:
    """Read an image file into a BGR array, raising ``ValueError`` on failure."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read image file: {path}")
    return image


def write_wav(
    path: PathLike,
    samples: np.ndarray,
    sample_rate: int,
    channels: int = 1,
) -> None:
    """Write ``int16`` PCM ``samples`` to a WAV file."""
    samples = np.asarray(samples)
    if samples.dtype != np.int16:
        samples = np.clip(samples, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(samples.astype("<i2", copy=False).tobytes())


def resample_frames(frames: Iterable, src_fps: float, target_fps: float) -> Iterator:
    """Resample a stream of frames from ``src_fps`` to ``target_fps``.

    Pure index arithmetic over an arbitrary iterable: a frame is emitted (and may
    be repeated when up-sampling, or dropped when down-sampling) so the output
    plays at ``target_fps`` while preserving real time.
    """
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    if not src_fps or src_fps <= 0:
        src_fps = target_fps

    step = src_fps / target_fps
    out_index = 0
    next_source = 0
    for source_index, frame in enumerate(frames):
        while next_source <= source_index:
            yield frame
            out_index += 1
            next_source = round(out_index * step)


def iter_video_frames(path: PathLike, target_fps: float) -> Iterator[np.ndarray]:
    """Yield BGR frames from a video, resampled to ``target_fps``.

    Raises ``FileNotFoundError`` if the video cannot be opened.  The underlying
    capture is always released, even if the consumer stops early.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise FileNotFoundError(f"could not open video file: {path}")

    src_fps = capture.get(cv2.CAP_PROP_FPS)
    # Guard against unknown (0) or NaN frame rates reported by some containers.
    if not src_fps or src_fps <= 0 or src_fps != src_fps:
        src_fps = target_fps

    def _raw_frames() -> Iterator[np.ndarray]:
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                yield frame
        finally:
            capture.release()

    yield from resample_frames(_raw_frames(), src_fps, target_fps)
