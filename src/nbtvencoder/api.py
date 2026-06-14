"""High-level helpers that tie image/video input to the encoder and WAV output."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

from .emulator import DEFAULT_SCALE, NBTVDecoder
from .encoder import DEFAULT_STILL_DURATION, NBTVConfig, NBTVEncoder, RngLike
from .media import (
    PathLike,
    is_video_path,
    iter_video_frames,
    read_image,
    read_wav,
    write_video,
    write_wav,
)


def _resolve_encoder(
    encoder: NBTVEncoder | None, config: NBTVConfig | None, seed: int | None
) -> NBTVEncoder:
    if encoder is not None:
        return encoder
    return NBTVEncoder(config, seed=seed)


def encode_file(
    path: PathLike,
    encoder: NBTVEncoder | None = None,
    *,
    duration: float = DEFAULT_STILL_DURATION,
    frames: int | None = None,
    rng: RngLike = None,
    force_video: bool | None = None,
    config: NBTVConfig | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Encode a single image or video file into NBTV samples.

    Whether the input is treated as video is decided by ``force_video`` when
    given, otherwise by the file extension.  For still images, ``frames`` (if
    given) overrides ``duration``.
    """
    encoder = _resolve_encoder(encoder, config, seed)
    path = Path(path)
    as_video = is_video_path(path) if force_video is None else force_video

    if as_video:
        video_frames = iter_video_frames(path, encoder.frame_rate)
        return encoder.encode_frames(video_frames, rng)
    return encoder.encode_still(read_image(path), duration=duration, frames=frames, rng=rng)


def encode_files(
    paths: Iterable[PathLike],
    encoder: NBTVEncoder | None = None,
    *,
    duration: float = DEFAULT_STILL_DURATION,
    frames: int | None = None,
    rng: RngLike = None,
    force_video: bool | None = None,
    config: NBTVConfig | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Encode several files and concatenate the result into one sample stream."""
    encoder = _resolve_encoder(encoder, config, seed)
    rng = encoder.make_rng(rng)
    chunks = [
        encode_file(
            path,
            encoder,
            duration=duration,
            frames=frames,
            rng=rng,
            force_video=force_video,
        )
        for path in paths
    ]
    chunks = [chunk for chunk in chunks if chunk.size]
    if not chunks:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(chunks)


def encode_to_wav(
    inputs: PathLike | Iterable[PathLike],
    output: PathLike,
    encoder: NBTVEncoder | None = None,
    *,
    duration: float = DEFAULT_STILL_DURATION,
    frames: int | None = None,
    rng: RngLike = None,
    force_video: bool | None = None,
    config: NBTVConfig | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Encode one or more inputs and write the samples to ``output`` as WAV.

    Returns the generated samples.
    """
    encoder = _resolve_encoder(encoder, config, seed)
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    samples = encode_files(
        inputs,
        encoder,
        duration=duration,
        frames=frames,
        rng=rng,
        force_video=force_video,
    )
    write_wav(output, samples, encoder.config.sample_rate, encoder.config.channels)
    return samples


def emulate_wav(
    wav_path: PathLike,
    output: PathLike,
    *,
    fps: float | None = None,
    scale: int = DEFAULT_SCALE,
    frame: int = 0,
    threshold: float | None = None,
    config: NBTVConfig | None = None,
) -> list:
    """Decode an NBTV WAV signal and render it to a PNG still or a video.

    The output format is chosen from ``output``'s extension: a recognised video
    extension produces a video at ``fps`` (defaulting to the NBTV frame rate),
    anything else writes a single still for frame index ``frame``.  The WAV
    header's sample rate is authoritative and overrides ``config.sample_rate``.

    Returns the list of decoded BGR frames.
    """
    samples, sample_rate = read_wav(wav_path)
    base = config or NBTVConfig()
    config = dataclasses.replace(base, sample_rate=sample_rate)

    decoder = NBTVDecoder(config, threshold=threshold)
    images = decoder.decode_images(samples, scale=scale)
    if not images:
        raise ValueError("no NBTV frames found in signal")

    output = Path(output)
    if is_video_path(output):
        write_video(images, output, fps or decoder.frame_rate)
    else:
        index = max(0, min(frame, len(images) - 1))
        if not cv2.imwrite(str(output), images[index]):
            raise ValueError(f"could not write image: {output}")
    return images
