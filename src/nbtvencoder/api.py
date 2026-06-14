"""High-level helpers that tie image/video input to the encoder and WAV output."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .encoder import DEFAULT_STILL_DURATION, NBTVConfig, NBTVEncoder, RngLike
from .media import PathLike, is_video_path, iter_video_frames, read_image, write_wav


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
