import wave

import numpy as np
import pytest

from nbtvencoder import (
    is_image_path,
    is_video_path,
    read_image,
    resample_frames,
    write_wav,
)


def test_write_wav_roundtrip(tmp_path):
    samples = np.arange(-200, 200, dtype=np.int16)
    path = tmp_path / "out.wav"
    write_wav(path, samples, 44100)

    with wave.open(str(path), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 44100
        assert wav.getnframes() == len(samples)
        data = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    assert np.array_equal(data, samples)


def test_write_wav_casts_non_int16(tmp_path):
    samples = np.array([0, 1000, -1000], dtype=np.int32)
    path = tmp_path / "out.wav"
    write_wav(path, samples, 8000)
    with wave.open(str(path), "rb") as wav:
        assert wav.getframerate() == 8000
        data = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    assert np.array_equal(data, samples.astype(np.int16))


def test_extension_detection():
    assert is_video_path("clip.MP4")
    assert is_video_path("movie.mkv")
    assert not is_video_path("photo.png")
    assert is_image_path("photo.JPG")
    assert not is_image_path("clip.mov")


def test_read_image_missing(tmp_path):
    with pytest.raises(ValueError):
        read_image(tmp_path / "does-not-exist.png")


def test_resample_passthrough_when_equal():
    frames = list(range(10))
    out = list(resample_frames(iter(frames), 30, 30))
    assert out == frames


def test_resample_downsample_reduces_count():
    frames = list(range(100))
    out = list(resample_frames(iter(frames), 30, 12.7))
    assert 0 < len(out) < 100
    assert 30 < len(out) < 55  # ~ 100 * 12.7 / 30
    # Selected source indices must be non-decreasing.
    assert out == sorted(out)


def test_resample_upsample_increases_count():
    frames = list(range(10))
    out = list(resample_frames(iter(frames), 10, 25))
    assert len(out) > 10
    assert set(out) <= set(frames)


def test_resample_handles_unknown_src_fps():
    frames = list(range(5))
    out = list(resample_frames(iter(frames), 0, 12.7))
    assert out == frames


def test_resample_rejects_bad_target():
    with pytest.raises(ValueError):
        list(resample_frames(iter([1, 2, 3]), 30, 0))


def test_resample_empty():
    assert list(resample_frames(iter([]), 30, 12.7)) == []
