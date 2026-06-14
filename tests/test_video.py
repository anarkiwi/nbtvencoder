import pytest

from _helpers import write_test_video
from nbtvencoder import NBTVEncoder, encode_file, iter_video_frames


def test_iter_video_frames_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_video_frames(tmp_path / "missing.avi", 12.7))


def test_iter_and_encode_video(tmp_path):
    path = tmp_path / "clip.avi"
    if not write_test_video(path, frames=24, fps=30):
        pytest.skip("no OpenCV video writer codec available")

    encoder = NBTVEncoder(seed=1)
    frames = list(iter_video_frames(path, encoder.frame_rate))
    # 24 source frames at 30 fps down to ~12.7 fps -> noticeably fewer frames.
    assert 1 <= len(frames) < 24

    samples = encoder.encode_frames(iter(frames))
    assert samples.size == len(frames) * encoder.samples_per_frame


def test_encode_file_video_branch(tmp_path):
    path = tmp_path / "clip.avi"
    if not write_test_video(path, frames=18, fps=30):
        pytest.skip("no OpenCV video writer codec available")

    encoder = NBTVEncoder(seed=2)
    samples = encode_file(path, encoder)
    assert samples.size > 0
    assert samples.size % encoder.samples_per_frame == 0
