import numpy as np

from nbtvencoder import NBTVEncoder, encode_file, encode_files, encode_to_wav


def test_encode_file_still(image_file):
    encoder = NBTVEncoder(seed=1)
    samples = encode_file(image_file, encoder, duration=1.0)
    assert samples.dtype == np.int16
    expected_frames = round(1.0 * encoder.frame_rate)
    assert samples.size == expected_frames * encoder.samples_per_frame


def test_encode_file_explicit_frames(image_file):
    encoder = NBTVEncoder(seed=1)
    samples = encode_file(image_file, encoder, frames=3)
    assert samples.size == 3 * encoder.samples_per_frame


def test_encode_files_concatenates(image_file):
    encoder = NBTVEncoder(seed=1)
    samples = encode_files([image_file, image_file], encoder, frames=2)
    assert samples.size == 2 * 2 * encoder.samples_per_frame


def test_encode_files_empty_input():
    samples = encode_files([], seed=1)
    assert samples.size == 0


def test_encode_to_wav_writes_file(image_file, tmp_path):
    import wave

    out = tmp_path / "result.wav"
    samples = encode_to_wav(image_file, out, frames=2, seed=1)
    assert out.exists()
    with wave.open(str(out), "rb") as wav:
        assert wav.getframerate() == 44100
        assert wav.getnframes() == samples.size


def test_encode_to_wav_single_path_string(image_file, tmp_path):
    out = tmp_path / "single.wav"
    samples = encode_to_wav(str(image_file), out, frames=1, seed=1)
    assert samples.size > 0
