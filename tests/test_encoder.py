import numpy as np
import pytest

from _helpers import frame_regions
from nbtvencoder import NBTVConfig, NBTVEncoder, make_rng


def test_samples_per_frame(config):
    assert config.samples_per_line == 102 + 6
    assert config.samples_per_frame == 16 + 32 * (102 + 6)
    assert config.samples_per_frame == 3472


def test_frame_rate(config):
    assert config.frame_rate == pytest.approx(44100 / 3472)
    assert 12.0 < config.frame_rate < 13.0


def test_encode_image_array_shape_and_dtype(gradient_image):
    encoder = NBTVEncoder(seed=1)
    frame = encoder.encode_image_array(gradient_image)
    assert frame.dtype == np.int16
    assert frame.shape == (encoder.samples_per_frame,)


def test_encode_accepts_grayscale_input():
    encoder = NBTVEncoder(seed=1)
    gray = np.full((40, 40), 128, dtype=np.uint8)
    frame = encoder.encode_image_array(gray)
    assert frame.shape == (encoder.samples_per_frame,)


def test_rejects_bad_dimensions():
    encoder = NBTVEncoder(seed=1)
    with pytest.raises(ValueError):
        encoder.encode_image_array(np.zeros((2, 2, 2, 2), dtype=np.uint8))


def test_seed_is_deterministic(gradient_image):
    a = NBTVEncoder(seed=7).encode_image_array(gradient_image)
    b = NBTVEncoder(seed=7).encode_image_array(gradient_image)
    assert np.array_equal(a, b)


def test_different_seeds_differ(gradient_image):
    a = NBTVEncoder(seed=1).encode_image_array(gradient_image)
    b = NBTVEncoder(seed=2).encode_image_array(gradient_image)
    assert not np.array_equal(a, b)


def test_framesync_value_range(config, gradient_image):
    frame = NBTVEncoder(config, seed=3).encode_image_array(gradient_image)
    framesync, _ = frame_regions(config, frame)
    assert framesync.min() >= config.framesync_center - config.framesync_range + 1
    assert framesync.max() <= config.framesync_center


def test_linesync_value_range(config, gradient_image):
    frame = NBTVEncoder(config, seed=3).encode_image_array(gradient_image)
    _, lines = frame_regions(config, frame)
    linesync = np.concatenate([ls for _, ls in lines])
    assert linesync.max() <= -config.linesync_level
    assert linesync.min() >= -config.linesync_level - config.linesync_jitter + 1


def test_black_image_pixels_are_bright(config):
    black = np.zeros((40, 40, 3), dtype=np.uint8)
    frame = NBTVEncoder(config, seed=0).encode_image_array(black)
    _, lines = frame_regions(config, frame)
    pixels = np.concatenate([px for px, _ in lines])
    assert pixels.min() >= config.pixel_scale
    assert pixels.max() <= config.pixel_scale + config.pixel_noise - 1


def test_white_image_pixels_are_dark(config):
    white = np.full((40, 40, 3), 255, dtype=np.uint8)
    frame = NBTVEncoder(config, seed=0).encode_image_array(white)
    _, lines = frame_regions(config, frame)
    pixels = np.concatenate([px for px, _ in lines])
    assert pixels.min() >= 0
    assert pixels.max() <= config.pixel_noise - 1


def test_encode_still_repeats_identical_frame(gradient_image):
    encoder = NBTVEncoder(seed=1)
    out = encoder.encode_still(gradient_image, frames=5)
    spf = encoder.samples_per_frame
    assert out.shape == (5 * spf,)
    blocks = out.reshape(5, spf)
    for i in range(1, 5):
        assert np.array_equal(blocks[0], blocks[i])


def test_encode_still_duration(gradient_image):
    encoder = NBTVEncoder(seed=1)
    out = encoder.encode_still(gradient_image, duration=2.0)
    expected = round(2.0 * encoder.frame_rate)
    assert out.shape == (expected * encoder.samples_per_frame,)


def test_encode_frames_concatenates(gradient_image):
    encoder = NBTVEncoder(seed=1)
    frames = [gradient_image, gradient_image, gradient_image]
    out = encoder.encode_frames(iter(frames))
    assert out.shape == (3 * encoder.samples_per_frame,)


def test_encode_frames_empty_is_empty():
    encoder = NBTVEncoder(seed=1)
    out = encoder.encode_frames(iter([]))
    assert out.shape == (0,)
    assert out.dtype == np.int16


def test_custom_config_geometry():
    config = NBTVConfig(lines=16, dots=50)
    encoder = NBTVEncoder(config, seed=1)
    frame = encoder.encode_image_array(np.zeros((30, 30, 3), dtype=np.uint8))
    assert frame.shape == (config.samples_per_frame,)
    assert config.samples_per_frame == 16 + 16 * (50 + 6)


def test_make_rng_variants():
    gen = np.random.default_rng(123)
    assert make_rng(gen) is gen  # an existing generator passes through
    # An integer is treated as a seed and is reproducible.
    a = make_rng(5).integers(0, 1000, size=10)
    b = make_rng(5).integers(0, 1000, size=10)
    assert np.array_equal(a, b)


def test_explicit_rng_seeds_image(gradient_image):
    encoder = NBTVEncoder()  # no default seed
    a = encoder.encode_image_array(gradient_image, rng=11)
    b = encoder.encode_image_array(gradient_image, rng=11)
    assert np.array_equal(a, b)


def test_noise_can_be_disabled(gradient_image):
    config = NBTVConfig(pixel_noise=0, framesync_range=0, linesync_jitter=0)
    # With no dithering noise the output is independent of the seed.
    a = NBTVEncoder(config, seed=1).encode_image_array(gradient_image)
    b = NBTVEncoder(config, seed=999).encode_image_array(gradient_image)
    assert np.array_equal(a, b)

    framesync, lines = frame_regions(config, a)
    assert np.all(framesync == config.framesync_center)
    linesync = np.concatenate([ls for _, ls in lines])
    assert np.all(linesync == -config.linesync_level)
