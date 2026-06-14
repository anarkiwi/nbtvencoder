import numpy as np
import pytest

from nbtvencoder import (
    NBTVConfig,
    NBTVDecoder,
    NBTVEncoder,
    apply_scan_geometry,
    find_line_syncs,
    group_frame_syncs,
    restore_scan_geometry,
)


def test_geometry_restore_inverts_apply():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, size=(7, 5)).astype(np.uint8)
    assert np.array_equal(restore_scan_geometry(apply_scan_geometry(x)), x)


def test_find_line_syncs():
    samples = np.array([0, 0, -5000, -5000, -5000, 10, 20, -4500, 5, -3000], dtype=np.int64)
    starts = find_line_syncs(samples, threshold=-2000)
    assert list(starts) == [2, 7, 9]


def test_find_line_syncs_none():
    samples = np.array([0, 100, 200, 6000], dtype=np.int64)
    assert find_line_syncs(samples, threshold=-2000).size == 0


def test_group_frame_syncs_two_frames():
    cfg = NBTVConfig()
    starts = []
    for frame in range(2):
        for line in range(cfg.lines):
            starts.append(
                frame * cfg.samples_per_frame
                + cfg.framesync_samples
                + line * cfg.samples_per_line
                + cfg.dots
            )
    groups = group_frame_syncs(np.array(starts), cfg)
    assert len(groups) == 2
    assert all(len(group) == cfg.lines for group in groups)


def test_decode_grid_matches_encoder_grid(gradient_image):
    encoder = NBTVEncoder(seed=1)
    decoder = NBTVDecoder()
    samples = encoder.encode_image_array(gradient_image)

    grids = decoder.decode_grids(samples)
    assert len(grids) == 1

    expected = encoder._to_nbtv_gray(gradient_image)  # intensity grid the encoder sampled
    assert grids[0].shape == expected.shape
    # Recovered intensity matches within the dithering noise / quantisation.
    assert np.allclose(grids[0], expected, atol=0.05)


def test_decode_multiframe_order_and_levels():
    encoder = NBTVEncoder(seed=2)
    decoder = NBTVDecoder()
    levels = [40, 128, 220]
    frames = [np.full((30, 30, 3), v, dtype=np.uint8) for v in levels]
    samples = encoder.encode_frames(frames)

    grids = decoder.decode_grids(samples)
    assert len(grids) == 3
    means = [float(g.mean()) for g in grids]
    assert means[0] < means[1] < means[2]
    for v, mean in zip(levels, means):
        assert mean == pytest.approx(v / 255, abs=0.05)


def test_render_frame_shape_and_scale(gradient_image):
    encoder = NBTVEncoder(seed=1)
    decoder = NBTVDecoder()
    samples = encoder.encode_image_array(gradient_image)
    grid = decoder.decode_grids(samples)[0]

    image = decoder.render_frame(grid, scale=4)
    # Restored orientation is (dots, lines); upscaled by 4 and expanded to BGR.
    assert image.shape == (102 * 4, 32 * 4, 3)
    assert image.dtype == np.uint8

    unscaled = decoder.render_frame(grid, scale=1)
    assert unscaled.shape == (102, 32, 3)


def test_decode_images_white_is_bright_black_is_dark():
    encoder = NBTVEncoder(seed=1)
    decoder = NBTVDecoder()

    white = encoder.encode_image_array(np.full((20, 20, 3), 255, dtype=np.uint8))
    black = encoder.encode_image_array(np.zeros((20, 20, 3), dtype=np.uint8))

    white_img = decoder.decode_images(white, scale=1)[0]
    black_img = decoder.decode_images(black, scale=1)[0]
    assert white_img.mean() > 200
    assert black_img.mean() < 55


def test_decode_empty_signal():
    decoder = NBTVDecoder()
    assert decoder.decode_grids(np.zeros(100, dtype=np.int16)) == []


def test_custom_threshold():
    decoder = NBTVDecoder(threshold=-1000)
    assert decoder.threshold == -1000
    assert NBTVDecoder().threshold == -NBTVConfig().linesync_level / 2
