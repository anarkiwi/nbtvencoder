import cv2
import numpy as np
import pytest

from nbtvencoder import NBTVConfig


@pytest.fixture
def config():
    return NBTVConfig()


@pytest.fixture
def gradient_image():
    """A 64x48 BGR image with a horizontal brightness ramp."""
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    ramp = np.linspace(0, 255, 64, dtype=np.uint8)
    image[:, :] = ramp[None, :, None]
    return image


@pytest.fixture
def image_file(tmp_path, gradient_image):
    path = tmp_path / "gradient.png"
    assert cv2.imwrite(str(path), gradient_image)
    return path
