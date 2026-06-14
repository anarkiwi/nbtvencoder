"""Encode still images and video into NBTV televisor WAV signals.

NBTV (Narrow-Bandwidth Television) is the Baird-style 32-line mechanical
television format.  This package transcodes ordinary images and video into the
16-bit PCM signal that drives an NBTV televisor, replacing the original Perl /
ImageMagick proof of concept with an OpenCV pipeline.
"""

from ._version import __version__
from .api import encode_file, encode_files, encode_to_wav
from .encoder import (
    DEFAULT_STILL_DURATION,
    NBTVConfig,
    NBTVEncoder,
    make_rng,
)
from .media import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    is_image_path,
    is_video_path,
    iter_video_frames,
    read_image,
    resample_frames,
    write_wav,
)

__all__ = [
    "__version__",
    "DEFAULT_STILL_DURATION",
    "NBTVConfig",
    "NBTVEncoder",
    "make_rng",
    "encode_file",
    "encode_files",
    "encode_to_wav",
    "read_image",
    "write_wav",
    "iter_video_frames",
    "resample_frames",
    "is_image_path",
    "is_video_path",
    "IMAGE_EXTENSIONS",
    "VIDEO_EXTENSIONS",
]
