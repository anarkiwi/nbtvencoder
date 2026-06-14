"""Encode still images and video into NBTV televisor WAV signals.

NBTV (Narrow-Bandwidth Television) is the Baird-style 32-line mechanical
television format.  This package transcodes ordinary images and video into the
16-bit PCM signal that drives an NBTV televisor, replacing the original Perl /
ImageMagick proof of concept with an OpenCV pipeline.
"""

from ._version import __version__
from .api import emulate_wav, encode_file, encode_files, encode_to_wav
from .emulator import (
    DEFAULT_SCALE,
    NBTVDecoder,
    find_line_syncs,
    group_frame_syncs,
)
from .encoder import (
    DEFAULT_STILL_DURATION,
    NBTVConfig,
    NBTVEncoder,
    make_rng,
)
from .geometry import apply_scan_geometry, restore_scan_geometry
from .media import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    is_image_path,
    is_video_path,
    iter_video_frames,
    read_image,
    read_wav,
    resample_frames,
    write_video,
    write_wav,
)

__all__ = [
    "__version__",
    "DEFAULT_STILL_DURATION",
    "DEFAULT_SCALE",
    "NBTVConfig",
    "NBTVEncoder",
    "NBTVDecoder",
    "make_rng",
    "encode_file",
    "encode_files",
    "encode_to_wav",
    "emulate_wav",
    "find_line_syncs",
    "group_frame_syncs",
    "apply_scan_geometry",
    "restore_scan_geometry",
    "read_image",
    "read_wav",
    "write_wav",
    "write_video",
    "iter_video_frames",
    "resample_frames",
    "is_image_path",
    "is_video_path",
    "IMAGE_EXTENSIONS",
    "VIDEO_EXTENSIONS",
]
