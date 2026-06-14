# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - 2026-06-14

First Python release. Reimplements the original Perl/ImageMagick proof of
concept (`legacy/make-nbtv-still.pl`) as an installable package.

### Added
- `nbtvencoder` Python package using OpenCV (`cv2`) and NumPy.
- Faithful port of the 32-line NBTV still-image signal, preserving the original
  sync structure and signal levels so existing televisor kits keep working.
- **Video support**: encode any OpenCV-readable video, resampled to the NBTV
  frame rate, into a continuous signal.
- **Emulator**: `NBTVDecoder` and the `nbtv-emulate` command decode an NBTV WAV
  back into frames and render them to a PNG still or an MP4/AVI video, inverting
  the sync framing, level mapping, and scan geometry of the encoder.
- `nbtvencoder` command-line tool (and `python -m nbtvencoder`) accepting any
  mix of image and video inputs.
- Public API: `NBTVEncoder`, `NBTVDecoder`, `NBTVConfig`, `encode_file`,
  `encode_files`, `encode_to_wav`, `emulate_wav`, `read_wav`, `write_wav`,
  `write_video`, and frame/geometry helpers.
- Reproducible output via an optional RNG seed.
- pytest test suite, GitHub Actions CI (lint, test matrix, build), and
  Dependabot configuration.
