"""Command-line interface for the NBTV emulator (signal -> PNG/video)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ._version import __version__
from .api import emulate_wav
from .emulator import DEFAULT_SCALE
from .encoder import NBTVConfig
from .media import is_video_path


def build_parser() -> argparse.ArgumentParser:
    defaults = NBTVConfig()
    parser = argparse.ArgumentParser(
        prog="nbtv-emulate",
        description=(
            "Emulate an NBTV televisor: decode an NBTV WAV signal and render it "
            "to a PNG still or a video (MP4/AVI)."
        ),
    )
    parser.add_argument("input", help="NBTV WAV file to decode")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="output file; .png/.jpg -> still, .mp4/.avi -> video (default: <input>.png)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=DEFAULT_SCALE,
        help="integer upscale factor for the rendered frames (default: %(default)s)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="output video frame rate (default: the NBTV frame rate)",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="frame index to render for still output (default: %(default)s)",
    )
    parser.add_argument(
        "--lines", type=int, default=defaults.lines, help="scan lines (default: %(default)s)"
    )
    parser.add_argument(
        "--dots", type=int, default=defaults.dots, help="dots per line (default: %(default)s)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="line-sync detection threshold (default: half the line-sync level)",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress progress output")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"error: no such file: {args.input}", file=sys.stderr)
        return 2

    output = Path(args.output) if args.output else input_path.with_suffix(".png")
    config = NBTVConfig(lines=args.lines, dots=args.dots)

    try:
        images = emulate_wav(
            input_path,
            output,
            fps=args.fps,
            scale=args.scale,
            frame=args.frame,
            threshold=args.threshold,
            config=config,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        height, width = images[0].shape[:2]
        if is_video_path(output):
            print(f"rendered {len(images)} frames to {output} ({width}x{height})")
        else:
            index = max(0, min(args.frame, len(images) - 1))
            print(f"rendered frame {index} of {len(images)} to {output} ({width}x{height})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
