"""Command-line interface for the NBTV encoder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ._version import __version__
from .api import encode_file
from .encoder import DEFAULT_STILL_DURATION, NBTVConfig, NBTVEncoder
from .media import is_video_path, write_wav


def build_parser() -> argparse.ArgumentParser:
    defaults = NBTVConfig()
    parser = argparse.ArgumentParser(
        prog="nbtvencoder",
        description=(
            "Encode still images and video into an NBTV (Narrow-Bandwidth "
            "Television) WAV signal for a mechanical televisor."
        ),
    )
    parser.add_argument("inputs", nargs="+", help="image or video file(s) to encode")
    parser.add_argument(
        "-o", "--output", default="nbtv.wav", help="output WAV path (default: nbtv.wav)"
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=DEFAULT_STILL_DURATION,
        help="seconds to hold each still image (default: %(default)s)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="explicit frame count per still image (overrides --duration)",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="treat every input as video regardless of file extension",
    )
    parser.add_argument(
        "--lines", type=int, default=defaults.lines, help="scan lines (default: %(default)s)"
    )
    parser.add_argument(
        "--dots", type=int, default=defaults.dots, help="dots per line (default: %(default)s)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=defaults.sample_rate,
        help="output sample rate in Hz (default: %(default)s)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=defaults.channels,
        help="output channel count (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible output")
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress progress output")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = NBTVConfig(
        lines=args.lines,
        dots=args.dots,
        sample_rate=args.sample_rate,
        channels=args.channels,
    )
    encoder = NBTVEncoder(config, seed=args.seed)
    rng = encoder.make_rng()

    chunks = []
    for raw_path in args.inputs:
        path = Path(raw_path)
        if not path.exists():
            print(f"error: no such file: {raw_path}", file=sys.stderr)
            return 2

        as_video = args.video or is_video_path(path)
        if not args.quiet:
            print(f"encoding {'video' if as_video else 'image'}: {raw_path}")

        try:
            chunk = encode_file(
                path,
                encoder,
                duration=args.duration,
                frames=args.frames,
                rng=rng,
                force_video=as_video,
            )
        except (ValueError, FileNotFoundError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        chunks.append(chunk)

    samples = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
    write_wav(args.output, samples, config.sample_rate, config.channels)

    if not args.quiet:
        seconds = len(samples) / config.sample_rate / max(config.channels, 1)
        print(
            f"wrote {args.output}: {len(samples)} samples "
            f"({seconds:.1f}s @ {config.sample_rate} Hz)"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
