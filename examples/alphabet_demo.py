#!/usr/bin/env python3
"""NBTV alphabet demo.

Generate the alphabet A-Z as test frames, run them through the NBTV *encoder* to
produce a televisor signal, then *emulate* the televisor to render that signal
back out as:

  * ``alphabet.wav``  -- the NBTV signal (play this into a real televisor)
  * ``alphabet.mp4``  -- the decoded picture as video
  * ``alphabet.apng`` -- the decoded picture as a looping animated PNG you can
                         embed straight into a README

Run it::

    pip install "nbtvencoder[demo]"     # nbtvencoder + Pillow (for the APNG)
    python examples/alphabet_demo.py

Tweak ``HOLD`` (frames per letter), ``SCALE`` (upscaling) and ``SEED`` below.
"""

from __future__ import annotations

import string

import cv2
import numpy as np
from PIL import Image

from nbtvencoder import NBTVDecoder, NBTVEncoder, write_wav

HOLD = 3  # NBTV frames each letter is held on screen
SCALE = 6  # integer upscaling of the 32x102 picture
SEED = 1  # fix the dithering noise so the demo is reproducible


def make_letter(char: str) -> np.ndarray:
    """Draw one capital letter, sized to the NBTV display aspect (32 x 102)."""
    image = np.zeros((306, 96, 3), dtype=np.uint8)  # 3x the 32x102 display
    (text_w, text_h), _ = cv2.getTextSize(char, cv2.FONT_HERSHEY_SIMPLEX, 3.2, 8)
    origin = ((image.shape[1] - text_w) // 2, (image.shape[0] + text_h) // 2)
    cv2.putText(image, char, origin, cv2.FONT_HERSHEY_SIMPLEX, 3.2, (255, 255, 255), 8, cv2.LINE_AA)
    return image


def main() -> None:
    letters = [make_letter(char) for char in string.ascii_uppercase]

    # 1. Encode: each letter becomes HOLD identical NBTV frames, looped A -> Z.
    encoder = NBTVEncoder(seed=SEED)
    rng = encoder.make_rng()
    samples = np.concatenate(
        [encoder.encode_still(letter, frames=HOLD, rng=rng) for letter in letters]
    )
    write_wav("alphabet.wav", samples, encoder.config.sample_rate)
    print(f"encoded {len(letters)} letters -> alphabet.wav ({samples.size} samples)")

    # 2. Emulate the televisor: decode the signal back into rendered frames.
    decoder = NBTVDecoder()
    frames = decoder.decode_images(samples, scale=SCALE)
    fps = decoder.frame_rate
    height, width = frames[0].shape[:2]
    print(f"decoded {len(frames)} frames ({width}x{height}) at {fps:.1f} fps")

    # 3a. Write the MP4 video.
    writer = cv2.VideoWriter("alphabet.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()
    print("wrote alphabet.mp4")

    # 3b. Write the looping animated PNG (BGR -> RGB for Pillow).
    rgb_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    rgb_frames[0].save(
        "alphabet.apng",
        save_all=True,
        append_images=rgb_frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print("wrote alphabet.apng")


if __name__ == "__main__":
    main()
