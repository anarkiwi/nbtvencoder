# nbtvencoder

[![CI](https://github.com/anarkiwi/nbtvencoder/actions/workflows/ci.yml/badge.svg)](https://github.com/anarkiwi/nbtvencoder/actions/workflows/ci.yml)

Encode still images **and video** into an [NBTV](http://www.nbtv.wyenet.co.uk/sound.htm)
(Narrow-Bandwidth Television) signal packaged as a WAV file — and **emulate** a
televisor by decoding that signal back into a PNG or MP4.

NBTV is the modern continuation of John Logie Baird's 32-line mechanical
television format. Playing the generated WAV into an NBTV televisor displays the
picture. This encoder is known to work with the
[Mindsets televisor kit](https://mindsetsonline.co.uk/shop/televisor/).

This is a Python/OpenCV (`cv2`) reimplementation of the original Perl proof of
concept, which is preserved in [`legacy/`](legacy/make-nbtv-still.pl).

## Demo

![NBTV alphabet demo](examples/alphabet.apng)

The alphabet A–Z, run through the **encoder** into an NBTV signal and then
decoded back by the **emulator** — what a televisor would show. Reproduce it:

```bash
pip install "nbtvencoder[demo]"
python examples/alphabet_demo.py     # writes alphabet.wav, alphabet.mp4, alphabet.apng
```

The full pipeline is in [examples/alphabet_demo.py](examples/alphabet_demo.py)
(or the copy/paste shell version, [examples/alphabet_demo.sh](examples/alphabet_demo.sh)):

```bash
# 1. generate the alphabet as test frames, then encode them to one NBTV signal
nbtvencoder letters/*.png -o alphabet.wav --frames 3 --seed 1
# 2. emulate the televisor: render the signal back to video
nbtv-emulate alphabet.wav -o alphabet.mp4 --scale 6
```

## Install

```bash
pip install nbtvencoder
```

This pulls in `numpy` and `opencv-python-headless`. If you already use the full
`opencv-python` (with GUI support), that works too — the encoder only needs
OpenCV's decoding and image-processing routines.

## Command-line usage

```bash
# Encode a still image, held for 10 seconds (the default):
nbtvencoder photo.jpg -o nbtv.wav

# Encode a video — frames are resampled to the NBTV frame rate (~12.7 fps):
nbtvencoder clip.mp4 -o nbtv.wav

# Hold a still for 30 seconds:
nbtvencoder photo.jpg -d 30 -o nbtv.wav

# Combine several inputs into one signal (a slideshow / playlist):
nbtvencoder intro.png clip.mp4 outro.png -o show.wav

# Reproducible output (fixes the dithering noise):
nbtvencoder photo.jpg --seed 1 -o nbtv.wav
```

Image vs. video is detected from the file extension; pass `--video` to force
every input through the video decoder. Run `nbtvencoder --help` for the full
option list (geometry, sample rate, channels, frame count, …).

You can also run it as a module:

```bash
python -m nbtvencoder photo.jpg -o nbtv.wav
```

## Emulator (signal → PNG / MP4)

`nbtv-emulate` decodes an NBTV WAV the way a televisor would and renders the
picture to an image or a video. The output format is chosen from the extension:

```bash
# Render the first frame of a signal to a PNG still:
nbtv-emulate nbtv.wav -o frame.png

# Render the whole signal to an MP4 at the NBTV frame rate:
nbtv-emulate nbtv.wav -o out.mp4

# Pick a frame, and upscale the chunky 32-line picture more:
nbtv-emulate nbtv.wav -o frame.png --frame 10 --scale 12
```

The decoder separates the line/frame sync pulses, slices each scan line, maps
sample levels back to brightness, and undoes the scan geometry — so encoding an
image and emulating it back round-trips to a recognisable picture (at the
format's native 32×102 resolution). Run `nbtv-emulate --help` for all options.

## Python API

```python
from nbtvencoder import NBTVEncoder, NBTVDecoder, encode_to_wav, emulate_wav

# One-liner: encode any mix of images/videos straight to a WAV file.
encode_to_wav(["photo.jpg", "clip.mp4"], "nbtv.wav", seed=1)

# Emulate a televisor: decode the WAV back to a PNG still or an MP4.
emulate_wav("nbtv.wav", "frame.png")
emulate_wav("nbtv.wav", "out.mp4")

# Or work with the samples directly (a NumPy int16 array).
encoder = NBTVEncoder(seed=1)
samples = encoder.encode_still(cv2_image, duration=10.0)   # still image
samples = encoder.encode_frames(frame_iterable)            # any frame source

grids = NBTVDecoder().decode_grids(samples)                # intensity frames back out
```

Key entry points:

| Function / class | Purpose |
| --- | --- |
| `NBTVEncoder` | Core encoder: `encode_image_array`, `encode_still`, `encode_frames`. |
| `NBTVDecoder` | Emulator: `decode_grids`, `render_frame`, `decode_images`. |
| `NBTVConfig` | Frozen dataclass of all timing/level parameters. |
| `encode_file` / `encode_files` | Encode file(s) to a sample array (auto-detects video). |
| `encode_to_wav` | Encode input(s) and write a WAV in one call. |
| `emulate_wav` | Decode a WAV and render a PNG still or a video. |
| `read_wav` / `write_wav` | Read/write a 16-bit PCM WAV as an `int16` array. |
| `iter_video_frames` | Iterate a video's frames resampled to a target fps. |

## How it works

For each frame the encoder emits a 16-bit PCM stream:

```
[ frame sync ][ line 0 dots ][ line sync ] ... [ line 31 dots ][ line sync ]
```

The image is converted to grayscale, flipped, rotated, and resized to
`dots × lines` (102 × 32 by default) — NBTV scans vertically, so each image row
becomes one scan line. Pixel intensity is inverted and scaled into the sample
range, with small dithering noise, exactly as the original encoder did. At
44.1 kHz this produces roughly 12.7 frames per second.

For **video**, the source is resampled to that frame rate so playback runs at
real time; for a **still image** the single frame is repeated to fill the
requested duration.

The **emulator** runs this in reverse: it detects the line-sync pulses (the
strongly negative end-of-line samples), groups them into frames using the wider
gap left by the frame sync, reads the `dots` samples before each sync as one
scan line, maps the levels back to brightness, and applies the inverse of the
scan geometry to produce an upright picture.

## Development

```bash
git clone https://github.com/anarkiwi/nbtvencoder
cd nbtvencoder
pip install -e ".[dev]"

pytest                 # run the test suite
ruff check .           # lint
ruff format .          # format
```

CI runs the linter, the test suite across Python 3.9–3.13, and a package build
on every push and pull request. Dependencies are kept current by Dependabot.

## License

MIT — see [LICENSE](LICENSE).
