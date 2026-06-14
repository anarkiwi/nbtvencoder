import cv2
import numpy as np
import pytest

from nbtvencoder import emulate_wav, encode_to_wav, write_wav
from nbtvencoder.cli import main as encode_main
from nbtvencoder.emulator_cli import build_parser
from nbtvencoder.emulator_cli import main as emulate_main


def _count_video_frames(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return -1
    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    return count


def test_emulate_wav_to_png(tmp_path):
    white = np.full((40, 40, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "white.png"), white)
    wav = tmp_path / "white.wav"
    encode_to_wav(tmp_path / "white.png", wav, frames=2, seed=1)

    out = tmp_path / "render.png"
    images = emulate_wav(wav, out, scale=4)
    assert out.exists()
    assert len(images) == 2
    rendered = cv2.imread(str(out))
    assert rendered.shape == (102 * 4, 32 * 4, 3)
    assert rendered.mean() > 200  # a white still stays bright


def test_emulate_wav_to_video(tmp_path):
    levels = [30, 150, 240]
    for i, v in enumerate(levels):
        cv2.imwrite(str(tmp_path / f"f{i}.png"), np.full((30, 30, 3), v, dtype=np.uint8))
    wav = tmp_path / "seq.wav"
    encode_to_wav([tmp_path / f"f{i}.png" for i in range(3)], wav, frames=1, seed=1)

    out = tmp_path / "render.avi"
    images = emulate_wav(wav, out, scale=2)
    assert out.exists()
    assert len(images) == 3
    assert _count_video_frames(out) == 3


def test_emulate_wav_to_mp4(tmp_path):
    cv2.imwrite(str(tmp_path / "g.png"), np.full((30, 30, 3), 128, dtype=np.uint8))
    wav = tmp_path / "g.wav"
    encode_to_wav(tmp_path / "g.png", wav, frames=3, seed=1)

    out = tmp_path / "render.mp4"
    try:
        images = emulate_wav(wav, out, scale=2)
    except RuntimeError:
        pytest.skip("no mp4 video writer codec available")
    assert out.exists()
    assert len(images) == 3
    assert _count_video_frames(out) >= 1


def test_emulate_cli_png_default_output(tmp_path):
    cv2.imwrite(str(tmp_path / "src.png"), np.full((40, 40, 3), 255, dtype=np.uint8))
    wav = tmp_path / "clip.wav"
    encode_to_wav(tmp_path / "src.png", wav, frames=1, seed=1)

    rc = emulate_main([str(wav), "--scale", "2", "-q"])
    assert rc == 0
    # default output is <input>.png
    assert (tmp_path / "clip.png").exists()


def test_emulate_cli_video_output(tmp_path, capsys):
    cv2.imwrite(str(tmp_path / "src.png"), np.full((30, 30, 3), 128, dtype=np.uint8))
    wav = tmp_path / "clip.wav"
    encode_to_wav(tmp_path / "src.png", wav, frames=4, seed=1)

    out = tmp_path / "out.avi"
    rc = emulate_main([str(wav), "-o", str(out), "--scale", "2"])
    assert rc == 0
    assert "rendered 4 frames" in capsys.readouterr().out
    assert _count_video_frames(out) == 4


def test_emulate_cli_png_reports_progress(tmp_path, capsys):
    cv2.imwrite(str(tmp_path / "src.png"), np.full((40, 40, 3), 255, dtype=np.uint8))
    wav = tmp_path / "clip.wav"
    encode_to_wav(tmp_path / "src.png", wav, frames=2, seed=1)

    rc = emulate_main([str(wav), "-o", str(tmp_path / "f.png"), "--scale", "2"])
    assert rc == 0
    assert "rendered frame 0 of 2" in capsys.readouterr().out


def test_emulate_cli_missing_file(tmp_path, capsys):
    rc = emulate_main([str(tmp_path / "nope.wav")])
    assert rc == 2
    assert "no such file" in capsys.readouterr().err


def test_emulate_cli_signal_without_frames(tmp_path, capsys):
    # A silent WAV has no sync pulses, so no frames can be decoded.
    silent = tmp_path / "silent.wav"
    write_wav(silent, np.zeros(5000, dtype=np.int16), 44100)
    rc = emulate_main([str(silent), "-o", str(tmp_path / "out.png")])
    assert rc == 1
    assert "error" in capsys.readouterr().err


def test_full_roundtrip_via_both_clis(tmp_path):
    # encode an image to a WAV, then emulate it back to a PNG.
    cv2.imwrite(str(tmp_path / "in.png"), np.full((40, 40, 3), 200, dtype=np.uint8))
    wav = tmp_path / "mid.wav"
    assert encode_main([str(tmp_path / "in.png"), "-o", str(wav), "--frames", "2", "-q"]) == 0

    out = tmp_path / "out.png"
    assert emulate_main([str(wav), "-o", str(out), "--scale", "3", "-q"]) == 0
    rendered = cv2.imread(str(out))
    assert rendered is not None
    # mid-grey in, recognisably mid-grey out
    assert 150 < rendered.mean() < 230


def test_emulate_parser_defaults():
    args = build_parser().parse_args(["x.wav"])
    assert args.scale == 8
    assert args.lines == 32
    assert args.dots == 102
    assert args.frame == 0
