import wave

import pytest

from _helpers import write_test_video
from nbtvencoder.cli import build_parser, main


def test_cli_encodes_image(image_file, tmp_path):
    out = tmp_path / "out.wav"
    rc = main([str(image_file), "-o", str(out), "--frames", "3", "--seed", "1", "-q"])
    assert rc == 0
    with wave.open(str(out), "rb") as wav:
        assert wav.getframerate() == 44100
        assert wav.getnchannels() == 1
        assert wav.getnframes() == 3 * 3472


def test_cli_reports_progress(image_file, tmp_path, capsys):
    out = tmp_path / "out.wav"
    rc = main([str(image_file), "-o", str(out), "--frames", "1", "--seed", "1"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "encoding image" in captured.out
    assert "wrote" in captured.out


def test_cli_missing_file_returns_2(tmp_path, capsys):
    out = tmp_path / "out.wav"
    rc = main([str(tmp_path / "nope.png"), "-o", str(out)])
    assert rc == 2
    assert "no such file" in capsys.readouterr().err


def test_cli_unreadable_image_returns_1(tmp_path, capsys):
    # A file with an image extension but invalid contents.
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"not really a png")
    out = tmp_path / "out.wav"
    rc = main([str(bad), "-o", str(out)])
    assert rc == 1
    assert "error" in capsys.readouterr().err


def test_cli_custom_geometry(image_file, tmp_path):
    out = tmp_path / "out.wav"
    rc = main(
        [
            str(image_file),
            "-o",
            str(out),
            "--lines",
            "16",
            "--dots",
            "50",
            "--frames",
            "1",
            "--seed",
            "1",
            "-q",
        ]
    )
    assert rc == 0
    with wave.open(str(out), "rb") as wav:
        assert wav.getnframes() == 16 + 16 * (50 + 6)


def test_cli_video_input(tmp_path):
    clip = tmp_path / "clip.avi"
    if not write_test_video(clip, frames=18, fps=30):
        pytest.skip("no OpenCV video writer codec available")
    out = tmp_path / "out.wav"
    rc = main([str(clip), "-o", str(out), "--seed", "1", "-q"])
    assert rc == 0
    with wave.open(str(out), "rb") as wav:
        assert wav.getnframes() % 3472 == 0
        assert wav.getnframes() > 0


def test_cli_force_video_flag_on_image(image_file, tmp_path, capsys):
    out = tmp_path / "out.wav"
    # --video forces the still to be opened as a video, which fails to decode.
    rc = main([str(image_file), "-o", str(out), "--video"])
    captured = capsys.readouterr()
    assert "encoding video" in captured.out
    # cv2 can sometimes read a single image as a 1-frame "video"; accept either
    # a clean encode or a decode error, but never a crash.
    assert rc in (0, 1)


def test_parser_defaults():
    args = build_parser().parse_args(["a.png"])
    assert args.output == "nbtv.wav"
    assert args.lines == 32
    assert args.dots == 102
    assert args.sample_rate == 44100
