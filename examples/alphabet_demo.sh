#!/usr/bin/env bash
# NBTV alphabet demo -- copy/paste into a terminal.
#
#   pip install "nbtvencoder[demo]"   # nbtvencoder + Pillow (for the APNG)
#   bash examples/alphabet_demo.sh
#
# Encodes the alphabet A-Z into an NBTV signal, then emulates a televisor to
# render it back out. Produces, in the current directory:
#   alphabet.wav   the NBTV signal (play this into a real televisor)
#   alphabet.mp4   the decoded picture as video
#   alphabet.apng  the decoded picture as a looping animated PNG
set -euo pipefail

letters_dir="$(mktemp -d)"
trap 'rm -rf "$letters_dir"' EXIT

# 1. Generate the alphabet A-Z as test frames (sized to the 32x102 NBTV aspect).
python3 - "$letters_dir" <<'PY'
import sys, string, cv2, numpy as np
out = sys.argv[1]
for i, ch in enumerate(string.ascii_uppercase):
    img = np.zeros((306, 96, 3), np.uint8)
    (tw, th), _ = cv2.getTextSize(ch, cv2.FONT_HERSHEY_SIMPLEX, 3.2, 8)
    cv2.putText(img, ch, ((96 - tw) // 2, (306 + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 3.2, (255, 255, 255), 8, cv2.LINE_AA)
    cv2.imwrite(f"{out}/{i:02d}_{ch}.png", img)
print("generated 26 letters")
PY

# 2. Encode the alphabet into one NBTV signal (each letter held for 3 frames).
nbtvencoder "$letters_dir"/*.png -o alphabet.wav --frames 3 --seed 1

# 3. Emulate the televisor: render the signal back to an MP4 video.
nbtv-emulate alphabet.wav -o alphabet.mp4 --scale 6

# 4. ...and to a looping animated PNG you can embed in Markdown.
python3 - alphabet.wav alphabet.apng <<'PY'
import sys, cv2
from PIL import Image
from nbtvencoder import NBTVDecoder, read_wav
wav, apng = sys.argv[1], sys.argv[2]
samples, _ = read_wav(wav)
decoder = NBTVDecoder()
frames = decoder.decode_images(samples, scale=6)
rgb = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
rgb[0].save(apng, save_all=True, append_images=rgb[1:],
            duration=int(1000 / decoder.frame_rate), loop=0)
print(f"wrote {apng} ({len(frames)} frames)")
PY

echo "done -> alphabet.wav, alphabet.mp4, alphabet.apng"
