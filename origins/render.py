"""Render video frames (optionally a frame range) and pipe them to ffmpeg.

    python -m origins.render --start 0 --end 9000 --out build/part0.mp4
    python -m origins.render --still 123.4 --out frame.png
"""
import argparse
import subprocess
import sys
import time
import numpy as np
from PIL import Image

from . import timeline as TL
from . import fx
from . import overlay as O
from . import scenes_cosmic, scenes_life, scenes_human

SCENES = {}
for mod in (scenes_cosmic, scenes_life, scenes_human):
    for p in TL.PHASES:
        if hasattr(mod, p.key):
            SCENES[p.key] = getattr(mod, p.key)

NO_CROSSFADE = {"bigbang", "dying", "asteroid"}
XFADE = 0.5

# Bottom-left darkening behind the title card (internal resolution).
_g = np.clip((fx.YY - 300) / 240, 0, 1) ** 1.3 * (1 - np.clip((fx.XX - 120) / 840, 0, 1) * 0.55)
CARD_SHADE = (_g * 0.62)[..., None].astype(np.float32)
_c = np.clip(1 - np.hypot((fx.XX - 890) / 260, (fx.YY - 50) / 110), 0, 1) ** 1.5
CAL_SHADE = (_c * 0.4)[..., None].astype(np.float32)
_rng = np.random.default_rng(5)
DITHER = (_rng.uniform(-0.5, 0.5, (fx.H, fx.W, 3)) / 255.0).astype(np.float32)


def scene_at(T):
    p = TL.phase_at(T)
    C = SCENES[p.key](T - p.start, p.dur, T)
    i = TL.PHASES.index(p)
    t = T - p.start
    if i > 0 and t < XFADE and p.key not in NO_CROSSFADE:
        prev = TL.PHASES[i - 1]
        a = fx.smooth(t, 0, XFADE)
        C = SCENES[prev.key](T - prev.start, prev.dur, T) * (1 - a) + C * a
    return C, p


def render_frame(T):
    C, p = scene_at(T)
    fin = TL.BY_KEY["finale"]
    ui = O.smooth(T, 6.0, 7.2) * (1 - O.smooth(T, fin.start + 3.5, fin.start + 5.5))
    card = O.card_alpha(T, p)
    C = C * (1 - CARD_SHADE * card) * (1 - CAL_SHADE * ui)
    C = fx.tonemap(C) * fx.VIGNETTE + DITHER
    img = Image.fromarray((np.clip(C, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")
    img = img.resize((O.OW, O.OH), Image.BICUBIC).convert("RGBA")
    if p.key == "intro":
        O.intro_titles(img, T)
    O.title_card(img, T, p)
    O.calendar(img, T, ui)
    O.timeline_bar(img, T, ui, p)
    if p.key == "finale":
        O.finale_titles(img, T)
    return img.convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=int(TL.DURATION * TL.FPS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--still", type=float, default=None)
    ap.add_argument("--crf", type=int, default=18)
    a = ap.parse_args()
    if a.still is not None:
        render_frame(a.still).save(a.out)
        return
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{O.OW}x{O.OH}",
           "-r", str(TL.FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf),
           "-pix_fmt", "yuv420p", a.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for i in range(a.start, a.end):
        img = render_frame(i / TL.FPS)
        proc.stdin.write(img.tobytes())
        if (i - a.start) % 300 == 0:
            el = time.time() - t0
            print(f"[{a.start}-{a.end}] frame {i} ({el:.0f}s)", file=sys.stderr, flush=True)
    proc.stdin.close()
    proc.wait()


if __name__ == "__main__":
    main()
