#!/usr/bin/env python3
"""FOLM brand reel renderer.

Renders 1080x1920 / 30fps / 16s frames with Pillow + numpy and pipes them to
ffmpeg (H.264, no audio).

Usage:
    python3 render.py                 # -> output/folm_brand_01.mp4
    python3 render.py --stills DIR    # also export check frames to DIR
"""
import argparse
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(ROOT, "photos")
OUT = os.path.join(ROOT, "output", "folm_brand_01.mp4")

W, H = 1080, 1920
FPS = 30
DURATION = 16.0
N_FRAMES = int(DURATION * FPS)

XFADE = 1.0          # crossfade length (s), centred on each cut point
ZOOM_FROM, ZOOM_TO = 1.00, 1.04

BG = (0xD8, 0xD3, 0xCA)
TEXT_COLOR = (0x5C, 0x56, 0x4C)
TAGLINE = "時を越えて届く、美しいフォルム"

# (file, nominal start, nominal end). Neighbouring scenes overlap by XFADE
# around each boundary.
SCENES = [
    ("DSCF0631.jpeg", 0.0, 3.5),
    ("DSCF0647.jpeg", 3.5, 7.0),
    ("DSCF0666.jpeg", 7.0, 10.5),
    ("DSCF0668.jpeg", 10.5, 12.5),
    (None, 12.5, 16.0),  # end card
]

LOGO_FILE = "folm_logo_2x.png"
LOGO_WIDTH = 440               # px on screen
LOGO_CENTER_Y = 830            # slightly above centre (960)
TEXT_GAP = 190                 # logo bottom -> text top
FONT_SIZE = 43                 # ~4% of 1080
TRACKING = 0.20                # extra letter spacing, in em

LOGO_FADE_START = 12.5
LOGO_FADE_LEN = 1.0
TEXT_FADE_START = LOGO_FADE_START + 0.8
TEXT_FADE_LEN = 1.0

FONT_CANDIDATES = [
    # macOS (Hiragino Mincho)
    "/System/Library/Fonts/ヒラギノ明朝 ProN.ttc",
    "/System/Library/Fonts/ヒラギノ明朝 ProN W3.otf",
    "/System/Library/Fonts/Hiragino Mincho ProN.ttc",
    "/Library/Fonts/ヒラギノ明朝 ProN W3.otf",
    # Linux fallbacks
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Light.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSerifCJK-Regular.ttc",
]


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def find_font():
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    sys.exit("No Mincho/Serif CJK font found; install fonts-noto-cjk.")


def load_photo(name):
    im = Image.open(os.path.join(PHOTOS, name))
    im = ImageOps.exif_transpose(im)  # honour EXIF orientation
    return im.convert("RGB")


class PhotoScene:
    """Centre-cropped cover fit with a very slow linear zoom."""

    def __init__(self, img, t0, t1):
        self.img = img
        self.t0, self.t1 = t0, t1
        sw, sh = img.size
        # source pixels per output pixel at zoom 1.0 (cover fit)
        self.scale = min(sw / W, sh / H)
        self.cx, self.cy = sw / 2.0, sh / 2.0

    def render(self, t):
        p = (t - self.t0) / (self.t1 - self.t0)
        p = min(max(p, 0.0), 1.0)
        z = ZOOM_FROM + (ZOOM_TO - ZOOM_FROM) * p
        s = self.scale / z
        # output (x, y) -> source (a*x + b*y + c, d*x + e*y + f)
        coeffs = (s, 0, self.cx - s * W / 2.0,
                  0, s, self.cy - s * H / 2.0)
        return self.img.transform((W, H), Image.AFFINE, coeffs,
                                  resample=Image.BICUBIC)


class EndCard:
    def __init__(self):
        logo = Image.open(os.path.join(PHOTOS, LOGO_FILE))
        logo = ImageOps.exif_transpose(logo).convert("RGBA")
        alpha = np.asarray(logo.getchannel("A"), dtype=np.float32)
        if alpha.min() > 250:
            # opaque logo on white: derive alpha from darkness, treating
            # near-white (compression noise / off-white paper) as transparent
            lum = np.asarray(logo.convert("L"), dtype=np.float32)
            alpha = np.clip((230.0 - lum) * (255.0 / 200.0), 0, 255)
            ink = np.zeros((*alpha.shape, 3), dtype=np.uint8)
            logo = Image.fromarray(np.dstack([ink, alpha.astype(np.uint8)]),
                                   "RGBA")
        bbox = logo.getchannel("A").point(lambda a: 255 if a > 8 else 0).getbbox()
        logo = logo.crop(bbox)
        lh = round(logo.height * LOGO_WIDTH / logo.width)
        self.logo = logo.resize((LOGO_WIDTH, lh), Image.LANCZOS)
        self.logo_pos = ((W - LOGO_WIDTH) // 2, LOGO_CENTER_Y - lh // 2)

        self.text = self._render_text(self.logo_pos[1] + lh + TEXT_GAP)
        self.bg = Image.new("RGB", (W, H), BG)

    def _render_text(self, top):
        font = ImageFont.truetype(find_font(), FONT_SIZE)
        track = FONT_SIZE * TRACKING
        advances = [font.getlength(c) for c in TAGLINE]
        # trailing tracking excluded so the line is optically centred
        total = sum(advances) + track * (len(TAGLINE) - 1)
        layer = Image.new("RGBA", (W, H), TEXT_COLOR + (0,))
        draw = ImageDraw.Draw(layer)
        x = (W - total) / 2.0
        for c, adv in zip(TAGLINE, advances):
            draw.text((x, top), c, font=font, fill=TEXT_COLOR + (255,))
            x += adv + track
        return layer

    def render(self, t):
        frame = self.bg.copy()
        la = smoothstep((t - LOGO_FADE_START) / LOGO_FADE_LEN)
        ta = smoothstep((t - TEXT_FADE_START) / TEXT_FADE_LEN)
        if la > 0:
            logo = self.logo.copy()
            logo.putalpha(logo.getchannel("A").point(lambda a: round(a * la)))
            frame.paste(logo, self.logo_pos, logo)
        if ta > 0:
            text = self.text.copy()
            text.putalpha(text.getchannel("A").point(lambda a: round(a * ta)))
            frame.paste(text, (0, 0), text)
        return frame


def build_scenes():
    scenes = []
    for i, (name, start, end) in enumerate(SCENES):
        vis0 = start - XFADE / 2 if i > 0 else start
        vis1 = end + XFADE / 2 if i < len(SCENES) - 1 else end
        if name is None:
            r = EndCard()
        else:
            r = PhotoScene(load_photo(name), vis0, vis1)
        scenes.append((r, vis0, vis1))
    return scenes


def frame_at(scenes, t):
    out = None
    for r, v0, v1 in scenes:
        if not (v0 <= t < v1 or (t >= v1 and v1 == DURATION)):
            continue
        img = np.asarray(r.render(t), dtype=np.float32)
        # fade in over the first XFADE of visibility (not for the first scene)
        w = 1.0 if v0 == 0 else smoothstep((t - v0) / XFADE)
        out = img if out is None else out * (1 - w) + img * w
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", help="directory to write check frames into")
    args = ap.parse_args()

    scenes = build_scenes()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
        "-r", str(FPS), "-i", "-",
        "-an",
        "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
        "-c:v", "libx264", "-preset", "slow", "-crf", "16",
        "-profile:v", "high", "-tune", "film",
        "-colorspace", "bt709", "-color_primaries", "bt709",
        "-color_trc", "bt709", "-color_range", "tv",
        "-movflags", "+faststart",
        OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for n in range(N_FRAMES):
        proc.stdin.write(frame_at(scenes, n / FPS).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit("ffmpeg failed")
    print("wrote", OUT)

    if args.stills:
        os.makedirs(args.stills, exist_ok=True)
        checks = {"scene1_mid": 1.75, "scene2_mid": 5.25, "scene3_mid": 8.75,
                  "scene4_mid": 11.5, "endcard_mid": 14.25,
                  "endcard_last": (N_FRAMES - 1) / FPS}
        for label, t in checks.items():
            path = os.path.join(args.stills, f"{label}.png")
            Image.fromarray(frame_at(scenes, t)).save(path)
            print("wrote", path)


if __name__ == "__main__":
    main()
