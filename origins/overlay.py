"""Crisp 1080p UI drawn on top of the upscaled scene: title cards, the
cosmic calendar, the timeline bar and the opening / closing titles."""
import math
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from . import timeline as TL

OW, OH = 1920, 1080
FONT_DIR = "/usr/share/fonts/opentype/inter/"
FONTS = {
    "display": FONT_DIR + "InterDisplay-Bold.otf",
    "display_semi": FONT_DIR + "InterDisplay-SemiBold.otf",
    "display_light": FONT_DIR + "InterDisplay-Light.otf",
    "semi": FONT_DIR + "Inter-SemiBold.otf",
    "medium": FONT_DIR + "Inter-Medium.otf",
    "regular": FONT_DIR + "Inter-Regular.otf",
    "italic": FONT_DIR + "Inter-Italic.otf",
    "light": FONT_DIR + "Inter-Light.otf",
    "serif_italic": "/usr/share/fonts/truetype/crosextra/Caladea-Italic.ttf",
}


@lru_cache(maxsize=64)
def font(key, size):
    return ImageFont.truetype(FONTS[key], size)


def smooth(x, a, b):
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


def window(t, t0, t1, fin=0.5, fout=0.5):
    return smooth(t, t0, t0 + fin) * (1 - smooth(t, t1 - fout, t1))


@lru_cache(maxsize=1024)
def text_img(text, key, size, rgba=(255, 255, 255, 255), tracking=0, shadow=True):
    f = font(key, size)
    if tracking:
        widths = [f.getlength(ch) + tracking for ch in text]
        w = int(sum(widths)) + 1
    else:
        w = int(f.getlength(text)) + 1
    asc, desc = f.getmetrics()
    pad = 16
    im = Image.new("RGBA", (w + pad * 2, asc + desc + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if tracking:
        x = pad
        for ch, cw in zip(text, widths):
            d.text((x, pad), ch, font=f, fill=rgba)
            x += cw
    else:
        d.text((pad, pad), text, font=f, fill=rgba)
    if shadow:
        a = im.getchannel("A").filter(ImageFilter.GaussianBlur(6)).point(lambda v: int(v * 0.75))
        sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
        sh.putalpha(a)
        im = Image.alpha_composite(sh, im)
    return im


def wrap(text, key, size, maxw):
    f = font(key, size)
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if f.getlength(trial) <= maxw:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def put(base, im, x, y, alpha):
    """Composite a cached RGBA text image (top-left incl. its padding) with opacity."""
    if alpha <= 0.003:
        return
    if alpha < 0.997:
        im = im.copy()
        im.putalpha(im.getchannel("A").point(lambda v, a=alpha: int(v * a)))
    base.alpha_composite(im, (int(x) - 16, int(y) - 16))


def width(text, key, size, tracking=0):
    f = font(key, size)
    if tracking:
        return sum(f.getlength(ch) + tracking for ch in text)
    return f.getlength(text)


# ------------------------------------------------------------------ title card
LEFT = 110
CARD_BOTTOM = 925
# Cards wait until a white-out flash has passed so the text is readable.
CARD_DELAY = {"bigbang": 3.0, "asteroid": 5.2}


def card_alpha(T, p):
    if not p.title:
        return 0.0
    t0 = CARD_DELAY.get(p.key, 0.25)
    return window(T - p.start, t0, p.dur - 0.05, 0.6, 0.45)


def title_card(base, T, p):
    if not p.title:
        return 0.0
    t = T - p.start
    short = p.group != ""
    a = card_alpha(T, p)
    if a <= 0:
        return 0.0
    t0 = CARD_DELAY.get(p.key, 0.25)
    slide = (1 - smooth(t, t0, t0 + 0.7)) * 14
    acc = tuple(p.accent) + (255,)
    blocks = []  # (img, height_advance, alpha_mult)
    label = p.number + ("   ·   " + p.group if p.group else "")
    blocks.append((text_img(label, "semi", 21, acc, tracking=3), 34, 1.0))
    blocks.append((text_img(p.title, "display", 62 if short else 66), 74, 1.0))
    blocks.append((text_img(p.date, "medium", 30, acc), 48, 1.0))
    for ln in wrap(p.desc, "regular", 28, 880):
        blocks.append((text_img(ln, "regular", 28, (255, 255, 255, 235)), 38, 1.0))
    if p.fact:
        ft = max(p.dur * 0.38, t0 + 2.5)
        fa = smooth(t, ft, ft + 0.8)
        fl = wrap(p.fact, "italic", 25, 860)
        blocks.append((None, 12, 0))
        for i, ln in enumerate(fl):
            prefix = "—  " if i == 0 else "    "
            blocks.append((text_img(prefix + ln, "italic", 25, (255, 235, 200, 215)), 34, fa))
    total = sum(h for _, h, _ in blocks)
    y = CARD_BOTTOM - total + slide
    for im, h, am in blocks:
        if im is not None:
            put(base, im, LEFT, y, a * am)
        y += h
    return a


# ------------------------------------------------------------------ cosmic calendar
def calendar(base, T, alpha):
    if alpha <= 0:
        return
    ya = TL.years_ago(T)
    date, tm = TL.cosmic_calendar(ya)
    right = OW - LEFT
    lab = text_img("COSMIC CALENDAR", "semi", 17, (255, 255, 255, 150), tracking=3)
    put(base, lab, right - width("COSMIC CALENDAR", "semi", 17, 3), 66, alpha)
    d = text_img(date, "display_semi", 46)
    put(base, d, right - width(date, "display_semi", 46), 92, alpha)
    y = 150
    if tm:
        ti = text_img(tm, "medium", 30, (255, 220, 160, 255))
        put(base, ti, right - width(tm, "medium", 30), y, alpha)
        y += 42
    cap = "if all 13.8 billion years were one year"
    put(base, text_img(cap, "regular", 16, (255, 255, 255, 120)), right - width(cap, "regular", 16), y + 2, alpha)


# ------------------------------------------------------------------ timeline bar
BAR_Y = 1018
INSET_Y = 966
X0, X1 = LEFT, OW - LEFT


def timeline_bar(base, T, alpha, phase):
    if alpha <= 0:
        return
    ov = Image.new("RGBA", (OW, 170), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    oy = OH - 170
    ya = TL.years_ago(T)
    by = BAR_Y - oy
    d.line([(X0, by), (X1, by)], fill=(255, 255, 255, 80), width=2)
    for p in TL.PHASES:
        if p.title and not p.group:
            fx = X0 + (X1 - X0) * (1 - p.ya / TL.AGE_OF_UNIVERSE)
            c = p.accent + ((235,) if p is phase else (130,))
            r = 4 if p is phase else 3
            d.ellipse([fx - r, by - r, fx + r, by + r], fill=c)
    f = 1 - ya / TL.AGE_OF_UNIVERSE
    mx = X0 + (X1 - X0) * f
    d.line([(X0, by), (mx, by)], fill=(255, 255, 255, 170), width=2)
    inset = smooth(T, TL.LAST100K_START - 0.6, TL.LAST100K_START + 0.6)
    iy = INSET_Y - oy
    if inset > 0:
        ia = int(255 * inset)
        d.line([(X1, by), (X0, iy + 2)], fill=(255, 255, 255, int(40 * inset)), width=1)
        d.line([(X1, by), (X1, iy + 2)], fill=(255, 255, 255, int(40 * inset)), width=1)
        d.line([(X0, iy), (X1, iy)], fill=(255, 255, 255, int(90 * inset)), width=2)
        f2 = min(max(1 - ya / 1e5, 0), 1)
        ix = X0 + (X1 - X0) * f2
        d.line([(X0, iy), (ix, iy)], fill=(255, 220, 160, int(200 * inset)), width=2)
        for p in TL.PHASES:
            if p.group:
                px = X0 + (X1 - X0) * (1 - p.ya / 1e5)
                c = p.accent + ((240,) if p is phase else (int(150 * inset),))
                d.ellipse([px - 3, iy - 3, px + 3, iy + 3], fill=c)
    # marker glow
    for rr, aa in ((14, 40), (9, 80), (5, 255)):
        d.ellipse([mx - rr, by - rr, mx + rr, by + rr], fill=(255, 240, 210, aa))
    if inset > 0:
        for rr, aa in ((14, 40), (9, 80), (5, 255)):
            d.ellipse([ix - rr, iy - rr, ix + rr, iy + rr], fill=(255, 220, 160, int(aa * inset)))
    if alpha < 1:
        ov.putalpha(ov.getchannel("A").point(lambda v: int(v * alpha)))
    base.alpha_composite(ov, (0, oy))
    # labels
    put(base, text_img("BIG BANG", "semi", 14, (255, 255, 255, 130), tracking=2), X0, BAR_Y + 12, alpha)
    put(base, text_img("TODAY", "semi", 14, (255, 255, 255, 130), tracking=2), X1 - width("TODAY", "semi", 14, 2),
        BAR_Y + 12, alpha)
    lbl = TL.format_years(ya)
    lw = width(lbl, "semi", 20)
    if inset > 0:
        put(base, text_img("THE LAST 100,000 YEARS", "semi", 14, (255, 220, 160, 170), tracking=2),
            X0, INSET_Y - 26, alpha * inset)
        lx = min(max(ix - lw / 2, X0 + 250), X1 - lw)
        put(base, text_img(lbl, "semi", 20), lx, INSET_Y - 34, alpha * inset)
    else:
        lx = min(max(mx - lw / 2, X0), X1 - lw)
        put(base, text_img(lbl, "semi", 20), lx, BAR_Y - 38, alpha)


# ------------------------------------------------------------------ intro / finale titles
def centered(base, text, key, size, y, alpha, rgba=(255, 255, 255, 255), tracking=0):
    w = width(text, key, size, tracking)
    put(base, text_img(text, key, size, rgba, tracking=tracking), (OW - w) / 2, y, alpha)


def intro_titles(base, T):
    a = window(T, 0.6, 5.6, 0.9, 0.9)
    centered(base, "13.8 BILLION YEARS", "display", 104, 380, a, tracking=10)
    centered(base, "in 5 minutes", "display_light", 46, 512, a * smooth(T, 1.2, 2.2))
    centered(base, "The story of how life began on Earth", "light", 28, 600, a * smooth(T, 1.8, 2.8),
             (255, 255, 255, 190))


def finale_titles(base, T):
    t = T - TL.BY_KEY["finale"].start
    a = window(t, 6.0, 12.0, 1.0, 0.8)
    centered(base, "Every atom in you was made in a star.", "serif_italic", 62, 800, a)
    b = window(t, 11.6, 15.2, 0.8, 0.7)
    if b > 0:
        cx, cy = OW / 2, OH / 2
        ov = Image.new("RGBA", (OW, OH), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        d.line([(cx + 12, cy - 12), (cx + 90, cy - 90), (cx + 150, cy - 90)], fill=(255, 255, 255, int(170 * b)), width=2)
        base.alpha_composite(ov)
        put(base, text_img("You are here", "medium", 30), cx + 160, cy - 110, b)
    c = window(t, 15.4, 21.4, 1.0, 1.2)
    centered(base, "13.8 Billion Years in 5 Minutes", "display_semi", 66, 330, c)
    centered(base, "From the Big Bang to you", "light", 32, 425, c * smooth(t, 16.0, 17.0), (255, 255, 255, 200))
