"""Scene 16 (the last 100,000 years) and the finale."""
import math
from functools import lru_cache
import numpy as np
from .fx import *
from . import shapes as S
from . import timeline as TL
from .scenes_cosmic import STARS
from .scenes_life import grad, snow

PROJ = MapProj()


# ------------------------------------------------------------------ map helpers
@lru_cache(maxsize=2)
def map_base(night=False):
    lon, lat = PROJ.lonlat_grid()
    inside = (YY > PROJ.top) & (YY < PROJ.xy(0, PROJ.lat1)[1])
    land = PROJ.land() * inside
    edge = np.clip(land - blur(land, 3), 0, 1) * 4
    if night:
        C = grad((0, (2, 5, 14)), (1, (1, 3, 9)))
        land_col = col((16, 18, 26)) + (NOISE[(YY.astype(int)) % 512, (XX.astype(int)) % 512] * 0.03)[..., None]
        edge_col = col((60, 110, 170))
    else:
        C = grad((0, (8, 20, 38)), (1, (4, 10, 22)))
        n = sample(NOISE, XX * 0.8, YY * 0.8)
        land_col = (col((92, 78, 58)) * (0.75 + 0.5 * n)[..., None])
        edge_col = col((255, 210, 150))
    grid = ((np.abs(((lon + 180) % 30) - 15) > 14.6) | (np.abs(((lat + 90) % 30) - 15) > 14.6)) & inside
    C += grid[..., None] * 0.035
    a = land[..., None]
    C = C * (1 - a) + land_col * a
    C += edge[..., None] * edge_col * 0.25
    C *= np.where(inside, 1.0, 0.6)[..., None]
    return C.astype(np.float32)


def route_xy(pts):
    return [PROJ.xy(lon, lat) for lon, lat in pts]


def partial(pts, u):
    """First fraction u (by arc length) of a polyline."""
    pts = np.asarray(pts, np.float32)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    L = seg.sum() * min(max(u, 0), 1)
    out = [tuple(pts[0])]
    acc = 0.0
    for i, s in enumerate(seg):
        if acc + s >= L:
            f = (L - acc) / (s + 1e-9)
            out.append(tuple(pts[i] + (pts[i + 1] - pts[i]) * f))
            return out
        acc += s
        out.append(tuple(pts[i + 1]))
    return out


def draw_routes(C, routes, t, rgb=(255, 190, 90), offsets=(0, -360)):
    m = Mask()
    heads = []
    for pts, t0, t1 in routes:
        u = smooth(t, t0, t1)
        if u <= 0:
            continue
        for off in offsets:
            xy = route_xy([(lon + off, lat) for lon, lat in pts])
            p = partial(xy, u)
            if len(p) > 1:
                m.line(p, 2.2)
            if u < 1:
                heads.append(p[-1])
            else:
                heads.append((p[-1][0], p[-1][1], 1))
    add(C, m.arr(), rgb, 1.0, glow_r=10, glow_k=1.2)
    for h in heads:
        if len(h) == 3:
            blob(C, h[0], h[1], 9, rgb, 0.8, 2)
        else:
            blob(C, h[0], h[1], 16, (255, 240, 200), 1.6, 2)
    return C


OUT_OF_AFRICA = [
    ([(36, 2), (40, 8), (43, 12.5), (48, 15), (55, 20), (62, 25), (70, 22), (76, 14), (80, 12), (88, 21),
      (95, 18), (100, 10), (103, 2), (108, -4), (118, -8), (126, -10), (132, -14), (135, -22)], 0.2, 4.6),
    ([(40, 8), (37, 20), (35, 30), (36, 36), (30, 40), (22, 42), (12, 46), (2, 47), (-4, 42)], 1.6, 4.8),
    ([(62, 25), (60, 36), (68, 42), (80, 48), (95, 50)], 2.4, 5.0),
    ([(95, 18), (105, 25), (112, 32), (118, 38), (126, 42)], 3.2, 5.2),
]
INTO_AMERICAS = [
    ([(126, 42), (135, 50), (150, 58), (165, 64), (180, 66), (192, 65), (205, 62), (222, 59), (232, 54),
      (236, 48), (242, 38), (255, 28), (265, 18), (280, 8), (290, -5), (295, -25), (290, -45)], 0.5, 5.6),
]


def africa(t, d, T):
    C = map_base().copy()
    ox, oy = PROJ.xy(36, 2)
    for k in range(3):
        rr = ((t * 0.7 + k / 3) % 1.0)
        add(C, Mask().ring(ox, oy, 6 + rr * 40, 6 + rr * 40, 1.4).arr(), (255, 200, 120), 0.8 * (1 - rr))
    blob(C, ox, oy, 26, (255, 200, 120), 1.2, 2)
    return draw_routes(C, OUT_OF_AFRICA, t)


ICE_SHEETS = [
    [(-170, 70), (-140, 72), (-110, 76), (-80, 80), (-60, 82), (-20, 82), (-20, 70), (-45, 58), (-58, 50),
     (-70, 42), (-80, 40), (-95, 39), (-110, 46), (-125, 48), (-135, 55), (-150, 60), (-165, 62)],
    [(-12, 56), (-8, 62), (5, 70), (20, 75), (45, 78), (70, 76), (60, 68), (45, 62), (32, 56), (22, 53),
     (10, 52), (0, 52), (-6, 51)],
]


def iceage(t, d, T):
    C = map_base().copy()
    C = draw_routes(C, [(p, -2, -1) for p, _, _ in OUT_OF_AFRICA], t)
    g = smooth(t, 0, 2.5)
    m = Mask()
    for poly in ICE_SHEETS:
        for off in (0, 360):
            m.poly([PROJ.xy(lon + off, lat) for lon, lat in poly])
    near_land = np.clip(blur(PROJ.land(), 6) * 3, 0, 1)
    lon_, lat_ = PROJ.lonlat_grid()
    ice = blur(m.arr() * near_land, 3) * g * (lat_ < PROJ.lat0 - 1)
    paint(C, ice, (225, 240, 255), 0.85)
    bx, by = PROJ.xy(183, 65)
    br = Mask().ellipse(bx, by, 30, 12).arr() * smooth(t, 0.5, 2.0)
    paint(C, br, (110, 95, 75))
    C = draw_routes(C, INTO_AMERICAS, t, rgb=(160, 220, 255))
    snow(C, t, 260, 171, (230, 240, 255), speed=22, k=0.6, dx=6)
    return C


# ------------------------------------------------------------------ cave art
@lru_cache(maxsize=1)
def _rock():
    n = sample(NOISE, XX * 0.45, YY * 0.45) * 0.8 + sample(NOISE_FINE, XX * 0.6, YY * 0.6) * 0.2
    gy, gx = np.gradient(blur(n, 4))
    shade = np.clip(0.9 + (gx * -1 + gy * -0.6) * 22, 0.6, 1.2)
    base = np.stack([0.50 + 0.25 * n, 0.36 + 0.18 * n, 0.25 + 0.12 * n], -1)
    return (base * shade[..., None]).astype(np.float32)


BISON = [(-95, 10), (-100, -5), (-92, -20), (-80, -28), (-70, -45), (-50, -58), (-25, -60), (0, -52), (30, -45),
         (60, -42), (85, -38), (100, -30), (108, -20), (110, -5), (104, 10), (100, 35), (96, 60), (88, 60),
         (86, 35), (70, 20), (40, 22), (20, 24), (0, 25), (-20, 26), (-30, 40), (-34, 62), (-42, 62),
         (-44, 40), (-55, 30), (-70, 30), (-80, 25), (-90, 22), (-95, 10)]


def _hand(m, x, y, s, ang):
    m.poly(rot(ellipse_pts(0, 0, 17 * s, 20 * s, 24), ang, x, y))
    for fa, fl in [(-0.75, 30), (-0.3, 40), (0.0, 44), (0.3, 40), (0.62, 32)]:
        a = ang - math.pi / 2 + fa
        bx, by = x + math.cos(a) * 12 * s, y + math.sin(a) * 12 * s
        m.line([(bx, by), (bx + math.cos(a) * fl * s, by + math.sin(a) * fl * s)], 9 * s)
    a = ang + 0.2
    m.line([(x - 12 * s * math.cos(a), y), (x - 36 * s * math.cos(a), y + 12 * s)], 10 * s)


def caveart(t, d, T):
    C = _rock().copy()
    tx, ty = 860 + math.sin(t * 2.1) * 6, 470 + math.cos(t * 3.3) * 4
    flick = 0.9 + 0.07 * math.sin(t * 17) + 0.05 * math.sin(t * 31)
    light = np.exp(-(((XX - tx) / 620) ** 2 + ((YY - ty) / 480) ** 2)) * flick
    lit = 0.15 + 1.0 * light
    C *= lit[..., None] * np.array([1.15, 0.95, 0.75], np.float32)
    ochre = col((165, 50, 25))
    for i, (hx, hy, s, ang, t0) in enumerate([(460, 150, 1.4, -0.2, 0.3), (580, 115, 1.25, 0.15, 1.0),
                                              (370, 250, 1.2, -0.4, 1.7)]):
        g = smooth(t, t0, t0 + 1.0)
        if g <= 0:
            continue
        hm = Mask()
        _hand(hm, hx, hy, s, ang)
        hand = hm.arr()
        spray = np.clip(blur(hand, 14) * 3.0 - hand * 3.0, 0, 1) * g * 0.95
        C[:] = C * (1 - spray[..., None]) + ochre * lit[..., None] * spray[..., None]
    # a bison drawn stroke by stroke
    bx, by, sc = 690, 270, 1.25
    pts = [(bx + x * sc, by + y * sc) for x, y in BISON]
    u = smooth(t, 1.2, 4.6)
    fill = smooth(t, 4.0, 5.5)
    if fill > 0:
        fm = Mask().poly(pts).arr() * fill * 0.6
        C[:] = C * (1 - fm[..., None]) + ochre * 1.2 * lit[..., None] * fm[..., None]
    if u > 0:
        p = partial(pts, u)
        if len(p) > 1:
            om = Mask().line(p, 5.5).arr()
            paint(C, om, (20, 14, 12), 0.85)
        if u > 0.99:
            horn = Mask().line([(bx - 80 * sc, by - 28 * sc), (bx - 86 * sc, by - 42 * sc), (bx - 74 * sc, by - 48 * sc)], 3)
            horn.line([(bx + 108 * sc, by - 20 * sc), (bx + 120 * sc, by), (bx + 116 * sc, by + 22 * sc)], 3)
            paint(C, horn.arr(), (20, 14, 12), 0.85)
    dots = Mask()
    for k in range(int(smooth(t, 2, 5) * 9)):
        dots.ellipse(560 + k * 18, 420 + math.sin(k) * 6, 5, 5)
    paint(C, dots.arr(), (150, 40, 20), 0.8)
    blob(C, tx, ty, 160, (255, 150, 60), 0.35 * flick, 2)
    return C


# ------------------------------------------------------------------ farming
def farming(t, d, T):
    C = grad((0, (90, 100, 150)), (0.35, (240, 170, 110)), (0.55, (255, 210, 130)), (0.56, (120, 90, 60)),
             (1.0, (60, 40, 20)))
    blob(C, 300, 285, 200, (255, 210, 140), 0.9, 2)
    disc(C, 300, 290, 40, (255, 235, 190))
    hills = Mask().poly([(float(x), 300 - 18 * math.sin(x * 0.008 + 2) - 6 * math.sin(x * 0.03))
                         for x in np.linspace(-10, W + 10, 40)] + [(W + 10, H), (-10, H)])
    for hx in (640, 700, 770, 840):
        hy = 300 - 18 * math.sin(hx * 0.008 + 2) - 6 * math.sin(hx * 0.03)
        hills.rect(hx - 18, hy - 22, hx + 18, hy + 4)
        hills.poly([(hx - 24, hy - 20), (hx, hy - 46), (hx + 24, hy - 20)])
    paint(C, hills.arr(), (70, 50, 40))
    for hx in (700, 840):
        sm = noise_field(1.0, -t * 20, hx, NOISE_B)
        hy = 300 - 18 * math.sin(hx * 0.008 + 2) - 70
        plume = np.exp(-((XX - hx - (hy - YY) * 0.3) / (6 + np.clip(hy - YY, 0, 200) * 0.25)) ** 2) * (YY < hy + 25)
        C += (plume * sm * 0.25)[..., None] * col((230, 210, 190))
    r = rng(181)
    n = 700
    y = np.sort(r.uniform(305, 560, n))
    x = r.uniform(-10, W + 10, n)
    sc = (y - 290) / 260
    stalks = Mask()
    ears = Mask()
    for i in range(n):
        h = 70 * sc[i] + 10
        sw = math.sin(t * 1.6 + x[i] * 0.015) * 8 * sc[i] + math.sin(t * 3.1 + i) * 1.5
        tip = (x[i] + sw, y[i] - h)
        stalks.line([(x[i], y[i]), tip], max(0.6, 1.4 * sc[i]))
        ears.poly(rot(ellipse_pts(0, 0, 9 * sc[i] + 2, 2.6 * sc[i] + 0.8, 10), -math.pi / 2 + sw * 0.03, tip[0], tip[1]))
    paint(C, stalks.arr(), (150, 110, 50))
    paint(C, ears.arr(), (255, 205, 110))
    return C


# ------------------------------------------------------------------ writing & cities
@lru_cache(maxsize=1)
def _clay():
    n = sample(NOISE_FINE, XX * 2.0, YY * 2.0)
    return np.stack([0.62 + 0.12 * n, 0.46 + 0.1 * n, 0.32 + 0.08 * n], -1).astype(np.float32)


def writing(t, d, T):
    C = grad((0, (40, 30, 70)), (0.5, (190, 100, 80)), (0.63, (250, 170, 100)), (0.64, (70, 50, 40)),
             (1.0, (30, 22, 18)))
    blob(C, 160, 330, 150, (255, 180, 110), 0.7, 2)
    m = Mask()
    base = 345
    for k, (w, h) in enumerate([(300, 40), (220, 38), (140, 36)]):
        y0 = base - sum(hh for _, hh in [(300, 40), (220, 38), (140, 36)][:k])
        m.rect(300 - w / 2, y0 - h, 300 + w / 2, y0)
    m.rect(285, base - 150, 315, base)
    m.rect(-5, base, W + 5, H)
    for px, ph in [(40, 90), (520, 110), (575, 80)]:
        m.line([(px, base), (px + 6, base - ph)], 4)
        for k in range(7):
            a = -math.pi + k * math.pi / 6
            m.poly(tapered(bezier((px + 6, base - ph), (px + 6 + math.cos(a) * 20, base - ph - 12),
                                  (px + 6 + math.cos(a) * 36, base - ph + 14), n=8), np.linspace(4, 0.5, 8)))
    paint(C, m.arr(), (30, 18, 16))
    # clay tablet
    cx, cy, tw, th, ang = 690, 285, 250, 330, -0.06
    tab = Mask().poly(rot(ellipse_pts(0, 0, tw / 2, th / 2, 64, wobble=lambda a: 0.9 * (
        (abs(math.cos(a)) ** 6 + abs(math.sin(a)) ** 6) ** (-1 / 6)) - 0.9 + 0.0), ang, cx, cy)).arr()
    shade = np.clip(1.15 - ((XX - cx + 60) ** 2 + (YY - cy + 80) ** 2) / 200000, 0.6, 1.2)
    C[:] = C * (1 - tab[..., None]) + _clay() * shade[..., None] * tab[..., None]
    # cuneiform appearing row by row
    rows, per = 9, 8
    total = rows * per
    shown = int(smooth(t, 0.3, 5.4) * total)
    r = rng(191)
    dark = Mask()
    hi = Mask()
    for i in range(shown):
        row, k = divmod(i, per)
        x = cx - tw / 2 + 30 + k * 25 + r.uniform(-3, 3)
        y = cy - th / 2 + 32 + row * 33
        x, y = rot([(x - cx, y - cy)], ang, cx, cy)[0]
        kind = r.integers(0, 3)
        if kind == 0:
            dark.poly([(x - 6, y - 6), (x + 6, y - 6), (x, y + 2)])
            dark.line([(x, y), (x, y + 14)], 1.6)
        elif kind == 1:
            dark.poly([(x - 6, y - 5), (x - 6, y + 5), (x + 2, y)])
            dark.line([(x, y), (x + 12, y)], 1.6)
        else:
            dark.poly([(x - 5, y - 5), (x + 5, y - 5), (x, y + 1)])
            dark.poly([(x + 3, y + 3), (x + 13, y + 3), (x + 8, y + 9)])
        hi.line([(x - 6, y - 7), (x + 6, y - 7)], 1)
    paint(C, dark.arr(), (70, 45, 30), 0.9)
    paint(C, hi.arr(), (240, 200, 160), 0.4)
    return C


# ------------------------------------------------------------------ empires & ideas
def _temple(m, cx, base):
    m.rect(cx - 210, base - 22, cx + 210, base)
    m.rect(cx - 195, base - 34, cx + 195, base - 22)
    for k in range(8):
        x = cx - 175 + k * 50
        m.rect(x - 11, base - 190, x + 11, base - 34)
    m.rect(cx - 200, base - 215, cx + 200, base - 190)
    m.poly([(cx - 210, base - 215), (cx, base - 280), (cx + 210, base - 215)])


def _cathedral(m, cx, base):
    m.rect(cx - 170, base - 150, cx + 170, base)
    for sx in (-120, 120):
        m.rect(cx + sx - 40, base - 260, cx + sx + 40, base)
        m.poly([(cx + sx - 40, base - 260), (cx + sx, base - 370), (cx + sx + 40, base - 260)])
    m.poly([(cx - 80, base - 150), (cx, base - 220), (cx + 80, base - 150)])


def _dome(m, cx, base):
    m.rect(cx - 220, base - 90, cx + 220, base)
    m.rect(cx - 90, base - 150, cx + 90, base - 90)
    m.ellipse(cx, base - 150, 95, 120)
    m.rect(cx - 12, base - 300, cx + 12, base - 260)
    m.poly([(cx - 10, base - 300), (cx, base - 330), (cx + 10, base - 300)])


def empires(t, d, T):
    C = canvas((226, 206, 168))
    n = noise_field(1.0, 50, 50)
    C *= (0.8 + 0.3 * n)[..., None]
    C *= np.clip(1.15 - ((XX - W / 2) ** 2 / 300000 + (YY - H / 2) ** 2 / 120000), 0.45, 1)[..., None]
    geo = Mask()
    for k, rr in enumerate((90, 150, 210, 260)):
        geo.ring(700, 240, rr, rr, 1)
        a = t * (0.3 - k * 0.05) + k
        geo.ellipse(700 + math.cos(a) * rr, 240 + math.sin(a) * rr, 5, 5)
    geo.line([(700 - 280, 240), (700 + 280, 240)], 0.8)
    geo.line([(700, 240 - 280), (700, 240 + 280)], 0.8)
    paint(C, geo.arr(), (120, 90, 60), 0.35)
    ink = (60, 42, 30)
    for fn, t0, t1 in [(_temple, -1, 2.3), (_cathedral, 1.7, 4.3), (_dome, 3.7, 99)]:
        a = smooth(t, t0, t0 + 0.6) * (1 - smooth(t, t1 - 0.6, t1))
        if a > 0:
            m = Mask()
            fn(m, 640, 470)
            paint(C, m.arr(), ink, a * 0.9)
    paint(C, Mask().rect(-5, 470, W + 5, H).arr(), ink, 0.9)
    return C


# ------------------------------------------------------------------ industrial revolution
def industry(t, d, T):
    C = grad((0, (60, 50, 60)), (0.5, (170, 100, 70)), (0.7, (230, 140, 70)), (0.71, (40, 30, 30)),
             (1.0, (20, 16, 16)))
    m = Mask()
    chimneys = [(430, 150), (520, 190), (700, 130), (860, 170)]
    for x, h in chimneys:
        m.rect(x - 10, 380 - h, x + 10, 380)
    for k in range(6):
        x0 = 380 + k * 95
        m.poly([(x0, 380), (x0, 320), (x0 + 60, 290), (x0 + 60, 320), (x0 + 95, 320), (x0 + 95, 380)])
    m.rect(-5, 380, W + 5, H)
    paint(C, m.arr(), (30, 22, 22))
    win = Mask()
    for k in range(6):
        for j in range(3):
            win.rect(395 + k * 95 + j * 22, 340, 405 + k * 95 + j * 22, 352)
    add(C, win.arr(), (255, 170, 70), 0.9 + 0.1 * math.sin(t * 5), glow_r=6, glow_k=0.6)
    dens = np.zeros((H, W, 3), np.float32)
    r = rng(201)
    n = 900
    ph = r.uniform(0, 1, n)
    for i, (x, h) in enumerate(chimneys):
        age = ((t * 0.25 + ph + i * 0.3) % 1.0)
        px = x + age * 160 + np.sin(age * 8 + ph * 10) * 15 * age
        py = 380 - h - age * 120
        dens += splat(px, py, (1, 1, 1), (1 - age) * 1.5)
    dens = blur(dens, 9)
    a = np.clip(dens[..., :1] * 0.9, 0, 0.85)
    C[:] = C * (1 - a) + a * col((70, 60, 60))
    # steam train
    tx = -200 + t * 120
    tr = Mask()
    tr.rect(tx, 395, tx + 70, 420)
    tr.rect(tx - 30, 385, tx, 420)
    tr.rect(tx + 52, 380, tx + 62, 395)
    for c in range(3):
        tr.rect(tx - 40 - (c + 1) * 70, 392, tx - 40 - c * 70 - 6, 420)
    for k in range(6):
        tr.ellipse(tx + 60 - k * 22, 424, 7, 7)
    tr.rect(-5, 430, W + 5, 434)
    paint(C, tr.arr(), (15, 10, 10))
    r2 = rng(202)
    sp = r2.uniform(0, 1, 200)
    age = (t * 0.8 + sp) % 1.0
    sx = tx + 57 - age * 140 + np.sin(sp * 20) * 10
    sy = 378 - age * 80
    sd = blur(splat(sx, sy, (1, 1, 1), (1 - age) * 2), 6)
    a = np.clip(sd[..., :1], 0, 0.8)
    C[:] = C * (1 - a) + a * col((200, 190, 180))
    # gears
    gm = Mask()
    gm.poly(gear_pts := S.gear_pts(820, 470, 120, 14, t * 0.6))
    gm.poly(S.gear_pts(638, 515, 75, 9, -t * 0.6 * 120 / 75 + 0.17))
    paint(C, gm.arr(), (24, 18, 18))
    hole = Mask().ellipse(820, 470, 30, 30).ellipse(638, 515, 20, 20).arr()
    paint(C, hole, (60, 40, 30))
    rim = Mask().line(gear_pts + [gear_pts[0]], 2).arr()
    add(C, rim, (255, 160, 80), 0.35)
    return C


# ------------------------------------------------------------------ into the sky
def space(t, d, T):
    C = grad((0, (2, 4, 14)), (0.75, (14, 24, 60)), (0.83, (60, 60, 90)), (0.84, (10, 10, 16)), (1, (6, 6, 10)))
    STARS.draw(C, T, k=0.7, glow_k=0.0)
    blob(C, 790, 105, 90, (200, 210, 255), 0.35, 2)
    disc(C, 790, 105, 34, (225, 228, 235))
    disc(C, 778, 96, 7, (195, 198, 205))
    disc(C, 800, 116, 5, (200, 202, 210))
    # 1903: a biplane crosses
    if t < 3.0:
        px = -60 + t * 260
        py = 170 + math.sin(t * 2) * 6
        bp = Mask().rect(px - 26, py - 2, px + 18, py + 2).rect(px - 20, py - 14, px + 10, py - 11)
        bp.rect(px - 20, py + 9, px + 10, py + 12).line([(px - 15, py - 12), (px - 15, py + 10)], 1.4)
        bp.line([(px + 5, py - 12), (px + 5, py + 10)], 1.4).poly([(px - 26, py), (px - 34, py - 8), (px - 30, py)])
        paint(C, bp.arr(), (20, 20, 30))
    # 1969: launch
    a = max(T - TL.LAUNCH_T, 0)
    rise = 18 * a ** 2
    gx = 600
    pad = Mask().rect(gx + 30, 290, gx + 42, 455).rect(-5, 455, W + 5, H)
    for k in range(8):
        pad.line([(gx + 30, 300 + k * 20), (gx + 42, 310 + k * 20)], 1.2)
    paint(C, pad.arr(), (12, 12, 18))
    ry = 455 - rise
    rk = Mask()
    rk.rect(gx - 9, ry - 150, gx + 9, ry - 10)
    rk.poly([(gx - 9, ry - 150), (gx, ry - 182), (gx + 9, ry - 150)])
    rk.poly([(gx - 9, ry - 30), (gx - 22, ry - 5), (gx - 9, ry - 10)])
    rk.poly([(gx + 9, ry - 30), (gx + 22, ry - 5), (gx + 9, ry - 10)])
    rkm = rk.arr()
    paint(C, rkm, (215, 215, 220))
    bands = Mask().rect(gx - 9, ry - 100, gx + 9, ry - 94).rect(gx - 9, ry - 60, gx + 9, ry - 54).arr()
    paint(C, bands * rkm, (30, 30, 35))
    if T > TL.LAUNCH_T - 0.6:
        ig = smooth(T, TL.LAUNCH_T - 0.6, TL.LAUNCH_T + 0.2)
        fl = 50 + 20 * math.sin(t * 40) + 30 * smooth(a, 0, 2)
        fm = Mask().poly(tapered(bezier((gx, ry - 8), (gx, ry + fl * 0.5), (gx + math.sin(t * 30) * 2, ry + fl), n=10),
                                 np.linspace(8, 1, 10))).arr()
        add(C, fm, (255, 190, 90), 2.0 * ig, glow_r=20, glow_k=2.0)
        blob(C, gx, ry + 10, 90, (255, 170, 80), 1.2 * ig, 2)
        r = rng(211)
        n = 900
        ph = r.uniform(0, 1, n)
        ang = r.uniform(-0.35, 0.35, n) + np.where(r.uniform(0, 1, n) < 0.5, 0, math.pi)
        age = (t * 0.5 + ph) % 1.0
        dist = age * r.uniform(80, 260, n) * smooth(T, TL.LAUNCH_T - 0.4, TL.LAUNCH_T + 1.5)
        sx = gx + np.cos(ang) * dist
        sy = 452 - np.abs(np.sin(ang)) * dist * 0.3 - age * 30
        sd = blur(splat(sx, sy, (1, 1, 1), (1.2 - age)), 10)
        aa = np.clip(sd[..., :1] * 0.5, 0, 0.9)
        C[:] = C * (1 - aa) + aa * col((230, 200, 180))
    return C


# ------------------------------------------------------------------ today: the connected world
HOTSPOTS = [(-74, 40.7, 6, 3), (-87, 41, 6, 2), (-118, 34, 4, 1.5), (-122, 37.7, 3, 1), (-95, 30, 5, 1),
            (-80, 26, 3, 0.6), (-99, 19.4, 4, 1.5), (-46.6, -23.5, 5, 1.5), (-43, -22.9, 3, 0.8),
            (-58, -34.6, 3, 0.8), (-77, -12, 2, 0.4), (-74, 4.6, 3, 0.5), (0, 51, 8, 3), (10, 50, 8, 3),
            (25, 50, 8, 1.5), (37.6, 55.7, 5, 1.2), (12, 42, 5, 1), (-3.7, 40.4, 4, 1), (31, 30, 3, 1.2),
            (3.4, 6.5, 3, 1), (36.8, -1.3, 3, 0.5), (28, -26, 3, 0.6), (51, 35.7, 4, 0.8), (46.7, 24.7, 3, 0.4),
            (55, 25, 2, 0.4), (77, 28.6, 5, 3), (72.8, 19, 3, 2), (80, 13, 3, 1.5), (88.4, 22.6, 3, 2),
            (90.4, 23.8, 3, 1.5), (67, 24.9, 3, 1), (116.4, 39.9, 4, 2), (121.5, 31.2, 4, 3),
            (113.3, 23.1, 4, 2.5), (104, 30.6, 4, 1.2), (114, 30.6, 4, 1.5), (127, 37.5, 2, 1.5),
            (139.7, 35.7, 3, 2.5), (135.5, 34.7, 2, 1.2), (106.8, -6.2, 3, 1.5), (100.5, 13.7, 3, 1),
            (121, 14.6, 2, 1), (106.7, 10.8, 2, 0.8), (151.2, -33.9, 2, 0.6), (145, -37.8, 2, 0.5)]


@lru_cache(maxsize=1)
def city_points():
    r = rng(221)
    w = np.array([h[3] for h in HOTSPOTS], np.float64)
    idx = r.choice(len(HOTSPOTS), 9000, p=w / w.sum())
    lon = np.array([HOTSPOTS[i][0] for i in idx]) + r.normal(0, 1, 9000) * np.array([HOTSPOTS[i][2] for i in idx]) * 0.6
    lat = np.array([HOTSPOTS[i][1] for i in idx]) + r.normal(0, 1, 9000) * np.array([HOTSPOTS[i][2] for i in idx]) * 0.45
    lon2 = r.uniform(-180, 180, 4000)
    lat2 = r.uniform(-50, 70, 4000)
    lon = np.concatenate([lon, lon2])
    lat = np.concatenate([lat, lat2])
    bright = np.concatenate([r.uniform(0.4, 1.0, 9000), r.uniform(0.05, 0.25, 4000)])
    u = (lon + 180) / 360 * LAND.shape[1]
    v = np.clip((90 - lat) / 180 * LAND.shape[0], 0, LAND.shape[0] - 1.01)
    ok = sample(LAND, u, v) > 0.5
    return lon[ok], lat[ok], bright[ok]


HUBS = {"nyc": (-74, 40.7), "lon": (0, 51.5), "lag": (3.4, 6.5), "del": (77, 28.6), "sha": (121.5, 31.2),
        "tok": (139.7, 35.7), "sao": (-46.6, -23.5), "syd": (151.2, -33.9), "la": (-118, 34), "cai": (31, 30),
        "mos": (37.6, 55.7), "sin": (103.8, 1.35), "jnb": (28, -26), "mex": (-99, 19.4)}
LINKS = [("nyc", "lon"), ("lon", "del"), ("del", "sha"), ("sha", "tok"), ("la", "nyc"), ("la", "tok"),
         ("sao", "nyc"), ("sao", "lag"), ("lag", "lon"), ("cai", "lon"), ("mos", "lon"), ("sin", "syd"),
         ("sin", "del"), ("jnb", "lag"), ("mex", "la"), ("cai", "del"), ("mos", "sha"), ("syd", "la")]


def today(t, d, T):
    C = map_base(night=True).copy()
    lon, lat, b = city_points()
    x, y = PROJ.xy(lon, lat)
    on = smooth(t, 0, 1.5)
    tw = 0.85 + 0.15 * np.sin(T * 3 + lon * 7)
    glowing_points(C, x, y, col((255, 200, 120)), b * tw * on * 1.4, halo=0.5, r=6)
    lm = Mask()
    pulses = []
    for i, (a_, b_) in enumerate(LINKS):
        p0 = PROJ.xy(*HUBS[a_])
        p1 = PROJ.xy(*HUBS[b_])
        dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 - dist * 0.25)
        arc = bezier(p0, mid, p1, n=30)
        u = smooth(t, 0.5 + i * 0.12, 2.0 + i * 0.12)
        if u > 0:
            lm.line(partial(arc, u), 1.2)
            if u >= 1:
                k = ((t * 0.5 + i * 0.37) % 1.0)
                pulses.append(partial(arc, k)[-1])
    add(C, lm.arr(), (90, 200, 255), 0.6, glow_r=8, glow_k=0.8)
    for px, py in pulses:
        blob(C, px, py, 8, (200, 240, 255), 1.4, 2)
    return C


# ------------------------------------------------------------------ finale: the pale blue dot
@lru_cache(maxsize=1)
def _city_tex():
    lon, lat, b = city_points()
    w, h = 1024, 512
    tex = np.zeros((h, w), np.float32)
    xi = ((lon + 180) / 360 * w).astype(int) % w
    yi = np.clip(((90 - lat) / 180 * h).astype(int), 0, h - 1)
    np.add.at(tex, (yi, xi), b)
    return np.clip(blur(tex, 1) * 1.5, 0, 1.5)


def _earth_tex(T):
    def f(lat, lon):
        land = tex_lookup(LAND, lat, lon + math.pi)
        n = tex_lookup(NOISE, lat, lon, 2.0)
        alat = np.abs(lat)
        desert = np.exp(-((np.degrees(alat) - 24) / 9) ** 2) * (n > 0.45)
        green = np.stack([0.16 + 0.1 * n, 0.32 + 0.12 * n, np.full_like(n, 0.12)], -1)
        sand = np.array([0.62, 0.52, 0.34])
        lc = green * (1 - desert[:, None]) + sand * desert[:, None]
        ocean = np.stack([0.02 + 0.02 * n, 0.12 + 0.05 * n, 0.32 + 0.08 * n], -1)
        c = ocean * (1 - land[:, None]) + lc * land[:, None]
        ice = smooth(np.degrees(alat), 68, 74)[:, None]
        c = c * (1 - ice) + np.array([0.92, 0.95, 1.0]) * ice
        cl = tex_lookup(NOISE, lat, lon + T * 0.02, 3.0) * 0.7 + tex_lookup(NOISE_FINE, lat, lon, 2.0) * 0.3
        cl = smooth(cl, 0.58, 0.8)[:, None] * 0.8
        return c * (1 - cl) + np.array([1.0, 1.0, 1.0]) * cl
    return f


def _earth_emit(lat, lon, lam):
    tex = _city_tex()
    c = tex_lookup(tex, lat, lon + math.pi)
    night = np.clip(1 - lam * 4, 0, 1) ** 2
    v = c * night * 0.9
    return np.stack([v, v * 0.75, v * 0.4], -1)


def finale(t, d, T):
    C = canvas((0, 0, 0))
    z = smooth(t, 3.0, 15.0)
    STARS_W = STARS
    STARS_W.draw(C, T, k=0.25 + 0.6 * z, zoom=1.8 - 0.8 * z, glow_k=0.1)
    r = 230 * math.exp(-z * 4.8)
    cx = 480 + 60 * (1 - z)
    cy = 270
    sphere(C, cx, cy, r, _earth_tex(T), lon0=0.5 - (T - 278) * 0.06, tilt=0.4, light=(0.75, -0.3, 0.6),
           ambient=0.02, atmo=(90, 160, 255), atmo_k=0.8, emit_fn=_earth_emit)
    if r < 6:
        blob(C, cx, cy, 10, (120, 180, 255), 0.8 * smooth(t, 11, 13), 2)
    C *= 1 - smooth(t, d - 1.5, d)
    return C
