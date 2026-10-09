"""Scenes 5-15: from the first cells to the first humans."""
import math
from functools import lru_cache
import numpy as np
from .fx import *
from . import shapes as S
from . import timeline as TL
from .scenes_cosmic import STARS


# ------------------------------------------------------------------ shared helpers
def rays(C, t, k=0.25, ang=0.35, depth=1.0, rgb=(150, 220, 255)):
    u = XX * math.cos(ang) + YY * math.sin(ang)
    r = (np.sin(u * 0.021 + t * 0.4) * np.sin(u * 0.047 - t * 0.27 + 1.3) + np.sin(u * 0.011 + t * 0.15))
    r = np.clip(r - 0.4, 0, 2) * (1 - YY / (H * depth)).clip(0, 1) ** 1.5
    C += r[..., None] * col(rgb) * k
    return C


@lru_cache(maxsize=8)
def _floor(seed, base, amp):
    r = rng(seed)
    xs = np.linspace(-10, W + 10, 60)
    ys = base + amp * (sample(NOISE, xs * 0.6 + seed * 37, np.full_like(xs, seed * 13.0)) - 0.5) * 2
    return [(float(x), float(y)) for x, y in zip(xs, ys)] + [(W + 10, H + 10), (-10, H + 10)]


def floor_y(seed, base, amp, x):
    pts = _floor(seed, base, amp)[:60]
    xs = [p[0] for p in pts]
    return float(np.interp(x, xs, [p[1] for p in pts]))


def snow(C, t, n, seed, rgb, speed=20, k=0.5, size_soft=0, dx=0.0):
    r = rng(seed)
    x0 = r.uniform(0, W, n)
    y0 = r.uniform(0, H, n)
    sp = r.uniform(0.5, 1.5, n)
    x = (x0 + np.sin(t * 0.7 + x0) * 8 + dx * t * sp) % W
    y = (y0 + t * speed * sp) % H
    return glowing_points(C, x, y, col(rgb), r.uniform(0.3, 1, n) * k, halo=0.2, r=4, soft=size_soft)


@lru_cache(maxsize=16)
def _grad(stops):
    return vgrad(list(stops))


def grad(*stops):
    return _grad(tuple(stops)).copy()


# ------------------------------------------------------------------ 5. oceans & first life
_r = rng(50)
_SM = 3500
_sm_u = _r.uniform(0, 1, _SM)
_sm_s = _r.uniform(0, 6.28, _SM)
_sm_v = _r.uniform(0.7, 1.3, _SM)
_VENTS = [(300, 330, 1.0), (165, 395, 0.6)]


def _cells(t, t0, period, pos, depth, seed, out):
    r = rng(seed)
    if t < t0:
        return
    if depth == 4 or t < t0 + period:
        out.append((pos[0], pos[1], t - t0, min((t - t0 - period) / 0.0001, 0) if depth < 4 else 0))
        return
    a = r.uniform(0, 2 * math.pi)
    sep = 26 * smooth(t, t0 + period, t0 + period + 1.2)
    for i, sgn in enumerate((-1, 1)):
        p = (pos[0] + math.cos(a) * sep * sgn + (t - t0 - period) * r.uniform(-3, 3),
             pos[1] + math.sin(a) * sep * sgn * 0.8 - (t - t0 - period) * r.uniform(0, 3))
        _cells(t, t0 + period, period * 1.0, p, depth + 1, seed * 3 + i + 1, out)


def life(t, d, T):
    C = grad((0.0, (16, 72, 100)), (0.45, (6, 34, 58)), (1.0, (2, 10, 22)))
    rays(C, t, 0.22)
    snow(C, t, 300, 51, (170, 210, 220), speed=6, k=0.35)
    fl = _floor(52, 470, 18)
    m = Mask().poly(fl)
    for vx, vy, s in _VENTS:
        m.poly([(vx - 38 * s, 480), (vx - 22 * s, vy + 60 * s), (vx - 14 * s, vy + 10 * s), (vx - 9 * s, vy),
                (vx + 9 * s, vy), (vx + 13 * s, vy + 25 * s), (vx + 24 * s, vy + 70 * s), (vx + 40 * s, 480)])
    paint(C, m.arr(), (6, 12, 16))
    # black smoker plumes
    dens = np.zeros((H, W, 3), np.float32)
    for i, (vx, vy, s) in enumerate(_VENTS):
        L = 6.0
        age = (t + _sm_u * L + i * 1.7) % L
        x = vx + np.sin(age * 1.1 + _sm_s) * age * 7 * s + age * (6 + 10 * np.sin(_sm_s * 3)) * s + age ** 2 * 1.5
        y = vy - age * 45 * _sm_v * s - age ** 2 * 1.5
        w = np.clip(1 - age / L, 0, 1) * s
        dens += splat(x, y, (1, 1, 1), w)
    dens = blur(dens, 7)
    a = np.clip(dens[..., :1] * 1.1, 0, 0.95)
    C *= 1 - a
    C += a * col((78, 70, 66))
    for vx, vy, s in _VENTS:
        blob(C, vx, vy + 6, 60 * s, (255, 120, 40), 0.9 * s * (0.85 + 0.15 * math.sin(t * 7)), 2.2)
    # bubbles
    r = rng(53)
    n = 50
    bx = r.uniform(120, 700, n)
    by0 = r.uniform(0, 1, n)
    ph = (t * 0.12 + by0) % 1.0
    x = bx + np.sin(ph * 30 + bx) * 4
    y = 470 - ph * 470
    glowing_points(C, x, y, col((190, 235, 255)), 0.9 * np.ones(n), halo=0.3, r=4, soft=1)
    # the first cells: grow, divide
    cells = []
    _cells(t, 6.0, 2.5, (430, 330), 0, 7, cells)
    if cells:
        mm = Mask()
        for cx, cy, age, _ in cells:
            cy -= age * 2
            cx += math.sin(age + cx) * 3
            rr = 9 + 3 * smooth(age, 0, 1.5)
            mm.ring(cx, cy, rr, rr * 0.93, 1.6)
            blob(C, cx, cy, rr * 2.2, (90, 255, 210), 0.35 * smooth(age, 0, 0.8), 1.8)
            blob(C, cx - 2, cy - 1, rr * 0.5, (210, 255, 200), 0.5 * smooth(age, 0, 0.8), 1.5)
        add(C, mm.arr(), (140, 255, 220), 1.1, glow_r=10, glow_k=1.2)
    return C


# ------------------------------------------------------------------ 6. great oxidation
_DOMES = [(130, 70, 32), (260, 90, 44), (400, 60, 30), (520, 110, 52), (660, 80, 40), (790, 100, 48),
          (905, 70, 34)]
HORIZON = 205


def oxygen(t, d, T):
    sk = smooth(t, 1.5, 11.0)
    C = np.empty((H, W, 3), np.float32)
    sky = grad((0, (140, 85, 60)), (1.0, (205, 140, 95))) * (1 - sk) + grad((0, (40, 95, 190)), (1.0, (170, 205, 240))) * sk
    sea = grad((0, (55, 95, 60)), (1.0, (12, 30, 24))) * (1 - sk) + grad((0, (30, 120, 150)), (1.0, (4, 36, 60))) * sk
    C[:HORIZON] = sky[: HORIZON]
    C[HORIZON:] = sea[HORIZON:]
    sun = mix_col((255, 170, 90), (255, 250, 230), sk)
    blob(C, 720, 110, 160, sun, 0.8, 2.2)
    disc(C, 720, 110, 24, tuple(min(255, c + 20) for c in sun))
    C[HORIZON:] = C[HORIZON:]  # (keep sun above water)
    rays(C, t, 0.12 + 0.1 * sk, ang=0.3)
    C[HORIZON - 1:HORIZON + 2] += 0.25 + 0.1 * math.sin(t * 3)
    # stromatolites
    fl = 490
    m = Mask().rect(-5, fl, W + 5, H + 5)
    for x, h, w in _DOMES:
        m.ellipse(x, fl, w, h)
    paint(C, m.arr(), (34, 30, 24))
    bands = Mask()
    for x, h, w in _DOMES:
        for k in (0.82, 0.62, 0.42):
            bands.ring(x, fl, w * k, h * k, 2.2)
    paint(C, bands.arr(), (70, 58, 40), 0.7)
    mat = Mask()
    for x, h, w in _DOMES:
        mat.ring(x, fl, w, h, 3.5)
    add(C, mat.arr(), (60, 230, 90), 0.5 + 0.2 * math.sin(t * 2), glow_r=8, glow_k=0.8)
    # oxygen bubbles, more and more over time
    r = rng(61)
    n = 420
    src = r.integers(0, len(_DOMES), n)
    ang = r.uniform(-1.3, 1.3, n)
    birth = r.uniform(-4, d, n)
    gate = r.uniform(0, 1, n) < 0.2 + 0.8 * smooth(t, 0, d * 0.8)
    age = t - birth
    dx = np.array([_DOMES[i][0] for i in src]) + np.sin(ang) * np.array([_DOMES[i][2] for i in src])
    dy = fl - np.cos(ang) * np.array([_DOMES[i][1] for i in src])
    y = dy - age * 55
    x = dx + np.sin(age * 3 + ang * 9) * 5
    ok = gate & (age > 0) & (y > HORIZON + 3)
    glowing_points(C, x[ok], y[ok], col((210, 255, 240)), np.full(ok.sum(), 1.1), halo=0.4, r=5, soft=1)
    return C


# ------------------------------------------------------------------ 7. complex cells
HOST = (470, 280, 175)


def _capsule(m, x, y, L, Wd, ang):
    m.poly(rot(ellipse_pts(0, 0, L / 2, Wd / 2, 32), ang, x, y))


def eukaryote(t, d, T):
    C = canvas((12, 6, 24))
    n = noise_field(0.7, t * 8, 10)
    C += np.stack([n * 0.10, n * 0.03, n * 0.16], -1)
    for i, (bx, by, br) in enumerate([(120, 90, 90), (860, 440, 120), (820, 80, 70), (90, 470, 80)]):
        blob(C, bx + math.sin(t * 0.3 + i) * 20, by, br, (110, 60, 160), 0.18, 2)
    hx, hy, R = HOST
    # bacterium path
    a0 = -0.35
    touch = (hx + math.cos(a0) * (R + 14), hy + math.sin(a0) * (R + 14))
    inside = (565, 235)
    if t < 3.0:
        u = smooth(t, 0, 3.0)
        bx, by = 960 + (touch[0] - 960) * u, 120 + (touch[1] - 120) * u
    else:
        u = smooth(t, 3.0, 6.0)
        bx, by = touch[0] + (inside[0] - touch[0]) * u, touch[1] + (inside[1] - touch[1]) * u
    eng = smooth(t, 2.2, 3.4) * (1 - smooth(t, 4.0, 6.0))

    def wob(a):
        da = math.atan2(math.sin(a - a0), math.cos(a - a0))
        return (0.025 * math.sin(4 * a + 1.3 * t) + 0.015 * math.sin(7 * a - t)
                + eng * 0.16 * math.exp(-(da / 0.32) ** 2) - eng * 0.06 * math.exp(-(da / 0.08) ** 2))

    pts = ellipse_pts(hx, hy, R, R, 120, wobble=wob)
    m = Mask().poly(pts)
    paint(C, m.arr(), (90, 45, 140), 0.35)
    edge = Mask().line(pts + [pts[0]], 3)
    add(C, edge.arr(), (210, 150, 255), 1.0, glow_r=10, glow_k=1.0)
    # vesicles drifting inside
    r = rng(71)
    for i in range(14):
        a = r.uniform(0, 6.28) + t * r.uniform(-0.1, 0.1)
        rr = r.uniform(30, R - 30)
        vx, vy = hx + math.cos(a) * rr, hy + math.sin(a) * rr
        blob(C, vx, vy, 10, (190, 140, 255), 0.25, 2)
    # nucleus forms
    nu = smooth(t, 4.5, 7.5)
    if nu > 0:
        nm = Mask().ellipse(415, 300, 58 * nu, 54 * nu)
        paint(C, nm.arr(), (60, 25, 95), 0.8)
        add(C, Mask().ring(415, 300, 58 * nu, 54 * nu, 2.5).arr(), (230, 170, 255), 0.9, glow_r=6, glow_k=0.6)
        blob(C, 405, 292, 22 * nu, (255, 160, 230), 0.4, 2)
    # bacterium -> mitochondrion (and its descendants)
    mi = smooth(t, 6.0, 8.0)
    green, orange = np.array([90, 230, 120]), np.array([255, 150, 70])
    c = tuple(green * (1 - mi) + orange * mi)
    split = smooth(t, 9.0, 11.0)
    bodies = [(bx, by, 0.0)]
    if t > 9.0:
        bodies = [(bx - 30 * split, by - 20 * split, -0.4 * split), (bx + 25 * split, by + 35 * split, 0.5 * split)]
    for (x, y, aa) in bodies:
        ang = -0.6 + aa + math.sin(t * 0.8) * 0.1
        mb = Mask()
        _capsule(mb, x, y, 50, 22, ang)
        paint(C, mb.arr(), tuple(np.array(c) * 0.55), 0.95)
        add(C, Mask().line(ellipse_pts(x, y, 25, 11, 32, ang) + [ellipse_pts(x, y, 25, 11, 32, ang)[0]], 1.6).arr(),
            c, 1.0, glow_r=6, glow_k=0.8)
        if mi > 0.1:
            cr = [(x + math.cos(ang) * s_ - math.sin(ang) * 6 * (1 if j % 2 else -1),
                   y + math.sin(ang) * s_ + math.cos(ang) * 6 * (1 if j % 2 else -1))
                  for j, s_ in enumerate(np.linspace(-18, 18, 9))]
            add(C, Mask().line(cr, 1.5).arr(), (255, 220, 160), 0.7 * mi)
        if t < 3.4:
            tail = [(x + math.cos(ang) * (25 + k * 4) + math.sin(k * 0.9 - t * 12) * 3,
                     y + math.sin(ang) * (25 + k * 4)) for k in range(10)]
            add(C, Mask().line(tail, 1.2).arr(), c, 0.8)
    return C


# ------------------------------------------------------------------ 8. snowball earth -> ediacaran
def _snow_tex(ice_lat):
    def f(lat, lon):
        land = tex_lookup(NOISE_B, lat, lon, 1.0)
        base = np.where((land > 0.6)[:, None], np.array([0.42, 0.36, 0.26]), np.array([0.05, 0.2, 0.42]))
        jag = (tex_lookup(NOISE_FINE, lat, lon, 2.0) - 0.5) * 0.25
        ice = smooth(np.abs(lat) + jag, ice_lat - 0.05, ice_lat + 0.05)[:, None]
        return base * (1 - ice) + np.array([0.92, 0.96, 1.0]) * ice
    return f


def _ediacaran(t):
    C = grad((0, (40, 120, 130)), (0.55, (14, 60, 75)), (1.0, (6, 26, 36)))
    rays(C, t, 0.2, ang=0.25)
    snow(C, t, 200, 81, (200, 240, 230), speed=5, k=0.3)
    paint(C, Mask().poly(_floor(82, 455, 12)).arr(), (52, 64, 58))
    r = rng(83)
    fronds = Mask()
    ribs = Mask()
    for i in range(10):
        x = 330 + i * 64 + r.uniform(-20, 20)
        base = floor_y(82, 455, 12, x)
        h = r.uniform(90, 190)
        sw = math.sin(t * 0.9 + i) * 10
        cl = bezier((x, base), (x + sw * 0.5, base - h * 0.5), (x + sw, base - h), n=14)
        wd = np.sin(np.linspace(0.15, 1.0, 14) * math.pi) * h * 0.13
        wd[:3] = 1.5
        fronds.poly(tapered(cl, wd))
        for j in range(4, 14):
            px, py = cl[j]
            ribs.line([(px - wd[j], py + 3), (px + wd[j], py + 3)], 0.8)
    paint(C, fronds.arr(), (190, 130, 120), 0.85)
    paint(C, ribs.arr(), (120, 70, 70), 0.6)
    dick = Mask()
    dr = Mask()
    for x, y, s in [(250, 470, 26), (600, 485, 20), (780, 475, 30), (880, 495, 16)]:
        dick.ellipse(x, y, s, s * 0.35)
        for k in np.linspace(-0.85, 0.85, 9):
            dr.line([(x + k * s, y - s * 0.3 * math.sqrt(1 - k * k)), (x + k * s, y + s * 0.3 * math.sqrt(1 - k * k))], 0.7)
    paint(C, dick.arr(), (170, 120, 90))
    paint(C, dr.arr(), (110, 70, 55), 0.7)
    return C


def snowball(t, d, T):
    cf = smooth(t, 7.0, 8.8)
    out = None
    if cf < 1:
        C = canvas((1, 2, 6))
        STARS.draw(C, T, k=0.5, glow_k=0.0)
        ice = math.radians(62 - 56 * smooth(t, 0, 5.0) + 70 * smooth(t, 5.6, 8.5))
        sphere(C, 590, 265, 195, _snow_tex(ice), lon0=t * 0.25, tilt=0.35, light=(-0.6, -0.35, 0.7),
               ambient=0.04, atmo=(200, 230, 255), atmo_k=0.5)
        out = C
    if cf > 0:
        E = _ediacaran(t)
        out = E if out is None else out * (1 - cf) + E * cf
    return out


# ------------------------------------------------------------------ 9. cambrian explosion
def _jelly(C, x, y, s, t, rgb):
    pul = 1 + 0.12 * math.sin(t * 3)
    bell = [(x + math.cos(a) * 26 * s * pul, y - math.sin(a) * 20 * s / pul) for a in np.linspace(0, math.pi, 18)]
    m = Mask().poly(bell)
    tm = Mask()
    for k in range(5):
        tx = x + (k - 2) * 8 * s
        tm.line([(tx + math.sin(t * 2 + j * 0.6 + k) * 4 * s, y + j * 9 * s) for j in range(8)], 1.2)
    add(C, m.arr(), rgb, 0.45, glow_r=10, glow_k=0.8)
    add(C, tm.arr(), rgb, 0.4)


def _anomalocaris(C, x, y, s, t):
    m = Mask()
    body = [(x + k * 14 * s, y + math.sin(t * 4 - k * 0.7) * 2 * s) for k in range(11)]
    wd = np.concatenate([np.linspace(10, 15, 3), np.linspace(15, 3, 8)]) * s
    m.poly(tapered(body, wd))
    for k in range(1, 10):
        bx, by = body[k]
        flap = math.sin(t * 6 - k * 0.8) * 5 * s
        m.ellipse(bx, by - wd[k] - 3 * s + flap * 0.3, 7 * s, 4 * s + abs(flap) * 0.3)
        m.ellipse(bx, by + wd[k] + 3 * s - flap * 0.3, 7 * s, 4 * s + abs(flap) * 0.3)
    m.poly([(x + 150 * s, y), (x + 175 * s, y - 14 * s), (x + 168 * s, y), (x + 175 * s, y + 14 * s)])
    for sg in (-1, 1):
        cl = bezier((x - 2 * s, y + sg * 4 * s), (x - 30 * s, y + sg * 8 * s), (x - 30 * s, y + sg * 26 * s), n=10)
        m.poly(tapered(cl, np.linspace(4, 1.5, 10) * s))
    m.ellipse(x + 4 * s, y - 16 * s, 5 * s, 4 * s)
    paint(C, m.arr(), (150, 60, 50))
    blob(C, x + 4 * s, y - 16 * s, 4 * s, (255, 220, 120), 0.6, 1)


def _fish(m, x, y, s, ang):
    pts = [(0, 0), (8, -3), (20, -2), (28, 0), (20, 2), (8, 3)]
    pts = [(px - 14, py) for px, py in pts] + [(-14, 0), (-20, -5), (-18, 0), (-20, 5)]
    m.poly(rot([(px * s, py * s) for px, py in pts[:6]], ang, x, y))
    m.poly(rot([(px * s, py * s) for px, py in pts[6:]], ang, x, y))


def cambrian(t, d, T):
    C = grad((0, (30, 110, 150)), (0.5, (12, 60, 92)), (1.0, (4, 22, 38)))
    rays(C, t, 0.25)
    snow(C, t, 220, 91, (200, 240, 255), speed=5, k=0.3)
    fl = _floor(92, 462, 14)
    paint(C, Mask().poly(fl).arr(), (60, 62, 52))
    # sponges and stalked animals
    sp = Mask()
    r = rng(93)
    for i in range(12):
        x = r.uniform(380, 940)
        base = floor_y(92, 462, 14, x) + 4
        h = r.uniform(40, 120)
        sw = math.sin(t + i) * 4
        cl = bezier((x, base), (x + sw * 0.4, base - h * 0.5), (x + sw, base - h), n=10)
        sp.poly(tapered(cl, np.linspace(6, 12, 10) * r.uniform(0.7, 1.3)))
    paint(C, sp.arr(), (190, 140, 80), 0.85)
    # trilobites crawling
    tm = Mask()
    seg = Mask()
    for i in range(6):
        x = (100 + i * 150 + t * (8 + i)) % (W + 120) - 60
        base = floor_y(92, 462, 14, x) + 2
        s = 0.8 + 0.15 * (i % 3)
        tm.poly([(x + math.cos(a) * 26 * s, base - math.sin(a) * 15 * s) for a in np.linspace(0, math.pi, 20)])
        for k in np.linspace(-0.7, 0.7, 7):
            seg.line([(x + k * 26 * s, base), (x + k * 24 * s, base - 14 * s * math.sqrt(1 - k * k))], 0.9)
        for k in range(6):
            lx = x - 18 * s + k * 7 * s
            seg.line([(lx, base), (lx + math.sin(t * 12 + k) * 3, base + 3)], 1)
    paint(C, tm.arr(), (140, 95, 60))
    paint(C, seg.arr(), (70, 45, 30), 0.8)
    # anomalocaris patrolling
    _anomalocaris(C, 980 - (t * 55) % 1300, 230 + math.sin(t * 0.7) * 20, 1.1, T)
    # a school of early fish
    fm = Mask()
    for i in range(16):
        ph = i * 0.7
        x = (t * 70 + i * 23) % (W + 300) - 150
        y = 140 + math.sin(t * 1.3 + ph) * 25 + (i % 4) * 14 + (i // 4) * 6
        _fish(fm, x, y, 0.9, math.sin(t * 1.3 + ph) * 0.3)
    paint(C, fm.arr(), (170, 200, 210), 0.9)
    for i, (x, y, rgb) in enumerate([(650, 160, (255, 140, 200)), (820, 300, (140, 220, 255)),
                                      (520, 330, (255, 190, 140)), (900, 120, (200, 160, 255))]):
        _jelly(C, x + math.sin(T * 0.3 + i) * 15, y - (t * 6) % 40 + math.sin(t + i) * 8, 0.9, T + i, rgb)
    return C


# ------------------------------------------------------------------ 10. life moves onto land
SHORE_H = 310


def _ground_land(x):
    return SHORE_H + 3 + np.clip((x - 330) / 80, 0, 1) * (np.sin(x * 0.02) * 3 + 3)


def land(t, d, T):
    C = grad((0, (35, 45, 95)), (0.45, (200, 120, 100)), (0.575, (255, 175, 100)), (0.58, (90, 80, 90)),
             (1.0, (14, 22, 34)))
    blob(C, 690, SHORE_H, 170, (255, 190, 110), 0.9, 2.2)
    disc(C, 690, SHORE_H, 34, (255, 225, 170))
    # sea surface with glitter
    sea = np.zeros((H, W), np.float32)
    sea[SHORE_H:] = 1
    gl = np.clip(np.sin(XX * 0.25 + t * 4) * np.sin(YY * 1.3 - t * 2) - 0.6, 0, 1) * \
        np.exp(-((XX - 690) / (40 + np.abs(YY - SHORE_H) * 1.0)) ** 2) * sea
    C += gl[..., None] * col((255, 210, 150)) * 2
    # land
    xs = np.linspace(330, W + 10, 50)
    ground = [(330, SHORE_H + 2)] + [(float(x), float(_ground_land(x))) for x in xs] + [(W + 10, H), (300, H)]
    ground_m = Mask().poly([(330, SHORE_H + 0.5)] + ground[1:-1] + [(110, H)]).arr()
    gshade = grad((0, (70, 60, 48)), (0.6, (58, 50, 40)), (1.0, (30, 26, 22)))
    paint(C, ground_m, gshade)
    foam = Mask().line([(330 - s_ * 220 + math.sin(t * 2 + s_ * 9) * 5, SHORE_H + s_ * 230)
                        for s_ in np.linspace(0, 1, 30)], 1.8)
    add(C, foam.arr(), (220, 230, 240), 0.35 + 0.15 * math.sin(t * 2))
    # plants grow in waves: mosses & cooksonia, ferns, then trees
    r = rng(101)
    pm = Mask()
    for i in range(46):
        x = r.uniform(360, 950)
        gy = float(_ground_land(x))
        kind = 0 if i < 20 else (1 if i < 36 else 2)
        start = [0.5, 3.5, 6.5][kind] + r.uniform(0, 3)
        g = smooth(t, start, start + 3.5)
        if g <= 0:
            continue
        sw = math.sin(t * 1.3 + i) * 2
        if kind == 0:
            h = r.uniform(10, 22) * g
            pm.line([(x, gy), (x + sw * 0.3, gy - h * 0.6)], 1.4)
            for sgn in (-1, 1):
                tip = (x + sgn * 5 * g + sw * 0.5, gy - h)
                pm.line([(x + sw * 0.3, gy - h * 0.6), tip], 1.2)
                pm.ellipse(tip[0], tip[1], 2.2 * g, 2.2 * g)
        elif kind == 1:
            h = r.uniform(30, 60) * g
            for k in range(6):
                a = -math.pi / 2 + (k - 2.5) * 0.35
                cl = bezier((x, gy), (x + math.cos(a) * h * 0.6 + sw, gy + math.sin(a) * h * 0.6),
                            (x + math.cos(a) * h + sw * 1.5 + (k - 2.5) * 8, gy + math.sin(a) * h * 0.8 + 10), n=8)
                pm.poly(tapered(cl, np.linspace(2.5, 0.6, 8)))
        else:
            h = r.uniform(90, 170) * g
            pm.line([(x, gy), (x + sw * 0.3, gy - h)], 4 * g)
            for k in range(7):
                yy = gy - h * (0.45 + k * 0.08)
                ww = h * 0.3 * (1 - k / 8)
                pm.line([(x - ww + sw, yy + 6), (x + sw * 0.5, yy), (x + ww + sw, yy + 6)], 2.5 * g)
    paint(C, pm.arr(), (14, 24, 14))
    # insects take flight
    if t > 7:
        r2 = rng(102)
        for i in range(7):
            ph = r2.uniform(0, 6.28)
            ix = 600 + math.sin(t * 0.8 + ph) * 200 + i * 20
            iy = 200 + math.sin(t * 1.7 + ph * 2) * 50
            blob(C, ix, iy, 5, (255, 230, 180), 0.5 * smooth(t, 7, 9), 2)
            disc(C, ix, iy, 1.6, (20, 20, 20))
    # tiktaalik crawls ashore
    u = smooth(t, 6.0, d)
    tx = 230 + u * 240
    ty = SHORE_H + 70 + (1 - u) * 20
    m = Mask()
    body = [(tx - 70 + k * 7.5, ty + math.sin(t * 5 - k * 0.5) * 1.5 * (1 - u) + k * 0.2) for k in range(14)]
    wd = np.concatenate([np.linspace(1, 7, 6), np.linspace(7, 6, 4), np.linspace(6, 3, 4)])
    m.poly(tapered(body, wd))
    for k, sg in ((8, 1), (5, -1)):
        bx, by = body[k]
        a = math.sin(t * 2.5 + k) * 0.5
        m.line([(bx, by), (bx + math.sin(a) * 8, by + 9)], 2.5)
    paint(C, m.arr(), (20, 26, 20))
    if u < 0.6:
        add(C, Mask().ring(tx - 40, ty + 6, 30 + (t * 20) % 30, 5, 1).arr(), (255, 220, 180), 0.25)
    blob(C, tx - 10, ty - 6, 60, (255, 200, 140), 0.12, 2)
    return C


# ------------------------------------------------------------------ 11. the great dying
def dying(t, d, T):
    dark = smooth(t, 2, d) * 0.5
    C = grad((0, (40, 6, 4)), (0.55, (150, 40, 14)), (0.62, (200, 70, 20)), (1.0, (20, 6, 4)))
    cl = noise_field(0.6, t * 25, 30, NOISE)
    C *= (1 - np.clip(cl - 0.35, 0, 1)[..., None] * 1.2 * (YY < 360)[..., None])
    C *= 1 - dark
    gnd = [(float(x), 360 + 12 * math.sin(x * 0.01) + 8 * math.sin(x * 0.037)) for x in np.linspace(-10, W + 10, 40)]
    paint(C, Mask().poly(gnd + [(W + 10, H), (-10, H)]).arr(), (12, 6, 5))
    # glowing fissures
    fis = Mask()
    r = rng(111)
    fiss = []
    for k in range(3):
        y0 = 395 + k * 45
        pts = [(float(x), y0 + r.normal(0, 4) + 10 * math.sin(x * 0.01 + k)) for x in np.linspace(80 + k * 90, 900 - k * 60, 25)]
        fis.line(pts, 3 - k * 0.6)
        fiss.append(pts)
    add(C, fis.arr(), (255, 120, 30), 2.0, glow_r=14, glow_k=2.0)
    # lava fountains
    n = 2500
    src = r.integers(0, 3, n)
    pos = r.integers(0, 25, n)
    v0 = r.uniform(60, 180, n)
    vx = r.normal(0, 18, n)
    ph = r.uniform(0, 1, n)
    L = 2.2
    age = ((t / L + ph) % 1.0) * L
    x0 = np.array([fiss[s][p][0] for s, p in zip(src, pos)])
    y0 = np.array([fiss[s][p][1] for s, p in zip(src, pos)])
    x = x0 + vx * age
    y = y0 - v0 * age + 90 * age ** 2
    ok = y <= y0 + 1
    hot = np.clip(1 - age / L, 0, 1)
    colr = np.stack([np.ones(n), 0.35 + 0.5 * hot, 0.1 * hot], 1).astype(np.float32)
    glowing_points(C, x[ok], y[ok], colr[ok], (0.6 + hot[ok]) * 1.2, halo=0.5, r=8)
    tm = Mask()
    for i, (x, h) in enumerate([(70, 160), (230, 120), (760, 190), (900, 140)]):
        S.bare_tree(tm, x, 362 + 6 * math.sin(x * 0.01), h, 112 + i)
    paint(C, tm.arr(), (8, 4, 4))
    snow(C, t, 350, 113, (160, 140, 130), speed=25, k=0.5, dx=8)
    for lt in (4.2, 7.6):
        a = t - lt
        if 0 < a < 0.4:
            C += 0.5 * math.exp(-a * 10)
    return C


# ------------------------------------------------------------------ 12. dinosaurs
DINO_HORIZON = 330


def dino_landscape(t, T, dusk=0.0):
    C = grad((0, (45, 40, 85)), (0.45, (200, 110, 90)), (0.61, (255, 165, 80)), (0.62, (60, 40, 50)),
             (1.0, (14, 10, 14)))
    if dusk:
        C *= 1 - dusk * 0.55
    blob(C, 300, 300, 200, (255, 170, 90), 0.8 * (1 - dusk), 2.2)
    disc(C, 300, 300, 52, (255, 210, 150), alpha=1 - dusk * 0.5)
    # volcano and its plume
    v = Mask().poly([(640, DINO_HORIZON + 5), (760, 230), (790, 230), (930, DINO_HORIZON + 5)])
    paint(C, v.arr(), (70, 45, 60))
    sm = noise_field(1.2, -T * 4, T * 18, NOISE_B)
    plume = np.exp(-((XX - 775 - (230 - YY) * 0.5) / (20 + (230 - YY) * 0.45)) ** 2) * (YY < 232)
    C *= 1 - (plume * np.clip(sm * 1.1, 0, 1) * 0.6)[..., None]
    hills = [(float(x), DINO_HORIZON + 10 - 25 * math.sin(x * 0.006 + 1) - 12 * math.sin(x * 0.019))
             for x in np.linspace(-10, W + 10, 40)]
    paint(C, Mask().poly(hills + [(W + 10, H), (-10, H)]).arr(), (60, 38, 50))
    return C


def _dino_cast(C, t, T, freeze=None):
    tt = t if freeze is None else freeze
    mid = Mask()
    S.sauropod(mid, 820 - tt * 9, 400, 0.85, tt * 2.2, -1)
    S.sauropod(mid, 640 - tt * 7, 392, 0.55, tt * 2.6 + 1, -1)
    S.sauropod(mid, 980 - tt * 8, 396, 0.45, tt * 2.4 + 2, -1)
    paint(C, mid.arr(), (40, 24, 32), 0.92)
    front = Mask()
    gnd = [(float(x), 455 + 10 * math.sin(x * 0.012)) for x in np.linspace(-10, W + 10, 30)]
    front.poly(gnd + [(W + 10, H), (-10, H)])
    for x, h in [(30, 180), (95, 120), (900, 210)]:
        S.conifer(front, x, 470, h, h * 0.45)
    for x, h in [(170, 40), (760, 55), (845, 45), (600, 30)]:
        S.cycad(front, x, 468, h, T)
    if tt > 5:
        S.theropod(front, -120 + (tt - 5) * 48, 468, 0.85, (tt - 5) * 4.5, 1)
    paint(C, front.arr(), (12, 8, 10))
    sky = Mask()
    for i in range(3):
        S.pterosaur(sky, 1000 - ((tt * 40 + i * 260) % 1200), 110 + i * 35 + math.sin(tt + i) * 10, 0.42 - i * 0.07,
                    tt * 5 + i)
    paint(C, sky.arr(), (30, 20, 30))


def dinos(t, d, T):
    C = dino_landscape(t, T)
    _dino_cast(C, t, T)
    # the first flowers
    fg = smooth(t, 10, 15)
    if fg > 0:
        r = rng(121)
        for i in range(16):
            x, y = r.uniform(520, 940), r.uniform(478, 530)
            g = smooth(t, 10 + i * 0.25, 12 + i * 0.25)
            pc = [(255, 120, 170), (255, 240, 240), (255, 200, 90)][i % 3]
            for k in range(5):
                a = k * 1.2566 + i
                disc(C, x + math.cos(a) * 3.5 * g, y + math.sin(a) * 3.5 * g, 2.6 * g, pc, alpha=g)
            disc(C, x, y, 1.5 * g, (255, 230, 120), alpha=g)
    return C


# ------------------------------------------------------------------ 13. the asteroid
IMPACT_XY = (420, DINO_HORIZON + 4)


def asteroid(t, d, T):
    ti = T - TL.IMPACT_T
    dusk = smooth(ti, 0.5, 3.5) if ti > 0 else 0.15
    C = dino_landscape(17 + t, T, dusk=dusk * 0.9)
    _dino_cast(C, 17 + t, T, freeze=17 + min(t, 4.0))
    if ti < 0:
        u = smooth(t, 0.2, 4.0) ** 2
        sx, sy = 1050 + (IMPACT_XY[0] - 1050) * u, -80 + (IMPACT_XY[1] - -80) * u
        # trail
        n = 400
        k = np.linspace(0, 1, n)
        trail_u = np.clip(u - k * 0.25, 0, 1)
        tx = 1050 + (IMPACT_XY[0] - 1050) * trail_u + rng(131).normal(0, 3, n) * k * 6
        ty = -80 + (IMPACT_XY[1] + 80) * trail_u + rng(132).normal(0, 3, n) * k * 6
        glowing_points(C, tx, ty, col((255, 170, 90)), (1 - k) * 2.0 * smooth(t, 0.2, 1.0), halo=0.6, r=10, soft=1)
        blob(C, sx, sy, 30 + 60 * u, (255, 220, 160), 1.5 + 2.0 * u, 2)
        disc(C, sx, sy, 4 + 6 * u, (255, 250, 230))
        return C
    # impact
    C *= 1 - 0.7 * smooth(ti, 0.5, 5)
    fl = math.exp(-ti * 1.6)
    rr = ti * 260
    sw = Mask().ring(IMPACT_XY[0], IMPACT_XY[1], rr, rr * 0.55, 5 + ti * 3).arr()
    sw[int(IMPACT_XY[1]) + 2:] = 0
    add(C, sw, (255, 200, 140), 1.6 * math.exp(-ti * 0.6), glow_r=16, glow_k=1.5)
    fire = np.exp(-((YY - IMPACT_XY[1]) / (20 + ti * 20)) ** 2) * np.exp(-((XX - IMPACT_XY[0]) / (60 + ti * 220)) ** 2)
    C += fire[..., None] * col((255, 110, 40)) * (1.5 * math.exp(-ti * 0.25))
    blob(C, IMPACT_XY[0], IMPACT_XY[1], 150 + ti * 120, (255, 150, 70), 2.5 * fl + 0.4, 2)
    C += np.float32(4.0 * math.exp(-ti * 2.2))
    snow(C, t, 300, 133, (255, 120, 50), speed=-30, k=0.8 * smooth(ti, 1.0, 2.5), dx=10)
    return C


# ------------------------------------------------------------------ 14. rise of mammals
def mammals(t, d, T):
    dawn = smooth(t, 0, d * 0.8)
    C = grad((0, (20, 25, 60)), (0.5, (120, 80, 120)), (0.66, (255, 170, 130)), (1.0, (30, 20, 30))) * (0.5 + 0.5 * dawn)
    blob(C, 760, 360, 260, (255, 200, 140), 0.4 + 0.8 * dawn, 2.0)
    disc(C, 760, 380 - 40 * dawn, 30, (255, 225, 180), alpha=dawn)
    # god rays
    ang = np.arctan2(YY - 360, XX - 760)
    rr = np.clip(np.sin(ang * 25 + 0.3 * math.sin(t * 0.2)) * np.sin(ang * 9 + 1) - 0.3, 0, 1)
    C += (rr * np.clip(1 - np.hypot(XX - 760, YY - 360) / 700, 0, 1))[..., None] * col((255, 210, 160)) * 0.18 * dawn
    far = Mask()
    for i in range(30):
        far.ellipse(i * 34, 395 + 10 * math.sin(i * 1.7), 32, 30)
    far.rect(-5, 395, W + 5, H)
    paint(C, far.arr(), (60, 40, 55), 0.9)
    trees = Mask()
    for i, (x, h) in enumerate([(70, 250), (240, 210), (520, 230), (900, 260)]):
        S.broadleaf(trees, x, 470, h, 141 + i, sway=math.sin(t * 0.6 + i) * 3)
    trees.rect(-5, 468, W + 5, H)
    # branches for the primates
    trees.line([(520, 375), (600, 360), (665, 362)], 6)
    trees.line([(240, 385), (190, 372), (140, 374)], 5)
    # small mammals scurrying
    for i in range(4):
        x = (t * (60 + i * 10) + i * 230) % (W + 100) - 50
        y = 466
        hop = abs(math.sin(t * 9 + i)) * 4
        trees.ellipse(x, y - 7 - hop, 11, 6)
        trees.ellipse(x + 10, y - 9 - hop, 5, 4)
        trees.line([(x - 10, y - 7 - hop), (x - 22, y - 4 - hop)], 1.5)
    # primates appear
    pa = smooth(t, 5, 7)
    if pa > 0:
        for (bx, by, f, s) in [(625, 358, 1, 1.0), (170, 370, -1, 0.85)]:
            yy = by - (1 - pa) * 30
            trees.ellipse(bx, yy - 16 * s, 9 * s, 13 * s)
            trees.ellipse(bx + 3 * f * s, yy - 34 * s, 7 * s, 7 * s)
            trees.line([(bx, yy - 22 * s), (bx + 14 * f * s, yy - 2 * s)], 3 * s)
            tail = bezier((bx - 4 * f * s, yy - 6 * s), (bx - 8 * f * s, yy + 30 * s),
                          (bx + math.sin(t * 1.5) * 14 * s, yy + 50 * s), n=10)
            trees.line(tail, 2.2 * s)
    paint(C, trees.arr(), (14, 10, 16))
    # fireflies fade with dawn
    r = rng(142)
    n = 40
    x = r.uniform(0, W, n) + np.sin(t * 0.7 + r.uniform(0, 6, n)) * 20
    y = r.uniform(300, 470, n) + np.cos(t * 0.9 + r.uniform(0, 6, n)) * 10
    w = (np.sin(t * 3 + r.uniform(0, 6, n)) * 0.5 + 0.5) * (1 - dawn)
    glowing_points(C, x, y, col((200, 255, 120)), w * 2, halo=0.8, r=6)
    # birds
    if t > 7:
        bm = Mask()
        for i in range(7):
            bx = 1000 - (t - 7) * 90 + abs(i - 3) * 22
            by = 120 + abs(i - 3) * 12 + i * 2
            fl = math.sin(t * 8 + i) * 4
            bm.line([(bx - 7, by - fl), (bx, by), (bx + 7, by - fl)], 1.6)
        paint(C, bm.arr(), (30, 20, 30))
    return C


# ------------------------------------------------------------------ 15. the first humans
def _savanna(t, T):
    C = grad((0, (60, 45, 90)), (0.4, (220, 110, 70)), (0.665, (255, 180, 90)), (0.67, (70, 40, 30)),
             (1.0, (25, 14, 10)))
    blob(C, 600, 360, 230, (255, 170, 90), 1.0, 2.0)
    disc(C, 600, 360, 70, (255, 200, 120))
    m = Mask()
    m.poly([(float(x), 362 + 6 * math.sin(x * 0.02)) for x in np.linspace(-10, W + 10, 30)] + [(W + 10, H), (-10, H)])
    for i, (x, h) in enumerate([(120, 150), (820, 190), (930, 110)]):
        S.acacia(m, x, 365, h, 151 + i)
    for i in range(140):
        gx = i * 7 + (i * 37) % 5
        gh = 6 + (i * 13) % 9
        m.line([(gx, 366), (gx + math.sin(t * 1.5 + i * 0.3) * 2, 366 - gh)], 1.2)
    paint(C, m.arr(), (22, 12, 10))
    walkers = Mask()
    for i in range(5):
        posture = i / 4
        x = 270 + i * 95 + t * 16
        S.human(walkers, x, 368, 1.0 + 0.12 * posture, posture, t * 4.2 + i * 1.3, 1)
    paint(C, walkers.arr(), (14, 8, 6))
    return C


def _campfire(t, T):
    C = grad((0, (4, 6, 20)), (0.6, (14, 16, 40)), (0.62, (10, 8, 10)), (1.0, (6, 4, 4)))
    STARS.draw(C, T, k=0.6, glow_k=0.0)
    fx_, fy_ = 600, 440
    flick = 0.85 + 0.1 * math.sin(t * 13) + 0.05 * math.sin(t * 29)
    light = np.exp(-(((XX - fx_) / 340) ** 2 + ((YY - fy_) / 170) ** 2)) * (YY > 330)
    C += light[..., None] * col((255, 130, 50)) * 0.55 * flick
    fm = Mask()
    for k in range(7):
        ph = k * 0.9
        h = 50 + 20 * math.sin(t * 7 + ph)
        cx = fx_ + (k - 3) * 7
        fm.poly(tapered(bezier((cx, fy_), (cx + math.sin(t * 6 + ph) * 8, fy_ - h * 0.5),
                               (cx + math.sin(t * 9 + ph) * 10, fy_ - h), n=10), np.linspace(8, 0.5, 10)))
    add(C, fm.arr(), (255, 150, 50), 1.6, glow_r=20, glow_k=1.8)
    blob(C, fx_, fy_ - 15, 40, (255, 230, 160), 1.5 * flick, 2)
    r = rng(161)
    n = 120
    ph = r.uniform(0, 1, n)
    age = (t * 0.6 + ph) % 1.0
    x = fx_ + r.normal(0, 12, n) + np.sin(age * 9 + ph * 20) * 18 * age
    y = fy_ - 20 - age * 220
    glowing_points(C, x, y, col((255, 170, 70)), (1 - age) * 1.5, halo=0.5, r=6)
    sm = Mask()
    for x, f in [(480, 1), (720, -1), (530, 1), (680, -1)]:
        S.sitting_human(sm, x, 455 + (10 if x in (530, 680) else 0), 1.05, f)
    sm.line([(fx_ - 30, fy_ + 4), (fx_ + 30, fy_ - 2)], 5)
    sm.line([(fx_ - 28, fy_ - 3), (fx_ + 28, fy_ + 5)], 5)
    paint(C, sm.arr(), (10, 6, 6))
    return C


def humans(t, d, T):
    cf = smooth(t, 8.0, 9.8)
    if cf <= 0:
        return _savanna(t, T)
    if cf >= 1:
        return _campfire(t, T)
    return _savanna(t, T) * (1 - cf) + _campfire(t, T) * cf
