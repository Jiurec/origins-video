"""Scenes 0-4: intro, Big Bang, first stars, solar system, Hadean Earth."""
import math
import numpy as np
from .fx import *
from . import timeline as TL

STARS = Starfield(1100, seed=3)
STARS_FAR = Starfield(1600, seed=4)


# ------------------------------------------------------------------ intro
def intro(t, d, T):
    C = canvas((0, 0, 0))
    k = smooth(t, 1.5, d) * (0.6 + 0.4 * math.sin(t * 9) ** 2)
    blob(C, W / 2, H / 2, 6 + 10 * k, (255, 245, 230), k * 1.5, 2)
    return C


# ------------------------------------------------------------------ big bang
_r = rng(10)
_BB_N = 9000
_bb_ang = _r.uniform(0, 2 * math.pi, _BB_N)
_bb_z = _r.uniform(-1, 1, _BB_N)
_bb_spd = _r.uniform(0.05, 1.0, _BB_N) ** 0.6
_bb_w = _r.uniform(0.2, 1.0, _BB_N)
_bb_web = sample(NOISE, np.cos(_bb_ang) * 120 + 256 + _bb_spd * 80, np.sin(_bb_ang) * 120 + 256 + _bb_spd * 140)
_bb_w *= np.clip((_bb_web - 0.35) * 3.0, 0.05, 1.5)


def _temp_color(u):
    """Cooling fireball: white-blue -> yellow -> orange -> deep red."""
    stops = [(0.0, (1.6, 1.6, 1.9)), (0.12, (1.4, 1.25, 0.9)), (0.3, (1.2, 0.75, 0.35)),
             (0.6, (0.8, 0.3, 0.12)), (1.0, (0.3, 0.08, 0.06))]
    u = min(max(u, 0), 1)
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if u <= b:
            f = (u - a) / (b - a)
            return np.array([ca[i] + (cb[i] - ca[i]) * f for i in range(3)], np.float32)
    return np.array(stops[-1][1], np.float32)


def bigbang(t, d, T):
    C = canvas((0, 0, 0))
    tb = T - TL.BANG_T
    cx, cy = W / 2, H / 2
    if tb < 0:
        k = 1.5 + 2.0 * smooth(tb, -2, 0)
        jit = 1.5 * smooth(tb, -1, 0)
        blob(C, cx + math.sin(t * 50) * jit, cy + math.cos(t * 43) * jit, 16 + 14 * smooth(tb, -2, 0),
             (255, 245, 230), k, 2)
        return C
    u = tb / (d - 2.0)
    # expanding particle cloud (3-D directions projected)
    R = 620 * (1 - math.exp(-tb * 0.42)) + 18 * tb
    rad = _bb_spd * R
    sz = np.sqrt(np.clip(1 - _bb_z ** 2, 0, 1))
    x = cx + np.cos(_bb_ang) * rad * sz
    y = cy + np.sin(_bb_ang) * rad * sz * 0.92
    colr = _temp_color(u * 1.15)
    fade = math.exp(-max(tb - 8.0, 0) * 0.45)
    w = _bb_w * (0.8 + 0.5 * _bb_spd) * fade * 2.6
    glowing_points(C, x, y, colr, w, core=1.0, halo=0.6, r=12, soft=1)
    # fireball core
    core = math.exp(-tb * 0.9)
    blob(C, cx, cy, 80 + tb * 140, tuple(colr / 2 * 255), 2.2 * core, 1.6)
    # cosmic microwave background mottling (the afterglow)
    cmb = smooth(tb, 4.0, 7.0) * (1 - smooth(tb, 8.5, 11.5)) * 0.16
    if cmb > 0.002:
        n = noise_field(0.9, tb * 6, 40, NOISE_FINE)
        C += (np.stack([n * 1.0, n * 0.55, 0.35 - n * 0.2], -1) * cmb).astype(np.float32)
    # the initial flash
    flash = math.exp(-tb * 3.2) * 3.0
    C += flash
    return C


# ------------------------------------------------------------------ first stars & galaxy
_r = rng(20)
_IGN = [(_r.uniform(80, 880), _r.uniform(60, 470), _r.uniform(0.6, 1.3)) for _ in TL.STAR_IGNITIONS]
_GN = 12000
_g_r = _r.uniform(0, 1, _GN) ** 1.6
_g_arm = _r.integers(0, 3, _GN)
_g_off = _r.normal(0, 0.32, _GN) * (0.4 + _g_r)
_g_th0 = 1.9 * np.log(_g_r * 8 + 0.4) + _g_arm * (2 * math.pi / 3) + _g_off
_g_diff = _r.uniform(0, 1, _GN) < 0.3
_g_th0[_g_diff] = _r.uniform(0, 2 * math.pi, _g_diff.sum())
_g_sx = _r.uniform(-1, 1, _GN)
_g_sy = _r.uniform(-1, 1, _GN)
_g_w = _r.uniform(0.2, 1.0, _GN)
_g_col = np.where((_g_r < 0.18)[:, None], np.array([1.0, 0.85, 0.6]),
                  np.where((_r.uniform(0, 1, _GN) < 0.25)[:, None], np.array([1.0, 0.6, 0.75]),
                           np.array([0.6, 0.75, 1.0]))).astype(np.float32)


def stars(t, d, T):
    C = canvas((2, 3, 10))
    zoom = 1.0 + t * 0.012
    STARS_FAR.draw(C, T, k=0.25 + 0.5 * smooth(t, 0, 8), zoom=zoom, glow_k=0.0)
    # ignitions of the first stars
    for (x, y, s), ti in zip(_IGN, TL.STAR_IGNITIONS):
        a = T - ti
        if a < 0:
            continue
        flash = math.exp(-a * 2.5)
        steady = 0.6 * s
        blob(C, x, y, 30 * s + 60 * flash, (170, 200, 255), steady * 0.7 + 2.5 * flash, 3.0)
        blob(C, x, y, 4 + 3 * s, (255, 255, 255), 1.5, 1.0)
    # galaxy assembling from scattered gas
    g = smooth(t, 4.0, 13.0)
    if g > 0:
        gx, gy, size = 520, 270, 300
        th = _g_th0 + t * 0.12
        sp_x = np.cos(th) * _g_r * size
        sp_y = np.sin(th) * _g_r * size
        px = sp_x * (g) + _g_sx * 520 * (1 - g)
        py = sp_y * (g) + _g_sy * 300 * (1 - g)
        tilt = -0.4
        ct, st = math.cos(tilt), math.sin(tilt)
        x = gx + (px * ct - py * 0.62 * st)
        y = gy + (px * st + py * 0.62 * ct)
        w = _g_w * (0.35 + 0.65 * g) * smooth(t, 2.0, 6.0) * 0.55
        glowing_points(C, x, y, _g_col, w, core=1.0, halo=0.7, r=14, soft=1)
        blob(C, gx, gy, 120, (255, 220, 160), 0.9 * g, 2.2)
        blob(C, gx, gy, 30, (255, 240, 210), 1.5 * g, 1.5)
    # supernova
    a = T - TL.SUPERNOVA_T
    if a > 0:
        sx, sy = 205, 145
        fl = math.exp(-a * 1.4)
        blob(C, sx, sy, 40 + 160 * fl, (220, 230, 255), 3.0 * fl, 2.5)
        rr = 10 + a * 45
        m = Mask().ring(sx, sy, rr, rr * 0.92, 3 + a * 1.5).arr()
        add(C, m, (255, 150, 110), 0.8 * math.exp(-a * 0.6), glow_r=12, glow_k=1.2)
        m = Mask().ring(sx, sy, rr * 0.8, rr * 0.75, 2).arr()
        add(C, m, (120, 200, 255), 0.6 * math.exp(-a * 0.6), glow_r=8, glow_k=1.0)
    return C


# ------------------------------------------------------------------ solar system formation
_r = rng(30)
_DN = 16000
_d_r = 40 + _r.uniform(0, 1, _DN) ** 0.8 * 400
_d_th = _r.uniform(0, 2 * math.pi, _DN)
_d_h = _r.normal(0, 1, _DN)
_d_w = _r.uniform(0.2, 1.0, _DN)
_PLANETS = [(95, 0.3, 4), (140, 2.0, 5), (195, 4.1, 6), (265, 1.2, 5), (360, 3.3, 10)]  # radius, phase, size
EARTH_ORBIT = 2


def solar(t, d, T):
    C = canvas((2, 2, 6))
    STARS.draw(C, T, k=0.5, glow_k=0.0)
    cx, cy = 480, 270
    zoom = 1.0 + 0.25 * smooth(t, 0, d)
    flat = 0.3
    # disk particles with co-rotating clumps and planet gaps
    om = 9.0 / (_d_r / 100) ** 1.5 * 0.12
    th = _d_th + om * t
    form = smooth(t, 2.0, 12.0)
    dens = sample(NOISE, _d_r * 0.9, (_d_th % (2 * math.pi)) * 80)
    w = _d_w * (0.5 + dens)
    for pr, _, _ in _PLANETS:
        w *= 1 - 0.9 * form * np.exp(-((_d_r - pr) / 9) ** 2)
    w *= 1 - 0.55 * smooth(t, 8, d)
    x = cx + np.cos(th) * _d_r * zoom
    y = cy + (np.sin(th) * _d_r * flat + _d_h * (3 + _d_r * 0.02)) * zoom
    temp = np.clip(1 - _d_r / 300, 0, 1)
    colr = np.stack([0.75 + 0.25 * temp, 0.4 + 0.45 * temp, 0.25 + 0.4 * temp ** 2], 1).astype(np.float32)
    back = np.sin(th) < 0
    glowing_points(C, x[back], y[back], colr[back], w[back] * 0.8, halo=0.35, r=8)
    # the Sun ignites
    ig = smooth(t, 2.5, 5.5)
    sun_col = mix_col((255, 90, 40), (255, 235, 190), ig)
    blob(C, cx, cy, (45 + 70 * ig) * zoom, tuple(sun_col), 0.5 + 1.0 * ig, 2.4)
    blob(C, cx, cy, (14 + 8 * ig) * zoom, (255, 250, 235), 1.0 + 2.0 * ig, 1.2)
    flare = math.exp(-max(t - 2.8, 0) * 1.5) * (t > 2.8)
    blob(C, cx, cy, 260 * zoom, (255, 230, 180), 1.2 * flare, 2.5)
    front = ~back
    glowing_points(C, x[front], y[front], colr[front], w[front] * 0.8, halo=0.35, r=8)
    # planets accreting
    for i, (pr, ph, size) in enumerate(_PLANETS):
        a = ph + 9.0 / (pr / 100) ** 1.5 * 0.12 * t
        px = cx + math.cos(a) * pr * zoom
        py = cy + math.sin(a) * pr * flat * zoom
        g = smooth(t, 3 + i * 0.7, 11 + i * 0.5)
        if g <= 0:
            continue
        pc = (130, 170, 255) if i == EARTH_ORBIT else (230, 180, 140)
        blob(C, px, py, size * 3 * zoom, pc, 0.6 * g, 2)
        disc(C, px, py, max(size * 0.6 * g * zoom, 0.8), tuple(min(255, c + 60) for c in pc), alpha=g)
        if i == EARTH_ORBIT:
            hl = smooth(t, 10, 12) * (1 - smooth(t, d - 1.0, d))
            if hl > 0:
                m = Mask().ring(px, py, 16 * zoom, 16 * zoom, 1.2).arr()
                add(C, m, (150, 210, 255), 0.9 * hl)
    return C


# ------------------------------------------------------------------ Hadean Earth & Theia
EC = (585, 285)
ER = 150
_r = rng(40)
_JN = 6000
_j_spread = _r.normal(0, 0.35, _JN)
_j_a = _r.uniform(1.25, 2.6, _JN)
_j_spd = _r.uniform(0.6, 1.6, _JN)
_j_w = _r.uniform(0.3, 1.0, _JN)
_j_h = _r.normal(0, 1, _JN)
_CONTACT = math.radians(205)  # direction from Earth centre to the impact point


def _lava_tex(phase):
    def f(lat, lon):
        n = tex_lookup(NOISE, lat, lon + phase * 0.05, 1.0)
        m = tex_lookup(NOISE_B, lat, lon - phase * 0.03, 1.5)
        crust = np.stack([0.16 + 0.14 * n, 0.10 + 0.07 * n, 0.08 + 0.05 * n], -1)
        return crust * (0.7 + 0.5 * m[:, None])
    return f


def _lava_emit(phase, heat):
    def f(lat, lon, lam):
        n = tex_lookup(NOISE, lat, lon + phase * 0.05, 1.0)
        m = tex_lookup(NOISE_B, lat, lon - phase * 0.03, 1.5)
        veins = np.clip(1 - np.abs(m - 0.5) / (0.025 + 0.05 * heat), 0, 1) ** 1.5
        pools = np.clip((n - 0.68 + 0.35 * heat) * 4, 0, 1)
        v = np.clip(veins + pools, 0, 1) * (0.55 + 0.6 * heat)
        return np.stack([v * 1.5, v * 0.55, v * 0.12], -1)
    return f


def hadean(t, d, T):
    C = canvas((1, 1, 4))
    STARS.draw(C, T, k=0.55, glow_k=0.0, dx=-t * 3)
    ti = T - TL.THEIA_T
    heat = 0.25 + 0.75 * smooth(ti, 0, 0.8) * (1 - 0.5 * smooth(ti, 3, 9)) if ti > 0 else 0.25
    cx, cy = EC
    contact = (cx + math.cos(_CONTACT) * ER, cy + math.sin(_CONTACT) * ER)
    # debris ring split into back / front relative to Earth
    moon_ang = _CONTACT + 0.9 + max(ti, 0) * 0.55
    moon_r = ER * 2.15
    if ti > 0:
        age = ti
        out = smooth(age, 0, 1.6)
        th = _CONTACT + _j_spread + age * 0.55 * _j_spd / _j_a ** 1.5 * 2.2
        rad = ER * (1 + (_j_a - 1) * out)
        cl = smooth(ti, 3.0, 8.0)
        th = th * (1 - cl) + (moon_ang + _j_spread * 0.08) * cl
        rad = rad * (1 - cl) + (moon_r + _j_h * 6) * cl
        px = cx + np.cos(th) * rad
        py = cy + np.sin(th) * rad * 0.35 + _j_h * 6 * (1 - cl)
        w = _j_w * (1.2 - 0.9 * cl) * (1 - 0.7 * smooth(ti, 6, 9))
        hot = np.clip(1 - age * 0.25, 0.25, 1)
        colr = np.array([1.0, 0.45 + 0.4 * hot, 0.2 + 0.3 * hot], np.float32)
        back = np.sin(th) < 0
        glowing_points(C, px[back], py[back], colr, w[back], halo=0.6, r=10)
    else:
        back = None
    # moon behind?
    mx = cx + math.cos(moon_ang) * moon_r
    my = cy + math.sin(moon_ang) * moon_r * 0.35
    mg = smooth(ti, 5.5, 9.0)

    def moon_tex(lat, lon):
        n = tex_lookup(NOISE_FINE, lat, lon, 2.0)
        return np.stack([0.55 + 0.3 * n] * 3, -1) * np.array([1.0, 0.95, 0.9])

    def draw_moon():
        if mg > 0:
            r = 6 + 32 * mg
            sphere(C, mx, my, r, moon_tex, lon0=t * 0.2, light=(-0.7, -0.2, 0.6), ambient=0.02,
                   emit_fn=lambda la, lo, lam: np.stack([np.full_like(lam, 0.35 * (1 - mg))] * 3, -1)
                   * np.array([1, 0.4, 0.1]))

    if math.sin(moon_ang) < 0:
        draw_moon()
    # Earth
    sphere(C, cx, cy, ER, _lava_tex(T), lon0=T * 0.15, tilt=0.25, light=(-0.75, -0.35, 0.55),
           ambient=0.05, emit_fn=_lava_emit(T, heat), atmo=(255, 120, 60), atmo_k=0.35 + 0.4 * heat)
    # Theia approaching
    if ti < 0.15:
        u = smooth(t, 0, 6.0)
        u = u ** 1.6
        start = (-120.0, -60.0)
        tr = 62
        end = (cx + math.cos(_CONTACT) * (ER + tr * 0.6), cy + math.sin(_CONTACT) * (ER + tr * 0.6))
        tx, ty = start[0] + (end[0] - start[0]) * u, start[1] + (end[1] - start[1]) * u

        def theia_tex(lat, lon):
            n = tex_lookup(NOISE_B, lat, lon, 2.0)
            return np.stack([0.45 + 0.3 * n, 0.33 + 0.2 * n, 0.28 + 0.15 * n], -1)

        sphere(C, tx, ty, tr, theia_tex, lon0=t * 0.3, light=(-0.75, -0.35, 0.55), ambient=0.04,
               atmo=(255, 160, 110), atmo_k=0.15)
    if ti > 0:
        glowing_points(C, px[~back], py[~back], colr, w[~back], halo=0.6, r=10)
        fl = math.exp(-ti * 2.2)
        blob(C, contact[0], contact[1], 120 + 260 * (1 - fl), (255, 220, 170), 3.0 * fl, 2.0)
        C += 1.2 * math.exp(-ti * 5.0)
    if math.sin(moon_ang) >= 0:
        draw_moon()
    return C
