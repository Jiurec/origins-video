"""Procedural silhouettes drawn into a Mask (internal-pixel coordinates)."""
import math
import numpy as np
from .fx import Mask, tapered, bezier, rot, ellipse_pts, rng


def sauropod(m, x, y, s, phase, facing=-1):
    """Long-necked dinosaur. (x, y) = ground point under the body."""
    f = facing
    by = y - 62 * s
    m.ellipse(x, by, 62 * s, 27 * s)
    # legs (walk cycle)
    for i, (lx, off) in enumerate([(-38, 0), (-30, math.pi), (34, math.pi), (42, 0)]):
        a = math.sin(phase + off) * 0.28
        top = (x + lx * s * f, by + 8 * s)
        foot = (top[0] + math.sin(a) * 50 * s * f, y)
        m.line([top, foot], 13 * s if i in (0, 3) else 11 * s)
    sway = math.sin(phase * 0.5) * 6 * s
    neck = bezier((x + 45 * s * f, by - 6 * s), (x + 95 * s * f, by - 40 * s),
                  (x + 110 * s * f + sway, by - 120 * s), n=16)
    m.poly(tapered(neck, np.linspace(13 * s, 5 * s, 16)))
    hx, hy = neck[-1]
    m.ellipse(hx + 7 * s * f, hy, 11 * s, 6 * s)
    tail = bezier((x - 50 * s * f, by - 4 * s), (x - 110 * s * f, by + 6 * s),
                  (x - 170 * s * f, by + 26 * s + math.sin(phase) * 5 * s), n=18)
    m.poly(tapered(tail, np.linspace(15 * s, 1 * s, 18)))


def theropod(m, x, y, s, phase, facing=1):
    f = facing
    hip = (x, y - 70 * s)
    body = bezier((hip[0] - 10 * s * f, hip[1]), (hip[0] + 30 * s * f, hip[1] - 8 * s),
                  (hip[0] + 55 * s * f, hip[1] - 22 * s), n=10)
    m.poly(tapered(body, np.linspace(20 * s, 13 * s, 10)))
    hx, hy = body[-1]
    head = [(0, -12), (38, -6), (40, 4), (6, 12), (-8, 6)]
    m.poly([(hx + px * s * f, hy + py * s) for px, py in head])
    tail = bezier((hip[0] - 5 * s * f, hip[1] - 2 * s), (hip[0] - 60 * s * f, hip[1] - 8 * s),
                  (hip[0] - 120 * s * f, hip[1] - 2 * s + math.sin(phase) * 4 * s), n=14)
    m.poly(tapered(tail, np.linspace(16 * s, 1 * s, 14)))
    for off in (0, math.pi):
        a = math.sin(phase + off) * 0.45
        knee = (hip[0] + (10 + math.sin(a) * 25) * s * f, hip[1] + 32 * s)
        foot = (hip[0] + (math.sin(a) * 40 - 4) * s * f, y)
        m.line([hip, knee, foot], 11 * s)
    m.line([(hx - 12 * s * f, hy + 10 * s), (hx - 2 * s * f, hy + 20 * s)], 4 * s)


def pterosaur(m, x, y, s, phase):
    flap = math.sin(phase) * 0.6
    for side in (-1, 1):
        tip = (x + side * 70 * s, y - math.sin(flap) * 40 * s * 1.0 + 10 * s)
        elbow = (x + side * 32 * s, y - math.sin(flap) * 18 * s - 4 * s)
        m.poly([(x, y - 3 * s), elbow, tip, (x + side * 20 * s, y + 6 * s)])
    m.ellipse(x, y, 10 * s, 5 * s)
    m.poly([(x + 8 * s, y - 4 * s), (x + 30 * s, y - 2 * s), (x + 8 * s, y + 2 * s)])
    m.poly([(x - 2 * s, y - 4 * s), (x - 14 * s, y - 14 * s), (x + 4 * s, y - 5 * s)])


def conifer(m, x, y, h, w):
    m.line([(x, y), (x, y - h)], max(2, w * 0.08))
    for i in range(7):
        yy = y - h * (0.25 + 0.11 * i)
        ww = w * (1 - i / 8)
        m.poly([(x - ww / 2, yy + h * 0.05), (x, yy - h * 0.14), (x + ww / 2, yy + h * 0.05)])


def cycad(m, x, y, h, phase=0.0):
    m.line([(x, y), (x, y - h)], h * 0.18)
    for i in range(9):
        a = -math.pi + (i + 0.5) * math.pi / 9 + math.sin(phase + i) * 0.04
        tip = (x + math.cos(a) * h * 1.1, y - h + math.sin(a) * h * 0.7 + h * 0.35)
        mid = (x + math.cos(a) * h * 0.55, y - h + math.sin(a) * h * 0.6 - h * 0.1)
        m.poly(tapered(bezier((x, y - h), mid, tip, n=8), np.linspace(h * 0.09, 1, 8)))


def broadleaf(m, x, y, h, seed, sway=0.0):
    r = rng(seed)
    m.line([(x, y), (x + sway * 0.3, y - h * 0.6)], h * 0.06)
    for _ in range(9):
        cx = x + sway + r.normal(0, h * 0.18)
        cy = y - h * 0.75 + r.normal(0, h * 0.12)
        rr = h * r.uniform(0.16, 0.26)
        m.ellipse(cx, cy, rr * 1.2, rr)


def acacia(m, x, y, h, seed):
    r = rng(seed)
    m.line([(x, y), (x - h * 0.05, y - h * 0.55), (x - h * 0.25, y - h * 0.85)], h * 0.05)
    m.line([(x - h * 0.05, y - h * 0.55), (x + h * 0.22, y - h * 0.86)], h * 0.04)
    for i in range(7):
        cx = x - h * 0.02 + (i - 3) * h * 0.13 + r.normal(0, h * 0.03)
        m.ellipse(cx, y - h * 0.9 + r.normal(0, h * 0.02), h * 0.2, h * 0.07)


def bare_tree(m, x, y, h, seed):
    r = rng(seed)

    def branch(x0, y0, ang, ln, wd, depth):
        x1, y1 = x0 + math.cos(ang) * ln, y0 + math.sin(ang) * ln
        m.line([(x0, y0), (x1, y1)], wd)
        if depth > 0:
            for da in (-0.45, 0.4):
                branch(x1, y1, ang + da + r.normal(0, 0.15), ln * r.uniform(0.6, 0.8), wd * 0.65, depth - 1)

    branch(x, y, -math.pi / 2 + r.normal(0, 0.08), h * 0.38, h * 0.05, 4)


def human(m, x, y, s, posture, phase, facing=1, walk=True):
    """posture 0 = knuckle-walking ape, 1 = upright human. (x, y) ground."""
    f = facing
    lean = (1 - posture) * 0.9
    leg = (38 + 14 * posture) * s
    hip = (x, y - leg)
    torso = 44 * s
    sh = (hip[0] + math.sin(lean) * torso * f, hip[1] - math.cos(lean) * torso)
    head = (sh[0] + (6 + 10 * (1 - posture)) * s * f, sh[1] - (12 - 4 * (1 - posture)) * s)
    m.line([hip, sh], (12 + 4 * (1 - posture)) * s)
    m.ellipse(head[0], head[1], 9 * s, 10 * s)
    m.line([sh, ((sh[0] + head[0]) / 2, (sh[1] + head[1]) / 2)], 7 * s)
    for off in (0, math.pi):
        a = (math.sin(phase + off) * 0.4) if walk else 0.1 * (1 if off else -1)
        knee = (hip[0] + math.sin(a) * leg * 0.5 * f + 3 * s * f, hip[1] + leg * 0.5)
        foot = (hip[0] + math.sin(a) * leg * 0.8 * f, y)
        m.line([hip, knee, foot], 7 * s)
        arm_len = (44 + 16 * (1 - posture)) * s
        aa = -a * 0.8 + lean * 0.6
        elbow = (sh[0] + math.sin(aa) * arm_len * 0.5 * f, sh[1] + math.cos(aa) * arm_len * 0.5)
        hand = (sh[0] + math.sin(aa + 0.2) * arm_len * f, min(sh[1] + math.cos(aa + 0.2) * arm_len, y))
        m.line([sh, elbow, hand], 5.5 * s)


def sitting_human(m, x, y, s, facing=1):
    f = facing
    m.ellipse(x, y - 52 * s, 8 * s, 9 * s)
    m.line([(x, y - 44 * s), (x - 2 * s * f, y - 12 * s)], 13 * s)
    m.line([(x - 2 * s * f, y - 12 * s), (x + 22 * s * f, y - 18 * s), (x + 26 * s * f, y)], 7 * s)
    m.line([(x, y - 38 * s), (x + 18 * s * f, y - 24 * s), (x + 24 * s * f, y - 32 * s)], 5 * s)


def trilobite(m, x, y, s, ang):
    pts = ellipse_pts(0, 0, 24 * s, 14 * s, 30)
    m.poly(rot(pts, ang, x, y))


def gear_pts(cx, cy, r, teeth, ang, depth=0.18):
    pts = []
    n = teeth * 4
    for i in range(n):
        a = ang + 2 * math.pi * i / n
        rr = r if (i % 4) in (0, 1) else r * (1 - depth)
        pts.append((cx + math.cos(a) * rr, cy + math.sin(a) * rr))
    return pts
