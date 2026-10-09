"""Low-level drawing helpers. Scenes render into a float32 RGB canvas at
internal resolution (W x H); values may exceed 1.0 (HDR) and are tone
mapped at the end."""
import math
import numpy as np
from PIL import Image, ImageDraw

W, H = 960, 540
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)


def rng(seed):
    return np.random.default_rng(seed)


def smooth(x, a=0.0, b=1.0):
    """Smoothstep of x between a and b (works on scalars and arrays)."""
    u = np.clip((np.asarray(x, dtype=np.float32) - a) / (b - a + 1e-9), 0, 1)
    r = u * u * (3 - 2 * u)
    return float(r) if r.ndim == 0 else r


def lerp(a, b, u):
    return a + (b - a) * u


def mix_col(c1, c2, u):
    return tuple(lerp(np.float32(a), np.float32(b), u) for a, b in zip(c1, c2))


def col(c):
    """0-255 tuple -> float array."""
    return np.array(c, np.float32) / 255.0


def canvas(c=(0, 0, 0)):
    out = np.empty((H, W, 3), np.float32)
    out[:] = col(c)
    return out


def vgrad(stops):
    """Vertical gradient from [(y_frac, rgb255), ...]."""
    ys = np.linspace(0, 1, H, dtype=np.float32)
    out = np.empty((H, 3), np.float32)
    pos = [s[0] for s in stops]
    for ch in range(3):
        out[:, ch] = np.interp(ys, pos, [s[1][ch] / 255.0 for s in stops])
    return np.repeat(out[:, None, :], W, axis=1)


# ---------------------------------------------------------------- blurring
def _box1(a, r, axis):
    if r < 1:
        return a
    n = a.shape[axis]
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    p = np.pad(a, pad, mode="edge")
    c = np.cumsum(p, axis=axis, dtype=np.float32)
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return (hi - lo) / (2 * r + 1)


def blur(a, r):
    """Approximate gaussian blur (3 box passes)."""
    r = int(max(r, 0))
    if r == 0:
        return a
    b = max(1, int(r / 1.7))
    for _ in range(3):
        a = _box1(a, b, 0)
        a = _box1(a, b, 1)
    return a


def _resize_f(a, w, h):
    if a.ndim == 2:
        return np.asarray(Image.fromarray(a, "F").resize((w, h), Image.BILINEAR))
    return np.stack([_resize_f(np.ascontiguousarray(a[..., c]), w, h) for c in range(a.shape[2])], -1)


def glow(a, r):
    """Wide soft glow computed at quarter resolution."""
    hh, ww = a.shape[0] // 4, a.shape[1] // 4
    small = a.reshape(hh, 4, ww, 4, *a.shape[2:]).mean(axis=(1, 3))
    small = blur(small, max(1, r // 4))
    return _resize_f(small.astype(np.float32), a.shape[1], a.shape[0])


# ---------------------------------------------------------------- particles
def splat(x, y, rgb, w=None):
    """Accumulate points into a new HDR layer (nearest pixel)."""
    x = np.asarray(x)
    y = np.asarray(y)
    xi = np.round(x).astype(np.int64)
    yi = np.round(y).astype(np.int64)
    m = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    if w is None:
        w = np.ones(len(x), np.float32)
    w = np.broadcast_to(np.asarray(w, np.float32), x.shape)[m]
    idx = yi[m] * W + xi[m]
    rgb = np.asarray(rgb, np.float32)
    out = np.empty((H * W, 3), np.float32)
    for c in range(3):
        cw = w * (rgb[m, c] if rgb.ndim == 2 else rgb[c])
        out[:, c] = np.bincount(idx, weights=cw, minlength=H * W)
    return out.reshape(H, W, 3)


def glowing_points(C, x, y, rgb, w=None, core=1.0, halo=0.8, r=10, soft=1):
    layer = splat(x, y, rgb, w)
    if soft:
        layer = blur(layer, soft)
    C += layer * core
    if halo:
        C += glow(layer, r) * halo * 6
    return C


def blob(C, cx, cy, r, rgb, k=1.0, power=2.0):
    """Additive radial glow, computed only inside its bounding box."""
    x0, x1 = int(max(cx - r, 0)), int(min(cx + r + 1, W))
    y0, y1 = int(max(cy - r, 0)), int(min(cy + r + 1, H))
    if x0 >= x1 or y0 >= y1:
        return C
    d = np.sqrt((XX[y0:y1, x0:x1] - cx) ** 2 + (YY[y0:y1, x0:x1] - cy) ** 2) / r
    f = np.clip(1 - d, 0, 1) ** power * k
    C[y0:y1, x0:x1] += f[..., None] * col(rgb)
    return C


def disc(C, cx, cy, r, rgb, alpha=1.0, edge=1.2):
    """Antialiased solid disc (alpha blend)."""
    x0, x1 = int(max(cx - r - 2, 0)), int(min(cx + r + 3, W))
    y0, y1 = int(max(cy - r - 2, 0)), int(min(cy + r + 3, H))
    if x0 >= x1 or y0 >= y1:
        return C
    d = np.sqrt((XX[y0:y1, x0:x1] - cx) ** 2 + (YY[y0:y1, x0:x1] - cy) ** 2)
    a = np.clip((r - d) / edge + 0.5, 0, 1)[..., None] * alpha
    sub = C[y0:y1, x0:x1]
    C[y0:y1, x0:x1] = sub * (1 - a) + col(rgb) * a
    return C


# ---------------------------------------------------------------- vector shapes
SS = 2  # supersampling for antialiased vector masks


class Mask:
    """Draw with PIL at SS x resolution in internal-pixel coordinates."""

    def __init__(self):
        self.im = Image.new("L", (W * SS, H * SS), 0)
        self.d = ImageDraw.Draw(self.im)

    def _p(self, pts):
        return [(float(x) * SS, float(y) * SS) for x, y in pts]

    def poly(self, pts, v=255):
        if len(pts) >= 3:
            self.d.polygon(self._p(pts), fill=v)
        return self

    def line(self, pts, width, v=255, joint="curve"):
        self.d.line(self._p(pts), fill=v, width=max(1, int(width * SS)), joint=joint)
        return self

    def ellipse(self, cx, cy, rx, ry, v=255):
        self.d.ellipse([(cx - rx) * SS, (cy - ry) * SS, (cx + rx) * SS, (cy + ry) * SS], fill=v)
        return self

    def ring(self, cx, cy, rx, ry, width, v=255):
        self.d.ellipse([(cx - rx) * SS, (cy - ry) * SS, (cx + rx) * SS, (cy + ry) * SS],
                       outline=v, width=max(1, int(width * SS)))
        return self

    def rect(self, x0, y0, x1, y1, v=255):
        self.d.rectangle([x0 * SS, y0 * SS, x1 * SS, y1 * SS], fill=v)
        return self

    def arr(self):
        return np.asarray(self.im.resize((W, H), Image.BOX), np.float32) / 255.0


def tapered(pts, widths):
    """Polygon around a centre-line with per-point half widths."""
    pts = np.asarray(pts, np.float32)
    n = len(pts)
    widths = np.broadcast_to(np.asarray(widths, np.float32), (n,))
    tang = np.gradient(pts, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-6
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    left = pts + nrm * widths[:, None]
    right = pts - nrm * widths[:, None]
    return [tuple(p) for p in left] + [tuple(p) for p in right[::-1]]


def bezier(p0, p1, p2, n=20, p3=None):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2 = (np.asarray(p, np.float32) for p in (p0, p1, p2))
    if p3 is None:
        pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
    else:
        p3 = np.asarray(p3, np.float32)
        pts = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3
    return [tuple(p) for p in pts]


def add(C, m, rgb, k=1.0, glow_r=0, glow_k=0.0):
    """Additive light through a mask, optionally with a glow halo."""
    layer = m[..., None] * col(rgb) * k
    C += layer
    if glow_r:
        C += glow(layer, glow_r) * glow_k
    return C


def paint(C, m, rgb, alpha=1.0):
    """Alpha-blend a solid colour (or HxWx3 image) through a mask."""
    a = (m * alpha)[..., None]
    c = col(rgb) if not isinstance(rgb, np.ndarray) else rgb
    C *= (1 - a)
    C += c * a
    return C


def rot(pts, ang, cx=0.0, cy=0.0):
    ca, sa = math.cos(ang), math.sin(ang)
    return [(cx + x * ca - y * sa, cy + x * sa + y * ca) for x, y in pts]


def ellipse_pts(cx, cy, rx, ry, n=40, ang=0.0, wobble=None):
    out = []
    for i in range(n):
        a = 2 * math.pi * i / n
        k = 1.0 + (wobble(a) if wobble else 0.0)
        x, y = math.cos(a) * rx * k, math.sin(a) * ry * k
        out.append((x, y))
    return rot(out, ang, cx, cy)


# ---------------------------------------------------------------- noise
def fbm_texture(n=512, beta=2.2, seed=0):
    """Tileable fractal noise via spectral synthesis, normalised to 0..1."""
    r = rng(seed)
    white = r.standard_normal((n, n))
    f = np.fft.fftfreq(n)
    fx, fy = np.meshgrid(f, f)
    k = np.sqrt(fx ** 2 + fy ** 2)
    k[0, 0] = 1.0
    spec = np.fft.fft2(white) / k ** (beta / 2)
    spec[0, 0] = 0
    out = np.real(np.fft.ifft2(spec))
    out = (out - out.min()) / (out.max() - out.min())
    return out.astype(np.float32)


def sample(tex, u, v):
    """Bilinear wrap-around sample of a 2-D texture at texel coords (u, v)."""
    h, w = tex.shape[:2]
    u = np.mod(u, w)
    v = np.mod(v, h)
    x0 = np.floor(u).astype(np.int32)
    y0 = np.floor(v).astype(np.int32)
    fx = (u - x0)[..., None] if tex.ndim == 3 else (u - x0)
    fy = (v - y0)[..., None] if tex.ndim == 3 else (v - y0)
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h
    a = tex[y0, x0] * (1 - fx) + tex[y0, x1] * fx
    b = tex[y1, x0] * (1 - fx) + tex[y1, x1] * fx
    return a * (1 - fy) + b * fy


NOISE = fbm_texture(512, 2.4, 11)
NOISE_FINE = fbm_texture(512, 1.6, 12)
NOISE_B = fbm_texture(512, 2.8, 13)


def noise_field(scale=1.0, ox=0.0, oy=0.0, tex=None):
    tex = NOISE if tex is None else tex
    return sample(tex, XX * scale + ox, YY * scale + oy)


# ---------------------------------------------------------------- world map
CONTINENTS = {
    "namerica": [(-168, 65), (-165, 70), (-155, 71), (-140, 70), (-128, 70), (-115, 68), (-95, 72),
                 (-85, 70), (-80, 63), (-92, 58), (-82, 52), (-78, 58), (-70, 61), (-64, 60), (-56, 52),
                 (-60, 46), (-66, 44), (-70, 42), (-76, 38), (-76, 35), (-81, 31), (-80, 26), (-82, 27),
                 (-84, 30), (-90, 29), (-97, 27), (-97, 22), (-92, 18), (-90, 21), (-87, 21), (-88, 16),
                 (-83, 15), (-83, 10), (-79, 9), (-77, 8), (-80, 7.5), (-85, 10), (-88, 13), (-92, 14.5),
                 (-96, 16), (-105, 20), (-106, 23), (-110, 24), (-112, 29), (-114, 31), (-117, 32),
                 (-121, 35), (-124, 40), (-124, 46), (-127, 50), (-135, 57), (-145, 60), (-152, 59),
                 (-158, 57), (-165, 60)],
    "greenland": [(-73, 78), (-60, 82), (-30, 83), (-20, 80), (-20, 72), (-25, 68), (-40, 64), (-45, 60),
                  (-50, 64), (-55, 70), (-60, 76)],
    "samerica": [(-77, 8), (-72, 12), (-62, 11), (-52, 5), (-50, 0), (-44, -2), (-35, -5), (-35, -9),
                 (-39, -15), (-40, -22), (-48, -26), (-53, -34), (-58, -38), (-62, -40), (-65, -45),
                 (-68, -50), (-69, -55), (-74, -52), (-74, -45), (-73, -37), (-71, -30), (-70, -18),
                 (-76, -14), (-81, -6), (-80, 0), (-78, 2)],
    "eurasia": [(-9, 43), (-9, 37), (-6, 36), (-2, 37), (0, 39), (3, 42), (4, 43), (8, 44), (12, 42),
                (16, 38), (18, 40), (13, 45), (19, 42), (23, 38), (26, 40), (29, 41), (33, 37), (36, 36),
                (35, 33), (34, 31), (34, 28), (39, 21), (43, 13), (45, 13), (52, 16), (57, 19), (60, 22),
                (57, 24), (56, 26), (52, 24), (48, 29), (50, 30), (57, 26), (62, 25), (67, 25), (70, 21),
                (73, 16), (77, 8), (80, 10), (80, 15), (87, 21), (92, 22), (94, 17), (98, 16), (98, 10),
                (100, 4), (103, 1.5), (104, 3), (102, 7), (100, 13), (103, 11), (106, 9), (109, 12),
                (109, 16), (106, 20), (110, 21), (117, 23), (121, 28), (122, 31), (120, 35), (122, 37),
                (118, 38), (121, 40), (125, 40), (129, 35), (130, 42), (135, 44), (141, 48), (140, 53),
                (137, 54), (142, 59), (155, 59), (156, 51), (160, 54), (163, 60), (170, 60), (178, 64),
                (180, 66), (180, 69), (160, 70), (140, 72), (130, 71), (113, 74), (105, 78), (95, 76),
                (80, 73), (70, 73), (66, 69), (55, 68), (45, 68), (40, 66), (33, 69), (25, 71), (15, 69),
                (10, 63), (5, 61), (5, 58), (8, 58), (11, 59), (12, 56), (10, 54), (8, 54), (5, 53),
                (2, 51), (-2, 49), (-5, 48), (-1, 46), (-2, 44)],
    "britain": [(-5, 50), (1, 51), (2, 53), (-1, 55), (-2, 57), (-4, 59), (-6, 58), (-5, 55), (-3, 54),
                (-5, 52)],
    "ireland": [(-10, 52), (-6, 52), (-6, 55), (-8, 55), (-10, 54)],
    "africa": [(-17, 21), (-17, 15), (-15, 11), (-8, 4), (-2, 5), (5, 6), (9, 4), (9, -1), (12, -6),
               (13, -12), (12, -17), (15, -27), (18, -34), (20, -35), (26, -34), (32, -28), (35, -23),
               (40, -16), (40, -10), (39, -5), (42, 0), (48, 5), (51, 11), (44, 11), (43, 13), (39, 17),
               (37, 22), (34, 28), (32, 31), (29, 31), (20, 32), (19, 30), (10, 34), (11, 37), (3, 37),
               (-5, 36), (-10, 32), (-13, 27)],
    "madagascar": [(44, -25), (47, -25), (50, -15), (49, -12), (44, -17)],
    "australia": [(114, -22), (114, -34), (117, -35), (123, -34), (131, -31), (137, -35), (138, -34),
                  (141, -38), (146, -39), (150, -37), (153, -32), (153, -25), (146, -19), (145, -14),
                  (142, -11), (141, -17), (136, -12), (132, -11), (129, -15), (123, -17)],
    "sumatra": [(95, 5), (98, 4), (104, -2), (106, -6), (101, -3), (96, 2)],
    "borneo": [(109, 2), (111, -3), (116, -4), (119, 1), (117, 7), (113, 3)],
    "java": [(105, -6), (114, -7), (114, -8.5), (106, -7.5)],
    "newguinea": [(131, -1), (141, -3), (150, -10), (143, -9), (138, -8), (132, -4)],
    "japan": [(130, 31), (132, 34), (136, 35), (140, 36), (142, 40), (141, 45), (144, 44), (141, 42),
              (140, 38), (136, 33)],
    "nz": [(172, -35), (178, -38), (175, -41), (171, -46), (167, -46), (172, -41)],
    "antarctica": [(-180, -72), (-120, -74), (-60, -64), (0, -70), (60, -67), (120, -66), (180, -72),
                   (180, -90), (-180, -90)],
}


def land_texture(w=1024, h=512, blur_px=2):
    """Equirectangular land mask (0..1), lon -180..180, lat 90..-90."""
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    for pts in CONTINENTS.values():
        d.polygon([((lon + 180) / 360 * w, (90 - lat) / 180 * h) for lon, lat in pts], fill=255)
    a = np.asarray(im, np.float32) / 255.0
    return np.clip(blur(a, blur_px), 0, 1)


LAND = land_texture()


class MapProj:
    """Equirectangular map placed on the internal canvas."""

    def __init__(self, lon0=-170, lon1=190, lat0=80, lat1=-58, top=24):
        self.lon0, self.lon1, self.lat0, self.lat1 = lon0, lon1, lat0, lat1
        self.sx = W / (lon1 - lon0)
        self.top = top

    def xy(self, lon, lat):
        return (lon - self.lon0) * self.sx, self.top + (self.lat0 - lat) * self.sx

    def lonlat_grid(self):
        lon = self.lon0 + XX / self.sx
        lat = self.lat0 - (YY - self.top) / self.sx
        return lon, lat

    def land(self):
        lon, lat = self.lonlat_grid()
        u = (lon + 180) / 360 * LAND.shape[1]
        v = np.clip((90 - lat) / 180 * LAND.shape[0], 0, LAND.shape[0] - 1.001)
        return sample(LAND, u, v)


# ---------------------------------------------------------------- spheres
def sphere(C, cx, cy, r, tex_fn, lon0=0.0, tilt=0.3, light=(-0.6, -0.4, 0.7),
           ambient=0.06, atmo=None, atmo_k=0.6, emit_fn=None):
    """Orthographic textured sphere. tex_fn(lat, lon) -> (N,3) rgb 0..1+."""
    if r < 1:
        return C
    pad = r * 0.25 + 4
    x0, x1 = int(max(cx - r - pad, 0)), int(min(cx + r + pad + 1, W))
    y0, y1 = int(max(cy - r - pad, 0)), int(min(cy + r + pad + 1, H))
    if x0 >= x1 or y0 >= y1:
        return C
    x = (XX[y0:y1, x0:x1] - cx) / r
    y = (YY[y0:y1, x0:x1] - cy) / r
    d2 = x * x + y * y
    inside = d2 < 1.0
    sub = C[y0:y1, x0:x1]
    if inside.any():
        xi, yi = x[inside], y[inside]
        z = np.sqrt(1 - xi * xi - yi * yi)
        ct, st = math.cos(tilt), math.sin(tilt)
        y2 = yi * ct - z * st
        z2 = yi * st + z * ct
        lat = np.arcsin(np.clip(-y2, -1, 1))
        lon = np.arctan2(xi, z2) + lon0
        rgb = tex_fn(lat, lon)
        L = np.array(light, np.float32)
        L /= np.linalg.norm(L)
        lam = np.clip(xi * L[0] + yi * L[1] + z * L[2], 0, 1)
        shade = ambient + (1 - ambient) * lam
        rgb = rgb * shade[:, None]
        if emit_fn is not None:
            rgb = rgb + emit_fn(lat, lon, lam)
        if atmo is not None:
            rim = (1 - z) ** 2.5
            rgb = rgb + col(atmo) * (rim * (0.3 + 0.7 * lam) * atmo_k)[:, None]
        dd = np.sqrt(d2[inside])
        a = np.clip((1 - dd) * r / 1.2, 0, 1)[:, None]
        sub[inside] = sub[inside] * (1 - a) + rgb * a
    if atmo is not None:
        dd = np.sqrt(d2)
        halo = np.clip(1 - (dd - 1) / 0.12, 0, 1) ** 2 * (dd >= 0.98)
        L = np.array(light, np.float32)
        side = np.clip(0.4 + (x * L[0] + y * L[1]) / (dd + 1e-6), 0, 1.2)
        sub += col(atmo)[None, None, :] * (halo * side * atmo_k)[..., None]
    return C


def tex_lookup(tex, lat, lon, scale=1.0):
    h, w = tex.shape[:2]
    u = (lon / (2 * math.pi)) * w * scale
    v = (0.5 - lat / math.pi) * h * scale
    return sample(tex, u, v)


# ---------------------------------------------------------------- stars
class Starfield:
    def __init__(self, n=900, seed=1, spread=1.0):
        r = rng(seed)
        self.x = r.uniform(-W * 0.1, W * 1.1, n) * spread
        self.y = r.uniform(-H * 0.1, H * 1.1, n) * spread
        self.b = r.uniform(0.15, 1.0, n) ** 3
        self.tw = r.uniform(0, 6.28, n)
        self.fr = r.uniform(0.5, 2.5, n)
        temp = r.uniform(0, 1, n)
        self.c = np.stack([0.75 + 0.25 * temp, 0.8 + 0.15 * np.ones(n), 1.0 - 0.3 * temp], 1).astype(np.float32)

    def draw(self, C, t, k=1.0, zoom=1.0, cx=W / 2, cy=H / 2, dx=0.0, dy=0.0, glow_k=0.25):
        x = cx + (self.x - W / 2) * zoom + dx
        y = cy + (self.y - H / 2) * zoom + dy
        w = self.b * (0.75 + 0.25 * np.sin(self.tw + t * self.fr)) * k
        return glowing_points(C, x, y, self.c, w * 1.6, core=1.0, halo=glow_k, r=6, soft=0)


# ---------------------------------------------------------------- finishing
def tonemap(C):
    knee = 0.8
    over = C > knee
    out = C.copy()
    out[over] = knee + (1 - knee) * np.tanh((C[over] - knee) / (1 - knee))
    return out


_vd = np.sqrt(((XX - W / 2) / (W / 2)) ** 2 + ((YY - H / 2) / (H / 2)) ** 2)
VIGNETTE = (1 - 0.35 * np.clip(_vd - 0.45, 0, 1) ** 1.5)[..., None].astype(np.float32)
