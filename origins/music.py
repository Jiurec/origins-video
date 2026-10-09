"""Procedural soundtrack. Every phase gets its own instrumentation, and a
single recurring theme (D - A - G - F - E - D) is re-orchestrated each time:
a cold synth after the Big Bang, a choir among the first stars, pizzicato in
the Cambrian, brass for the dinosaurs, a bone flute for the first humans and
an electronic lead for today.

    python -m origins.music build/soundtrack.wav
"""
import math
import sys
import wave
import numpy as np

from . import timeline as TL

SR = 44100
RNG = np.random.default_rng(2024)


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def tarr(dur):
    return np.arange(int(dur * SR), dtype=np.float32) / SR


# ------------------------------------------------------------------ filters & envelopes
def fft_filter(x, lo=0.0, hi=SR / 2, soft=0.15):
    n = len(x)
    if n == 0:
        return x
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    g = np.ones_like(f)
    if lo > 0:
        g *= 1 / (1 + (lo / np.maximum(f, 1e-3)) ** (2 / soft * 0.15 * 4))
    if hi < SR / 2:
        g *= 1 / (1 + (f / hi) ** (2 / soft * 0.15 * 4))
    return np.fft.irfft(X * g, n).astype(np.float32)


def noise(dur, seed=None):
    r = RNG if seed is None else np.random.default_rng(seed)
    return r.standard_normal(int(dur * SR)).astype(np.float32)


def env(dur, a=0.01, r=0.3, total=None, curve=3.0):
    """Linear attack, sustain, exponential-ish release that starts at `dur`."""
    total = dur + r if total is None else total
    t = tarr(total)
    e = np.minimum(t / max(a, 1e-4), 1.0)
    rel = t > dur
    e[rel] *= np.exp(-curve * (t[rel] - dur) / max(r, 1e-4))
    return e


def perc(total, decay):
    t = tarr(total)
    return np.exp(-t / decay).astype(np.float32) * np.minimum(t / 0.002, 1)


# ------------------------------------------------------------------ oscillators
def additive(f, total, amps, vib_rate=0.0, vib_depth=0.0, vib_delay=0.0, detune=0.0, phase=None):
    t = tarr(total)
    fr = f * 2 ** (detune / 1200.0)
    inst = np.full_like(t, fr)
    if vib_depth:
        ramp = np.clip((t - vib_delay) / 0.4, 0, 1)
        inst = inst * (1 + vib_depth * ramp * np.sin(2 * np.pi * vib_rate * t))
    ph = 2 * np.pi * np.cumsum(inst) / SR + (RNG.uniform(0, 6.28) if phase is None else phase)
    out = np.zeros_like(t)
    for k, a in enumerate(amps, start=1):
        if a == 0 or k * fr > 16000:
            continue
        out += a * np.sin(k * ph)
    return out


def saw_amps(n, roll=8.0):
    return [1.0 / k * math.exp(-k / roll) for k in range(1, n + 1)]


def odd_amps(n, roll=6.0):
    return [(1.0 / k if k % 2 else 0.08 / k) * math.exp(-k / roll) for k in range(1, n + 1)]


FORMANTS = {"a": [(800, 1.0, 110), (1150, 0.5, 120), (2900, 0.18, 180)],
            "o": [(450, 1.0, 90), (800, 0.45, 100), (2830, 0.08, 160)],
            "u": [(325, 1.0, 80), (700, 0.25, 90), (2530, 0.05, 150)]}


def formant_amps(f, vowel, n=40):
    out = []
    for k in range(1, n + 1):
        h = k * f
        g = sum(A * math.exp(-((h - F) / (B * 1.6)) ** 2) for F, A, B in FORMANTS[vowel]) + 0.015
        out.append(g / k ** 0.3)
    return out


# ------------------------------------------------------------------ instruments
def pad(m, dur, bright=6.0, a=1.2, r=2.0, voices=3, spread=9.0):
    f = midi(m)
    total = dur + r
    out = sum(additive(f, total, saw_amps(24, bright), 0.25, 0.002, detune=(i - (voices - 1) / 2) * spread)
              for i in range(voices))
    return out * env(dur, a, r, total) / voices * 0.5


def choir(m, dur, vowel="a", a=0.8, r=1.5, voices=3):
    f = midi(m)
    total = dur + r
    amps = formant_amps(f, vowel)
    out = sum(additive(f, total, amps, 5.0 + i * 0.4, 0.006, 0.2, detune=(i - 1) * 7) for i in range(voices))
    br = fft_filter(noise(total), 600, 3000) * 0.015
    return (out / voices * 0.6 + br) * env(dur, a, r, total)


def strings(m, dur, a=0.35, r=0.8, voices=3, bright=10.0, trem=0.0):
    f = midi(m)
    total = dur + r
    out = sum(additive(f, total, saw_amps(26, bright), 5.5 + i * 0.3, 0.004, 0.15, detune=(i - 1) * 6)
              for i in range(voices))
    out = out / voices * 0.45
    if trem:
        t = tarr(total)
        out *= 0.65 + 0.35 * np.sin(2 * np.pi * trem * t)
    return out * env(dur, a, r, total)


def brass(m, dur, a=0.08, r=0.4, power=1.0):
    f = midi(m)
    total = dur + r
    t = tarr(total)
    e = env(dur, a, r, total)
    swell = np.clip(e * (0.75 + 0.25 * np.minimum(t / (dur + 0.01), 1)) * power, 0, 1.3)
    ph = 2 * np.pi * np.cumsum(f * (1 + 0.003 * np.sin(2 * np.pi * 5 * t) * np.clip(t - 0.3, 0, 1))) / SR
    out = np.zeros_like(t)
    for k in range(1, 22):
        if k * f > 14000:
            break
        out += (1 / k) * np.sin(k * ph) * swell ** (0.6 + 0.22 * k)
    return out * 0.45


def flute(m, dur, a=0.06, r=0.15, breath=0.12, vib=0.005):
    f = midi(m)
    total = dur + r
    tone = additive(f, total, [1, 0.22, 0.07, 0.02], 5.0, vib, 0.25)
    nz = fft_filter(noise(total), f * 0.8, f * 3.5) * breath * 2
    chiff = fft_filter(noise(total), 1500, 6000) * perc(total, 0.03) * 0.2
    return (tone * 0.5 + nz + chiff) * env(dur, a, r, total)


def reed(m, dur, a=0.04, r=0.12):
    f = midi(m)
    total = dur + r
    return additive(f, total, odd_amps(15, 5), 5.2, 0.004, 0.3) * env(dur, a, r, total) * 0.4


def bell(m, dur=3.0, idx=2.5, ratio=3.5):
    f = midi(m)
    t = tarr(dur)
    I = idx * np.exp(-t * 2.5)
    out = np.sin(2 * np.pi * f * t + I * np.sin(2 * np.pi * f * ratio * t))
    return out * perc(dur, dur / 4) * 0.35


def glass(m, dur=4.0):
    f = midi(m)
    t = tarr(dur)
    out = sum(a * np.sin(2 * np.pi * f * k * t) * np.exp(-t / (dur / (1 + i * 1.5)))
              for i, (k, a) in enumerate([(1, 1), (2.76, 0.4), (5.40, 0.2), (8.93, 0.08)]))
    return out * np.minimum(t / 0.004, 1) * 0.25


def mallet(m, dur=1.2, bright=1.0):
    f = midi(m)
    t = tarr(dur)
    out = np.sin(2 * np.pi * f * t) * np.exp(-t / (dur / 3)) + \
        0.35 * bright * np.sin(2 * np.pi * f * 3.98 * t) * np.exp(-t / 0.08) + \
        0.15 * bright * np.sin(2 * np.pi * f * 9.2 * t) * np.exp(-t / 0.03)
    return out * np.minimum(t / 0.002, 1) * 0.4


def pluck(m, dur=1.5, bright=0.6, damp=0.996, seed=None):
    """Karplus-Strong, vectorised one period at a time."""
    f = midi(m)
    N = max(2, int(SR / f))
    n = int(dur * SR)
    r = np.random.default_rng(seed if seed is not None else int(f * 1000) % 100000)
    burst = r.uniform(-1, 1, N).astype(np.float32)
    if bright < 1:
        k = max(1, int((1 - bright) * 6))
        for _ in range(k):
            burst = 0.5 * (burst + np.roll(burst, 1))
    out = np.zeros(n + 2 * N, np.float32)
    out[:N] = burst
    s = N
    while s < n + N:
        e = min(s + N, n + 2 * N)
        prev1 = out[s - N:e - N]
        prev2 = out[s - N - 1:e - N - 1] if s - N - 1 >= 0 else np.concatenate([[0.0], out[s - N:e - N - 1]])
        out[s:e] = damp * 0.5 * (prev1 + prev2)
        s = e
    out = out[:n]
    out -= out.mean()
    return out * 0.5 * np.minimum(tarr(dur) / 0.002, 1)


def tom(f, dur=0.6, drop=0.75, decay=0.18, slap=0.3):
    t = tarr(dur)
    fr = f * (drop + (1 - drop) * np.exp(-t / 0.05))
    ph = 2 * np.pi * np.cumsum(fr) / SR
    body = np.sin(ph) * np.exp(-t / decay)
    sl = fft_filter(noise(dur), 300, 3000) * perc(dur, 0.015) * slap
    return (body + sl) * 0.7


def kick(dur=0.5):
    t = tarr(dur)
    fr = 45 + 90 * np.exp(-t / 0.04)
    ph = 2 * np.pi * np.cumsum(fr) / SR
    return (np.sin(ph) * np.exp(-t / 0.22) + fft_filter(noise(dur), 1500, 8000) * perc(dur, 0.004) * 0.3) * 0.9


def hat(dur=0.08, open_=False):
    d = 0.3 if open_ else dur
    return fft_filter(noise(d), 7000, 16000) * perc(d, d / 3) * 0.25


def shaker(dur=0.12):
    return fft_filter(noise(dur), 4000, 12000) * np.sin(np.linspace(0, math.pi, int(dur * SR))) ** 2 * 0.18


def clap(dur=0.25):
    n = fft_filter(noise(dur), 900, 5000)
    t = tarr(dur)
    e = np.exp(-t / 0.06) + 0.6 * np.exp(-np.maximum(t - 0.012, 0) / 0.01) * (t > 0.012)
    return n * e * 0.4


def woodblock(f=900, dur=0.15):
    t = tarr(dur)
    return (np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * f * 2.7 * t)) * np.exp(-t / 0.025) * 0.4


def timpani(m, dur=2.0):
    f = midi(m)
    t = tarr(dur)
    out = sum(a * np.sin(2 * np.pi * f * k * t * (1 + 0.01 * np.exp(-t / 0.1))) * np.exp(-t / (0.7 / k ** 0.5))
              for k, a in [(1, 1), (1.5, 0.5), (1.99, 0.35), (2.44, 0.2)])
    out += fft_filter(noise(dur), 60, 1500) * perc(dur, 0.04) * 0.6
    return out * 0.6


def clank(f=320, dur=0.5):
    t = tarr(dur)
    out = sum(a * np.sin(2 * np.pi * f * k * t) * np.exp(-t / (0.25 / k))
              for k, a in [(1, 1), (2.31, 0.7), (3.72, 0.5), (5.1, 0.4), (7.4, 0.2)])
    return (out + fft_filter(noise(dur), 2000, 9000) * perc(dur, 0.01) * 0.8) * 0.3


def boom(dur=5.0, f0=95, f1=26, nz=1.0):
    t = tarr(dur)
    fr = f1 + (f0 - f1) * np.exp(-t / 0.6)
    ph = 2 * np.pi * np.cumsum(fr) / SR
    sub = np.sin(ph) * np.exp(-t / 1.6)
    crack = fft_filter(noise(dur), 40, 2500) * np.exp(-t / 0.9) * nz
    hi = fft_filter(noise(dur), 2500, 12000) * np.exp(-t / 0.12) * 0.5 * nz
    return (sub * 1.2 + crack * 0.7 + hi) * np.minimum(t / 0.003, 1)


def riser(dur=2.0, lo=200, hi=6000):
    t = tarr(dur)
    out = np.zeros_like(t)
    bands = 8
    for b in range(bands):
        f0 = lo * (hi / lo) ** (b / bands)
        f1 = lo * (hi / lo) ** ((b + 1) / bands)
        w = np.exp(-((t / dur * bands - b - 0.5) / 0.9) ** 2)
        out += fft_filter(noise(dur), f0, f1) * w
    fr = 80 * (30 ** (t / dur))
    sweep = np.sin(2 * np.pi * np.cumsum(fr) / SR) * 0.15
    return (out * 0.6 + sweep) * (t / dur) ** 2


def wind(dur, lo=200, hi=1200, k=1.0, seed=7):
    t = tarr(dur)
    n = fft_filter(noise(dur, seed), lo, hi)
    mod = 0.55 + 0.45 * np.sin(2 * np.pi * 0.13 * t + 1) * np.sin(2 * np.pi * 0.07 * t)
    return n * mod * k


def rumble(dur, k=1.0, hi=140, seed=9):
    return fft_filter(noise(dur, seed), 20, hi) * k * 3


def bubble(dur=0.08):
    t = tarr(dur)
    f0 = RNG.uniform(500, 1100)
    fr = f0 * (1 + 2.5 * t / dur)
    return np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.sin(np.pi * t / dur) ** 2 * 0.25


def synth_lead(m, dur, a=0.01, r=0.25, bright=7.0):
    f = midi(m)
    total = dur + r
    out = additive(f, total, saw_amps(30, bright), 5.0, 0.003, 0.2) + \
        additive(f, total, saw_amps(30, bright), 5.0, 0.003, 0.2, detune=9)
    return out * env(dur, a, r, total) * 0.3


def sub(m, dur, a=0.005, r=0.1):
    f = midi(m)
    total = dur + r
    return additive(f, total, [1, 0.15]) * env(dur, a, r, total) * 0.8


def crackle(dur, rate=25, seed=3):
    r = np.random.default_rng(seed)
    out = np.zeros(int(dur * SR), np.float32)
    for _ in range(int(dur * rate)):
        i = r.integers(0, len(out) - 400)
        out[i:i + 400] += fft_filter(r.standard_normal(400).astype(np.float32), 1500, 9000) * \
            np.exp(-np.arange(400) / 40) * r.uniform(0.2, 1.0)
    return out * 0.5


# ------------------------------------------------------------------ mixing buses
def pan_gains(p):
    a = (p + 1) * math.pi / 4
    return math.cos(a), math.sin(a)


class Bus:
    def __init__(self, dur, tail=4.0):
        self.dur = dur
        self.n = int((dur + tail) * SR)
        self.x = np.zeros((2, self.n), np.float32)

    def add(self, t, sig, pan=0.0, gain=1.0):
        i = int(round(t * SR))
        if i < 0:
            sig = sig[-i:]
            i = 0
        n = min(len(sig), self.n - i)
        if n <= 0:
            return
        gl, gr = pan_gains(pan)
        self.x[0, i:i + n] += sig[:n] * gl * gain
        self.x[1, i:i + n] += sig[:n] * gr * gain

    def add_stereo(self, t, sig_l, sig_r, gain=1.0):
        self.add(t, sig_l, -1, gain * math.sqrt(2))
        self.add(t, sig_r, 1, gain * math.sqrt(2))


_IR = {}


def impulse(size):
    if size not in _IR:
        n = int(size * SR)
        t = np.arange(n) / SR
        irs = []
        for seed in (1, 2):
            r = np.random.default_rng(seed + int(size * 10))
            nz = r.standard_normal(n).astype(np.float32)
            dark = fft_filter(nz, 0, 3500)
            mixd = nz * np.exp(-t / (size * 0.08)) + dark * (1 - np.exp(-t / (size * 0.08)))
            ir = mixd * np.exp(-t * 6.9 / size) * np.minimum(t / 0.01, 1)
            irs.append(ir / np.sqrt(np.sum(ir ** 2)))
        _IR[size] = irs
    return _IR[size]


def convolve(x, ir):
    n = len(x) + len(ir) - 1
    nfft = 1 << (n - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(ir, nfft), nfft)[:len(x)]
    return y.astype(np.float32)


def finish(bus, wet=0.25, size=2.8, fout=1.2, fin=0.0):
    """Fade the dry signal at the phase end, then add reverb (whose tail rings on)."""
    t = np.arange(bus.n) / SR
    e = np.ones(bus.n, np.float32)
    if fin:
        e *= np.clip(t / fin, 0, 1)
    if fout is not None:
        e *= np.clip(1 - (t - bus.dur) / max(fout, 1e-3), 0, 1)
    dry = bus.x * e
    if wet > 0:
        irs = impulse(size)
        wetsig = np.stack([convolve(dry[0], irs[0]), convolve(dry[1], irs[1])])
        return dry + wetsig * wet
    return dry


# ------------------------------------------------------------------ musical material
D = 62  # D4
THEME = {"minor": [0, 7, 5, 3, 2, 0], "major": [0, 7, 5, 4, 2, 0], "dorian": [0, 7, 5, 3, 2, 0],
         "penta": [0, 7, 5, 3, 0], "phrygian": [0, 7, 5, 3, 1, 0]}
RHYTHM = {5: [1, 1, 0.5, 0.5, 2], 6: [1, 1, 0.5, 0.5, 1, 2]}

CHORDS_MINOR = [[50, 57, 62, 65, 69], [46, 53, 58, 62, 65], [41, 48, 53, 57, 60], [48, 55, 60, 64, 67]]  # Dm Bb F C
CHORDS_MAJOR = [[50, 57, 62, 66, 69], [47, 54, 59, 62, 66], [43, 50, 55, 59, 62], [45, 52, 57, 61, 64]]  # D Bm G A


def theme(bus, t0, beat, root, inst, variant="minor", gain=1.0, pan=0.0, legato=1.0, rhythm=None, **kw):
    notes = THEME[variant]
    rh = rhythm or RHYTHM[len(notes)]
    t = t0
    for n, b in zip(notes, rh):
        dur = b * beat * legato
        bus.add(t, inst(root + n, dur, **kw), pan, gain)
        t += b * beat
    return t


def chord(bus, t, dur, notes, inst, gain=1.0, spread=0.6, **kw):
    for i, n in enumerate(notes):
        p = (i / max(len(notes) - 1, 1) - 0.5) * 2 * spread
        bus.add(t, inst(n, dur, **kw), p, gain / math.sqrt(len(notes)))


# ------------------------------------------------------------------ phases
def m_intro(d):
    b = Bus(d)
    b.add(1.0, pad(38, d - 1.0, bright=2.5, a=3.5, r=1.0) * 0.6)
    b.add(1.0, pad(45, d - 1.0, bright=2.5, a=4.0, r=1.0) * 0.3)
    for t, m in [(1.8, 81), (3.2, 86), (4.4, 88)]:
        b.add(t, glass(m, 3.0), RNG.uniform(-0.6, 0.6), 0.25)
    return finish(b, 0.4, 3.5, fout=0.05)


def m_bigbang(d):
    b = Bus(d)
    tb = TL.BANG_T - TL.BY_KEY["bigbang"].start
    b.add(0.0, riser(tb, 150, 8000), 0, 0.7)
    b.add(tb, boom(6.0), 0, 1.6)
    b.add(tb, timpani(38, 3.0), 0, 0.8)
    whoosh = fft_filter(noise(7.0), 300, 5000) * np.exp(-tarr(7.0) / 2.0)
    b.add_stereo(tb + 0.05, whoosh, np.roll(whoosh, 4000), 0.35)
    b.add(tb + 2.5, pad(50, d - tb - 2.5, bright=3, a=3.0, r=3), -0.3, 0.5)
    b.add(tb + 2.5, pad(57, d - tb - 2.5, bright=3, a=3.5, r=3), 0.3, 0.4)
    b.add(tb + 3.5, pad(64, d - tb - 3.5, bright=2.5, a=3.0, r=3), 0.5, 0.3)
    b.add(tb + 3.5, pad(65, d - tb - 3.5, bright=2.5, a=3.0, r=3), -0.5, 0.25)
    cold = lambda m, dur: (additive(midi(m), dur + 1.5, [1, 0.0, 0.12, 0, 0.05], 4.5, 0.004, 0.3) *
                           env(dur, 0.25, 1.5) * 0.35)
    theme(b, tb + 5.5, 1.0, D + 12, cold, gain=0.9)
    return finish(b, 0.45, 4.0)


def m_stars(d):
    b = Bus(d)
    for i, ch in enumerate([CHORDS_MINOR[0], CHORDS_MINOR[1], CHORDS_MINOR[2], CHORDS_MINOR[3]]):
        chord(b, i * 4.0, 4.6, [n + 12 for n in ch[1:4]], choir, 0.75, vowel="a", a=1.2, r=1.8)
        b.add(i * 4.0, pad(ch[0] - 12, 4.4, bright=2.5, a=1.0, r=1.5), 0, 0.35)
    st = TL.BY_KEY["stars"].start
    penta = [74, 77, 79, 81, 84, 86, 89, 91]
    for i, ti in enumerate(TL.STAR_IGNITIONS):
        b.add(ti - st, bell(penta[(i * 3) % len(penta)], 3.0, idx=1.5), ((i * 0.37) % 1.4) - 0.7, 0.35)
    b.add(TL.SUPERNOVA_T - st, boom(4.0, 70, 30, 0.4), 0, 0.5)
    for k, m in enumerate([86, 93, 98]):
        b.add(TL.SUPERNOVA_T - st + k * 0.07, bell(m, 3.5, idx=3), (k - 1) * 0.5, 0.3)
    theme(b, 4.0, 1.05, D + 12, lambda m, dur: choir(m, dur, "o", a=0.25, r=0.9), gain=0.9)
    return finish(b, 0.5, 4.5)


def m_solar(d):
    b = Bus(d)
    for i, ch in enumerate(CHORDS_MINOR[:3] + [CHORDS_MINOR[0]]):
        chord(b, i * 3.75, 4.2, ch[:3], pad, 0.55, bright=3, a=0.8, r=1.5)
    arp = [50, 57, 62, 64, 65, 69, 65, 64]
    step = 0.15
    n = int(d / step)
    for i in range(n):
        t = i * step
        m = arp[i % len(arp)] + (12 if (i // 16) % 2 else 0)
        b.add(t, bell(m, 1.2, idx=1.2, ratio=2.0), math.sin(2 * math.pi * 0.2 * t) * 0.8,
              0.22 * min(1, t / 2.5 + 0.2))
    b.add(2.6, riser(0.6, 400, 6000), 0, 0.3)
    b.add(3.0, boom(3.0, 70, 35, 0.2), 0, 0.35)
    theme(b, 6.0, 0.8, D + 12, lambda m, dur: flute(m, dur, breath=0.05), gain=0.9)
    return finish(b, 0.35, 3.0)


def m_hadean(d):
    b = Bus(d)
    b.add(0, rumble(d + 1, 0.5), 0, 1.0)
    b.add(0, pad(38, d, bright=4, a=1.5, r=1.5), -0.2, 0.45)
    b.add(0.5, pad(39, d - 1, bright=4, a=2.5, r=1.5), 0.2, 0.25)
    r = np.random.default_rng(41)
    ti = TL.THEIA_T - TL.BY_KEY["hadean"].start
    t = 0.0
    while t < d - 0.3:
        dens = 0.45 if t < ti else (0.18 if t < ti + 3 else 0.3)
        b.add(t, tom(r.uniform(55, 95), 0.7, decay=0.25), r.uniform(-0.6, 0.6), r.uniform(0.35, 0.7))
        t += r.uniform(0.6, 1.4) * dens
    b.add(ti - 1.5, riser(1.5, 100, 3000), 0, 0.35)
    b.add(ti, boom(6.0, 110, 28), 0, 1.5)
    b.add(ti, timpani(33, 3.0), 0, 0.8)
    theme(b, ti + 2.2, 0.9, D - 24, lambda m, dur: brass(m, dur, a=0.15, power=0.9), gain=1.0)
    return finish(b, 0.35, 3.0)


def m_life(d):
    b = Bus(d)
    for i, ch in enumerate([CHORDS_MINOR[0], CHORDS_MINOR[1], CHORDS_MINOR[0], CHORDS_MINOR[2]]):
        chord(b, i * 4.5, 5.0, ch[:4], lambda m, dur, **k: fft_filter(pad(m, dur, bright=2.0, a=1.5, r=1.8), 0, 900),
              0.7)
    r = np.random.default_rng(51)
    for _ in range(int(d * 3)):
        b.add(r.uniform(0, d), bubble(r.uniform(0.04, 0.09)), r.uniform(-0.8, 0.8), r.uniform(0.15, 0.35))
    for k in range(int((d - 6) / 1.2)):
        t = 6 + k * 1.2
        b.add(t, tom(48, 0.4, decay=0.12, slap=0.0), 0, 0.35)
        b.add(t + 0.22, tom(46, 0.4, decay=0.12, slap=0.0), 0, 0.25)
    mar = lambda m, dur: mallet(m, max(dur, 0.8) + 0.6, bright=0.6)
    end = theme(b, 6.8, 0.75, D, mar, gain=0.9, pan=-0.2)
    theme(b, end + 0.4, 0.7, D + 12, mar, gain=0.7, pan=0.25)
    b.add(0, wind(d + 2, 100, 500, 0.15), 0, 1)
    return finish(b, 0.45, 3.5)


def m_oxygen(d):
    b = Bus(d)
    seq = [(CHORDS_MINOR[0], 2.5), (CHORDS_MINOR[1], 5.0), (CHORDS_MINOR[2], 7.5), (CHORDS_MAJOR[0], 10.5)]
    t = 0.0
    for ch, br in seq:
        dur = 3.5
        chord(b, t, dur + 0.4, ch[:4], pad, 0.6, bright=br, a=0.6, r=1.2)
        t += dur
    r = np.random.default_rng(61)
    up = [74, 76, 77, 81, 84, 86, 88, 89, 93]
    for i in range(int(d / 0.25)):
        tt = i * 0.25
        if r.uniform() < 0.25 + 0.5 * tt / d:
            m = up[r.integers(0, len(up))]
            if tt > 10.5:
                m = m + 1 if m % 12 == 5 else m
            b.add(tt, glass(m, 2.0), r.uniform(-0.7, 0.7), 0.12)
    theme(b, 7.0, 0.65, D + 12, lambda m, dur: bell(m, dur + 1.5, idx=1.8, ratio=2.0) * 1.2,
          variant="major" if False else "minor", gain=0.8)
    theme(b, 10.5, 0.55, D + 24, lambda m, dur: bell(m, dur + 1.5, idx=1.5, ratio=2.0), variant="major", gain=0.6)
    return finish(b, 0.4, 3.0)


def m_eukaryote(d):
    b = Bus(d)
    for i, ch in enumerate([[43, 50, 55, 58], [50, 57, 62, 65], [46, 53, 58, 62]]):
        chord(b, i * 4.0, 4.5, ch, pad, 0.45, bright=3, a=0.7, r=1.2)
    beat = 0.55
    e1 = theme(b, 0.8, beat, D, lambda m, dur: reed(m, dur), gain=0.9, pan=-0.35)
    theme(b, 0.8 + 1.5 * beat, beat, D + 9, lambda m, dur: flute(m, dur, breath=0.06), gain=0.7, pan=0.35)
    theme(b, e1 + 0.6, beat, D + 12, lambda m, dur: flute(m, dur, breath=0.06), gain=0.7, pan=0.35)
    theme(b, e1 + 0.6 + 1.5 * beat, beat, D + 3, lambda m, dur: reed(m, dur), gain=0.8, pan=-0.35)
    b.add(6.0, pad(62, 3.0, bright=4, a=1.0, r=2.0), 0, 0.25)
    return finish(b, 0.4, 2.8)


def m_snowball(d):
    b = Bus(d)
    b.add(0, wind(d + 2, 400, 3000, 0.25, 81), 0, 1)
    for t, m in [(0.3, 93), (1.6, 88), (2.9, 98), (4.0, 86), (5.1, 93), (6.0, 91)]:
        b.add(t, glass(m, 4.5), RNG.uniform(-0.8, 0.8), 0.25)
    b.add(0, pad(74, 7.0, bright=1.5, a=1.5, r=2.0), 0, 0.15)
    for i, ch in enumerate([CHORDS_MAJOR[0], CHORDS_MAJOR[2], CHORDS_MAJOR[0]]):
        chord(b, 6.2 + i * 2.7, 3.2, ch[:4], strings, 0.75, a=1.2 if i == 0 else 0.6, r=1.5)
    theme(b, 8.4, 0.85, D - 12, lambda m, dur: strings(m, dur, a=0.25, voices=2), variant="major", gain=0.85)
    return finish(b, 0.45, 3.5)


def m_cambrian(d):
    b = Bus(d)
    beat = 60 / 132
    e = beat / 2
    bass = [38, 50, 45, 50, 34, 46, 41, 46, 36, 48, 43, 48, 33, 45, 40, 45]
    pz = lambda m, dur=0.6: pluck(m, 0.6, bright=0.75, damp=0.985)
    for i in range(int(d / e)):
        t = i * e
        b.add(t, pz(bass[i % len(bass)]), -0.2, 0.55)
        if i % 2 == 1:
            b.add(t, woodblock(1100 if i % 4 == 1 else 850), 0.5, 0.35)
        if i % 4 == 0:
            ch = CHORDS_MINOR[(i // 4) % 4]
            for k, m in enumerate(ch[2:5]):
                b.add(t + k * 0.01, pz(m), 0.3, 0.22)
    mar = lambda m, dur: mallet(m + 12, 0.6, 1.0) + pluck(m, 0.6, 0.8, 0.985) * 0.6
    end = theme(b, 8 * e, beat, D, mar, gain=0.85, pan=0.1)
    theme(b, end + 2 * beat, beat * 0.75, D + 12, mar, gain=0.8, pan=-0.1)
    return finish(b, 0.25, 2.0)


def m_land(d):
    b = Bus(d)
    drone = lambda m, dur: reed(m, dur, a=0.6, r=0.3) * 0.7
    b.add(0, drone(50, d), -0.3, 0.6)
    b.add(0, drone(57, d), 0.3, 0.45)
    beat = 60 / 90
    for i in range(int(d / beat)):
        b.add(i * beat, tom(80, 0.4, decay=0.15, slap=0.15), 0, 0.35)
        if i % 2:
            b.add(i * beat + beat / 2, tom(140, 0.3, decay=0.08, slap=0.3), 0.3, 0.15)
    fl = lambda m, dur: flute(m, dur, breath=0.1)
    end = theme(b, 1.5, beat, D + 12, fl, variant="dorian", gain=0.9)
    t = end + 0.3
    for m, du in [(74, 0.5), (76, 0.5), (77, 0.5), (79, 0.5), (81, 1.5), (79, 0.5), (77, 0.5), (76, 0.5), (74, 1.5)]:
        b.add(t, fl(m, du * beat * 0.95), 0, 0.8)
        t += du * beat
    return finish(b, 0.3, 2.5, fout=0.12)


def m_dying(d):
    b = Bus(d)
    b.add(0, boom(4.0, 60, 25, 0.8), 0, 1.0)
    for m, p in [(26, -0.3), (27, 0.3), (33, 0.0), (38, -0.5), (39, 0.5)]:
        b.add(0, strings(m, d, a=0.05, r=2.0, bright=6), p, 0.4)
    b.add(0, rumble(d + 2, 0.7, 180, 111), 0, 1)
    r = np.random.default_rng(112)
    for t in r.uniform(0.8, d - 1, 5):
        b.add(t, boom(3.0, 60, 30, 0.6), r.uniform(-0.6, 0.6), 0.4)
    for lt in (4.2, 7.6):
        b.add(lt, fft_filter(noise(1.2), 500, 9000) * perc(1.2, 0.15), r.uniform(-0.5, 0.5), 0.6)
    theme(b, 1.5, 1.45, D - 24, lambda m, dur: brass(m, dur, a=0.4, power=0.8), gain=0.9)
    return finish(b, 0.4, 3.5, fout=0.6)


def m_dinos(d):
    b = Bus(d)
    beat = 60 / 96
    nb = int(d / beat)
    prog = [CHORDS_MINOR[0], CHORDS_MINOR[1], CHORDS_MINOR[3], CHORDS_MINOR[0]]
    for bar in range(0, nb, 4):
        ch = prog[(bar // 4) % 4]
        chord(b, bar * beat, 4 * beat, [ch[0] - 12, ch[0], ch[1]], strings, 0.7, a=0.15, r=0.5, bright=9)
    for i in range(nb * 2):
        t = i * beat / 2
        ch = prog[(i // 8) % 4]
        b.add(t, strings(ch[0] + (12 if i % 2 else 0), beat / 2 * 0.6, a=0.01, r=0.08), -0.3, 0.3)
    for i in range(nb):
        root = prog[(i // 4) % 4][0] - 12
        b.add(i * beat, timpani(root + (0 if i % 2 == 0 else 7) if root + 7 < 50 else root, 1.5), 0,
              0.7 if i % 4 == 0 else 0.4)
    horn = lambda m, dur: brass(m, dur, a=0.06, power=1.1)
    end = theme(b, 4 * beat, beat, D - 12, horn, gain=1.0, pan=-0.15)
    theme(b, end + 2 * beat, beat, D, horn, gain=1.0, pan=0.15)
    theme(b, end + 2 * beat, beat, D - 12, horn, gain=0.7, pan=-0.2)
    b.add(d - 1.2, timpani(38, 2.0), 0, 0.6)
    return finish(b, 0.3, 2.8, fout=1.0)


def m_asteroid(d):
    b = Bus(d)
    ti = TL.IMPACT_T - TL.BY_KEY["asteroid"].start
    for m, p in [(50, -0.5), (51, 0.5), (56, 0), (62, -0.3), (63, 0.3)]:
        s = strings(m, ti, a=ti * 0.9, r=0.02, trem=12)
        b.add(0, s * np.linspace(0.2, 1.2, len(s)), p, 0.4)
    b.add(0, riser(ti, 150, 9000), 0, 0.9)
    b.add(ti, boom(7.0, 120, 22, 1.5), 0, 1.8)
    b.add(ti, timpani(26, 3.0), 0, 0.9)
    b.add(ti + 0.4, rumble(6.0, 0.6, 120, 131) * np.exp(-tarr(6.0) / 2.5), 0, 1)
    b.add(ti + 1.5, wind(d - ti, 150, 700, 0.18, 132), 0, 1)
    tone = additive(3520, d - ti, [1]) * 0.03 * np.exp(-tarr(d - ti) / 2.0)
    b.add(ti + 0.3, tone, 0.2, 1)
    return finish(b, 0.4, 4.0, fout=0.8)


def m_mammals(d):
    b = Bus(d)
    for i, ch in enumerate(CHORDS_MAJOR):
        chord(b, i * 3.5, 4.0, ch[:4], strings, 0.75, a=0.8, r=1.0, bright=7)
        for k, m in enumerate([ch[0], ch[1], ch[2], ch[3], ch[2] + 12, ch[3]]):
            b.add(i * 3.5 + k * 0.29, pluck(m + 12, 1.6, bright=0.5, damp=0.997), 0.4, 0.2)
    end = theme(b, 1.8, 0.8, D - 12, lambda m, dur: strings(m, dur, a=0.2, voices=2, bright=6), variant="major",
                gain=0.8, pan=-0.2)
    theme(b, end + 0.2, 0.7, D + 12, lambda m, dur: strings(m, dur, a=0.15, voices=2, bright=12), variant="major",
          gain=0.6, pan=0.25)
    return finish(b, 0.45, 3.0)


def m_humans(d):
    b = Bus(d)
    beat = 0.6
    pat = [(0, 70, 0.8), (0.5, 160, 0.35), (1.0, 120, 0.5), (1.5, 160, 0.35), (1.75, 160, 0.3)]
    for bar in range(int(d / (2 * beat))):
        for off, f, g in pat:
            t = bar * 2 * beat + off * beat * 2 / 2
            gg = g * (0.6 if t > 9 else 1.0)
            b.add(t, tom(f, 0.5, decay=0.18 if f < 100 else 0.08, slap=0.35), -0.2 if f > 100 else 0.1, gg)
        for k in range(4):
            b.add(bar * 2 * beat + k * beat / 2, shaker(), 0.5, 0.25)
    hum = lambda m, dur: choir(m, dur, "u", a=0.8, r=1.0)
    b.add(0, hum(38, d), -0.2, 0.5)
    b.add(0, hum(45, d), 0.2, 0.35)
    theme(b, 2.4, 1.0, D - 12, lambda m, dur: choir(m, dur, "o", a=0.15, r=0.5), gain=0.85)
    b.add(8.5, crackle(d - 8.5, 30), 0, 0.5)
    return finish(b, 0.35, 2.5)


def m_africa(d):
    b = Bus(d)
    b.add(0, reed(50, d, a=1.0, r=0.5) * 0.5, 0, 0.6)
    b.add(0, wind(d + 1, 300, 1500, 0.08, 171), 0, 1)
    theme(b, 0.3, 0.55, D + 12, lambda m, dur: flute(m, dur, breath=0.22, vib=0.008), variant="penta", gain=1.0)
    theme(b, 3.4, 0.5, D + 12, lambda m, dur: flute(m, dur, breath=0.22, vib=0.008), variant="penta", gain=0.7,
          rhythm=[0.5, 0.5, 0.5, 0.5, 2])
    return finish(b, 0.45, 3.0, fout=0.6)


def m_caveart(d):
    b = Bus(d)
    b.add(0, choir(62, d, "a", a=0.6, r=0.8, voices=1), -0.2, 0.55)
    b.add(0, choir(50, d, "u", a=0.6, r=0.8, voices=1), 0.2, 0.35)
    theme(b, 0.6, 0.5, D + 12, lambda m, dur: flute(m, dur, breath=0.2), variant="penta", gain=0.85, pan=0.2,
          rhythm=[1, 1, 0.5, 0.5, 2])
    for i in range(10):
        b.add(i * 0.55 + (0.27 if i % 3 == 2 else 0), woodblock(1400 + 300 * (i % 2), 0.08), -0.4, 0.2)
    return finish(b, 0.6, 3.5, fout=0.6)


def m_iceage(d):
    b = Bus(d)
    b.add(0, wind(d + 1, 250, 2500, 0.35, 181), 0, 1)
    b.add(0, pad(38, d, bright=2, a=1.0, r=1.0), 0, 0.4)
    b.add(0, pad(45, d, bright=2, a=1.0, r=1.0), 0, 0.25)
    for t, m in [(0.5, 74), (1.5, 81), (2.8, 79), (4.0, 77)]:
        for k in range(3):
            b.add(t + k * 0.38, flute(m, 0.6, breath=0.25), (-0.5, 0.5, 0)[k], 0.7 * 0.45 ** k)
    return finish(b, 0.5, 3.5, fout=0.6)


def m_farming(d):
    b = Bus(d)
    beat = 0.5
    for i in range(int(d / beat)):
        b.add(i * beat, tom(90, 0.4, decay=0.14, slap=0.25), 0, 0.45 if i % 2 == 0 else 0.25)
    b.add(0, reed(50, d, a=0.4) * 0.5, -0.2, 0.5)
    b.add(0, reed(57, d, a=0.4) * 0.5, 0.2, 0.35)
    theme(b, 0.5, 0.5, D + 12, lambda m, dur: flute(m, dur, breath=0.1), variant="major", gain=0.9)
    theme(b, 3.6, 0.4, D + 12, lambda m, dur: flute(m, dur, breath=0.1), variant="major", gain=0.7)
    return finish(b, 0.3, 2.5, fout=0.6)


def m_writing(d):
    b = Bus(d)
    b.add(0, reed(50, d, a=0.3) * 0.5, 0, 0.45)
    lyre = lambda m, dur: pluck(m, 2.0, bright=0.9, damp=0.997)
    theme(b, 0.3, 0.5, D, lyre, variant="phrygian", gain=1.0, pan=0.2)
    theme(b, 3.3, 0.4, D + 12, lyre, variant="phrygian", gain=0.8, pan=-0.2)
    for i in range(12):
        b.add(i * 0.5, tom(110 if i % 4 else 75, 0.4, decay=0.12, slap=0.4), -0.3, 0.3)
    return finish(b, 0.35, 2.5, fout=0.6)


def m_empires(d):
    b = Bus(d)
    chant = lambda m, dur: choir(m, dur, "a", a=0.1, r=0.4)
    b.add(0, choir(38, 3.3, "o", a=0.3, r=0.5), 0, 0.5)
    theme(b, 0.1, 0.5, D - 12, chant, gain=0.85)
    lute = lambda m, dur: pluck(m, 1.6, bright=0.55, damp=0.995)
    for i, ch in enumerate([CHORDS_MINOR[0], CHORDS_MINOR[3]]):
        for k, m in enumerate(ch[:4]):
            b.add(3.0 + i * 1.5 + k * 0.06, lute(m, 1.0), -0.3, 0.35)
    theme(b, 3.0, 0.42, D + 12, lute, gain=0.9, pan=0.2)
    return finish(b, 0.45, 3.0, fout=0.6)


def m_industry(d):
    b = Bus(d)
    chord(b, 0, 2.6, CHORDS_MINOR[0][:4], strings, 0.7, a=0.3, r=0.5)
    beat = 0.4
    for i in range(int(d / (beat / 2))):
        t = i * beat / 2
        if t < 2.0:
            continue
        if i % 2 == 0:
            b.add(t, clank(260 if i % 8 == 0 else 330, 0.4), 0.3, 0.35)
        if i % 4 == 2:
            b.add(t, fft_filter(noise(0.35), 3000, 12000) * perc(0.35, 0.1), -0.4, 0.35)
        if i % 4 == 0:
            b.add(t, kick(0.4), 0, 0.5)
    for i in range(int((d - 2.0) / beat)):
        b.add(2.0 + i * beat, brass(38 if i % 4 < 2 else 41, beat * 0.6, a=0.02, power=0.8), 0, 0.45)
    theme(b, 2.0, 0.4, D - 12, lambda m, dur: brass(m, dur, a=0.03, power=1.0), gain=0.8)
    return finish(b, 0.25, 2.0, fout=0.5)


def m_space(d):
    b = Bus(d)
    lt = TL.LAUNCH_T - TL.BY_KEY["space"].start
    for m, p in [(50, -0.5), (57, 0.5), (62, -0.2), (66, 0.2), (69, 0)]:
        s = strings(m, d, a=d * 0.8, r=0.5, trem=10)
        b.add(0, s, p, 0.45)
    roar = fft_filter(noise(d - lt + 1, 211), 25, 900)
    roar *= np.minimum(tarr(d - lt + 1) / 1.5, 1)
    b.add(lt, roar, 0, 0.6)
    b.add(lt, boom(3.0, 80, 30, 0.4), 0, 0.6)
    for i in range(16):
        b.add(lt - 1.0 + i * 0.06, timpani(38, 0.5), 0, 0.15 + 0.02 * i)
    theme(b, lt + 0.4, 0.55, D, lambda m, dur: brass(m, dur, a=0.05, power=1.2), variant="major", gain=1.0)
    theme(b, lt + 0.4, 0.55, D - 12, lambda m, dur: brass(m, dur, a=0.05, power=1.0), variant="major", gain=0.6)
    return finish(b, 0.35, 3.0, fout=0.6)


def m_today(d):
    b = Bus(d)
    beat = 0.5
    for i in range(int(d / beat)):
        t = i * beat
        b.add(t, kick(), 0, 0.75)
        b.add(t, sub(26 if (i // 4) % 2 == 0 else 22, beat * 0.8), 0, 0.5)
        b.add(t + beat / 2, hat(), 0.3, 0.6)
        if i % 2 == 1:
            b.add(t, clap(), -0.1, 0.4)
    arp = [62, 65, 69, 74, 69, 65]
    for i in range(int(d / (beat / 4))):
        t = i * beat / 4
        duck = 0.5 + 0.5 * min(1, (t % beat) / 0.2)
        b.add(t, synth_lead(arp[i % len(arp)], 0.08, r=0.08, bright=5), math.sin(t * 3) * 0.6, 0.18 * duck)
    theme(b, 0.5, 0.5, D + 12, lambda m, dur: synth_lead(m, dur, bright=9), gain=0.8)
    theme(b, 3.5, 0.375, D + 12, lambda m, dur: synth_lead(m, dur, bright=9), gain=0.7)
    return finish(b, 0.25, 2.0, fout=0.5)


def m_finale(d):
    b = Bus(d, tail=3.0)
    prog = [CHORDS_MINOR[0], CHORDS_MINOR[1], CHORDS_MINOR[2], CHORDS_MINOR[3]]
    for i, ch in enumerate(prog):
        chord(b, i * 3.5, 4.0, ch[:4], strings, 0.8, a=0.8, r=1.0)
        chord(b, i * 3.5, 4.0, [n + 12 for n in ch[1:4]], choir, 0.35, vowel="a", a=1.0, r=1.0)
    for i in range(int(14 / 0.5)):
        t = i * 0.5
        b.add(t, sub(38, 0.3), 0, 0.25)
        b.add(t, synth_lead([62, 69, 74, 69][i % 4], 0.1, r=0.1, bright=3), math.sin(t) * 0.5, 0.06)
    theme(b, 1.0, 1.0, D - 12, lambda m, dur: brass(m, dur, a=0.15, power=1.0), gain=0.9)
    # resolution: D major, everything together
    chord(b, 14.0, 6.5, CHORDS_MAJOR[0], strings, 1.0, a=0.6, r=2.5)
    chord(b, 14.0, 6.5, [n + 12 for n in CHORDS_MAJOR[0][1:4]], choir, 0.5, vowel="a", a=1.0, r=2.5)
    b.add(14.0, timpani(38, 2.5), 0, 0.6)
    theme(b, 8.0, 0.9, D + 12, lambda m, dur: strings(m, dur, a=0.15, voices=3, bright=11), variant="major",
          gain=0.8, pan=0.1)
    theme(b, 8.0, 0.9, D + 12, lambda m, dur: bell(m, dur + 2.0, idx=1.5, ratio=2.0), variant="major", gain=0.4,
          pan=-0.3)
    for k, m in enumerate([74, 78, 81, 86]):
        b.add(14.2 + k * 0.25, bell(m, 4.0, idx=1.2, ratio=2.0), (k - 1.5) * 0.4, 0.3)
    out = finish(b, 0.5, 4.5, fout=None)
    t = np.arange(out.shape[1]) / SR
    out *= np.clip((d - 0.6 - t) / 4.5, 0, 1)
    return out


# ------------------------------------------------------------------ master
# Loudness target (dB RMS over the phase itself) - shapes the dynamics of the film.
TARGET_DB = {"intro": -30, "bigbang": -18, "stars": -21, "solar": -21, "hadean": -18, "life": -22,
             "oxygen": -21, "eukaryote": -21, "snowball": -21, "cambrian": -19, "land": -19, "dying": -17,
             "dinos": -16, "asteroid": -17, "mammals": -20, "humans": -19, "africa": -20, "caveart": -20,
             "iceage": -21, "farming": -19, "writing": -19, "empires": -20, "industry": -18, "space": -16,
             "today": -16, "finale": -18}


def limiter(x, thr=0.9, block=128, release=0.15):
    n = x.shape[1] // block * block
    pk = np.abs(x[:, :n]).max(0).reshape(-1, block).max(1)
    g = np.minimum(1.0, thr / np.maximum(pk, 1e-9))
    g = np.minimum(g, np.concatenate([g[1:], [1.0]]))  # look ahead one block
    rel = math.exp(-block / (release * SR))
    out = np.empty_like(g)
    cur = 1.0
    for i, v in enumerate(g):
        cur = v if v < cur else v + (cur - v) * rel
        out[i] = cur
    gs = np.repeat(out, block)
    y = x.copy()
    y[:, :n] *= gs
    return y


def build():
    total = int((TL.DURATION + 1) * SR)
    mix = np.zeros((2, total), np.float32)
    for p in TL.PHASES:
        fn = globals()["m_" + p.key]
        part = fn(p.dur)
        body = part[:, :int(p.dur * SR)].mean(0)
        rms_db = 20 * math.log10(math.sqrt(float(np.mean(body ** 2))) + 1e-9)
        gain = 10 ** ((TARGET_DB[p.key] - rms_db) / 20)
        print(f"  music: {p.key:10s} {rms_db:6.1f} dB -> {TARGET_DB[p.key]} dB", flush=True)
        i = int(p.start * SR)
        n = min(part.shape[1], total - i)
        mix[:, i:i + n] += part[:, :n] * gain
    mix = mix[:, :int(TL.DURATION * SR)]
    mix = limiter(mix, 0.85)
    mix = np.tanh(mix * 1.1) / np.tanh(1.1)
    mix *= 0.89 / np.max(np.abs(mix))
    return mix


def write_wav(path, mix):
    data = (np.clip(mix.T, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "build/soundtrack.wav"
    write_wav(out, build())
    print("wrote", out)
