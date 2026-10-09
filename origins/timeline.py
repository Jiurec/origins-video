"""Master timeline: every phase, its on-screen text, dates and timing.

All times are in seconds of the final video. `ya` = "years ago" at the
moment the phase starts; the on-screen counter interpolates between them.
"""
import math
from dataclasses import dataclass, field

DURATION = 300.0
FPS = 30
AGE_OF_UNIVERSE = 13.8e9


@dataclass
class Phase:
    key: str            # scene + music identifier
    start: float
    end: float
    ya: float           # years ago at phase start
    title: str = ""
    date: str = ""
    desc: str = ""
    fact: str = ""
    accent: tuple = (255, 255, 255)
    number: str = ""
    group: str = ""     # header label for the sub-events of the last 100k years

    @property
    def dur(self):
        return self.end - self.start


P = Phase
PHASES = [
    P("intro", 0, 6, 13.8e9),
    P("bigbang", 6, 22, 13.8e9, "The Big Bang", "13.8 billion years ago",
      "All of space, time, matter and energy burst out of an unimaginably hot, dense point and begin to expand.",
      "Within about 20 minutes, the first atomic nuclei, hydrogen and helium, had formed.",
      (255, 214, 150)),
    P("stars", 22, 38, 13.6e9, "First Stars & Galaxies", "~13.6 billion years ago",
      "Gravity gathers hydrogen and helium into the first stars. Their cores forge carbon, oxygen and iron.",
      "Exploding stars scattered those elements through space: the raw material for planets, and for you.",
      (170, 200, 255)),
    P("solar", 38, 53, 4.6e9, "The Sun & Earth Are Born", "4.6 billion years ago",
      "A cloud of gas and dust collapses. The Sun ignites at its centre and rocky planets grow in the disk around it.",
      "Earth was assembled from collisions of countless smaller bodies over tens of millions of years.",
      (255, 190, 110)),
    P("hadean", 53, 68, 4.5e9, "A World of Fire", "4.5 billion years ago",
      "Young Earth is a glowing magma ocean. A Mars-sized world, Theia, is thought to have slammed into it.",
      "Debris from that impact formed our Moon, which then loomed far larger and closer in the sky.",
      (255, 120, 60)),
    P("life", 68, 86, 4.0e9, "Oceans & the First Life", "~4 billion years ago",
      "Steam rains down into oceans. Somewhere, perhaps at deep-sea hydrothermal vents, chemistry becomes biology.",
      "Every living thing today descends from one common ancestor, nicknamed LUCA.",
      (90, 210, 230)),
    P("oxygen", 86, 100, 2.4e9, "The Great Oxidation", "2.4 billion years ago",
      "Cyanobacteria harness sunlight and release oxygen as waste. The air and oceans change forever.",
      "Oxygen rusted the iron dissolved in the seas, leaving red banded rocks we still mine today.",
      (120, 230, 150)),
    P("eukaryote", 100, 112, 2.0e9, "Complex Cells", "~2 billion years ago",
      "One cell engulfs another but doesn't digest it. The guest becomes the mitochondrion, the cell's power plant.",
      "Your cells still carry these ancient bacterial passengers, complete with their own DNA.",
      (230, 150, 255)),
    P("snowball", 112, 126, 720e6, "Snowball Earth & First Animals", "720 – 540 million years ago",
      "Ice spreads almost from pole to pole. After the great thaw, the first large soft-bodied animals appear.",
      "Many Ediacaran creatures had no mouth, eyes or shell. Some looked like fronds or quilted mats.",
      (180, 230, 255)),
    P("cambrian", 126, 142, 538e6, "The Cambrian Explosion", "538 million years ago",
      "In a geological blink, animals evolve eyes, shells, legs and jaws. Most major animal groups appear.",
      "Trilobites thrived in the seas for around 270 million years.",
      (100, 220, 255)),
    P("land", 142, 157, 470e6, "Life Moves onto Land", "470 – 375 million years ago",
      "Plants green the continents, insects follow, and lobe-finned fish begin to crawl ashore.",
      "Tiktaalik, a 375-million-year-old fish with a neck and wrists, bridges fish and four-legged animals.",
      (150, 230, 110)),
    P("dying", 157, 167, 252e6, "The Great Dying", "252 million years ago",
      "Colossal eruptions in Siberia poison the air and seas: the worst extinction in Earth's history.",
      "Roughly 90% of marine species and 70% of land vertebrate species vanish.",
      (255, 80, 60)),
    P("dinos", 167, 184, 230e6, "The Age of Dinosaurs", "230 – 66 million years ago",
      "Dinosaurs rule the land for over 160 million years, pterosaurs fill the skies, and the first flowers bloom.",
      "Birds are living dinosaurs, descendants of small feathered theropods.",
      (255, 170, 80)),
    P("asteroid", 184, 194, 66e6, "The Asteroid", "66 million years ago",
      "A 10 km asteroid strikes Mexico's Yucatán. Firestorms and years of darkness end the age of dinosaurs.",
      "About three-quarters of all species disappear. Mostly small animals make it through.",
      (255, 110, 50)),
    P("mammals", 194, 208, 66e6, "The Rise of Mammals", "66 – 7 million years ago",
      "Small, warm-blooded survivors inherit an empty world: whales, horses, bats, and primates in the trees.",
      "Grasping hands and forward-facing eyes evolved for a life among the branches.",
      (255, 200, 120)),
    P("humans", 208, 224, 7e6, "The First Humans", "7 million – 300,000 years ago",
      "African apes begin walking upright, shape stone tools, master fire, and Homo sapiens emerges.",
      "The oldest Homo sapiens fossils, from Jebel Irhoud in Morocco, are about 300,000 years old.",
      (255, 160, 90)),
    # ---- The last 100,000 years (6 s each) ----
    P("africa", 224, 230, 70e3, "Out of Africa", "~70,000 – 60,000 years ago",
      "Small bands of modern humans spread out of Africa, across Asia and on to Australia.",
      accent=(255, 200, 120)),
    P("caveart", 230, 236, 45e3, "The First Artists", "~45,000 – 40,000 years ago",
      "Cave paintings, carved figurines and bone flutes appear. Our Neanderthal cousins vanish.",
      accent=(230, 120, 80)),
    P("iceage", 236, 242, 20e3, "The Ice Age Peak", "~20,000 years ago",
      "Ice sheets bury the north. Humans cross from Asia into the Americas.",
      accent=(190, 230, 255)),
    P("farming", 242, 248, 12e3, "The First Farmers", "~12,000 years ago",
      "As the ice retreats, people plant wheat and tame animals. Villages grow.",
      accent=(240, 210, 110)),
    P("writing", 248, 254, 5.2e3, "Writing & Cities", "~5,200 years ago",
      "In Mesopotamia, scribes press wedge-shaped marks into clay. Recorded history begins.",
      accent=(230, 180, 120)),
    P("empires", 254, 260, 2.5e3, "Empires & Ideas", "2,500 – 500 years ago",
      "Philosophy, science and great empires rise and fall. The Renaissance revives ancient learning.",
      accent=(240, 220, 170)),
    P("industry", 260, 266, 250, "The Industrial Revolution", "~250 years ago",
      "Steam engines and factories transform how people live, work and travel.",
      accent=(220, 160, 110)),
    P("space", 266, 272, 120, "Into the Sky", "1903 – 1969",
      "From the first powered flight to footprints on the Moon in just 66 years.",
      accent=(170, 210, 255)),
    P("today", 272, 278, 25, "Today", "Now",
      "8 billion people, linked by a global network, still discovering where we came from.",
      accent=(120, 220, 255)),
    P("finale", 278, 300, 0),
]

_n = 0
for _p in PHASES:
    if _p.key in ("intro", "finale"):
        continue
    if _p.start < 224:
        _n += 1
        _p.number = f"PHASE {_n:02d}"
    else:
        _p.number = f"PHASE {_n + 1:02d}"
        _p.group = "THE LAST 100,000 YEARS"

LAST100K_START = 224.0
BY_KEY = {p.key: p for p in PHASES}

# Synchronised moments shared by the visuals and the soundtrack (global s).
BANG_T = BY_KEY["bigbang"].start + 2.0
THEIA_T = BY_KEY["hadean"].start + 6.0
IMPACT_T = BY_KEY["asteroid"].start + 4.0
LAUNCH_T = BY_KEY["space"].start + 1.2
STAR_IGNITIONS = [BY_KEY["stars"].start + x for x in
                  (0.6, 1.3, 1.9, 2.6, 3.0, 3.7, 4.5, 5.0, 5.8, 6.5, 7.4, 8.1)]
SUPERNOVA_T = BY_KEY["stars"].start + 12.0


def phase_at(t):
    for p in PHASES:
        if p.start <= t < p.end:
            return p
    return PHASES[-1]


def years_ago(t):
    """Continuous 'years ago' counter for video time t."""
    for i, p in enumerate(PHASES):
        if p.start <= t < p.end or i == len(PHASES) - 1:
            a = p.ya
            b = PHASES[i + 1].ya if i + 1 < len(PHASES) else 0.0
            if p.key == "finale":
                return 0.0
            u = min(max((t - p.start) / p.dur, 0.0), 1.0)
            if a <= 0 or b <= 0:
                return a + (b - a) * u
            return math.exp(math.log(a) + (math.log(b) - math.log(a)) * u)
    return 0.0


def format_years(ya):
    if ya < 1:
        return "Today"
    if ya >= 1e9:
        return f"{ya / 1e9:.2f}".rstrip("0").rstrip(".") + " billion years ago"
    if ya >= 1e6:
        v = ya / 1e6
        s = f"{v:.0f}" if v >= 100 else (f"{v:.1f}".rstrip("0").rstrip("."))
        return s + " million years ago"
    if ya >= 1e4:
        return f"{round(ya, -2):,.0f} years ago"
    if ya >= 100:
        return f"{round(ya, -1):,.0f} years ago"
    return f"{ya:,.0f} years ago"


_MONTHS = [("Jan", 31), ("Feb", 28), ("Mar", 31), ("Apr", 30), ("May", 31), ("Jun", 30),
           ("Jul", 31), ("Aug", 31), ("Sep", 30), ("Oct", 31), ("Nov", 30), ("Dec", 31)]


def cosmic_calendar(ya):
    """Map 'years ago' onto Carl Sagan's Cosmic Calendar (one year = 13.8 Gyr).

    Returns (date string, time string or None)."""
    frac = min(max(1.0 - ya / AGE_OF_UNIVERSE, 0.0), 1.0)
    day = frac * 365.0
    if day >= 365.0:
        day = 365.0 - 1e-9
    d = int(day)
    for name, n in _MONTHS:
        if d < n:
            date = f"{name} {d + 1}"
            break
        d -= n
    time = None
    if day >= 364.0:
        secs = (day - 364.0) * 86400.0
        hh, rem = divmod(int(secs), 3600)
        mm, ss = divmod(rem, 60)
        time = f"{hh:02d}:{mm:02d}:{ss:02d}"
    return date, time
