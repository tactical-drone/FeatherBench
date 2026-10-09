"""
Compartment: DATASETS  (2-D classification problems)

Contract:  fn(n, noise, rng) -> (X float32 (n, 2), y int64 (n,))
  n      requested point count (a dataset may ignore it, e.g. the XOR gate)
  noise  0..0.5, dataset-specific meaning
  rng    numpy Generator; use it for all randomness so seeds reproduce
Keep points roughly inside [-1.2, 1.2]; the number of classes is max(y) + 1.
A dataset whose layout is itself random (line positions, cell centres) also
accepts layout_rng=None and draws the layout from it (default: rng, so the
data is unchanged); sample(name, n, noise, rng, layout_rng) uses it, so a fresh
sample (Session.evaluate_fresh) keeps the training seed's layout.

Also here: SPLITTERS, which turn a dataset into train / test index sets.
Contract:  fn(n, test_frac, rng) -> (train_idx, test_idx)

And FAMILIES: datasets with a size knob (spiral arms, checkerboard cells, ...),
named "family[size]", e.g. DATASETS.get("spiral[7]"). Those names resolve on
demand and are not added to the dataset list (or the UI dropdown).
"""
import hashlib
import math
import re

import numpy as np

from .registry import Registry, call_accepting

DATASETS = Registry("dataset", "fn(n, noise, rng) -> (X (n,2) float32, y (n,) int64)")
SPLITTERS = Registry("splitter", "fn(n, test_frac, rng) -> (train_idx, test_idx)")


def _out(X, y):
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


@DATASETS.register("XOR gate (4 points)")
def xor_gate(n, noise, rng):
    X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0])
    return _out(X + rng.normal(0, noise * 0.3, X.shape), y)


@DATASETS.register("XOR quadrants")
def xor_quadrants(n, noise, rng):
    X = rng.uniform(-1.2, 1.2, (n, 2))
    X += np.sign(X) * 0.05
    y = (X[:, 0] * X[:, 1] < 0).astype(int)
    X += rng.normal(0, noise, X.shape)
    return _out(X, y)


def spiral(k):
    def make(n, noise, rng):
        if n < k:
            raise ValueError(f"Spiral ({k} arms) needs n_points >= {k}, got {n}")
        per = n // k
        Xs, ys = [], []
        for c in range(k):
            r = np.linspace(0.05, 1.0, per)
            t = r * 3.2 * math.pi + c * 2 * math.pi / k
            t = t + rng.normal(0, noise * 1.5, per)
            Xs.append(np.stack([r * np.sin(t), r * np.cos(t)], 1) * 1.15)
            ys.append(np.full(per, c))
        return _out(np.concatenate(Xs), np.concatenate(ys))
    return make


DATASETS.register("Spiral (2 arms)", spiral(2))
DATASETS.register("Spiral (3 arms)", spiral(3))
DATASETS.register("Spiral (4 arms)", spiral(4))
DATASETS.register("Spiral (5 arms)", spiral(5))

@DATASETS.register("Circles")
def circles(n, noise, rng):
    r = np.concatenate([rng.uniform(0, 0.45, n // 2), rng.uniform(0.75, 1.15, n - n // 2)])
    t = rng.uniform(0, 2 * math.pi, n)
    X = np.stack([r * np.cos(t), r * np.sin(t)], 1) + rng.normal(0, noise, (n, 2))
    y = np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)])
    return _out(X, y)


@DATASETS.register("Moons")
def moons(n, noise, rng):
    m = n // 2
    t1, t2 = rng.uniform(0, math.pi, m), rng.uniform(0, math.pi, n - m)
    a = np.stack([np.cos(t1), np.sin(t1)], 1)
    b = np.stack([1 - np.cos(t2), 0.5 - np.sin(t2)], 1)
    X = (np.concatenate([a, b]) - [0.5, 0.25]) * 0.8 + rng.normal(0, noise, (n, 2))
    y = np.concatenate([np.zeros(m), np.ones(n - m)])
    return _out(X, y)


@DATASETS.register("Checkerboard")
def checkerboard(n, noise, rng):
    X = rng.uniform(-1.2, 1.2, (n, 2))
    y = ((np.floor((X[:, 0] + 1.2) / 0.6) + np.floor((X[:, 1] + 1.2) / 0.6)) % 2).astype(int)
    X += rng.normal(0, noise * 0.5, X.shape)
    return _out(X, y)


@DATASETS.register("Gaussian blobs (5)")
def blobs5(n, noise, rng):
    k = 5
    ang = np.arange(k) * 2 * math.pi / k
    centers = np.stack([np.cos(ang), np.sin(ang)], 1) * 0.8
    y = rng.integers(0, k, n)
    return _out(centers[y] + rng.normal(0, 0.12 + noise, (n, 2)), y)


def sample(name, n, noise, rng, layout_rng=None):
    """Points from dataset `name`. layout_rng (a Generator seeded like the training
    data's) fixes the layout of random-structure datasets; others ignore it."""
    fn = DATASETS.get(name)
    return fn(n, noise, rng) if layout_rng is None else call_accepting(fn, n, noise, rng, layout_rng=layout_rng)


# --------------------------------------------------------------------------- splitters

def _check_frac(test_frac):
    if not 0 <= test_frac < 1:
        raise ValueError(f"test_frac must be in [0, 1), got {test_frac!r}")


@SPLITTERS.register("random")
def random_split(n, test_frac, rng):
    """Shuffled holdout. Tiny sets (<= 8 points) keep everything for training;
    at least one point always stays in train."""
    _check_frac(test_frac)
    idx = rng.permutation(n)
    n_test = 0 if n <= 8 else min(int(n * test_frac), n - 1)
    return idx[n_test:], idx[:n_test]


@SPLITTERS.register("none (train on all)")
def no_split(n, test_frac, rng):
    return np.arange(n), np.arange(0)


def stratified_split(n, test_frac, rng, y=None):
    """Per-class holdout: int(n_c * test_frac) test points from each class c (so small,
    many-class sets keep every class in test), shuffled. Needs the labels y; not
    registered as a SPLITTER until Session passes y= to splitters that accept it."""
    _check_frac(test_frac)
    if y is None:
        raise ValueError("the stratified splitter needs the labels (y=)")
    y = np.asarray(y)
    if n <= 8:
        return rng.permutation(n), np.arange(0)
    tr, te = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        k = min(int(len(idx) * test_frac), len(idx) - 1)
        te.append(idx[:k])
        tr.append(idx[k:])
    tr, te = np.concatenate(tr), np.concatenate(te)
    return tr[rng.permutation(len(tr))], te[rng.permutation(len(te))]


# =========================================================================== dataset zoo
# More pattern families, so a recipe can be scored on how *general* it is
# (see SUITES below). Everything here is new and registered after the
# originals, so the original datasets and their RNG use are untouched.
# Every zoo dataset returns exactly n points with every class present
# whenever n >= number of classes, and draws all randomness from rng.
# For n < number of classes it still returns n points, using classes
# 0..n-1 only (never raises).
#
# Noise: each zoo dataset scales its noise so that noise 0.3 moves roughly
# 10-25% of points across a class boundary (measured with a 15-NN vote
# trained on noise-0 data). Datasets with a single smooth boundary (Linear,
# Sine boundary) sit at the low end, since jitter rarely crosses one line.
# The classic datasets above keep their own noise meaning, so benchmark
# scores are only comparable at one fixed noise level (default 0.05).

def _quotas(n, k, weights=None):
    """Split n into k class counts proportional to weights, each >= 1."""
    w = np.ones(k) if weights is None else np.asarray(weights, dtype=float)
    raw = n * w / w.sum()
    q = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - q), kind="stable")[: n - q.sum()]:
        q[i] += 1
    while (q < 1).any() and n >= k:  # tiny n: borrow from the largest class
        q[np.argmax(q)] -= 1
        q[np.argmin(q)] += 1
    return q


def _box(m, rng, half=1.2):
    return rng.uniform(-half, half, (m, 2))


def _disk(m, rng, radius=1.15):
    r = radius * np.sqrt(rng.uniform(0, 1, m))
    t = rng.uniform(0, 2 * math.pi, m)
    return np.stack([r * np.cos(t), r * np.sin(t)], 1)


def _area_weights(k, label_fn, sampler, power=0.5, m=20000):
    """Class weights proportional to (area share) ** power, estimated on a fixed
    private sample, so the caller's rng stream is not touched. power=0.5 is a
    compromise between equal counts (dense small regions) and equal density
    (tiny regions almost empty)."""
    lab = np.asarray(label_fn(sampler(m, np.random.default_rng(12345))), dtype=int)
    share = np.bincount(lab[lab >= 0], minlength=k)[:k] / m
    if (share == 0).any():
        raise RuntimeError("a class has no area")
    return share ** power


def _stratified(n, k, label_fn, rng, sampler=_box, weights=None, jitter=0.0):
    """Rejection-sample candidates from sampler(m, rng), label them with
    label_fn(X) -> int array, and keep a fixed quota per class, so classes are
    balanced (or follow weights) whatever the regions' areas. Then jitter.
    weights="sqrt_area" uses _area_weights (class count ~ sqrt of its area)."""
    if isinstance(weights, str):
        assert weights == "sqrt_area", weights
        weights = _area_weights(k, label_fn, sampler)
    quota = _quotas(n, k, weights)
    got = [[] for _ in range(k)]
    have = np.zeros(k, dtype=int)
    for _ in range(1000):
        if (have >= quota).all():
            break
        cand = sampler(max(256, 4 * n), rng)
        lab = np.asarray(label_fn(cand), dtype=int)
        for c in range(k):
            need = quota[c] - have[c]
            if need > 0:
                pts = cand[lab == c][:need]
                got[c].append(pts)
                have[c] += len(pts)
    else:
        raise RuntimeError("could not fill every class")
    X = np.concatenate([np.concatenate(g) for g in got if g] or [np.empty((0, 2))])
    y = np.repeat(np.arange(k), quota)
    order = rng.permutation(n)
    X, y = X[order], y[order]
    return _out(X + rng.normal(0, jitter, X.shape), y)


def _gauss_mix(n, centers, stds, rng, weights=None, labels=None):
    """Gaussian clusters; labels[i] is cluster i's class (default: i)."""
    centers = np.asarray(centers, dtype=float)
    stds = np.broadcast_to(np.asarray(stds, dtype=float), (len(centers),))
    labels = np.arange(len(centers)) if labels is None else np.asarray(labels)
    q = _quotas(n, len(centers), weights)
    X = np.concatenate([c + rng.normal(0, s, (m, 2)) for c, s, m in zip(centers, stds, q)])
    y = np.repeat(labels, q)
    order = rng.permutation(n)
    return _out(X[order], y[order])


def _polar(X):
    return np.hypot(X[:, 0], X[:, 1]), np.arctan2(X[:, 1], X[:, 0]) % (2 * math.pi)


# --------------------------------------------------------------------------- radial

@DATASETS.register("Rings (3)")
def rings3(n, noise, rng):
    """Three concentric annuli with gaps: class = which ring."""
    bands = [(0.0, 0.3), (0.5, 0.72), (0.92, 1.15)]

    def ring(r):
        lab = np.full(len(r), -1)
        for c, (a, b) in enumerate(bands):
            lab[(r >= a) & (r < b)] = c
        return lab
    return _stratified(n, 3, lambda X: ring(_polar(X)[0]), rng, _disk, weights="sqrt_area", jitter=noise * 0.4)


@DATASETS.register("Bullseye (5 rings)")
def bullseye(n, noise, rng):
    """Filled disk cut into 5 equal-width rings, alternating 2 classes (radial parity)."""
    return _stratified(n, 2, lambda X: np.floor(_polar(X)[0] / (1.15 / 5)).astype(int) % 2,
                       rng, _disk, weights="sqrt_area", jitter=noise * 0.2)


# --------------------------------------------------------------------------- angular

@DATASETS.register("Pinwheel (4)")
def pinwheel(n, noise, rng):
    """Four short curved blades that widen outwards: class = blade."""
    k = 4
    q = _quotas(n, k)
    y = np.repeat(np.arange(k), q)
    r = rng.uniform(0.15, 1.1, n)
    t = y * 2 * math.pi / k + 1.4 * r + rng.normal(0, 0.12, n) * (0.4 + r) + rng.normal(0, noise * 1.6, n)
    X = np.stack([r * np.cos(t), r * np.sin(t)], 1) + rng.normal(0, noise * 0.1, (n, 2))
    order = rng.permutation(n)
    return _out(X[order], y[order])


@DATASETS.register("Sectors (8 wedges)")
def sectors(n, noise, rng):
    """Disk cut into 8 pie wedges; opposite wedges share a class (4 classes)."""
    return _stratified(n, 4, lambda X: np.floor(_polar(X)[1] / (math.pi / 4)).astype(int) % 4,
                       rng, _disk, jitter=noise * 0.4)


# --------------------------------------------------------------------------- periodic

@DATASETS.register("Sine boundary")
def sine_boundary(n, noise, rng):
    """Above / below the wave x2 = 0.45 sin(5 x1)."""
    return _stratified(n, 2, lambda X: (X[:, 1] > 0.45 * np.sin(5 * X[:, 0])).astype(int),
                       rng, jitter=noise * 0.6)


@DATASETS.register("Stripes (diagonal)")
def stripes(n, noise, rng):
    """Parallel stripes running at -60 degrees (upper left to lower right; their
    normal points at 30 degrees), 0.32 wide, 3 classes repeating."""
    a = math.radians(30)
    return _stratified(n, 3, lambda X: np.floor((X[:, 0] * math.cos(a) + X[:, 1] * math.sin(a)) / 0.32).astype(int) % 3,
                       rng, jitter=noise * 0.3)


@DATASETS.register("Egg crate (3)")
def egg_crate(n, noise, rng):
    """Smooth 2-D wave sin(5 x1) + sin(5 x2): peaks (1), pits (2), flat sea (0)."""
    def lab(X):
        h = np.sin(5 * X[:, 0]) + np.sin(5 * X[:, 1])
        return np.where(h > 0.9, 1, np.where(h < -0.9, 2, 0))
    return _stratified(n, 3, lab, rng, jitter=noise * 0.3)


# --------------------------------------------------------------------------- tilings

@DATASETS.register("Checkerboard (fine)")
def checkerboard_fine(n, noise, rng):
    """5 x 5 checkerboard: smaller cells than the classic 4 x 4, and a cell (not a
    corner) at the centre, so no boundary lies on the axes."""
    return _stratified(n, 2, lambda X: (np.floor((X[:, 0] + 1.2) / 0.48) + np.floor((X[:, 1] + 1.2) / 0.48)).astype(int) % 2,
                       rng, jitter=noise * 0.3)


def _hex_labels(s):
    """Label function for hexagons of radius s, 3-coloured so no two neighbours share a class."""
    def lab(X):
        q = (math.sqrt(3) / 3 * X[:, 0] - X[:, 1] / 3) / s
        r = (2 / 3 * X[:, 1]) / s
        x, z = q, r
        yy = -x - z
        rx, ry, rz = np.round(x), np.round(yy), np.round(z)
        dx, dy, dz = abs(rx - x), abs(ry - yy), abs(rz - z)
        fx = (dx > dy) & (dx > dz)
        fz = ~fx & (dz >= dy)
        rx = np.where(fx, -ry - rz, rx)
        rz = np.where(fz, -rx - ry, rz)
        return (rx - rz).astype(int) % 3
    return lab


@DATASETS.register("Hex tiling (3)")
def hex_tiling(n, noise, rng):
    """About 20 small hexagonal tiles (radius 0.33), 3-coloured. Under-sampled
    at n=600 (~24 points per tile): use n >= 2000 to see the tiles clearly."""
    return _stratified(n, 3, _hex_labels(0.33), rng, jitter=noise * 0.3)


@DATASETS.register("Hex tiling (3, coarse)")
def hex_tiling_coarse(n, noise, rng):
    """About 10 larger hexagonal tiles (radius 0.48), 3-coloured: readable at n=600."""
    return _stratified(n, 3, _hex_labels(0.48), rng, jitter=noise * 0.3)


# --------------------------------------------------------------------------- random structure (new layout per seed)

@DATASETS.register("Random lines (parity)")
def random_lines(n, noise, rng, layout_rng=None):
    """4 random lines (directions ~45 deg apart, offset from the centre);
    class = parity of how many lines a point lies above."""
    lr = rng if layout_rng is None else layout_rng
    th = lr.uniform(0, math.pi) + np.arange(4) * math.pi / 4 + lr.uniform(-0.25, 0.25, 4)
    d = lr.choice([-1.0, 1.0], 4) * lr.uniform(0.15, 0.7, 4)  # off-centre, so cells vary
    normals = np.stack([np.cos(th), np.sin(th)], 1)
    return _stratified(n, 2, lambda X: ((X @ normals.T) > d).sum(1) % 2, rng, jitter=noise * 0.45)


@DATASETS.register("Voronoi (6)")
def voronoi(n, noise, rng, layout_rng=None):
    """6 random centres (kept >= 0.55 apart); class = nearest centre."""
    lr = rng if layout_rng is None else layout_rng
    centers = [lr.uniform(-0.95, 0.95, 2)]
    while len(centers) < 6:
        c = lr.uniform(-0.95, 0.95, 2)
        if min(np.hypot(*(c - p)) for p in centers) >= 0.55:
            centers.append(c)
    C = np.array(centers)
    return _stratified(n, 6, lambda X: ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1),
                       rng, jitter=noise * 0.5)


# --------------------------------------------------------------------------- linear sanity baseline

@DATASETS.register("Linear")
def linear(n, noise, rng):
    """One tilted straight line splits the plane: the easiest possible task."""
    return _stratified(n, 2, lambda X: (0.6 * X[:, 0] + 0.8 * X[:, 1] > 0.1).astype(int),
                       rng, jitter=noise * 0.6)


# --------------------------------------------------------------------------- interleaved shapes

@DATASETS.register("Moons (4)")
def moons4(n, noise, rng):
    """A chain of 4 interlocking half-moons, alternately up and down: class = moon.
    Arc radius 0.8 at spacing 1, so no two moons touch (closest approach 0.2
    after scaling, for same-facing and neighbouring moons alike)."""
    k, R, h = 4, 0.8, 0.4
    q = _quotas(n, k)
    Xs = []
    for c, m in enumerate(q):
        t = rng.uniform(0, math.pi, m)
        sign = 1 if c % 2 == 0 else -1
        Xs.append(np.stack([c + R * np.cos(t), h * (c % 2) + sign * R * np.sin(t)], 1))
    X = (np.concatenate(Xs) - [1.5, h / 2]) * [0.5, 0.9] + rng.normal(0, noise * 0.5, (n, 2))
    y = np.repeat(np.arange(k), q)
    order = rng.permutation(n)
    return _out(X[order], y[order])


# --------------------------------------------------------------------------- harder spirals (exact n)

def _spiral_arms(n, k, turns, powers, noise, rng, noise_scale=0.3):
    q = _quotas(n, k)
    Xs = []
    for c, m in enumerate(q):
        r = 0.05 + 0.95 * np.linspace(0, 1, m) ** powers[c]
        t = r * turns * 2 * math.pi + c * 2 * math.pi / k
        Xs.append(np.stack([r * np.sin(t), r * np.cos(t)], 1) * 1.15)
    X = np.concatenate(Xs) + rng.normal(0, noise * noise_scale, (n, 2))
    y = np.repeat(np.arange(k), q)
    order = rng.permutation(n)
    return _out(X[order], y[order])


@DATASETS.register("Spiral (6 arms)")
def spiral6(n, noise, rng):
    """Six interleaved arms but only 1.1 turns: tests many classes rather than
    density (the 5-arm classic winds 1.6 turns)."""
    return _spiral_arms(n, 6, 1.1, [1] * 6, noise, rng, noise_scale=0.2)


@DATASETS.register("Spiral (2 arms, tight)")
def spiral_tight(n, noise, rng):
    """Two arms wound 3.5 times: thin, closely packed bands."""
    return _spiral_arms(n, 2, 3.5, [1, 1], noise, rng, noise_scale=0.2)


@DATASETS.register("Spiral (3 arms, uneven)")
def spiral_uneven(n, noise, rng):
    """Three arms with different point density: crowded centre, even, crowded rim."""
    return _spiral_arms(n, 3, 1.6, [2.0, 1.0, 0.5], noise, rng)


# --------------------------------------------------------------------------- multi-scale

@DATASETS.register("Blobs + tiny cluster")
def blobs_tiny(n, noise, rng):
    """Three broad blobs and a tiny tight cluster (10% of points) in the middle."""
    ang = np.radians([90, 210, 330])
    centers = np.concatenate([np.stack([np.cos(ang), np.sin(ang)], 1) * 0.75, [[0.0, 0.0]]])
    return _gauss_mix(n, centers, [0.15 + noise * 0.7] * 3 + [0.04 + noise * 0.2], rng,
                      weights=[0.3, 0.3, 0.3, 0.1])


@DATASETS.register("Ring and blobs")
def ring_and_blobs(n, noise, rng):
    """A thin ring (class 1) between a centre blob and 4 corner blobs (both class 0)."""
    q0, q1 = _quotas(n, 2)
    qc, qs = _quotas(q0, 2)
    centre = rng.normal(0, 0.12, (qc, 2))
    corners = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) * 0.9
    sat = corners[rng.integers(0, 4, qs)] + rng.normal(0, 0.08, (qs, 2))
    t = rng.uniform(0, 2 * math.pi, q1)
    r = 0.62 + rng.normal(0, 0.03, q1)
    ring = np.stack([r * np.cos(t), r * np.sin(t)], 1)
    X = np.concatenate([centre, sat, ring])
    X = X + rng.normal(0, noise * 0.6, X.shape)
    y = np.repeat([0, 1], [q0, q1])
    order = rng.permutation(n)
    return _out(X[order], y[order])


# --------------------------------------------------------------------------- fractal boundary

@DATASETS.register("Mandelbrot")
def mandelbrot(n, noise, rng):
    """Inside (1) / outside (0) the Mandelbrot set, window re [-1.95, 0.45], im [-1.2, 1.2].
    At n=600 this is mostly the main cardioid plus the period-2 disc: a curved
    boundary with a cusp and a pinch, not a fractal test."""
    def lab(X):
        c = (X[:, 0] - 0.75) + 1j * X[:, 1]
        z = np.zeros_like(c)
        inside = np.ones(len(c), dtype=bool)
        for _ in range(40):
            z = np.where(inside, z * z + c, z)
            inside &= np.abs(z) <= 2
        return inside.astype(int)
    return _stratified(n, 2, lab, rng, weights="sqrt_area", jitter=noise * 0.5)


# --------------------------------------------------------------------------- compositional shapes

@DATASETS.register("Yin-yang")
def yin_yang(n, noise, rng):
    """The yin-yang symbol: two interlocking halves, each with a dot of the other."""
    R = 1.1

    def lab(X):
        x, yv = X[:, 0], X[:, 1]
        dt = np.hypot(x, yv - R / 2)
        db = np.hypot(x, yv + R / 2)
        out = (x > 0).astype(int)
        out = np.where(dt < R / 2, 0, out)
        out = np.where(db < R / 2, 1, out)
        out = np.where(dt < R / 7, 1, out)
        out = np.where(db < R / 7, 0, out)
        return out
    return _stratified(n, 2, lab, rng, lambda m, g: _disk(m, g, R), jitter=noise * 0.5)


@DATASETS.register("Smiley")
def smiley(n, noise, rng):
    """Background (0), face (1), and eyes + mouth (2). Class counts follow the
    square root of each region's area, so the small features are denser than
    the face but not 7x denser."""
    def lab(X):
        x, yv = X[:, 0], X[:, 1]
        out = np.where(np.hypot(x, yv) < 1.0, 1, 0)
        eyes = (np.hypot(np.abs(x) - 0.35, yv - 0.3) < 0.15)
        dm = np.hypot(x, yv - 0.1)
        mouth = (dm > 0.5) & (dm < 0.66) & (yv < -0.08)
        return np.where(eyes | mouth, 2, out)
    return _stratified(n, 3, lab, rng, weights="sqrt_area", jitter=noise * 0.3)


# --------------------------------------------------------------------------- clusters, imbalance, label noise

@DATASETS.register("XOR blobs")
def xor_blobs(n, noise, rng):
    """Four Gaussian blobs at the corners; diagonal blobs share a class. A sanity
    check (close to XOR quadrants), not a discriminating benchmark entry."""
    centers = [[0.6, 0.6], [-0.6, 0.6], [-0.6, -0.6], [0.6, -0.6]]  # classes alternate, so n=2 has both
    return _gauss_mix(n, centers, 0.16 + noise * 0.8, rng, labels=[0, 1, 0, 1])


@DATASETS.register("Imbalanced (90/10)")
def imbalanced(n, noise, rng):
    """A big majority blob (90%) and a small minority cluster (10%) at its edge.
    Always predicting the majority already scores 0.90 accuracy, so score this
    with balanced accuracy (mean per-class recall) where possible."""
    return _gauss_mix(n, [[-0.2, 0.0], [0.55, 0.35]], [0.28 + noise * 0.5, 0.1 + noise * 0.4],
                      rng, weights=[0.9, 0.1])


@DATASETS.register("Moons (label noise)")
def moons_label_noise(n, noise, rng):
    """The classic Moons (fixed geometry, positional noise 0.08) with 10% + noise/2
    of labels flipped at random (10..35%). Test labels are flipped too, so the
    best possible accuracy is about 1 - flip rate (0.875 at noise 0.05); do not
    compare its raw accuracy across noise levels."""
    X, y = moons(n, 0.08, rng)
    k = int(round(n * (0.1 + 0.5 * min(max(noise, 0.0), 0.5))))
    flip = rng.choice(n, k, replace=False)
    y = y.copy()
    y[flip] = 1 - y[flip]
    if y.min() == y.max():  # keep both classes even for absurdly small n
        y[0] = 1 - y[0]
    return _out(X, y)


# Datasets that do not return exactly n points (everything else does).
INEXACT_N = {
    "XOR gate (4 points)": "fixed 4-point toy, ignores n",
    "Spiral (2 arms)": "rounds n down to a multiple of the arm count; n < arms raises",
    "Spiral (3 arms)": "rounds n down to a multiple of the arm count; n < arms raises",
    "Spiral (4 arms)": "rounds n down to a multiple of the arm count; n < arms raises",
    "Spiral (5 arms)": "rounds n down to a multiple of the arm count; n < arms raises",
}


# =========================================================================== suites
# A suite is a named list of datasets to score one recipe on. Benchmark
# scores (mean / min test accuracy over a suite) are only comparable for the
# same suite contents and the same noise level, so store suite_signature(name)
# next to every score, and change a published suite by adding a new name.
# With 120 test points per dataset (n=600) one score has a standard error of
# about 0.02-0.04, so score each dataset over >= 3 seeds and average per
# dataset before taking the suite mean / min.
#
#   general  recommended default: one or two per pattern family, chosen so a
#            plain 16-16 tanh MLP lands mostly in 0.75-0.95 (not at the ceiling)
#   stretch  hard for small nets (often near chance): report separately,
#            do not let these pin a pass/fail min
#   sanity   easy checks a working recipe should ace; a low score means broken
#   zoo      every dataset added after the classics; classic = the originals
#   all      every registered dataset except the fixed 4-point XOR gate

SUITES = Registry("suite", "list of dataset names")

_FIXED_TOYS = ("XOR gate (4 points)",)

SUITES.register("classic", ["XOR quadrants", "Spiral (2 arms)", "Spiral (3 arms)", "Spiral (4 arms)",
                            "Spiral (5 arms)", "Circles", "Moons", "Checkerboard", "Gaussian blobs (5)"])
SUITES.register("zoo", [
    "Rings (3)", "Bullseye (5 rings)", "Pinwheel (4)", "Sectors (8 wedges)", "Sine boundary",
    "Stripes (diagonal)", "Egg crate (3)", "Checkerboard (fine)", "Hex tiling (3)", "Hex tiling (3, coarse)",
    "Random lines (parity)", "Voronoi (6)", "Linear", "Moons (4)", "Spiral (6 arms)",
    "Spiral (2 arms, tight)", "Spiral (3 arms, uneven)", "Blobs + tiny cluster", "Ring and blobs",
    "Mandelbrot", "Yin-yang", "Smiley", "XOR blobs", "Imbalanced (90/10)", "Moons (label noise)"])
SUITES.register("general", [  # v2: no near-ceiling entries, no under-sampled ones
    "Bullseye (5 rings)", "Sectors (8 wedges)", "Egg crate (3)", "Checkerboard", "Hex tiling (3, coarse)",
    "Random lines (parity)", "Voronoi (6)", "Spiral (3 arms)", "Blobs + tiny cluster", "Mandelbrot",
    "Yin-yang", "Smiley"])
SUITES.register("stretch", [
    "Spiral (4 arms)", "Spiral (5 arms)", "Spiral (6 arms)", "Spiral (2 arms, tight)",
    "Checkerboard (fine)", "Hex tiling (3)", "Stripes (diagonal)"])
SUITES.register("sanity", [
    "Linear", "Circles", "Moons", "Rings (3)", "Moons (4)", "Pinwheel (4)", "XOR blobs", "Ring and blobs",
    "Imbalanced (90/10)"])
SUITES.register("all", lambda: [d for d in DATASETS if d not in _FIXED_TOYS])  # resolved at lookup


def suite_datasets(name):
    """Dataset names in suite `name`; callable entries (like "all") are
    resolved now, so datasets registered later (my_parts.py) are included."""
    entry = SUITES.get(name)
    return list(entry() if callable(entry) else entry)


def suite_signature(name):
    """'name#<8 hex>': a short hash of the suite's current dataset list. Store it
    with every benchmark score; scores with different signatures don't compare."""
    digest = hashlib.sha1("\n".join(suite_datasets(name)).encode()).hexdigest()[:8]
    return f"{name}#{digest}"


# =========================================================================== families
# A family is one pattern with a size knob, so a recipe's cost can be measured
# against problem size (rsi complexity). Instances are named "family[size]"
# ("spiral[7]", "checkerboard[5]"): DATASETS.get / Config(dataset=...) resolve
# them on demand (cached in DATASETS.resolved), without adding them to the
# dataset list, the UI dropdown or suite "all".
#
# Noise: apart from spiral (which reuses spiral(k) and its angular noise), a
# family's jitter std is noise x its feature size (cell, stripe or band width;
# blobs scale their whole spread, 0.12 + noise, with the blob spacing), so a
# given noise blurs the same share of each boundary at every size and only the
# structure grows. Every family except spiral returns
# exactly n points, every class present once n >= its class count.

class Family:
    """A dataset family: make(size) -> dataset fn(n, noise, rng)."""

    def __init__(self, name, build, *, sizes, size_label, symbol, description, min_size, max_size,
                 n_classes=lambda size: 2, exact_n=True):
        self.name, self._build, self._n_classes = name, build, n_classes
        self.sizes = tuple(sizes)          # default sizes to sweep
        self.size_label = size_label       # what size counts, e.g. "arms"
        self.symbol = symbol               # letter for formulas, e.g. "k"
        self.description = description
        self.min_size, self.max_size = min_size, max_size
        self.exact_n = exact_n             # False: may return fewer than n points

    def check_size(self, size):
        if isinstance(size, bool) or int(size) != size or not self.min_size <= size <= self.max_size:
            raise ValueError(f"{self.name} size must be an integer in {self.min_size}..{self.max_size}, got {size!r}")
        return int(size)

    def make(self, size):
        size = self.check_size(size)
        fn = self._build(size)
        fn.family, fn.size = self.name, size
        fn.__doc__ = f"{self.description} ({self.symbol}={size})"
        return fn

    def dataset_name(self, size):
        return f"{self.name}[{self.check_size(size)}]"

    def n_classes(self, size):
        return self._n_classes(self.check_size(size))

    def to_dict(self):
        return {"name": self.name, "sizes": list(self.sizes), "size_label": self.size_label, "symbol": self.symbol,
                "description": self.description, "min_size": self.min_size, "max_size": self.max_size,
                "exact_n": self.exact_n}

    def __repr__(self):
        return f"Family({self.name}[{self.symbol}], {self.min_size}..{self.max_size})"


FAMILIES = Registry("family", "Family: make(size) -> fn(n, noise, rng)")


def _checkerboard_c(c):
    cell = 2.4 / c

    def make(n, noise, rng):
        lab = lambda X: (np.floor((X[:, 0] + 1.2) / cell) + np.floor((X[:, 1] + 1.2) / cell)).astype(int) % 2
        return _stratified(n, 2, lab, rng, jitter=noise * cell)
    return make


def _rings_k(k):
    band = 1.15 / (k + 1)

    def make(n, noise, rng):
        return _stratified(n, 2, lambda X: np.floor(_polar(X)[0] / band).astype(int) % 2, rng, _disk,
                           weights="sqrt_area", jitter=noise * band)
    return make


def _stripes_k(k):
    width = 2.4 / k

    def make(n, noise, rng):
        return _stratified(n, 2, lambda X: np.floor((X[:, 0] + 1.2) / width).astype(int) % 2, rng,
                           jitter=noise * width)
    return make


def _blobs_k(k):
    ang = np.arange(k) * 2 * math.pi / k
    centers = np.stack([np.cos(ang), np.sin(ang)], 1) * 0.8
    spacing = 2 * 0.8 * math.sin(math.pi / k)  # neighbour distance; 0.94 for k=5

    def make(n, noise, rng):
        return _gauss_mix(n, centers, (0.12 + noise) * spacing / 0.94, rng)
    return make


FAMILIES.register("spiral", Family(
    "spiral", spiral, sizes=range(2, 9), size_label="arms", symbol="k", min_size=2, max_size=32,
    n_classes=lambda k: k, exact_n=False,
    description="k interleaved spiral arms, one class per arm (spiral[5] == 'Spiral (5 arms)')"))
FAMILIES.register("checkerboard", Family(
    "checkerboard", _checkerboard_c, sizes=range(2, 9), size_label="cells per side", symbol="c",
    min_size=2, max_size=32, description="c x c checkerboard on [-1.2, 1.2]^2, 2 classes"))
FAMILIES.register("rings", Family(
    "rings", _rings_k, sizes=range(1, 9), size_label="ring boundaries", symbol="k", min_size=1, max_size=32,
    description="disk cut into k+1 equal-width concentric bands (k circular boundaries), alternating 2 classes"))
FAMILIES.register("stripes", Family(
    "stripes", _stripes_k, sizes=range(2, 13), size_label="stripes", symbol="k", min_size=2, max_size=48,
    description="k equal vertical stripes (k-1 boundaries), alternating 2 classes"))
FAMILIES.register("blobs", Family(
    "blobs", _blobs_k, sizes=range(2, 9), size_label="blobs", symbol="k", min_size=2, max_size=32,
    n_classes=lambda k: k,
    description="k Gaussian blobs on a circle, one class each; spread scales with the spacing"))



def _parity_k(k):
    """k random lines (directions spread over 180 degrees, off-centre); class = parity of the
    number of lines a point lies above: a system of linear equations mod 2."""
    def make(n, noise, rng, layout_rng=None):
        lr = rng if layout_rng is None else layout_rng
        th = lr.uniform(0, math.pi) + np.arange(k) * math.pi / k + lr.uniform(-0.15, 0.15, k)
        d = lr.choice([-1.0, 1.0], k) * lr.uniform(0.05, 0.75, k)
        normals = np.stack([np.cos(th), np.sin(th)], 1)
        return _stratified(n, 2, lambda X: ((X @ normals.T) > d).sum(1) % 2, rng, jitter=noise * 0.45)
    return make


# appended: the one family whose exact algorithmic complexity is known (linear equations mod 2,
# the Maltsev case; Lagerkvist 2026 solves Maltsev CSPs in O(n^2 m)), so the empirical exponent a
# net needs can be read against a real big-O
FAMILIES.register("parity", Family(
    "parity", _parity_k, sizes=range(1, 9), size_label="lines", symbol="k", min_size=1, max_size=32,
    description="k random lines, class = parity of how many a point lies above (linear equations mod 2)"))

_FAMILY_NAME = re.compile(r"(.+)\[([1-9][0-9]*)\]\Z")


def parse_family_name(name):
    """'spiral[7]' -> ('spiral', 7); None if name is not family[size] of a known family
    (canonical form only: no spaces, no leading zeros)."""
    m = _FAMILY_NAME.match(name) if isinstance(name, str) else None
    if m is None or m.group(1) not in FAMILIES:
        return None
    return m.group(1), int(m.group(2))


def family_datasets(family, sizes=None):
    """Dataset names for a family: family_datasets('spiral', [2, 3]) -> ['spiral[2]', 'spiral[3]']."""
    fam = FAMILIES.get(family)
    return [fam.dataset_name(k) for k in (fam.sizes if sizes is None else sizes)]


def _resolve_family(name):
    parsed = parse_family_name(name)
    if parsed is None:
        return None
    fam = FAMILIES[parsed[0]]
    if not fam.min_size <= parsed[1] <= fam.max_size:
        return None
    return fam.make(parsed[1])


DATASETS.add_resolver(_resolve_family)
