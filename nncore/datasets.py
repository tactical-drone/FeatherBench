"""
Compartment: DATASETS  (2-D classification problems)

Contract:  fn(n, noise, rng) -> (X float32 (n, 2), y int64 (n,))
  n      requested point count (a dataset may ignore it, e.g. the XOR gate)
  noise  0..0.5, dataset-specific meaning
  rng    numpy Generator; use it for all randomness so seeds reproduce
Keep points roughly inside [-1.2, 1.2]; the number of classes is max(y) + 1.

Also here: SPLITTERS, which turn a dataset into train / test index sets.
Contract:  fn(n, test_frac, rng) -> (train_idx, test_idx)
"""
import math

import numpy as np

from .registry import Registry

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


# --------------------------------------------------------------------------- splitters

@SPLITTERS.register("random")
def random_split(n, test_frac, rng):
    """Shuffled holdout. Tiny sets (<= 8 points) keep everything for training."""
    idx = rng.permutation(n)
    n_test = 0 if n <= 8 else int(n * test_frac)
    return idx[n_test:], idx[:n_test]


@SPLITTERS.register("none (train on all)")
def no_split(n, test_frac, rng):
    return np.arange(n), np.arange(0)
