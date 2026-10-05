# Test oracle: verbatim copy of nncore/session.py as it was before the WP-core rewrite (class renamed
# LegacySession, imports made absolute). Do not edit. Part of nn-playground, AGPL-3.0; see COMMERCIAL.md.
"""
Session: one live experiment. Owns the data, model, optimizer and step count,
and knows nothing about windows, plots or timers.

    s = Session(Config(dataset="Moons", layers="4:sin"))
    s.train(500)
    print(s.evaluate())
    s.configure(lr=0.01)          # rebuilds only what the change requires
    logits, hidden = s.predict(points, collect=True)

Every part is looked up by name in its registry, so dropping in a new part is
a register() call plus a Config value; this file does not need to change.
"""
import copy
import math

import numpy as np
import torch

from nncore.config import DATA_KEYS, MODEL_KEYS, OPT_KEYS, SAMPLER_KEYS, Config
from nncore.datasets import DATASETS, SPLITTERS
from nncore.features import featurize
from nncore.initializers import INITIALIZERS
from nncore.losses import LOSSES
from nncore.metrics import METRICS
from nncore.models import MODELS
from nncore.optimizers import OPTIMIZERS, SCHEDULES
from nncore.samplers import SAMPLERS
from nncore.training import TRAIN_STEPS


class LegacySession:
    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        self.new_data()

    # ---------------------------------------------------------------- configuration
    def configure(self, **changes):
        """Apply Config changes and rebuild the minimum needed. Returns the level rebuilt
        ('data', 'model', 'optimizer', 'sampler' or None). On error the old state stays."""
        unknown = set(changes) - set(self.cfg.to_dict())
        if unknown:
            raise ValueError(f"unknown config keys: {', '.join(sorted(unknown))}")
        keys = {k for k, v in changes.items() if getattr(self.cfg, k) != v}
        if not keys:
            return None
        old = copy.copy(self.cfg)
        for k in keys:
            setattr(self.cfg, k, changes[k])
        try:
            if keys & DATA_KEYS:
                self.new_data()
                return "data"
            if keys & MODEL_KEYS:
                self.reset_model()
                return "model"
            if keys & OPT_KEYS:
                self.reset_optimizer()
                return "optimizer"
            if keys & SAMPLER_KEYS:
                self.reset_sampler()
                return "sampler"
            return None
        except Exception:
            self.cfg = old
            raise

    # ---------------------------------------------------------------- building
    def new_data(self, seed=None):
        c = self.cfg
        if seed is not None:
            c.seed = seed
        rng = np.random.default_rng(c.seed)
        X, y = DATASETS.get(c.dataset)(c.n_points, c.noise, rng)
        tr, te = SPLITTERS.get(c.splitter)(len(X), c.test_frac, rng)
        self.X_all, self.y_all = X, y
        self.train_idx, self.test_idx = tr, te
        self.Xtr_raw, self.ytr = torch.from_numpy(X[tr]), torch.from_numpy(y[tr])
        self.Xte_raw, self.yte = torch.from_numpy(X[te]), torch.from_numpy(y[te])
        self.n_classes = int(y.max()) + 1
        self.reset_model()

    def reset_model(self):
        c = self.cfg
        feats = list(c.features) or ["x", "y"]
        torch.manual_seed(c.seed)
        model = MODELS.get(c.model)(len(feats), self.n_classes, c)  # build before assigning
        INITIALIZERS.get(c.init)(model)
        self.features, self.model = feats, model
        self.Xtr = featurize(self.Xtr_raw, feats)
        self.Xte = featurize(self.Xte_raw, feats) if len(self.Xte_raw) else None
        self.step_count = 0
        self.reset_optimizer()

    def reset_optimizer(self):
        c = self.cfg
        opt = OPTIMIZERS.get(c.optimizer)(self.model.parameters(), float(c.lr),
                                          float(c.weight_decay))
        self.opt, self.scheduler = opt, SCHEDULES.get(c.schedule)(opt)
        self.reset_sampler()

    def reset_sampler(self):
        n, bs = len(self.Xtr), self.cfg.batch_size
        bs = None if bs is None else max(1, min(int(bs), n))
        self.next_batch = SAMPLERS.get(self.cfg.sampler)(n, bs)

    # ---------------------------------------------------------------- running
    def train(self, steps=1):
        step_fn, loss_fn = TRAIN_STEPS.get(self.cfg.train_step), LOSSES.get(self.cfg.loss)
        loss = math.nan
        for _ in range(steps):
            idx = self.next_batch()
            xb, yb = (self.Xtr, self.ytr) if idx is None else (self.Xtr[idx], self.ytr[idx])
            loss = step_fn(self.model, self.opt, loss_fn, xb, yb)
            if self.scheduler is not None:
                self.scheduler.step()
            self.step_count += 1
        return loss

    @torch.no_grad()
    def evaluate(self):
        """{'train': {'loss': .., 'acc': ..}, 'test': {...}}; test values are NaN without a test set."""
        loss_fn = LOSSES.get(self.cfg.loss)
        out = {}
        for split, X, y in (("train", self.Xtr, self.ytr), ("test", self.Xte, self.yte)):
            if X is None:
                out[split] = {"loss": math.nan, **{m: math.nan for m in METRICS}}
                continue
            logits = self.model(X)
            out[split] = {"loss": loss_fn(logits, y).item(),
                          **{m: fn(logits, y) for m, fn in METRICS.items()}}
        return out

    @torch.no_grad()
    def predict(self, points, collect=False):
        """points: raw (N, 2) coordinates. Returns logits, or (logits, hidden list)."""
        return self.model(featurize(points, self.features), collect=collect)

    # ---------------------------------------------------------------- introspection
    @property
    def hidden_sizes(self):
        return list(getattr(self.model, "hidden_sizes", []))

    @property
    def n_params(self):
        return sum(p.numel() for p in self.model.parameters())

    @property
    def lr(self):
        return self.opt.param_groups[0]["lr"]

    def describe(self):
        fn = getattr(self.model, "describe", None)
        return fn() if fn else type(self.model).__name__
