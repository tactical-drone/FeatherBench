"""
Session: one live experiment. Owns the data, model, optimizer and step count,
and knows nothing about windows, plots or timers.

    import my_parts                 # optional: registers your parts (loaded lazily otherwise)
    s = Session(Config(model="mlp", dataset="Moons", layers="4:sin"))
    s.train(500)
    print(s.evaluate())
    s.configure(lr=0.01)          # rebuilds only what the change requires
    logits, hidden = s.predict(points, collect=True)

Every part is looked up by name in its registry, so dropping in a new part is
a register() call plus a Config value; this file does not need to change.

configure() is transactional: everything is built into locals first and
committed only on success, so a failed change leaves the session as it was.

Randomness: seed drives the data, split, init and batch order. The data uses
a numpy Generator; the init and the samplers use a torch RNG stream private
to this session (same draws as seeding the global torch RNG, which is left
alone), so other sessions, evaluate(), predict() and foreign torch draws
never shift a run.

Concurrency: one Session per process. The private RNG relies on
torch.random.fork_rng, which is not thread-safe; run parallel experiments in
spawned processes (nncore.run.run_unit / rsi), not threads. torch uses
NNCORE_THREADS threads (default 1: these nets are tiny).
"""
import contextlib
import copy
import dataclasses
import math

import numpy as np
import torch

from .config import DATA_KEYS, MODEL_KEYS, OPT_KEYS, SAMPLER_KEYS, Config
from .configio import check_config, coerce_value, config_from_dict, normalise
from .datasets import DATASETS, SPLITTERS, sample
from .features import featurize
from .initializers import INITIALIZERS
from .losses import LOSSES
from .metrics import METRICS
from .models import MODELS
from .optimizers import OPTIMIZERS, SCHEDULES
from .registry import accepted, call_accepting
from .samplers import SAMPLERS
from .training import TRAIN_STEPS

LEVELS = ("data", "model", "optimizer", "sampler")


@contextlib.contextmanager
def _eval_mode(model):
    """model.eval() under no_grad and a forked torch RNG; the training mode is restored."""
    was = model.training
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        model.eval()
        try:
            yield
        finally:
            model.train(was)


def _metrics(names):
    if not names:
        return {}
    from . import metrics
    extra = getattr(metrics, "EXTRA_METRICS", None)
    if extra is None:
        raise ValueError("extra metrics are not available in this nncore (no metrics.EXTRA_METRICS)")
    return {m: extra.get(m) for m in names}


class Session:
    def __init__(self, cfg=None):
        if isinstance(cfg, dict):
            cfg = config_from_dict(cfg)[0]
        cfg = normalise(copy.deepcopy(cfg)) if cfg is not None else Config()
        check_config(cfg)
        self.diverged_at = None
        self._commit(cfg, self._build(cfg, "data"))

    # ---------------------------------------------------------------- configuration
    def configure(self, *, skip_irrelevant=False, keep_optimizer_state=False, **changes):
        """Apply Config changes and rebuild the minimum needed. Returns the level rebuilt
        ('data', 'model', 'optimizer', 'sampler' or None). On error nothing changes.
        skip_irrelevant: model fields the current model does not read only update cfg.
        keep_optimizer_state: an lr / weight_decay-only change edits the optimizer in place
        (Adam moments and the schedule position are kept)."""
        changes = {k: coerce_value(k, v) for k, v in changes.items()}
        cand = dataclasses.replace(self.cfg, **copy.deepcopy(changes))
        keys = {k for k in changes if getattr(cand, k) != getattr(self.cfg, k)}
        if not keys:
            return None
        check_config(cand)
        if skip_irrelevant and "model" not in keys:
            from .schema import inert_fields
            keys -= inert_fields(cand)
        level = next((lv for lv, ks in zip(LEVELS, (DATA_KEYS, MODEL_KEYS, OPT_KEYS, SAMPLER_KEYS))
                      if keys & ks), None)
        if level is None:  # live keys (loss, train_step) or skipped fields: config only
            self.cfg = cand
            return None
        if level == "optimizer" and keep_optimizer_state and keys <= {"lr", "weight_decay"}:
            self._set_hparams(cand)
            self.cfg = cand
            return "optimizer"
        self._commit(cand, self._build(cand, level))
        return level

    def replace_config(self, cfg):
        """Swap in a whole new Config; the state equals Session(cfg). Transactional."""
        if isinstance(cfg, dict):
            cfg = config_from_dict(cfg)[0]
        cfg = normalise(copy.deepcopy(cfg))
        check_config(cfg)
        self._commit(cfg, self._build(cfg, "data"))

    def _set_hparams(self, cfg):
        for g in self.opt.param_groups:
            g["lr"], g["weight_decay"] = float(cfg.lr), float(cfg.weight_decay)
            if "initial_lr" in g:
                g["initial_lr"] = float(cfg.lr)
        if self.scheduler is not None and hasattr(self.scheduler, "base_lrs"):
            self.scheduler.base_lrs = [float(cfg.lr)] * len(self.scheduler.base_lrs)

    def _commit(self, cfg, state):
        self.__dict__.update(state)
        self.cfg = cfg

    # ---------------------------------------------------------------- building
    # Each builder returns a dict and touches nothing on self; _build runs them in the
    # order the legacy code did (data -> model -> optimizer -> sampler), so the RNG
    # draws are the same.
    def _build(self, cfg, level):
        st = {}
        i = LEVELS.index(level)
        if i <= 0:
            st.update(self._build_data(cfg))
        data = {k: st.get(k, getattr(self, k, None)) for k in ("n_classes", "Xtr_raw", "Xte_raw")}
        if i <= 1:
            st.update(self._build_model(cfg, data))
        model = st.get("model", getattr(self, "model", None))
        if i <= 2:
            st.update(self._build_opt(cfg, model))
        n = len(st.get("Xtr", getattr(self, "Xtr", ())))
        st.update(self._build_sampler(cfg, n, st.get("_rng_state", getattr(self, "_rng_state", None))))
        return st

    def _build_data(self, cfg):
        c = cfg
        rng = np.random.default_rng(c.seed)
        X, y = DATASETS.get(c.dataset)(c.n_points, c.noise, rng)
        if len(X) == 0:
            raise ValueError(f"dataset '{c.dataset}' produced no points for n_points={c.n_points}")
        if len(np.unique(y)) < 2:
            raise ValueError(f"dataset '{c.dataset}' produced fewer than 2 classes for n_points={c.n_points}, "
                             f"seed={c.seed}; use more points")
        tr, te = SPLITTERS.get(c.splitter)(len(X), c.test_frac, rng)
        if len(tr) == 0:
            raise ValueError(f"the train split is empty (n_points={c.n_points}, test_frac={c.test_frac})")
        return dict(X_all=X, y_all=y, train_idx=tr, test_idx=te,
                    Xtr_raw=torch.from_numpy(X[tr]), ytr=torch.from_numpy(y[tr]),
                    Xte_raw=torch.from_numpy(X[te]), yte=torch.from_numpy(y[te]),
                    n_classes=int(y.max()) + 1)

    def _build_model(self, cfg, data):
        c = cfg
        feats = list(c.features) or ["x", "y"]
        state = torch.Generator().manual_seed(c.seed).get_state()  # == torch.manual_seed(seed)
        with self._rng(state) as box:
            model = MODELS.get(c.model)(len(feats), data["n_classes"], c)
            INITIALIZERS.get(c.init)(model)
        return dict(features=feats, model=model, Xtr=featurize(data["Xtr_raw"], feats),
                    Xte=featurize(data["Xte_raw"], feats) if len(data["Xte_raw"]) else None,
                    step_count=0, diverged_at=None, _rng_state=box[0])

    def _build_opt(self, cfg, model):
        c = cfg
        opt = OPTIMIZERS.get(c.optimizer)(model.parameters(), float(c.lr), float(c.weight_decay))
        return dict(opt=opt, scheduler=call_accepting(SCHEDULES.get(c.schedule), opt, cfg=c))

    def _build_sampler(self, cfg, n, rng_state):
        bs = cfg.batch_size
        bs = None if bs is None else max(1, min(int(bs), n))
        with self._rng(rng_state) as box:
            next_batch = SAMPLERS.get(cfg.sampler)(n, bs)
        return dict(next_batch=next_batch, _rng_state=box[0])

    @contextlib.contextmanager
    def _rng(self, state=None):
        """Run a block on this session's torch RNG stream. With an explicit `state` the
        caller keeps the result (box[0]); without, self._rng_state advances."""
        box = [self._rng_state if state is None else state]
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(box[0])
            try:
                yield box
            finally:
                box[0] = torch.get_rng_state()
                if state is None:
                    self._rng_state = box[0]

    # thin transactional wrappers (the UI calls these)
    def new_data(self, seed=None):
        cfg = self.cfg if seed is None else dataclasses.replace(self.cfg, seed=coerce_value("seed", seed))
        check_config(cfg)
        self._commit(cfg, self._build(cfg, "data"))

    def reset_model(self):
        self._commit(self.cfg, self._build(self.cfg, "model"))

    def reset_optimizer(self):
        self._commit(self.cfg, self._build(self.cfg, "optimizer"))

    def reset_sampler(self):
        self._commit(self.cfg, self._build(self.cfg, "sampler"))

    # ---------------------------------------------------------------- running
    def train(self, steps=1, *, stop_on_nonfinite=False):
        """Run `steps` optimisation steps; returns the last loss (NaN for 0 steps).
        The first non-finite loss sets diverged_at; stop_on_nonfinite stops right there."""
        steps = int(steps)
        if steps < 0:
            raise ValueError(f"steps must be >= 0, got {steps}")
        c = self.cfg
        step_fn, loss_fn = TRAIN_STEPS.get(c.train_step), LOSSES.get(c.loss)
        want = accepted(step_fn, {"cfg": c, "step": 0})
        loss = math.nan
        with self._rng():
            for _ in range(steps):
                idx = self.next_batch()
                xb, yb = (self.Xtr, self.ytr) if idx is None else (self.Xtr[idx], self.ytr[idx])
                if want:
                    kw = {k: (self.step_count if k == "step" else c) for k in want}
                    loss = step_fn(self.model, self.opt, loss_fn, xb, yb, **kw)
                else:
                    loss = step_fn(self.model, self.opt, loss_fn, xb, yb)
                if self.scheduler is not None:
                    self.scheduler.step()
                self.step_count += 1
                if not math.isfinite(loss):
                    if self.diverged_at is None:
                        self.diverged_at = self.step_count
                    if stop_on_nonfinite:
                        break
        return loss

    def _score(self, X, y, loss_fn, extra):
        logits = self.model(X)
        return {"loss": loss_fn(logits, y).item(), **{m: fn(logits, y) for m, fn in METRICS.items()},
                **{m: fn(logits, y) for m, fn in extra.items()}}

    def evaluate(self, *, extra_metrics=()):
        """{'train': {'loss': .., 'acc': ..}, 'test': {...}}; test values are NaN without a test set.
        Runs in eval mode on a forked RNG, so it never changes training."""
        loss_fn, extra = LOSSES.get(self.cfg.loss), _metrics(extra_metrics)
        out = {}
        with _eval_mode(self.model):
            for split, X, y in (("train", self.Xtr, self.ytr), ("test", self.Xte, self.yte)):
                if X is None:
                    out[split] = {"loss": math.nan, **{m: math.nan for m in METRICS},
                                  **{m: math.nan for m in extra}}
                    continue
                out[split] = self._score(X, y, loss_fn, extra)
        return out

    def evaluate_fresh(self, n=5000, seed_offset=10_000, *, extra_metrics=()):
        """{'loss', metrics...} on n new points from the same dataset and noise, drawn with
        numpy rng(seed + seed_offset): a big held-out set that does not touch training.
        Random-structure datasets keep the training layout (layout_rng = rng(seed))."""
        c = self.cfg
        X, y = sample(c.dataset, int(n), c.noise, np.random.default_rng(c.seed + seed_offset),
                      layout_rng=np.random.default_rng(c.seed))
        if len(y) and int(y.max()) >= self.n_classes:
            raise ValueError(f"fresh sample has class {int(y.max())} but the model has {self.n_classes} outputs")
        loss_fn, extra = LOSSES.get(c.loss), _metrics(extra_metrics)
        with _eval_mode(self.model):
            return self._score(featurize(torch.from_numpy(X), self.features), torch.from_numpy(y), loss_fn, extra)

    def predict(self, points, collect=False):
        """points: raw (N, 2) coordinates (tensor, numpy array or list). Returns logits,
        or (logits, hidden list)."""
        pts = torch.as_tensor(points, dtype=torch.float32)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"points must have shape (N, 2), got {tuple(pts.shape)}")
        with _eval_mode(self.model):
            return self.model(featurize(pts, self.features), collect=collect)

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
