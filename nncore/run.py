"""
run: one JSON-safe training cell (config x dataset x seed), the unit that the
console caches and farms out to worker processes.

    from nncore.run import RunSpec, run_unit
    cell = run_unit(RunSpec(config=config_to_dict(Config(model="mlp")), steps=3000, fresh_points=2000))
    cell["status"], cell["test"]["acc"], cell["fresh"]["acc"], cell["model"]["n_params"]

run_unit makes the same calls as headless.run (Session(cfg), train() in chunks,
evaluate()), and chunking is exact, so its numbers equal an interactive run.

Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md.
"""
import contextlib
import hashlib
import math
import platform
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .configio import canonical_json, config_from_dict, config_key, config_to_dict, json_safe
from .registry import REPO_ROOT, loaded_plugins
from .session import Session

CURVE_ROWS = 50
CURVE_COLS = ("train_loss", "train_acc", "test_loss", "test_acc")


@dataclass(frozen=True)
class EarlyStop:
    at_step: int
    min_test_acc: float


@dataclass(frozen=True)
class RunSpec:
    config: dict                     # config_to_dict form, seed + dataset already applied
    steps: int
    eval_every: int = 0              # curve granularity; never changes the numbers
    max_seconds: float | None = None
    early_stop: EarlyStop | None = None
    stop_on_nonfinite: bool = True
    fresh_points: int = 0
    extra_metrics: tuple = field(default=())

    def __post_init__(self):
        object.__setattr__(self, "extra_metrics", tuple(self.extra_metrics))
        if isinstance(self.early_stop, dict):
            object.__setattr__(self, "early_stop", EarlyStop(**self.early_stop))

    def key(self, code_fp):
        """'k_' + 16 hex: the cache key of this cell under code fingerprint code_fp."""
        try:
            cfg = config_to_dict(config_from_dict(self.config)[0])
        except Exception:
            cfg = self.config
        d = {"config": cfg, "steps": int(self.steps), "eval_every": int(self.eval_every),
             "early_stop": asdict(self.early_stop) if self.early_stop else None,
             "fresh_points": int(self.fresh_points), "extra_metrics": list(self.extra_metrics), "code_fp": code_fp}
        if not self.stop_on_nonfinite:  # only non-default values join the key
            d["stop_on_nonfinite"] = False
        return "k_" + hashlib.sha256(canonical_json(d).encode()).hexdigest()[:16]


@contextlib.contextmanager
def _preserve_rng():
    """Leave the caller's torch and numpy global RNG states as they were."""
    np_state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        try:
            yield
        finally:
            np.random.set_state(np_state)


def _subsample(curve):
    n = len(curve["step"])
    if n <= CURVE_ROWS:
        return curve
    idx = sorted({round(i * (n - 1) / (CURVE_ROWS - 1)) for i in range(CURVE_ROWS)})
    return {k: [v[i] for i in idx] for k, v in curve.items()}


def _nulls(ev):
    return {k: None for k in ev}


def run_unit(spec):
    """Train one cell and return a CellResult dict (JSON-safe). Never raises for build,
    train or eval failures (status 'error'); restores the caller's RNG state; anything
    parts print goes to stderr."""
    t0 = time.perf_counter()
    cell = {"key": None, "config_key": None, "dataset": (spec.config or {}).get("dataset"),
            "seed": (spec.config or {}).get("seed"), "steps": int(spec.steps), "steps_done": 0, "status": "ok",
            "diverged_at": None, "train": None, "test": None, "fresh": None, "model": None, "seconds": 0.0,
            "cached": False, "curve": None, "error": None}
    curve = {"step": [], **{c: [] for c in CURVE_COLS}} if spec.eval_every else None
    phase = "build"
    with _preserve_rng(), contextlib.redirect_stdout(sys.stderr):
        try:
            cfg, _ = config_from_dict(spec.config)
            cell.update(config_key=config_key(cfg), dataset=cfg.dataset, seed=cfg.seed)
            s = Session(cfg)
            cell["model"] = {"describe": s.describe(), "n_params": s.n_params, "hidden_sizes": s.hidden_sizes,
                             "n_classes": s.n_classes}
            phase = "train"
            es, done = spec.early_stop, 0
            chunk = max(1, int(spec.eval_every) or 250)
            while done < spec.steps:
                n = min(chunk, spec.steps - done)
                if es and done < es.at_step < done + n:
                    n = es.at_step - done
                s.train(n, stop_on_nonfinite=spec.stop_on_nonfinite)
                done = s.step_count
                if s.diverged_at is not None and spec.stop_on_nonfinite:
                    cell.update(status="diverged", diverged_at=s.diverged_at)
                    break
                ev = None
                if spec.eval_every:
                    phase = "eval"
                    ev = s.evaluate()
                    curve["step"].append(done)
                    for c in CURVE_COLS:
                        split, metric = c.split("_")
                        curve[c].append(ev[split][metric])
                    phase = "train"
                if es and done == es.at_step:
                    ev = ev or s.evaluate()
                    acc = ev["test"]["acc"]
                    if acc is None or not math.isfinite(acc):
                        acc = ev["train"]["acc"]  # no test split
                    if acc < es.min_test_acc:
                        cell["status"] = "early_stopped"
                        break
                if spec.max_seconds and time.perf_counter() - t0 > spec.max_seconds and done < spec.steps:
                    cell["status"] = "timeout"
                    break
            cell["steps_done"] = s.step_count
            if s.diverged_at is not None:
                cell["diverged_at"] = s.diverged_at
            phase = "eval"
            ev = s.evaluate(extra_metrics=spec.extra_metrics)
            if cell["status"] == "diverged":
                cell["train"], cell["test"] = _nulls(ev["train"]), _nulls(ev["test"])
            else:
                cell["train"], cell["test"] = ev["train"], ev["test"]
                if spec.fresh_points:
                    cell["fresh"] = s.evaluate_fresh(spec.fresh_points, extra_metrics=spec.extra_metrics)
        except Exception as e:
            cell.update(status="error", train=None, test=None, fresh=None,
                        error={"type": type(e).__name__, "message": str(e), "phase": phase,
                               "traceback": "".join(traceback.format_exception(e)[-6:])})
    if curve is not None and curve["step"]:
        cell["curve"] = _subsample(curve)
    try:
        cell["key"] = spec.key(code_fingerprint())
    except Exception:
        pass
    cell["seconds"] = round(time.perf_counter() - t0, 4)
    return json_safe(cell)


# ---------------------------------------------------------------- aggregation
def _stats(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "median": None, "min": None, "max": None, "std": None, "n": 0}
    return {"mean": statistics.fmean(vals), "median": statistics.median(vals), "min": min(vals), "max": max(vals),
            "std": statistics.pstdev(vals), "n": len(vals)}


def summarize(cells):
    """TrialRecord.summary over CellResults: counts per status, then <split>_<metric> stats
    over ok and early_stopped cells, per-dataset test (and fresh) acc, max n_params, seconds."""
    cells = [c for c in cells if c is not None]
    count = lambda st: sum(c.get("status") == st for c in cells)  # noqa: E731
    out = {"n_cells": len(cells), "n_ok": count("ok"), "n_diverged": count("diverged"), "n_error": count("error"),
           "n_timeout": count("timeout"), "n_early_stopped": count("early_stopped")}
    good = [c for c in cells if c.get("status") in ("ok", "early_stopped")]
    keys = []
    for c in good:
        for split in ("train", "test", "fresh"):
            for m in (c.get(split) or {}):
                if f"{split}_{m}" not in keys:
                    keys.append(f"{split}_{m}")
    for k in keys:
        split, m = k.split("_", 1)
        out[k] = _stats([(c.get(split) or {}).get(m) for c in good])
    per = {}
    for c in good:
        per.setdefault(c.get("dataset"), []).append(c)
    out["per_dataset"] = {}
    for d, cs in per.items():
        row = {"n": len(cs)}
        for split in ("test", "fresh", "train"):
            v = [(c.get(split) or {}).get("acc") for c in cs]
            v = [x for x in v if x is not None]
            if v:
                row[f"{split}_acc_mean"], row[f"{split}_acc_min"] = statistics.fmean(v), min(v)
        out["per_dataset"][d] = row
    params = [c["model"]["n_params"] for c in cells if c.get("model")]
    out["n_params"] = max(params) if params else None
    out["seconds"] = round(sum(c.get("seconds") or 0.0 for c in cells), 4)
    return out


# ---------------------------------------------------------------- provenance
_FP_CACHE = {}


def _part_file(name):
    mod = sys.modules.get(name)
    f = getattr(mod, "__file__", None)
    if f is None:
        import importlib.util
        try:
            spec = importlib.util.find_spec(name)
            f = spec.origin if spec else None
        except (ImportError, ValueError):
            f = None
    return Path(f) if f else None


def code_fingerprint(parts=None):
    """'c_' + 12 hex over the nncore sources, the loaded part files, torch/numpy versions
    and the OS / machine. Results are only comparable under the same fingerprint."""
    parts = list(loaded_plugins() if parts is None else parts)
    files = sorted((REPO_ROOT / "nncore").rglob("*.py"))
    stamp = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files)
    key = (tuple(parts), stamp)
    if key in _FP_CACHE:
        return _FP_CACHE[key]
    entries = []
    for p in files:
        entries.append((p.relative_to(REPO_ROOT).as_posix(), p.read_bytes()))
    for name in parts:
        f = _part_file(name) if not name.endswith(".py") else Path(name).resolve()
        if f and f.exists():
            try:
                rel = f.resolve().relative_to(REPO_ROOT).as_posix()
            except ValueError:
                rel = f"<ext>/{f.name}"
            entries.append((rel, f.read_bytes()))
        else:
            entries.append((f"<missing>/{name}", b""))
    h = hashlib.sha256()
    for rel, data in sorted(entries):
        h.update(rel.encode() + b"\0" + hashlib.sha256(data).digest())
    h.update(f"torch={torch.__version__};numpy={np.__version__};{platform.system()};{platform.machine()}".encode())
    _FP_CACHE[key] = fp = "c_" + h.hexdigest()[:12]
    return fp


def environment():
    """Versions, platform, threads, code fingerprint and loaded parts, for provenance."""
    from . import __version__
    return {"nncore": __version__, "torch": torch.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "platform": platform.platform(), "system": platform.system(),
            "machine": platform.machine(), "threads": torch.get_num_threads(), "code_fp": code_fingerprint(),
            "parts": loaded_plugins()}
