"""
schema: what every Config field means, which models read it, and side builds
(param counts, hidden sizes) that are safe to call mid-run.

Every function here leaves the torch RNG state as it found it and keeps
stdout clean (model builds run under fork_rng with stdout sent to stderr).

    fields_read("custom nn")      # {'width', 'classes', 'expand', ...} found by watching a build
    inert_fields(cfg)             # model fields cfg.model ignores
    build_info(cfg)               # {'describe', 'n_params', 'hidden_sizes', 'n_classes', 'n_in'}
    describe()                    # the lever table + registries, as JSON-ready dicts

Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md.
"""
import contextlib
import dataclasses
import sys

import torch

from .config import DATA_KEYS, MODEL_KEYS, Config
from .configio import (FIELD_BOUNDS, FIELD_HELP, FIELD_REGISTRY, FIELDS, FLOAT_FIELDS, INT_FIELDS, LEVEL,
                       PART_REGISTRIES, SPECIAL_VALUES, UI_LIMITS, ascii_name, check_config, normalise)
from .registry import ensure_plugins, loaded_plugins

GROUPS = {**{k: "data" for k in DATA_KEYS}, **{k: "model" for k in MODEL_KEYS},
          "loss": "optimisation", "optimizer": "optimisation", "lr": "optimisation", "weight_decay": "optimisation",
          "schedule": "optimisation", "train_step": "optimisation", "sampler": "batching", "batch_size": "batching",
          "extra": "extra"}
ALWAYS_READ = {"model", "features", "init", "extra"}  # used by Session itself, whatever the model
SUGGESTED = {"width": [4, 8, 16, 32, 64], "classes": [2, 3, 4, 5, 8], "fourier_freq": [1.0, 2.0, 3.0, 5.0, 8.0],
             "lr": [0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3], "weight_decay": [0, 1e-05, 0.0001, 0.001],
             "batch_size": [None, 8, 16, 32, 64, 128], "n_points": [200, 600, 2000], "noise": [0.0, 0.05, 0.1, 0.2]}
LOG_SCALE = {"lr", "fourier_freq"}
ACTIVE_WHEN = {"fourier_freq": {"expand": ["fourier (sin)"]}}
LAYERS_GRAMMAR = "'' | W[:ACT](,W[:ACT])*  W in 1..512"
COST_MS_PER_STEP = {"mlp": 1.2, "custom nn": 2.4}  # one core, default sizes; doctor --calibrate refines


@contextlib.contextmanager
def quiet_build():
    """Side builds: forked torch RNG, stdout -> stderr (parts may print)."""
    with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(sys.stderr):
        yield


class _Recorder:
    """Mixin for a Config subclass that records which fields are read."""
    def __getattribute__(self, name):
        if name in FIELDS:
            object.__getattribute__(self, "_seen").add(name)
        return object.__getattribute__(self, name)


def _recording(cfg):
    cls = type("RecordingConfig", (_Recorder, Config), {})
    rec = cls(**{k: getattr(cfg, k) for k in FIELDS})
    object.__setattr__(rec, "_seen", set())
    return rec


_READ_CACHE = {}


def fields_read(model, cfg=None, n_in=2, n_out=2):
    """Config fields the model's build reads, found by building it once with a recording
    Config (RNG- and stdout-safe). A build that fails still reports what it read so far."""
    factory = PART_REGISTRIES["model"].get(model)
    key = (model, id(factory), n_in, n_out)
    if cfg is None and key in _READ_CACHE:
        return set(_READ_CACHE[key])
    base = dataclasses.replace(cfg if cfg is not None else Config(), model=model)
    rec = _recording(base)
    with quiet_build():
        try:
            factory(n_in, n_out, rec)
        except Exception:
            pass
    seen = set(object.__getattribute__(rec, "_seen")) - {"model"}
    if cfg is None:
        _READ_CACHE[key] = frozenset(seen)
    return seen


def inert_fields(cfg):
    """MODEL_KEYS that cfg.model does not read (model, features, init and extra always count)."""
    return set(MODEL_KEYS) - fields_read(cfg.model, cfg) - ALWAYS_READ


def build_info(cfg):
    """{'describe', 'n_params', 'hidden_sizes', 'n_classes', 'n_in'} for cfg, built exactly as
    Session builds it (data from the numpy rng, model under a forked torch RNG)."""
    from .session import Session
    cfg = normalise(cfg)
    check_config(cfg)
    with quiet_build():
        s = Session.__new__(Session)
        data = s._build_data(cfg)
        m = s._build_model(cfg, data)
    model = m["model"]
    fn = getattr(model, "describe", None)
    return {"describe": fn() if fn else type(model).__name__,
            "n_params": sum(p.numel() for p in model.parameters()),
            "hidden_sizes": [int(h) for h in getattr(model, "hidden_sizes", [])],
            "n_classes": data["n_classes"], "n_in": len(m["features"])}


def count_params(cfg):
    return build_info(cfg)["n_params"]


def _models_reading():
    ensure_plugins()
    return {m: sorted(fields_read(m)) for m in PART_REGISTRIES["model"]}


def _ui(f):
    lim = UI_LIMITS.get(f)
    if isinstance(lim, tuple):
        return {"min": lim[0], "max": lim[1]}
    return dict(lim) if lim else None


def field_specs(model=None):
    """One lever entry per Config field (see describe()). With `model`, model fields that
    model does not read are left out."""
    defaults, reads = Config(), _models_reading()
    out = []
    for f in FIELDS:
        d = getattr(defaults, f)
        spec = {"name": f, "group": GROUPS[f], "default": list(d) if isinstance(d, tuple) else d,
                "rebuild": LEVEL[f], "search": f not in DATA_KEYS, "help": FIELD_HELP.get(f, "")}
        if f in FIELD_REGISTRY:
            reg = PART_REGISTRIES[FIELD_REGISTRY[f]]
            names = list(reg)
            spec.update(kind="multi_choice" if f == "features" else "choice", registry=FIELD_REGISTRY[f],
                        choices=names)
            aliases = {ascii_name(n): n for n in names if ascii_name(n) != n}
            if aliases:
                spec["ascii_aliases"] = aliases
            if f in SPECIAL_VALUES:
                spec["special"] = sorted(SPECIAL_VALUES[f])
            if f == "features":
                spec.update(min_items=1, grammar="NAME(,NAME)*  in the order given (';' also separates)",
                            example="x,y,sin x")
        elif f in INT_FIELDS:
            spec["kind"] = "int"
        elif f in FLOAT_FIELDS:
            spec["kind"] = "float"
        elif f == "batch_size":
            spec.update(kind="int_or_null", null_means="full batch")
        elif f == "layers":
            spec.update(kind="layers", grammar=LAYERS_GRAMMAR)
        else:
            spec["kind"] = "object"
        if f in FIELD_BOUNDS:
            spec["bounds"] = list(FIELD_BOUNDS[f])
        if _ui(f):
            spec["ui"] = _ui(f)
        if f in LOG_SCALE:
            spec["scale"] = "log"
        if f in SUGGESTED:
            spec["suggested"] = SUGGESTED[f]
        if f in ACTIVE_WHEN:
            spec["active_when"] = ACTIVE_WHEN[f]
        if f in MODEL_KEYS and f not in ALWAYS_READ:
            spec["read_by"] = [m for m, r in reads.items() if f in r]
            if model is not None and model not in spec["read_by"]:
                continue
        else:
            spec["read_by"] = "*"
        out.append(spec)
    return out


def _doc(reg, name):
    if name in reg.docs:
        return reg.docs[name]
    return ""


def catalog():
    """{'registries': {key: {kind, signature, names, docs}}, 'suites': {...}} (suites if present)."""
    ensure_plugins()
    regs = {k: {"kind": r.kind, "signature": r.signature, "names": list(r),
                "docs": {n: _doc(r, n) for n in r if _doc(r, n)}} for k, r in PART_REGISTRIES.items()}
    from . import datasets, metrics
    regs["metrics"] = {"kind": "metric", "signature": metrics.METRICS.signature, "names": list(metrics.METRICS),
                       "docs": {}}
    extra = getattr(metrics, "EXTRA_METRICS", None)
    if extra is not None:
        regs["extra_metrics"] = {"kind": "metric", "signature": extra.signature, "names": list(extra),
                                 "docs": {n: _doc(extra, n) for n in extra if _doc(extra, n)}}
    out = {"registries": regs}
    if getattr(datasets, "SUITES", None) is not None:
        out["suites"] = {s: datasets.suite_datasets(s) for s in datasets.SUITES}
    fams = getattr(datasets, "FAMILIES", None)
    if fams is not None:
        out["families"] = {k: {a: getattr(v, a) for a in ("sizes", "size_label", "description") if hasattr(v, a)}
                           for k, v in fams.items()}
    return out


def describe(model=None, section=None):
    """The describe document: levers, models (fields each reads), registries, suites,
    metrics, objectives, cost model. section picks one key (levers|models|parts|suites|
    metrics|objectives|env)."""
    from . import metrics
    from .run import code_fingerprint, environment
    cat = catalog()
    extra = getattr(metrics, "EXTRA_METRICS", None)
    doc = {"code_fp": code_fingerprint(), "parts": loaded_plugins(),
           "levers": field_specs(model),
           "models": {m: {"reads": r} for m, r in _models_reading().items() if model in (None, m)},
           "registries": cat["registries"], "suites": cat.get("suites", {}),
           "metrics": list(metrics.METRICS), "extra_metrics": list(extra) if extra is not None else [],
           "objectives": {"metrics": ["test_acc", "train_acc", "fresh_acc", "test_loss", "train_loss"],
                          "seed_agg": ["mean", "median", "min", "cvar50"], "reduce": ["mean", "min"]},
           "cost_model": {"ms_per_step": dict(COST_MS_PER_STEP), "source": "built-in estimate"}}
    if "families" in cat:
        doc["families"] = cat["families"]
    if section is None:
        return doc
    if section == "env":
        return {"env": environment()}
    pick = "registries" if section == "parts" else section
    if pick not in doc:
        raise ValueError(f"unknown section '{section}' (have: levers, models, parts, suites, metrics, "
                         f"objectives, env)")
    return {pick: doc[pick], "code_fp": doc["code_fp"], "parts": doc["parts"]}
