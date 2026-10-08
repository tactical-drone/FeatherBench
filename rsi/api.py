"""
api: the console's Python API, one function per command (re-exported by
rsi/__init__.py). Each returns the command's `result` dict and raises RsiError.
Warnings go to the active collect_warnings() list (the CLI puts them in the
envelope).

Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md.
"""
import contextlib
import dataclasses
import importlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

from nncore import configio, schema
from nncore.config import Config
from nncore.registry import DEFAULT_PLUGINS, REPO_ROOT, disable_lazy_plugins, load_plugins, loaded_plugins
from nncore.run import code_fingerprint, environment, summarize

from . import rundir as rd
from .errors import ERROR_HELP, EXIT_MEANING, RsiError, as_rsi_error
from .io import now_iso
from .pool import Evaluator, cell_specs, default_workers
from .store import Store, closes_stores
from .sweep import coerce_override, expand_values, grid, one_at_a_time

__version__ = "0.4.0"
_WARN_STACK = []
COMMAND_HELP = {
    "describe": "levers, models, parts, suites, metrics, objectives, commands, errors, env (start here)",
    "schema": "JSON Schema of an output or input document",
    "check": "resolve + validate a config and build it (n_params, inert fields), no training",
    "run": "train one config over seeds x datasets (cached cells)",
    "sweep": "grid (--vary), one-at-a-time (--oat) or random (--random --space) over configs",
    "evolve": "evolutionary search over the levers (resumable, background)",
    "bench": "FeatherBench, the fewest-params benchmark (fewest params that solves every pattern wins): "
             "run a recipe on a frozen suite, rank submissions (list|rank); alias: featherbench",
    "complexity": "empirical complexity exponent: smallest solving capacity per dataset-family size",
    "status": "status of a background run dir", "wait": "block until a run dir is no longer running",
    "stop": "ask a run to stop (STOP file), optionally wait", "leaderboard": "ranked entries of run dirs",
    "export": "write a UI-loadable settings file for a leaderboard entry or trial",
    "open": "open a settings file / run / trial in the playground window",
    "runs": "list run dirs, query/show/stats over stored trials", "replay": "re-run a trial bypassing the cache",
    "repl": "JSONL session on stdin/stdout (set/train/eval/map/predict/...)", "doctor": "environment checks",
    "gp": "gp check EXPR: validate a GP activation expression",
}


# ---------------------------------------------------------------- warnings
@contextlib.contextmanager
def collect_warnings():
    """with collect_warnings() as ws: ... -> the W_* dicts raised by API calls inside."""
    ws = []
    _WARN_STACK.append(ws)
    try:
        yield ws
    finally:
        _WARN_STACK.remove(ws)


def warn(code, message, **fields):
    w = {"code": code, "message": message, **fields}
    if _WARN_STACK and w not in _WARN_STACK[-1]:
        _WARN_STACK[-1].append(w)


def _warn_all(ws):
    for w in ws or ():
        if _WARN_STACK and w not in _WARN_STACK[-1]:
            _WARN_STACK[-1].append(w)


# ---------------------------------------------------------------- parts, store, evaluator
def setup_parts(parts=None):
    """Load part modules: None = DEFAULT_PLUGINS (my_parts); [] or ['none'] = nothing (no lazy load)."""
    mods = list(DEFAULT_PLUGINS) if parts is None else [p for p in parts if p]
    if "none" in mods:
        mods = [m for m in mods if m != "none"]
        disable_lazy_plugins()
    if not mods and parts is not None:
        disable_lazy_plugins()
    for m in mods:
        try:
            load_plugins(m)
        except ModuleNotFoundError as e:
            if (e.name and (m == e.name or m.startswith(e.name + "."))) or (m.endswith(".py") and not Path(m).exists()):
                raise RsiError(f"part module '{m}' not found", "E_PARTS_MISSING", value=m,
                               hint="--parts takes a module name (my_parts, agent_parts.x) or a .py path") from None
            raise _part_error(m, e) from None
        except Exception as e:
            raise _part_error(m, e) from None
    return loaded_plugins()


def _part_error(m, e):
    import traceback
    err = RsiError(f"error in {m} (your workbench): {type(e).__name__}: {e}", "E_INTERNAL", value=m)
    err.tb = "".join(traceback.format_exception(e)[-4:])
    return err


def open_store(store=None):
    return store if isinstance(store, Store) else Store(store)


def open_evaluator(workers=None, *, cache=True, store=None, gp=False, log=None, cache_scope="code"):
    """An Evaluator over the loaded parts and the store (create it after setup_parts)."""
    if cache_scope == "config":
        warn("W_CACHE_CONFIG_SCOPE", "cache keys ignore the code fingerprint: results may predate code edits")
    return Evaluator(workers, loaded_plugins(), store=open_store(store), gp=gp, cache=cache,
                     cache_scope=cache_scope, log=log)


def _emit(on_event, event, **fields):
    if on_event is not None:
        try:
            on_event(event, **fields)
        except Exception:
            pass


# ---------------------------------------------------------------- configs
def _apply_source(cfg, src, store=None):
    if src is None:
        return cfg, []
    if isinstance(src, Config):
        return configio.normalise(src), []
    if isinstance(src, (str, Path)):
        p = Path(src)
        if not p.is_file():
            raise RsiError(f"config file not found: {src}", "E_NOT_FOUND", value=str(src))
        s = configio.load_settings(p)
        return s.config, s.warnings
    if isinstance(src, dict):
        if "format" in src or isinstance(src.get("config"), dict):
            s = configio.load_settings(src)
            return s.config, s.warnings
        return configio.config_from_dict(src, base=cfg)
    raise RsiError(f"cannot use {type(src).__name__} as a config", "E_BAD_SETTINGS")


def resolve_config(config=None, overrides=None, *, store=None):
    """(Config, warnings, pinned_fields). config: None | Config | dict (settings doc, trial
    record or partial config) | path | a list of those applied in order. overrides: {field or
    'extra.NAME': value}, values coerced and part names resolved (aliases -> W_ALIAS_USED)."""
    try:
        warns, cfg = [], Config()
        for src in (config if isinstance(config, list) else [config]):
            cfg, w = _apply_source(cfg, src, store)
            warns += w
        values, extra, pinned = {}, None, []
        for k, v in (overrides or {}).items():
            f, val, w = coerce_override(k, v)
            warns += w
            if f.startswith("extra."):
                extra = dict(cfg.extra) if extra is None else extra
                extra[f[6:]] = val
            else:
                values[f] = val
            pinned.append(f)
        if extra is not None:
            values["extra"] = {**values.get("extra", cfg.extra), **extra}
        cfg = configio.normalise(dataclasses.replace(cfg, **values))
        configio.check_config(cfg)
    except RsiError:
        raise
    except Exception as e:
        raise as_rsi_error(e) from None
    _warn_all(warns)
    return cfg, warns, pinned


@closes_stores
def config_from_ref(ref, store=None):
    """--from: a trial id (prefix) from the store, or RUNNAME[:rank] -> {"config", "parts"} (a
    record: part modules that are not loaded give E_PARTS_MISSING, not an unknown part)."""
    store = open_store(store)
    name, _, rank = str(ref).partition(":")
    d = store.root / name
    if (d / "leaderboard.json").exists() or (d / "manifest.json").exists():
        e = _entry(d, rank=int(rank or 1))
        return {"config": e["config"], "parts": e.get("parts") or _run_parts(d)}
    rec = store.trial(ref)
    return {"config": rec["config"], **({"parts": rec["parts"]} if rec.get("parts") is not None else {})}


def _run_parts(path):
    return (rd.read_json(Path(path) / "manifest.json", {}) or {}).get("parts") or []


def parts_flags(prog):
    """prog plus --parts flags when the loaded part modules are not just the default ones."""
    loaded = loaded_plugins()
    if loaded == list(DEFAULT_PLUGINS):
        return prog
    return " ".join([prog] + [f"--parts {m}" for m in loaded or ["none"]])


def saved_steps(config):
    """(train steps, meta.expect.at_step) of the last settings doc / trial record among the config
    sources ((None, None) when there is none): `run --config winner.json` trains as saved."""
    steps = at = None
    for src in (config if isinstance(config, list) else [config]):
        doc = src
        if isinstance(src, (str, Path)):
            try:
                doc = json.loads(Path(src).read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                continue
        if not isinstance(doc, dict):
            continue
        if doc.get("format") == configio.SETTINGS_FORMAT:
            exp = ((doc.get("meta") or {}).get("expect") or {}).get("at_step")
            steps, at = (doc.get("train") or {}).get("steps") or exp, exp
        elif isinstance(doc.get("config"), dict) and isinstance(doc.get("budget"), dict):
            steps, at = doc["budget"].get("steps"), None
    return (int(steps) if isinstance(steps, (int, float)) else None), (int(at) if isinstance(at, (int, float)) else None)


def inert_warnings(cfg, pinned):
    default = Config()
    inert = schema.inert_fields(cfg)
    for f in pinned:
        if f in inert and getattr(cfg, f) != getattr(default, f):
            warn("W_INERT_FIELD", f"{f} is not read by model '{cfg.model}'", field=f)


def parse_seeds(spec, what="seed"):
    """'0,1,2' | '0-4' | '0-4,10' | [0, 1] -> list of ints (order kept, deduped). `what` names
    the flag in errors (seed, size, holdout seed)."""
    if spec is None:
        return None
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, (list, tuple)):
        return list(dict.fromkeys(int(s) for s in spec))
    out = []
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        try:
            if "-" in part[1:]:
                a, b = part.split("-", 1) if not part.startswith("-") else (part, None)
                out += list(range(int(a), int(b) + 1))
            else:
                out.append(int(part))
        except (TypeError, ValueError):
            raise RsiError(f"bad {what} spec {spec!r}: use 0,1,2 or 0-4 or 0-4,10", "E_USAGE", value=spec) from None
    if not out or any(s < 0 or s > 2**32 - 1 for s in out):
        raise RsiError(f"bad {what} spec {spec!r}: use 0,1,2 or 0-4 or 0-4,10 (a range runs low-high)", "E_USAGE",
                       value=spec)
    return list(dict.fromkeys(out))


def suite_names(name):
    """Dataset names of suite `name` (E_UNSUPPORTED if this nncore has no SUITES)."""
    from nncore import datasets
    if getattr(datasets, "SUITES", None) is None:
        raise RsiError("dataset suites are not available in this nncore", "E_UNSUPPORTED")
    if name not in datasets.SUITES:
        raise RsiError(f"unknown suite '{name}'", "E_UNKNOWN_PART", field="suite", value=name,
                       did_you_mean=datasets.SUITES.suggest(name), allowed=list(datasets.SUITES))
    return datasets.suite_datasets(name)


def resolve_datasets(datasets=None, suite=None, default=None):
    if datasets and suite:
        raise RsiError("--datasets and --suite are mutually exclusive", "E_USAGE")
    if suite:
        names = suite_names(suite)
    elif datasets:
        names = [d.strip() for d in datasets.split(";")] if isinstance(datasets, str) else list(datasets)
    else:
        return [default]
    out = []
    for n in names:
        if n:
            try:
                c, w = configio.resolve_name("dataset", n)
            except Exception as e:
                raise as_rsi_error(e) from None
            _warn_all([w] if w else [])
            out.append(c)
    return list(dict.fromkeys(out))


def resolve_seeds(seeds=None, n_seeds=None, base_seed=0):
    if seeds is not None and n_seeds is not None:
        raise RsiError("--seeds and --n-seeds are mutually exclusive", "E_USAGE")
    if n_seeds is not None:
        if int(n_seeds) < 1:
            raise RsiError("--n-seeds must be >= 1", "E_USAGE")
        return list(range(base_seed, base_seed + int(n_seeds)))
    return parse_seeds(seeds) or [base_seed]


def _check_budget(steps=None, fresh_points=None, max_seconds=None):
    """E_USAGE up front for budget values no cell could use."""
    if steps is not None and int(steps) < 0:
        raise RsiError("--steps must be >= 0", "E_USAGE", field="steps", value=steps)
    if fresh_points is not None and int(fresh_points) < 0:
        raise RsiError("--fresh-points must be >= 0", "E_USAGE", field="fresh_points", value=fresh_points)
    if max_seconds is not None and not float(max_seconds) > 0:
        raise RsiError("--max-seconds must be > 0", "E_USAGE", field="max_seconds", value=max_seconds)


def _early(early_stop):
    if early_stop is None:
        return None
    if isinstance(early_stop, dict):
        return (int(early_stop["at_step"]), float(early_stop["min_test_acc"]))
    return (int(early_stop[0]), float(early_stop[1]))


def _metrics(metrics):
    if isinstance(metrics, str):
        metrics = [m.strip() for m in metrics.split(",") if m.strip()]
    out = list(metrics or ())
    if out:
        from nncore import metrics as mm
        extra = getattr(mm, "EXTRA_METRICS", None)
        if extra is None:
            raise RsiError("extra metrics are not available in this nncore", "E_UNSUPPORTED")
        for m in out:
            if m not in extra:
                raise RsiError(f"unknown metric '{m}'", "E_UNKNOWN_PART", field="metrics", value=m,
                               did_you_mean=extra.suggest(m), allowed=list(extra))
    return tuple(out)


def _task_warnings(cfg, n_cells, steps, workers):
    if cfg.splitter.startswith("none") or cfg.test_frac == 0:
        warn("W_NO_TEST_SET", "no test split: fitness uses train metrics", field="splitter")
    est = steps * n_cells * schema.COST_MS_PER_STEP.get(cfg.model, 2.4) / 1000 / max(1, workers or 1)
    if est > 60:
        warn("W_SLOW", f"estimated {est:.0f} s ({n_cells} cells x {steps} steps)", estimated_s=round(est, 1))


# ---------------------------------------------------------------- trials
def new_trial_id():
    return "t_" + uuid.uuid4().hex[:12]


def trial_status(cells):
    good = [c for c in cells if c and c.get("status") in ("ok", "early_stopped")]
    if cells and all(c and c.get("status") == "ok" for c in cells):
        return "ok"
    return "partial" if good else "failed"


def fitness_of(summary, objective="test_acc:mean"):
    """Higher is better: 'METRIC[:AGG]' from a trial summary; losses negated; params -> -n_params.
    test_* falls back to train_* when there is no test set."""
    if objective in ("params", "n_params"):
        return -summary["n_params"] if summary.get("n_params") is not None else None
    metric, _, agg = objective.partition(":")
    st = summary.get(metric) or {}
    v = st.get(agg or "mean")
    if v is None and metric.startswith("test_"):
        v = (summary.get("train_" + metric[5:]) or {}).get(agg or "mean")
    if v is None:
        return None
    return -v if metric.endswith("_loss") else v


def make_trial(cfg, cells, *, budget, seeds, datasets, objective="test_acc:mean", tags=(), run=None, code_fp=None,
               base=None):
    """A TrialRecord (rsi/trial@1) from its cells (None entries = not run)."""
    got = [c for c in cells if c is not None]
    summary = summarize(got)
    status = trial_status(list(cells))
    fit = fitness_of(summary, objective) if status != "failed" else None
    return {"id": new_trial_id(), "created": now_iso(), "status": status,
            "cached": bool(got) and all(c.get("cached") for c in got), "config": configio.config_to_dict(cfg),
            "config_diff": configio.config_diff(cfg, base), "config_key": configio.config_key(cfg),
            "budget": budget, "seeds": list(seeds), "datasets": list(datasets), "cells": list(cells),
            "summary": summary, "objective": objective, "fitness": fit, "map": None, "png": None,
            "settings_file": None, "tags": list(tags or ()), "run": run, "code_fp": code_fp,
            "parts": loaded_plugins()}


def _budget(steps, eval_every, max_seconds, early_stop, fresh_points, metrics):
    es = _early(early_stop)
    return {"steps": int(steps), "eval_every": int(eval_every or 0), "max_seconds": max_seconds,
            "early_stop": {"at_step": es[0], "min_test_acc": es[1]} if es else None,
            "fresh_points": int(fresh_points or 0), "metrics": list(metrics or ())}


def train_session(spec):
    """A trained Session for one RunSpec (same numbers as run_unit: chunking is exact)."""
    from nncore.run import _preserve_rng
    from nncore.session import Session
    with _preserve_rng(), contextlib.redirect_stdout(sys.stderr):
        s = Session(configio.config_from_dict(spec.config)[0])
        s.train(spec.steps, stop_on_nonfinite=spec.stop_on_nonfinite)
    return s


def _expect(cell, cfg, steps):
    env = environment()
    return {"train_acc": (cell.get("train") or {}).get("acc"), "test_acc": (cell.get("test") or {}).get("acc"),
            "seed": cfg.seed, "at_step": steps, "code_fp": env["code_fp"], "platform": platform.platform()}


# ---------------------------------------------------------------- describe / check
def describe(*, section=None, lever=None, model=None, parts=None):
    setup_parts(parts)
    if model is not None:
        model = configio.resolve_name("model", model)[0]
    sec = None if section in (None, "commands", "errors", "objectives", "benchmarks") else section
    try:
        doc = schema.describe(model, sec)
    except ValueError as e:
        raise RsiError(str(e), "E_USAGE", field="section", value=section) from None
    if sec is None:
        doc["commands"] = dict(COMMAND_HELP)
        doc["exit_codes"] = dict(EXIT_MEANING)
        doc["errors"] = dict(ERROR_HELP)
        env_cost = rd.read_json(open_store_root() / "env.json")
        if env_cost and env_cost.get("ms_per_step"):
            doc["cost_model"] = {"ms_per_step": env_cost["ms_per_step"], "source": "runs/env.json (doctor --calibrate)"}
        else:
            doc["cost_model"]["source"] = "built-in estimate (doctor --calibrate measures this machine)"
        doc["objectives"]["search"] = ["scalar", "pareto"] + [f"bench:{b}" for b in _benchmark_ids()]
        doc["benchmarks"] = _benchmark_ids()
        doc["rsi"] = __version__
    if lever is not None:
        specs = {s["name"]: s for s in schema.field_specs(model)}
        if lever not in specs:
            import difflib
            raise RsiError(f"unknown lever '{lever}'", "E_UNKNOWN_FIELD", field=lever,
                           did_you_mean=difflib.get_close_matches(lever, list(specs), n=3), allowed=list(specs))
        return {"lever": specs[lever], "code_fp": doc["code_fp"], "parts": doc["parts"]}
    if section in ("commands", "errors", "objectives", "benchmarks"):
        key = {"errors": "errors"}.get(section, section)
        out = {key: doc[key], "code_fp": doc["code_fp"], "parts": doc["parts"]}
        if section == "errors":
            out["exit_codes"] = doc["exit_codes"]
        return out
    return doc


def open_store_root():
    from .store import default_root
    return default_root()


def _benchmark_ids():
    try:
        mod = optional_module("bench")
        return list(getattr(mod, "BENCHMARKS", {}))
    except RsiError:
        return []


def check(config=None, overrides=None, *, save=None, diff=False, parts=None):
    setup_parts(parts)
    cfg, _, pinned = resolve_config(config, overrides)
    inert_warnings(cfg, pinned)
    _warn_all(configio.ui_issues(cfg))
    try:
        info = schema.build_info(cfg)
    except Exception as e:
        err = as_rsi_error(e)
        if err.code != "E_INTERNAL":
            raise err from None
        raise RsiError(f"the config is valid but does not build: {type(e).__name__}: {e}", "E_RUN_FAILED",
                       details={"type": type(e).__name__, "config_diff": configio.config_diff(cfg)}) from None
    out = {"valid": True, "issues": [], "config_diff": configio.config_diff(cfg), "config_key": configio.config_key(cfg),
           **info, "inert_fields": sorted(schema.inert_fields(cfg)),
           "command": configio.config_to_command(cfg, prog=parts_flags("python -m rsi run")),
           "settings_file": None}
    if not diff:
        out["config"] = configio.config_to_dict(cfg)
    if save:
        configio.save_settings(save, cfg, meta={"saved_by": "rsi check", "created": now_iso(),
                                                "parts": loaded_plugins()})
        out["settings_file"] = str(save)
    return out


# ---------------------------------------------------------------- run
@closes_stores
def run(config=None, overrides=None, *, steps=None, seeds=None, n_seeds=None, datasets=None, suite=None, eval_every=0,
        max_seconds=None, early_stop=None, fresh_points=0, metrics=(), workers=1, map=None, png=None,
        save_settings=None, curve=False, tags=(), cache=True, store=None, parts=None, on_event=None):
    setup_parts(parts)
    _check_budget(steps, fresh_points, max_seconds)
    cfg, _, pinned = resolve_config(config, overrides)
    inert_warnings(cfg, pinned)
    saved, at = saved_steps(config)
    if steps is None:  # steps: --steps, else the file's train.steps, else 2000
        steps = saved or 2000
    elif at is not None and int(steps) != at:
        warn("W_STEPS_DIFFER", f"the config file was saved at {at} steps (meta.expect); running {int(steps)}, so the "
                               f"numbers will differ from meta.expect", field="steps", saved=at, steps=int(steps))
    seeds = resolve_seeds(seeds, n_seeds, cfg.seed)
    datasets = resolve_datasets(datasets, suite, cfg.dataset)
    metrics = _metrics(metrics)
    if map:
        from .render import parse_size
        try:
            map = parse_size(map)
        except ValueError as e:
            raise RsiError(f"--map: {e}", "E_USAGE") from None
    if curve and not eval_every:
        eval_every = max(1, int(steps) // 20)
    specs = cell_specs(cfg, datasets=datasets, seeds=seeds, steps=steps, eval_every=eval_every,
                       max_seconds=max_seconds, early_stop=_early(early_stop), fresh_points=fresh_points,
                       metrics=metrics)
    _task_warnings(cfg, len(specs), int(steps), workers)
    st = open_store(store)
    with Evaluator(workers, loaded_plugins(), store=st, cache=cache) as ev:
        def done(i, cell):
            _emit(on_event, "cell", i=i, n=len(specs), dataset=cell.get("dataset"), seed=cell.get("seed"),
                  status=cell.get("status"), test_acc=(cell.get("test") or {}).get("acc"), cached=cell.get("cached"))
        cells = ev.map(specs, on_result=done)
        rec = make_trial(cfg, cells, budget=_budget(steps, eval_every, max_seconds, early_stop, fresh_points, metrics),
                         seeds=seeds, datasets=datasets, tags=tags, code_fp=ev.fp_for(cfg))
    for c in cells:
        if c and c.get("status") == "diverged":
            warn("W_NONFINITE", f"{c['dataset']} seed {c['seed']} diverged at step {c.get('diverged_at')}; "
                                "its metrics are null", dataset=c["dataset"], seed=c["seed"])
    if (map or png or save_settings) and cells and cells[0] and cells[0].get("status") != "error":
        _render(rec, specs[0], cells[0], map, png, save_settings)
    st.put_trial(rec)
    if rec["status"] == "failed" and all(c and c.get("status") in ("error", "diverged") for c in cells):
        e = RsiError(f"every cell failed ({', '.join(sorted({c['status'] for c in cells}))})", "E_RUN_FAILED",
                     details={"trial_id": rec["id"]})
        e.result = rec
        raise e
    return rec


def _render(rec, spec, cell, map_size, png, save_settings):
    """--map / --png / --save-settings for the first cell (its own dataset and seed)."""
    from . import render
    cfg = configio.config_from_dict(spec.config)[0]
    s = train_session(spec) if map_size or png else None
    if map_size:
        rec["map"] = render.ascii_map(s, *map_size)
    if png:
        rec["png"] = render.png_map(s, png)
    if save_settings:
        configio.save_settings(save_settings, cfg, train={"steps": spec.steps, "eval_every": spec.eval_every},
                               meta={"saved_by": "rsi run", "created": now_iso(), "trial_id": rec["id"],
                                     "expect": _expect(cell, cfg, spec.steps), "parts": loaded_plugins()})
        rec["settings_file"] = str(save_settings)


# ---------------------------------------------------------------- sweep
def _rank_key(rank_by):
    """rank_by -> objective for fitness_of ('test_acc' -> 'test_acc:mean')."""
    if rank_by in ("params", "n_params"):
        return "params"
    return rank_by if ":" in rank_by else f"{rank_by}:mean"


def _vary_dict(vary):
    if vary is None:
        return {}
    if isinstance(vary, dict):
        return dict(vary)
    out = {}
    for item in ([vary] if isinstance(vary, str) else vary):
        field, eq, vals = item.partition("=")
        if not eq or not field.strip():
            raise RsiError(f"bad --vary {item!r}: expected FIELD=v1,v2", "E_USAGE", value=item)
        out[field.strip().replace("-", "_") if not field.startswith("extra.") else field.strip()] = vals
    return out


def _random_configs(cfg, space, n, sample_seed, pinned=()):
    try:
        sp = optional_module("space")
    except RsiError:
        raise RsiError("--random needs rsi.space (not in this build yet)", "E_UNSUPPORTED") from None
    import random
    rng = random.Random(sample_seed)
    Space = getattr(sp, "Space")
    try:  # fields set on the command line stay pinned, as in evolve
        s = Space.load(space or "default", cfg, pinned=pinned)
    except RsiError:
        raise
    except Exception as e:
        raise as_rsi_error(e) from None
    out = []
    for _ in range(int(n)):
        out.append(configio.config_diff(s.phenotype(s.sample(rng)), cfg))
    return out


@closes_stores
def sweep(config=None, overrides=None, *, vary=None, vary_json=None, oat=None, random_n=None, space=None,
          sample_seed=0, rank_by="test_acc", average_over=(), max_runs=500, top=20, name=None, out=None, force=False,
          **run_kw):
    parts, on_event = run_kw.pop("parts", None), run_kw.pop("on_event", None)
    setup_parts(parts)
    _check_budget(run_kw.get("steps"), run_kw.get("fresh_points"), run_kw.get("max_seconds"))
    cfg, _, pinned = resolve_config(config, overrides)
    inert_warnings(cfg, pinned)
    modes = [m for m in ("vary", "oat", "random") if {"vary": vary or vary_json, "oat": oat, "random": random_n}[m]]
    if len(modes) != 1:
        raise RsiError("sweep needs exactly one of --vary/--vary-json, --oat, --random N --space FILE", "E_USAGE")
    mode = {"vary": "grid", "oat": "oat", "random": "random"}[modes[0]]
    axes, trials = {}, []
    if mode == "grid":
        vd = _vary_dict(vary)
        for f, vals in (vary_json or {}).items():
            vd[f] = vals if isinstance(vals, list) else [vals]
        axes = {f: expand_values(f, v) for f, v in vd.items()}
        trials = grid(axes)
    elif mode == "oat":
        fields = [f.strip() for f in (oat.split(",") if isinstance(oat, str) else oat) if f.strip()]
        trials = [{}] + [o for _, _, o in one_at_a_time(cfg, fields)]
        axes = {f: [getattr(cfg, f)] + [o[f] for o in trials[1:] if f in o] for f in fields}
    else:
        trials = [{}] + _random_configs(cfg, space, random_n, sample_seed, pinned)
    if "model" not in axes:
        inert = schema.inert_fields(cfg)
        for f in axes:
            if f in inert:
                warn("W_INERT_FIELD", f"{f} is not read by model '{cfg.model}': its trials train the same net",
                     field=f)
    configs, seen = [], set()
    for ov in trials:
        tcfg, _, _ = resolve_config(cfg, ov)
        k = configio.config_key(tcfg)
        if k not in seen:
            seen.add(k)
            configs.append((ov, tcfg))
    if len(configs) > int(max_runs):
        raise RsiError(f"the sweep has {len(configs)} configs, more than --max-runs {max_runs}", "E_USAGE",
                       hint="narrow the axes or raise --max-runs")
    steps = 2000 if run_kw.get("steps") is None else int(run_kw["steps"])
    varied = {f for ov, _ in configs for f in ov} | set(axes)
    if "dataset" in varied and (run_kw.get("datasets") or run_kw.get("suite")):
        raise RsiError("the sweep varies dataset: drop --datasets / --suite (or vary something else)", "E_USAGE",
                       field="dataset")
    if "seed" in varied and (run_kw.get("seeds") is not None or run_kw.get("n_seeds") is not None):
        raise RsiError("the sweep varies seed: drop --seeds / --n-seeds (or vary something else)", "E_USAGE",
                       field="seed")
    seeds = resolve_seeds(run_kw.get("seeds"), run_kw.get("n_seeds"), cfg.seed)
    datasets = resolve_datasets(run_kw.get("datasets"), run_kw.get("suite"), cfg.dataset)

    def task_of(tcfg):  # each trial trains its own dataset / seed unless --datasets / --seeds say otherwise
        return (resolve_datasets(run_kw.get("datasets"), run_kw.get("suite"), tcfg.dataset),
                resolve_seeds(run_kw.get("seeds"), run_kw.get("n_seeds"), tcfg.seed))
    metrics = _metrics(run_kw.get("metrics", ()))
    eval_every = int(run_kw.get("eval_every") or 0)
    budget = _budget(steps, eval_every, run_kw.get("max_seconds"), run_kw.get("early_stop"),
                     run_kw.get("fresh_points", 0), metrics)
    workers = run_kw.get("workers")
    workers = default_workers() if workers is None else workers
    st = open_store(run_kw.get("store"))
    path = Path(out) if out else st.root / (name or time.strftime("sweep-%Y%m%d-%H%M%S"))
    specs, owner, tasks = [], [], []
    for ti, (ov, tcfg) in enumerate(configs):
        tasks.append(task_of(tcfg))
        for s in cell_specs(tcfg, datasets=tasks[ti][0], seeds=tasks[ti][1], steps=steps, eval_every=eval_every,
                            max_seconds=budget["max_seconds"], early_stop=_early(budget["early_stop"]),
                            fresh_points=budget["fresh_points"], metrics=metrics):
            specs.append(s)
            owner.append(ti)
    _task_warnings(cfg, len(specs), steps, workers)
    objective = _rank_key(rank_by)
    run_dir = rd.RunDir.create(path, "sweep", {
        "mode": mode, "axes": axes, "base": configio.config_to_dict(cfg), "budget": budget,
        "seeds": seeds if "seed" not in varied else "per trial",
        "datasets": datasets if "dataset" not in varied else "per trial", "rank_by": rank_by, "n_trials": len(configs), "n_cells": len(specs),
        "code_fp": code_fingerprint(), "parts": loaded_plugins(), "tags": list(run_kw.get("tags") or ())}, force=force)
    state = "crashed"
    try:
        run_dir.write_progress(state="running", n_trials=len(configs), n_cells=len(specs), cells_done=0)
        counter = {"done": 0}

        def done(i, cell):
            counter["done"] += 1
            _emit(on_event, "cell", i=counter["done"], n=len(specs), trial=owner[i], dataset=cell.get("dataset"),
                  seed=cell.get("seed"), status=cell.get("status"), cached=cell.get("cached"))
            run_dir._progress["cells_done"] = counter["done"]
            run_dir.heartbeat()

        def stop_check():
            run_dir.heartbeat()
            return run_dir.stop_requested()
        with Evaluator(workers, loaded_plugins(), store=st, cache=run_kw.get("cache", True),
                       log=lambda e, **f: _emit(on_event, e, **f)) as ev:
            cells = ev.map(specs, on_result=done, stop_check=stop_check)
        recs = []
        for ti, (ov, tcfg) in enumerate(configs):
            tc = [c for c, o in zip(cells, owner) if o == ti]
            if all(c is None for c in tc):
                continue
            rec = make_trial(tcfg, tc, budget=budget, seeds=tasks[ti][1], datasets=tasks[ti][0], objective=objective,
                             tags=run_kw.get("tags") or (), run=run_dir.name, code_fp=ev.fp_for(tcfg), base=cfg)
            rec["axes"] = {f: ov.get(f, getattr(tcfg, f) if f in configio.FIELDS else None) for f in axes} \
                if mode != "random" else configio.config_diff(tcfg, cfg)
            st.put_trial(rec)
            run_dir.append("trials.jsonl", rec)
            recs.append(rec)
        stopped = run_dir.stop_requested() or any(c is None for c in cells)
        state = "stopped" if stopped else "done"
        result = _sweep_result(run_dir, mode, axes, recs, cells, rank_by, objective, average_over, top, cfg, state,
                               steps)
        run_dir.write_json("result.json", result)
        return result
    except KeyboardInterrupt:
        state = "stopped"
        raise RsiError("interrupted", "E_INTERRUPTED", hint=f"python -m rsi sweep ... --name {path.name}") from None
    finally:
        run_dir.close(state)


def _sweep_result(run_dir, mode, axes, recs, cells, rank_by, objective, average_over, top, base, state, steps):
    order = {id(r): i for i, r in enumerate(recs)}  # ties keep the sweep's own order
    ranked = sorted(recs, key=lambda r: (r["fitness"] is None, -(r["fitness"] or 0),
                                         r["summary"].get("n_params") or 0, order[id(r)]))
    entries = []
    for i, r in enumerate(ranked, 1):
        sm = r["summary"]
        ta = sm.get("test_acc") or sm.get("train_acc") or {}
        gid = r["config_key"][:12]
        sf = run_dir.path / "top" / f"{i:02d}.settings.json"
        if i <= max(1, int(top)):
            cfg = configio.config_from_dict(r["config"])[0]
            configio.save_settings(sf, cfg, train={"steps": steps}, meta={
                "saved_by": "rsi sweep", "created": now_iso(), "run": str(run_dir.path), "trial_id": r["id"],
                "genome_id": gid, "rank": i, "objective": objective, "score": r["fitness"], "parts": loaded_plugins()})
        entries.append({"rank": i, "genome_id": gid, "trial_id": r["id"], "score": r["fitness"],
                        "fitness": {"acc_agg": ta.get("mean"), "seed_agg": "mean", "acc_min": ta.get("min"),
                                    "acc_std": ta.get("std"), "n_seeds": len(r["seeds"]), "n_params": sm.get("n_params"),
                                    "seconds_mean": (sm.get("seconds") or 0) / max(1, sm.get("n_cells") or 1)},
                        "holdout": None, "config_diff": r["config_diff"], "config": r["config"],
                        "describe": next((c["model"]["describe"] for c in r["cells"] if c and c.get("model")), None),
                        "settings_file": str(sf) if i <= max(1, int(top)) else None, "parts": r.get("parts"),
                        "repro": f"python -m rsi run --from {r['id']} --steps {steps}"})
    best = None
    if ranked:
        best_cfg = configio.config_from_dict(ranked[0]["config"])[0]
        bf = run_dir.path / "best.settings.json"
        configio.save_settings(bf, best_cfg, train={"steps": steps}, meta={
            "saved_by": "rsi sweep", "created": now_iso(), "run": str(run_dir.path), "trial_id": ranked[0]["id"],
            "genome_id": entries[0]["genome_id"], "rank": 1, "objective": objective, "score": ranked[0]["fitness"],
            "parts": loaded_plugins()})
        best = {"id": ranked[0]["id"], "settings_file": str(bf)}
    base_key = configio.config_key(base)
    baseline = next(({"genome_id": e["genome_id"], "score": e["score"], "holdout_score": None}
                     for e, r in zip(entries, ranked) if r["config_key"] == base_key), None)
    lb = {"name": run_dir.name, "kind": "sweep", "state": state, "sort": "score", "baseline": baseline,
          "entries": entries, "pareto": pareto_front(entries),
          "stats": {"n_trials": len(recs), "n_cells": len(cells), "unique_genomes": len(recs)}}
    run_dir.write_json("leaderboard.json", lb)
    groups = []
    over = [average_over] if isinstance(average_over, str) else list(average_over or ())
    if over:
        buckets = {}
        for r in recs:
            fixed = {k: v for k, v in (r.get("axes") or {}).items() if k not in over}
            key = configio.canonical_json(fixed)
            vals = buckets.setdefault(key, (fixed, []))[1]
            for c in r["cells"]:
                if c and c.get("status") in ("ok", "early_stopped"):
                    v = (c.get("test") or {}).get("acc")
                    vals.append(v if v is not None else (c.get("train") or {}).get("acc"))
        for fixed, vals in buckets.values():
            vals = [v for v in vals if v is not None]
            groups.append({"fixed": fixed, "over": ",".join(over),
                           "test_acc": {"mean": statistics.fmean(vals) if vals else None,
                                        "min": min(vals) if vals else None, "n": len(vals)}})
        groups.sort(key=lambda g: (g["test_acc"]["mean"] is None, -(g["test_acc"]["mean"] or 0)))
    n_cached = sum(1 for c in cells if c and c.get("cached"))
    return {"kind": "sweep", "name": run_dir.name, "dir": str(run_dir.path), "status": state, "axes": axes,
            "mode": mode, "n_trials": len(recs), "n_cells": sum(c is not None for c in cells), "n_cached": n_cached,
            "n_failed": sum(1 for c in cells if c and c.get("status") in ("error", "timeout")),
            "ranked_by": rank_by,
            "trials": [{"rank": e["rank"], "id": e["trial_id"], "fitness": e["score"], "config_diff": e["config_diff"],
                        "summary": {"test_acc": _pick(r["summary"].get("test_acc")),
                                    "n_params": r["summary"].get("n_params")}, "status": r["status"]}
                       for e, r in zip(entries[:int(top)], ranked)],
            "top": int(top), "truncated": len(entries) > int(top),  # trials holds the best `top` of n_trials
            "groups": groups, "best": best}


def _pick(st):
    return {k: (st or {}).get(k) for k in ("mean", "min", "std", "n")} if st else None


def pareto_front(entries):
    """Non-dominated entries on (acc_agg high, n_params low), fewest params first."""
    pts = [(e["fitness"].get("acc_agg"), e["fitness"].get("n_params"), e["genome_id"]) for e in entries
           if e.get("fitness") and e["fitness"].get("acc_agg") is not None and e["fitness"].get("n_params") is not None]
    front = [p for p in pts if not any((q[0] >= p[0] and q[1] <= p[1]) and (q[0] > p[0] or q[1] < p[1]) for q in pts)]
    front = sorted(set(front), key=lambda p: (p[1], -p[0], p[2]))
    return [{"genome_id": g, "acc_agg": a, "n_params": n} for a, n, g in front]


# ---------------------------------------------------------------- optional (Build B) modules
def optional_module(name):
    """Import rsi.<name> (a later build adds it); E_UNSUPPORTED when it is missing."""
    try:
        return importlib.import_module(f"rsi.{name}")
    except ModuleNotFoundError as e:
        if e.name in (f"rsi.{name}",):
            raise RsiError(f"'{name}' is not available in this build of rsi console (rsi/{name}.py missing)",
                           "E_UNSUPPORTED", hint="python -m rsi describe --section commands") from None
        raise


def evolve(config=None, overrides=None, *, settings=None, space=None, name=None, out=None, resume=None, force=False,
           allow_code_change=False, workers=None, cache=True, store=None, parts=None, on_event=None, **settings_kw):
    mod = optional_module("evolve")
    return mod.evolve(config, overrides, settings=settings, space=space, name=name, out=out, resume=resume,
                      force=force, allow_code_change=allow_code_change, workers=workers, cache=cache, store=store,
                      parts=parts, on_event=on_event, **settings_kw)


def bench(config=None, benchmark="featherbench-general-v1", workers=None, submit=None, *, overrides=None, steps=None, cache=True,
          store=None, parts=None, on_event=None):
    return optional_module("bench").bench(config, benchmark, workers, submit, overrides=overrides, steps=steps,
                                          cache=cache, store=store, parts=parts, on_event=on_event)


def bench_rank(files, *, top=None):
    return optional_module("bench").rank(files, top=top)


def bench_list():
    return optional_module("bench").list_benchmarks()


def complexity(config=None, family=None, sizes=None, threshold=0.95, ladder=None, knob=None, seeds=None, steps=3000,
               workers=None, *, overrides=None, cache=True, store=None, parts=None, on_event=None):
    return optional_module("complexity").complexity(config, family, sizes, threshold, ladder, knob, seeds, steps,
                                                    workers, overrides=overrides, cache=cache, store=store,
                                                    parts=parts, on_event=on_event)


def gp_check(expr):
    return optional_module("gp").check(expr)


# ---------------------------------------------------------------- run dirs
def _store_root(store=None):
    if store is None:
        return open_store_root()
    return store.root if isinstance(store, Store) else Path(store)


def _dir(target, store=None):
    return rd.resolve_target(target, _store_root(store))


def status(target, *, history=10, store=None):
    return rd.read_status(_dir(target, store), history=history)


def wait(target, *, timeout=540.0, store=None, poll=1.0):
    path = _dir(target, store)
    t0 = time.time()
    while True:
        s = rd.read_status(path)
        if s["state"] not in ("running", "stopping"):
            return s
        if time.time() - t0 >= float(timeout):
            e = RsiError(f"{path.name} still {s['state']} after {timeout} s", "E_TIMEOUT",
                         hint=f"python -m rsi wait {path.name} --timeout {int(timeout)}")
            e.result = s
            raise e
        time.sleep(min(poll, max(0.05, float(timeout) - (time.time() - t0))))


def stop(target, *, now=False, wait=False, timeout=600.0, store=None):
    path = _dir(target, store)
    s = rd.read_status(path)
    if s["state"] not in ("running", "stopping"):
        s["already"] = s["state"]
        return s
    rd.request_stop(path, now=now)
    if wait:
        return globals()["wait"](path, timeout=timeout)
    return rd.read_status(path)


def _entries(path):
    lb = rd.read_json(Path(path) / "leaderboard.json")
    if not lb:
        raise RsiError(f"{path} has no leaderboard.json yet", "E_NOT_FOUND", value=str(path),
                       hint=f"python -m rsi status {Path(path).name}")
    return lb


def _entry(path, rank=1, genome_id=None):
    lb = _entries(path)
    es = lb.get("entries") or []
    if genome_id:
        hit = [e for e in es if str(e.get("genome_id", "")).startswith(genome_id)]
        if len(hit) != 1:
            raise RsiError(f"genome '{genome_id}' {'is ambiguous' if hit else 'not found'} in {path}",
                           "E_AMBIGUOUS_ID" if hit else "E_NOT_FOUND", value=genome_id)
        return {**hit[0], "_lb": lb}
    if not 1 <= int(rank) <= len(es):
        raise RsiError(f"rank {rank} not in 1..{len(es)} for {path}", "E_NOT_FOUND", value=rank)
    return {**es[int(rank) - 1], "_lb": lb}


def leaderboard(targets, *, top=10, sort="holdout", pareto=False, store=None):
    targets = [targets] if isinstance(targets, (str, Path)) else list(targets)
    entries, docs = [], []
    for t in targets:
        lb = _entries(_dir(t, store))
        docs.append(lb)
        for e in lb.get("entries") or []:
            entries.append({**e, "run": lb.get("name")} if len(targets) > 1 else dict(e))

    def neg(v):
        return -(v if v is not None else -math.inf)
    keys = {"holdout": lambda e: ((e.get("holdout") or {}).get("score") is None, neg((e.get("holdout") or {}).get("score")),
                                  neg(e.get("score"))),
            "score": lambda e: (e.get("score") is None, neg(e.get("score"))),
            "params": lambda e: ((e.get("fitness") or {}).get("n_params") or math.inf, neg(e.get("score"))),
            "acc": lambda e: neg((e.get("fitness") or {}).get("acc_agg"))}
    if sort not in keys:
        raise RsiError(f"bad --sort {sort!r} (holdout|score|params|acc)", "E_USAGE")
    entries.sort(key=keys[sort])
    for i, e in enumerate(entries, 1):
        e["rank"] = i
    first = docs[0]
    out = {"name": first.get("name") if len(docs) == 1 else ",".join(d.get("name", "?") for d in docs),
           "kind": first.get("kind"), "state": first.get("state") if len(docs) == 1 else None, "sort": sort,
           "baseline": first.get("baseline") if len(docs) == 1 else None, "entries": entries[:int(top)],
           "stats": first.get("stats") if len(docs) == 1 else {"runs": len(docs)}}
    if pareto:
        out["pareto"] = pareto_front(entries) if len(docs) > 1 or not first.get("pareto") else first["pareto"]
    return out


# ---------------------------------------------------------------- export / open
def _export_source(target, rank, genome_id, store):
    """(config dict, steps, meta) for a run dir entry or a trial id."""
    st = open_store(store)
    path = None
    try:
        path = _dir(target, st)
    except RsiError:
        pass
    if path is not None:
        e = _entry(path, rank, genome_id)
        man = rd.read_json(path / "manifest.json", {}) or {}
        steps = (man.get("budget") or {}).get("steps") or (man.get("settings") or {}).get("steps") or 3000
        meta = {"run": str(path), "genome_id": e.get("genome_id"), "trial_id": e.get("trial_id"),
                "rank": e.get("rank"), "objective": man.get("objective") or man.get("rank_by"), "score": e.get("score"),
                "holdout_score": (e.get("holdout") or {}).get("score")}
        src = {"config": e["config"], "parts": e.get("parts") or man.get("parts") or []}
        return src, int(steps), meta, path.name + f"-rank{e.get('rank')}"
    rec = st.trial(target)
    meta = {"trial_id": rec["id"], "run": rec.get("run"), "objective": rec.get("objective"),
            "score": rec.get("fitness")}
    src = {"config": rec["config"], **({"parts": rec["parts"]} if rec.get("parts") is not None else {})}
    return src, int(rec["budget"]["steps"]), meta, rec["id"]


@closes_stores
def export(target, *, rank=1, genome_id=None, out=None, steps=None, open=False, store=None, parts=None, workers=0,
           dry_run=False):
    setup_parts(parts)
    st = open_store(store)
    cfg_dict, src_steps, meta, stem = _export_source(target, rank, genome_id, st)
    cfg, warns, _ = resolve_config(cfg_dict)
    steps = int(steps or src_steps)
    with Evaluator(workers, loaded_plugins(), store=st) as ev:
        cell = ev.evaluate(cell_specs(cfg, steps=steps)[0])
    out = Path(out) if out else st.root / "exports" / f"{stem}.settings.json"
    meta = {"saved_by": "rsi export", "created": now_iso(), **{k: v for k, v in meta.items() if v is not None},
            "expect": _expect(cell, cfg, steps), "parts": loaded_plugins(), "note": ""}
    configio.save_settings(out, cfg, train={"steps": steps, "eval_every": max(1, steps // 20)}, meta=meta)
    res = {"settings_file": str(out), "config_diff": configio.config_diff(cfg), "meta": meta,
           "status": cell.get("status"), "repro": parts_flags("python -m rsi run") + f" --config {out}"}
    if open:
        res["open"] = open_ui(str(out), dry_run=dry_run)
    return res


def open_ui(target, *, opengl=False, dry_run=False, store=None):
    """Launch `nn_playground.py --load FILE` detached (the launcher on V's PC supports --load)."""
    p = Path(target)
    if p.is_dir() or not p.exists():
        try:
            d = _dir(target, store)
            f = d / "best.settings.json"
            if not f.exists():
                raise RsiError(f"{d} has no best.settings.json", "E_NOT_FOUND", value=str(target))
            p = f
        except RsiError as e:
            if e.code != "E_NOT_FOUND" or p.exists():
                raise
            p = Path(export(target, store=store)["settings_file"])
    launcher = REPO_ROOT / "nn_playground.py"
    exe = sys.executable
    if os.name == "nt" and Path(exe).with_name("pythonw.exe").exists():
        exe = str(Path(exe).with_name("pythonw.exe"))
    cmd = [exe, str(launcher), "--load", str(p.resolve())] + (["--opengl"] if opengl else [])
    res = {"settings_file": str(p), "command": cmd, "pid": None, "launched": False}
    if dry_run or os.environ.get("RSI_NO_LAUNCH"):
        return res
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            cwd=str(REPO_ROOT), **kw)
    res.update(pid=proc.pid, launched=True)
    return res


# ---------------------------------------------------------------- runs
def runs_list(*, kind=None, limit=20, store=None):
    root = _store_root(store)
    rows = []
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        if not (d / "manifest.json").exists():
            continue
        s = rd.read_status(d, history=0)
        man = rd.read_json(d / "manifest.json", {}) or {}
        if kind and s.get("kind") != kind:
            continue
        lb = rd.read_json(d / "leaderboard.json") or {}
        top = (lb.get("entries") or [None])[0]
        rows.append({"name": d.name, "kind": s.get("kind"), "state": s["state"], "created": man.get("created"),
                     "dir": str(d), "best": {"genome_id": top.get("genome_id"), "score": top.get("score"),
                                             "describe": top.get("describe")} if top else None})
    rows.sort(key=lambda r: r["created"] or "", reverse=True)
    return {"store": str(root), "runs": rows[:int(limit)] if limit else rows, "n": len(rows)}


@closes_stores
def runs_query(*, where=(), sort=(), limit=20, fields=None, tag=None, run=None, objective=None, store=None):
    st = open_store(store)
    if isinstance(fields, str):
        fields = [f.strip() for f in fields.split(",") if f.strip()]
    rows = st.query(where=where, sort=sort, limit=limit, fields=fields, run=run, tag=tag, objective=objective)
    return {"rows": rows, "n": len(rows), "total": st.count(), "where": list(where), "sort": list(sort)}


@closes_stores
def runs_show(id_or_prefix, *, curve=False, store=None):
    rec = open_store(store).trial(id_or_prefix)
    if not curve:
        for c in rec.get("cells") or []:
            if c:
                c.pop("curve", None)
    return rec


@closes_stores
def runs_stats(*, group_by, where=(), metric="summary.test_acc.mean", store=None):
    from .query import expand_path
    return {"group_by": group_by, "metric": expand_path(metric),
            "groups": open_store(store).stats(group_by, where=where, metric=expand_path(metric))}


# ---------------------------------------------------------------- replay
_REPLAY_FIELDS = ("status", "steps_done", "diverged_at", "train", "test", "fresh")


@closes_stores
def replay(trial_id, *, workers=1, store=None, parts=None):
    setup_parts(parts)
    st = open_store(store)
    rec = st.trial(trial_id)
    b = rec["budget"]
    try:
        cfg = configio.load_settings(rec).config  # E_PARTS_MISSING when its part modules are not loaded
    except Exception as e:
        raise as_rsi_error(e) from None
    specs = cell_specs(cfg, datasets=rec["datasets"], seeds=rec["seeds"], steps=b["steps"],
                       eval_every=b.get("eval_every", 0), max_seconds=b.get("max_seconds"),
                       early_stop=_early(b.get("early_stop")), fresh_points=b.get("fresh_points", 0),
                       metrics=b.get("metrics", ()))
    with Evaluator(workers, loaded_plugins(), store=None, cache=False) as ev:
        cells = ev.map(specs)
        fp = ev.fp_for(rec["config"])
    if rec.get("code_fp") and rec["code_fp"] != fp:
        warn("W_PLATFORM_MISMATCH", f"trial was made under {rec['code_fp']}, this code is {fp}",
             stored=rec["code_fp"], current=fp)
    diffs, compared = [], 0
    for old, new in zip(rec["cells"], cells):
        if old is None or old.get("status") == "timeout":
            continue
        compared += 1
        for f in _REPLAY_FIELDS:
            if old.get(f) != new.get(f):
                diffs.append({"dataset": old.get("dataset"), "seed": old.get("seed"), "field": f,
                              "stored": old.get(f), "replayed": new.get(f)})
        if (old.get("model") or {}).get("n_params") != (new.get("model") or {}).get("n_params"):
            diffs.append({"dataset": old.get("dataset"), "seed": old.get("seed"), "field": "model.n_params",
                          "stored": (old.get("model") or {}).get("n_params"),
                          "replayed": (new.get("model") or {}).get("n_params")})
    out = {"trial_id": rec["id"], "identical": not diffs, "n_cells": compared, "diffs": diffs,
           "code_fp": {"stored": rec.get("code_fp"), "current": fp}, "cells": cells}
    if diffs:
        e = RsiError(f"replay of {rec['id']} differs in {len(diffs)} field(s)", "E_NOT_REPRODUCIBLE",
                     details={"diffs": diffs})
        e.result = out
        raise e
    return out


def doctor(*, golden=False, record_golden=False, calibrate=False, store=None, parts=None):
    from .doctor import run_doctor
    return run_doctor(golden=golden, record_golden=record_golden, calibrate=calibrate, store=store, parts=parts)
