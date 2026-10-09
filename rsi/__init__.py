"""
rsi console (RSI = Recursive Super Intelligence): drive NN Playground from scripts and AI agents.

    python -m rsi <command> [flags]        (== python headless.py <command>)
    import rsi; rec = rsi.run(overrides={"model": "mlp"}, steps=3000, seeds="0-4")

One function per command; each returns the command's `result` dict (the
envelope's "result") and raises RsiError (.code, .exit, .to_dict()):

    describe(*, section=None, lever=None, model=None, parts=None) -> dict
    check(config=None, overrides=None, *, save=None, diff=False, parts=None) -> dict
    run(config=None, overrides=None, *, steps=2000, seeds=None, n_seeds=None, datasets=None, suite=None,
        eval_every=0, max_seconds=None, early_stop=None, fresh_points=0, metrics=(), workers=1, map=None,
        png=None, save_settings=None, curve=False, tags=(), cache=True, store=None, parts=None,
        on_event=None) -> dict
    sweep(config=None, overrides=None, *, vary=None, vary_json=None, oat=None, random_n=None, space=None,
          sample_seed=0, rank_by="test_acc", average_over=(), max_runs=500, top=20, name=None, out=None,
          force=False, **run_kw) -> dict
    evolve(config=None, overrides=None, *, settings=None, space=None, name=None, out=None, resume=None,
           force=False, allow_code_change=False, workers=None, cache=True, store=None, parts=None,
           on_event=None, **settings_kw) -> dict                       (rsi/evolve.py: evolve(...))
    bench(config=None, benchmark="featherbench-general-v1", workers=None, submit=None, *, overrides=None, cache=True,
          store=None, parts=None, on_event=None) -> dict                (rsi/bench.py: bench(...))
    bench_rank(files, *, top=None) -> dict; bench_list() -> dict        (rsi/bench.py: rank, list_benchmarks)
    complexity(config=None, family=None, sizes=None, threshold=0.95, ladder=None, knob=None, seeds=None,
               steps=3000, workers=None, *, overrides=None, cache=True, store=None, parts=None,
               on_event=None) -> dict                                   (rsi/complexity.py: complexity(...))
    gp_check(expr) -> dict                                              (rsi/gp.py: check)
    status(target, *, history=10) -> dict; wait(target, *, timeout=540.0) -> dict
    stop(target, *, now=False, wait=False, timeout=600.0) -> dict
    leaderboard(targets, *, top=10, sort="holdout", pareto=False) -> dict
    export(target, *, rank=1, genome_id=None, out=None, steps=None, open=False) -> dict
    open_ui(target, *, opengl=False, dry_run=False) -> dict
    runs_list(*, kind=None, limit=20, store=None) -> dict
    runs_query(*, where=(), sort=(), limit=20, fields=None, tag=None, run=None, objective=None, store=None) -> dict
    runs_show(id_or_prefix, *, curve=False, store=None) -> dict
    runs_stats(*, group_by, where=(), metric="summary.test_acc.mean", store=None) -> dict
    replay(trial_id, *, workers=1, store=None) -> dict
    doctor(*, golden=False, record_golden=False, calibrate=False) -> dict

config: None | Config | dict (settings doc, trial record, partial config) | path | list of those;
overrides: {field or "extra.NAME": value} (strings are coerced, part names resolved). Commands whose
module is not in this build (evolve, bench, complexity, gp) raise E_UNSUPPORTED.

Note: rsi.sweep / rsi.evolve / rsi.bench / rsi.complexity / rsi.doctor are the functions above. To
reach the modules of the same name use `from rsi.evolve import Evolution` or
importlib.import_module("rsi.evolve") (not `import rsi.evolve as m`).

Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md.
"""
import importlib
import sys
import types

__version__ = "0.5.0"

_API = ("describe", "check", "run", "sweep", "evolve", "bench", "bench_rank", "bench_list", "complexity", "gp_check",
        "status", "wait", "stop", "leaderboard", "export", "open_ui", "runs_list", "runs_query", "runs_show",
        "runs_stats", "replay", "doctor", "collect_warnings", "resolve_config", "setup_parts", "open_evaluator",
        "make_trial", "parse_seeds", "resolve_datasets", "resolve_seeds", "suite_names", "optional_module")
_LAZY = {**{n: "rsi.api" for n in _API}, "RsiError": "rsi.errors", "Evaluator": "rsi.pool", "cell_specs": "rsi.pool",
         "default_workers": "rsi.pool", "Store": "rsi.store", "load_settings": "nncore.configio",
         "save_settings": "nncore.configio", "config_to_dict": "nncore.configio",
         "config_from_dict": "nncore.configio", "config_key": "nncore.configio",
         "Space": "rsi.space", "Genome": "rsi.space", "EvolveSettings": "rsi.evolve"}

__all__ = sorted(_LAZY) + ["__version__"]


class _Package(types.ModuleType):
    """Keeps API functions bound when a same-named submodule (rsi/sweep.py ...) is imported."""
    def __setattr__(self, name, value):
        if name in _API and isinstance(value, types.ModuleType):
            return
        super().__setattr__(name, value)


def __getattr__(name):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module 'rsi' has no attribute {name!r}")
    try:
        value = getattr(importlib.import_module(mod), name)
    except (ImportError, AttributeError):
        if mod in ("rsi.space", "rsi.evolve"):
            raise AttributeError(f"rsi.{name} needs {mod} (not in this build yet)") from None
        raise
    if not isinstance(value, types.ModuleType):
        globals()[name] = value
    return value


sys.modules[__name__].__class__ = _Package
