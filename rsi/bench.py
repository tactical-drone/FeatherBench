"""
bench: FeatherBench, the fewest-params benchmark. "Whoever solves it with the least
amount of params wins" (memory and training speed): fewest params that solves every
pattern wins. A benchmark is a frozen, versioned
task: datasets x seeds at fixed n_points / noise / split / steps, scored on
fresh points. A recipe is any config; the benchmark overrides its data fields.

    python -m rsi bench list                           (or: python -m rsi featherbench list)
    python -m rsi bench --benchmark featherbench-general-v1 --model mlp --layers 8:sin,8 --workers 7 --submit me.json
    python -m rsi bench rank me.json you.json          # rsi/bench-leaderboard@1

Scoring: per dataset, the mean fresh accuracy over the seeds (a failed cell counts
as chance, 1/n_classes); solved = mean >= threshold. A submission reports n_solved,
solved_all, params_max ("a net of at most P params solves everything"),
params_total and mean_acc. Ranking: solved_all first; solved_all submissions by
fewest params_max, then higher mean_acc; the others by n_solved, mean_acc, then
fewer params_max. rank_key holds that order as a list that sorts ascending.

Never edit a published benchmark: add "<name>-v2" instead (scores with a
different definition hash do not compare). The v1 benchmarks were published as
general-v1 / classic-v1 / quick-v1; those ids stay accepted aliases and stay in the
hashed definition (hash_id), so their hashes and old submissions are unchanged.

Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md.
"""
import dataclasses
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

from nncore import configio
from nncore.run import environment

from .errors import RsiError
from .io import now_iso
from .store import closes_stores

SCHEMA = "rsi/bench@1"
TASK_FIELDS = ("dataset", "n_points", "noise", "test_frac", "splitter", "seed")


@dataclasses.dataclass(frozen=True)
class Benchmark:
    id: str
    version: int
    datasets: tuple
    seeds: tuple = (0, 1, 2)
    steps: int = 3000
    n_points: int = 600
    noise: float = 0.05
    test_frac: float = 0.2
    splitter: str = "random"
    fresh_points: int = 2000
    threshold: float = 0.90
    metric: str = "fresh_acc"
    title: str = ""
    hash_id: str = ""  # the id the definition was first published (and hashed) under

    def definition(self):
        """Everything that decides the numbers (hashed); title and display id excluded."""
        d = dataclasses.asdict(self)
        d.pop("title")
        d["id"] = d.pop("hash_id") or self.id
        d["datasets"], d["seeds"] = list(self.datasets), list(self.seeds)
        return d

    @property
    def hash(self):
        return hashlib.sha256(configio.canonical_json(self.definition()).encode()).hexdigest()[:12]

    def data(self, dataset, seed):
        """The task fields of one cell."""
        return {"dataset": dataset, "n_points": self.n_points, "noise": self.noise, "test_frac": self.test_frac,
                "splitter": self.splitter, "seed": int(seed)}

    def to_dict(self):
        return {**self.definition(), "id": self.id, "title": self.title, "hash": self.hash,
                "aliases": [a for a, t in ALIASES.items() if t == self.id],
                "n_cells": len(self.datasets) * len(self.seeds)}


# Frozen copies: these lists never follow later edits of nncore SUITES.
_GENERAL_V1 = ("Bullseye (5 rings)", "Sectors (8 wedges)", "Egg crate (3)", "Checkerboard", "Hex tiling (3, coarse)",
               "Random lines (parity)", "Voronoi (6)", "Spiral (3 arms)", "Blobs + tiny cluster", "Mandelbrot",
               "Yin-yang", "Smiley")  # == SUITES["general"] on 2026-10-05
_CLASSIC_V1 = ("XOR quadrants", "Spiral (2 arms)", "Spiral (3 arms)", "Spiral (4 arms)", "Spiral (5 arms)", "Circles",
               "Moons", "Checkerboard", "Gaussian blobs (5)")  # == SUITES["classic"] on 2026-10-05
NAME = "FeatherBench"
DEFAULT = "featherbench-general-v1"
BENCHMARKS = {
    "featherbench-general-v1": Benchmark("featherbench-general-v1", 1, _GENERAL_V1, hash_id="general-v1",
                                         title="FeatherBench general: 12 pattern families, 3 seeds"),
    "featherbench-classic-v1": Benchmark("featherbench-classic-v1", 1, _CLASSIC_V1, hash_id="classic-v1",
                                         title="FeatherBench classic: the original playground datasets, 3 seeds"),
    "featherbench-quick-v1": Benchmark("featherbench-quick-v1", 1, ("XOR quadrants", "Circles", "Moons", "Spiral (2 arms)"),
                                       seeds=(0,), steps=1000, fresh_points=1000, hash_id="quick-v1",
                                       title="FeatherBench quick: smoke test, 4 easy datasets, 1 seed, 1000 steps"),
}
ALIASES = {"general-v1": "featherbench-general-v1", "classic-v1": "featherbench-classic-v1",
           "quick-v1": "featherbench-quick-v1"}  # the ids the v1 benchmarks were published under


def canonical_id(bid):
    """A benchmark id or alias -> its FeatherBench id (unknown ids pass through)."""
    return ALIASES.get(bid, bid)


def get_benchmark(bid):
    if isinstance(bid, Benchmark):
        return bid
    b = BENCHMARKS.get(canonical_id(bid))
    if b is None:
        import difflib
        raise RsiError(f"unknown benchmark '{bid}'", "E_UNKNOWN_PART", field="benchmark", value=bid,
                       did_you_mean=difflib.get_close_matches(str(bid), [*BENCHMARKS, *ALIASES], n=3),
                       allowed=list(BENCHMARKS), hint="python -m rsi bench list")
    missing = [d for d in b.datasets if not configio.PART_REGISTRIES["dataset"].has(d)]
    if missing:
        raise RsiError(f"benchmark {bid} needs datasets this nncore lacks: {', '.join(missing)}", "E_UNSUPPORTED",
                       value=missing)
    return b


# ---------------------------------------------------------------- scoring
_NC = {}


def n_classes(cfg_dict):
    """Classes in a config's dataset (built once per task, cached)."""
    key = tuple(cfg_dict.get(k) for k in TASK_FIELDS)
    if key not in _NC:
        from nncore.session import Session
        try:
            cfg = configio.config_from_dict({k: cfg_dict[k] for k in TASK_FIELDS if k in cfg_dict})[0]
            _NC[key] = Session.__new__(Session)._build_data(cfg)["n_classes"]
        except Exception:
            _NC[key] = 2
    return _NC[key]


def cell_acc(cell, metric, chance):
    """The cell's accuracy for scoring: its metric if it trained, else chance."""
    if cell is None or cell.get("status") not in ("ok", "early_stopped"):
        return chance
    split, _, m = metric.partition("_")
    v = (cell.get(split) or {}).get(m)
    if v is None and split == "test":
        v = (cell.get("train") or {}).get(m)
    return chance if v is None else v


def score(bench, cells):
    """Per-dataset and overall scores. cells: {(dataset, seed): CellResult}."""
    per = []
    for d in bench.datasets:
        cs = [cells.get((d, s)) for s in bench.seeds]
        chance = 1.0 / n_classes(bench.data(d, bench.seeds[0]))
        accs = [cell_acc(c, bench.metric, chance) for c in cs]
        tests = [(c.get("test") or {}).get("acc") for c in cs if c and c.get("status") in ("ok", "early_stopped")]
        tests = [t for t in tests if t is not None]
        params = [c["model"]["n_params"] for c in cs if c and c.get("model")]
        mean = statistics.fmean(accs)
        per.append({"dataset": d, "fresh_acc_mean": mean, "fresh_acc_min": min(accs),
                    "test_acc_mean": statistics.fmean(tests) if tests else None,
                    "n_params": max(params) if params else None, "solved": mean >= bench.threshold,
                    "statuses": [c.get("status") if c else "missing" for c in cs]})
    return summarize_rows(per, bench)


def summarize_rows(per, bench):
    n_solved = sum(r["solved"] for r in per)
    params = [r["n_params"] for r in per]
    pmax = max(params) if params and all(p is not None for p in params) else None
    mean_acc = statistics.fmean(r["fresh_acc_mean"] for r in per) if per else None
    solved_all = bool(per) and n_solved == len(per)
    return {"per_dataset": per, "n_datasets": len(per), "n_solved": n_solved, "solved_all": solved_all,
            "params_max": pmax, "params_total": sum(p for p in params if p is not None),
            "mean_acc": mean_acc, "min_acc": min((r["fresh_acc_mean"] for r in per), default=None),
            "rank_key": rank_key(solved_all, n_solved, pmax, mean_acc), "threshold": bench.threshold}


def rank_key(solved_all, n_solved, params_max, mean_acc):
    """Ascending sort key: solved_all first; then fewest params_max (solved) or most
    datasets solved (unsolved); then higher mean_acc; then fewer params."""
    p = params_max if params_max is not None else 10 ** 12
    acc = -(mean_acc or 0.0)
    return [0, p, acc, 0] if solved_all else [1, -n_solved, acc, p]


def scalar(summary):
    """A scalar that orders like the rank key (for evolve's 'best' and stall logic):
    unsolved in [0, n_datasets), solved in (n_datasets, n_datasets + 1], fewer params higher."""
    n = summary["n_datasets"]
    if summary["solved_all"]:
        import math
        return n + 1 - math.log10(1 + (summary["params_max"] or 0)) / 12
    return summary["n_solved"] + 0.999 * (summary["mean_acc"] or 0.0)


# ---------------------------------------------------------------- commands
def list_benchmarks():
    return {"name": NAME, "benchmarks": [b.to_dict() for b in BENCHMARKS.values()], "default": DEFAULT,
            "aliases": dict(ALIASES),
            "objective": f"{NAME}: fewest params that solves every pattern wins (bench rank)"}


@closes_stores
def bench(config=None, benchmark=DEFAULT, workers=None, submit=None, *, overrides=None, steps=None, cache=True,
          store=None, parts=None, on_event=None):
    """Run a recipe on a benchmark; returns the submission (rsi/bench@1), also written to `submit`.
    steps overrides the benchmark's budget: the result is then unofficial (another definition hash)."""
    from . import api
    from .pool import Evaluator, cell_specs
    api.setup_parts(parts)
    b = get_benchmark(benchmark or DEFAULT)
    official = steps is None or int(steps) == b.steps
    if not official:
        if int(steps) < 1:
            raise RsiError("--steps must be >= 1", "E_USAGE", field="steps", value=steps)
        b = dataclasses.replace(b, steps=int(steps))
        api.warn("W_UNOFFICIAL", f"--steps {steps} overrides the benchmark's budget: an unofficial result "
                                 f"(definition hash {b.hash})", field="steps")
    cfg, _, pinned = api.resolve_config(config, overrides)
    fixed = [f for f in pinned if f in TASK_FIELDS]
    if fixed:
        api.warn("W_TASK_OVERRIDDEN", f"the benchmark fixes {', '.join(fixed)}; your values are ignored",
                 fields=fixed)
    api.inert_warnings(cfg, pinned)
    # the recipe as trained: the benchmark's data fields, but its own dataset / seed (set per cell)
    own = configio.config_to_dict(cfg)
    recipe = configio.config_from_dict({**own, **b.data(b.datasets[0], b.seeds[0]),
                                        "dataset": own["dataset"], "seed": own["seed"]})[0]
    specs = cell_specs(recipe, datasets=list(b.datasets), seeds=list(b.seeds), steps=b.steps,
                       fresh_points=b.fresh_points)
    t0 = time.perf_counter()
    st = api.open_store(store)
    with Evaluator(workers, None, store=st, cache=cache) as ev:
        def done(i, cell):
            api._emit(on_event, "cell", i=i, n=len(specs), dataset=cell.get("dataset"), seed=cell.get("seed"),
                      status=cell.get("status"), fresh_acc=(cell.get("fresh") or {}).get("acc"),
                      cached=cell.get("cached"))
        cells = ev.map(specs, on_result=done)
        code_fp = ev.fp_for(recipe)
    by = {(c["dataset"], c["seed"]): c for c in cells if c}
    res = score(b, by)
    trial = api.make_trial(recipe, cells, budget=api._budget(b.steps, 0, None, None, b.fresh_points, ()),
                           seeds=list(b.seeds), datasets=list(b.datasets), objective="fresh_acc:mean",
                           tags=[f"bench:{b.id}"], code_fp=code_fp)
    st.put_trial(trial)
    env = environment()
    doc = {"format": "rsi/bench", "version": 1, "benchmark": {"id": b.id, "version": b.version, "hash": b.hash,
                                                              "definition": b.definition(), "official": official},
           "submitted": now_iso(), "config": configio.config_to_dict(recipe), "config_key": configio.config_key(recipe),
           "config_diff": configio.config_diff(recipe, configio.config_from_dict(
               {**configio.config_to_dict(configio.Config()), **b.data(b.datasets[0], b.seeds[0]),
                "dataset": configio.Config().dataset, "seed": configio.Config().seed})[0]),
           "describe": next((c["model"]["describe"] for c in cells if c and c.get("model")), None),
           **res,
           "cells": [{"dataset": c["dataset"], "seed": c["seed"], "status": c["status"],
                      "fresh_acc": (c.get("fresh") or {}).get("acc"), "test_acc": (c.get("test") or {}).get("acc"),
                      "n_params": (c.get("model") or {}).get("n_params"), "seconds": c.get("seconds")}
                     for c in cells if c],
           "environment": {"platform": platform.platform(), "python": env["python"], "torch": env["torch"],
                           "nncore": env["nncore"], "rsi": api.__version__, "code_fp": code_fp, "parts": env["parts"]},
           "timings": {"wall_s": round(time.perf_counter() - t0, 3),
                       "cell_seconds": round(sum(c.get("seconds") or 0 for c in cells if c), 3),
                       "n_cells": len(cells), "n_cached": sum(1 for c in cells if c and c.get("cached"))},
           "trial_id": trial["id"], "submission_file": None}
    if submit:
        from .rundir import atomic_write_json
        doc["submission_file"] = str(submit)
        atomic_write_json(Path(submit), doc)
    return doc


def _load(f):
    try:
        with open(f, encoding="utf-8") as fh:
            doc = json.load(fh)
    except FileNotFoundError:
        raise RsiError(f"submission not found: {f}", "E_NOT_FOUND", value=str(f)) from None
    except (OSError, ValueError) as e:
        raise RsiError(f"cannot read submission {f}: {e}", "E_BAD_SETTINGS", value=str(f)) from None
    if isinstance(doc, dict) and doc.get("schema") == SCHEMA and isinstance(doc.get("result"), dict):
        doc = doc["result"]  # a saved `rsi bench` envelope works too
    if not isinstance(doc, dict) or doc.get("format") != "rsi/bench" or not isinstance(doc.get("benchmark"), dict):
        raise RsiError(f"{f} is not an rsi/bench submission", "E_BAD_SETTINGS", value=str(f))
    return doc


def rank(files, *, top=None):
    """rsi/bench-leaderboard@1 over submission files (one benchmark definition)."""
    docs = [(str(f), _load(f)) for f in files]
    ids = {(canonical_id(d["benchmark"].get("id")), d["benchmark"].get("hash")) for _, d in docs}  # old ids rank too
    if len(ids) > 1:
        raise RsiError(f"submissions are for different benchmarks: {sorted(map(str, ids))}", "E_BAD_SETTINGS",
                       value=sorted(map(str, ids)), hint="rank each benchmark id / definition hash separately")
    bid, bhash = ids.pop()
    local = BENCHMARKS.get(bid)
    entries = []
    for f, d in docs:
        per = d.get("per_dataset") or []
        thr = d["benchmark"].get("definition", {}).get("threshold", d.get("threshold", 0.9))
        rows = [{**r, "solved": r.get("fresh_acc_mean") is not None and r["fresh_acc_mean"] >= thr} for r in per]
        check = summarize_rows(rows, Benchmark(bid, 0, (), threshold=thr))
        consistent = all(check[k] == d.get(k) for k in ("n_solved", "solved_all", "params_max"))
        env = d.get("environment") or {}
        entries.append({"file": f, "solved_all": check["solved_all"], "n_solved": check["n_solved"],
                        "n_datasets": check["n_datasets"], "params_max": check["params_max"],
                        "params_total": check["params_total"], "mean_acc": check["mean_acc"],
                        "min_acc": check["min_acc"], "rank_key": check["rank_key"], "consistent": consistent,
                        "config_key": d.get("config_key"), "config_diff": d.get("config_diff"),
                        "describe": d.get("describe"), "submitted": d.get("submitted"),
                        "platform": env.get("platform"), "code_fp": env.get("code_fp"), "torch": env.get("torch")})
    entries.sort(key=lambda e: (not e["consistent"], e["rank_key"], e["submitted"] or "", e["file"]))
    for i, e in enumerate(entries, 1):
        e["rank"] = i
    return {"benchmark": {"id": bid, "hash": bhash, "known": local is not None,
                          "matches_local": bool(local) and local.hash == bhash},
            "n_submissions": len(entries), "entries": entries[:int(top)] if top else entries,
            "order": "solved_all, then fewest params_max (solved) / most datasets solved (unsolved), then mean_acc"}
