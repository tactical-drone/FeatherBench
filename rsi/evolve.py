"""
evolve: evolutionary search over the levers (SPEC section 6), resumable and
deterministic for any --workers value.

    python -m rsi evolve --set dataset="Spiral (5 arms)" --generations 20 --background --name s5a
    rsi.evolve(overrides={"model": "mlp"}, pop=8, generations=3, steps=500, name="t")

Each generation is built completely (one random.Random(search_seed) in this
process), then its cells (genome x dataset x seed) go through the console
Evaluator (cache, worker pool). Scores follow 6.5; promising genomes are
escalated to more seeds; the hall of fame keeps the best 50; the run dir holds
evals.jsonl (every cell), generations.jsonl, state.json (checkpoint after every
generation), leaderboard.json, best.settings.json and genomes/<id>.json.
At the end the top genomes are re-scored on disjoint holdout seeds.

Objectives: scalar (default), pareto (NSGA-II on accuracy vs log2 params) and
bench:<id> (fewest params that solve the benchmark; see rsi/bench.py).

Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md.
"""
import copy
import dataclasses
import math
import random
import statistics
import time
from pathlib import Path

from nncore import configio, schema
from nncore.registry import loaded_plugins
from nncore.run import RunSpec, code_fingerprint, environment

from . import pareto as par
from . import rundir as rd
from .errors import RsiError, as_rsi_error
from .io import now_iso
from .bench import get_benchmark, n_classes
from .bench import scalar as bench_scalar
from .bench import score as bench_score
from .space import Space
from .store import closes_stores

HOF_SIZE = 50
LEADERBOARD_SIZE = 20
RESUMABLE = ("generations", "time", "max_evals", "stall", "max_seconds", "holdout_seeds", "holdout_top",
             "holdout_every")


@dataclasses.dataclass
class EvolveSettings:
    """One field per evolve flag (SPEC 6.9), same names and defaults. None = derived:
    eval_every -> max(1, steps // 20); seeds -> [0, 1, 2]; max_params / max_est_seconds ->
    the space's constraints (20000 / 120)."""
    steps: int = 3000
    eval_every: int | None = None
    seeds: list | None = None
    n_seeds: int | None = None
    datasets: object = None
    suite: str | None = None
    max_seconds: float | None = None
    early_stop: object = None
    fresh_points: int = 0
    metrics: list = dataclasses.field(default_factory=list)
    tags: list = dataclasses.field(default_factory=list)
    models: list | None = None
    genes: list | None = None
    freeze: list | None = None
    unfreeze: list | None = None
    pop: int = 24
    generations: int = 20
    elite: int = 2
    tournament: int = 3
    p_crossover: float = 0.7
    mutations: float = 1.5
    immigrants: float = 0.1
    init: str = "mixed"
    seed_config: list = dataclasses.field(default_factory=list)
    warm_start: list = dataclasses.field(default_factory=list)
    max_seeds: int = 5
    escalate_top: float = 0.25
    escalate_margin: float = 0.01
    holdout_seeds: list = dataclasses.field(default_factory=lambda: [1000, 1001, 1002, 1003, 1004])
    holdout_top: int = 5
    holdout_every: int = 0
    metric: str = "test_acc"
    seed_agg: str = "mean"
    reduce: str = "mean"
    std_weight: float = 0.0
    param_penalty: float = 0.005
    param_ref: float = 64.0
    max_params: int | None = None
    time_penalty: float = 0.0
    success_threshold: float = 0.9
    objective: str = "scalar"
    fidelity: list | None = None
    gp_activations: bool = False
    gp_max_depth: int = 4
    gp_max_nodes: int = 15
    gp_rate: float = 0.15
    gp_learnable_consts: bool = False
    search_seed: int = 0
    time: float | None = None
    max_evals: int | None = None
    stall: int = 8
    max_est_seconds: float | None = None
    cache_scope: str = "code"

    def to_dict(self):
        return configio.json_safe(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in dataclasses.fields(cls)}
        bad = [k for k in d if k not in names]
        if bad:
            import difflib
            raise RsiError(f"unknown evolve setting '{bad[0]}'", "E_USAGE", field=bad[0],
                           did_you_mean=difflib.get_close_matches(bad[0], sorted(names), n=3))
        return cls(**{k: copy.deepcopy(v) for k, v in d.items()})


DEFAULTS = EvolveSettings().to_dict()


def _check_settings(s):
    def need(ok, msg, field):
        if not ok:
            raise RsiError(msg, "E_USAGE", field=field, value=getattr(s, field, None))
    need(s.pop >= 2, "--pop must be >= 2", "pop")
    need(s.generations >= 0, "--generations must be >= 0", "generations")
    need(0 <= s.elite < s.pop, "--elite must be in 0..pop-1", "elite")
    need(s.tournament >= 1, "--tournament must be >= 1", "tournament")
    need(0 <= s.p_crossover <= 1, "--p-crossover must be in [0, 1]", "p_crossover")
    need(s.mutations >= 1, "--mutations must be >= 1", "mutations")
    need(0 <= s.immigrants < 1, "--immigrants must be in [0, 1)", "immigrants")
    need(s.init in ("mixed", "base", "random"), "--init must be mixed, base or random", "init")
    need(s.metric in ("test_acc", "train_acc", "fresh_acc", "test_loss", "train_loss"), "bad --metric", "metric")
    need(s.seed_agg in ("mean", "median", "min", "cvar50"), "bad --seed-agg", "seed_agg")
    need(s.reduce in ("mean", "min"), "bad --reduce", "reduce")
    need(s.objective in ("scalar", "pareto") or str(s.objective).startswith("bench:"),
         "--objective must be scalar, pareto or bench:<id>", "objective")
    need(s.cache_scope in ("code", "config"), "--cache-scope must be code or config", "cache_scope")
    need(int(s.steps) >= 1, "--steps must be >= 1", "steps")
    need(s.stall >= 1, "--stall must be >= 1", "stall")
    need(0 < s.escalate_top <= 1, "--escalate-top must be in (0, 1]", "escalate_top")
    need(s.param_ref > 0, "--param-ref must be > 0", "param_ref")
    need(int(s.fresh_points or 0) >= 0, "--fresh-points must be >= 0", "fresh_points")
    need(s.max_seconds is None or s.max_seconds > 0, "--max-seconds must be > 0", "max_seconds")


# ---------------------------------------------------------------- task
class Task:
    """What every genome is scored on: datasets x seeds at a fixed budget (and, for
    bench:<id>, the benchmark's data fields)."""

    def __init__(self, datasets, steps, *, eval_every=0, max_seconds=None, early_stop=None, fresh_points=0,
                 metrics=(), bench=None):
        self.datasets, self.steps, self.eval_every = list(datasets), int(steps), int(eval_every or 0)
        self.max_seconds, self.early_stop, self.fresh_points = max_seconds, early_stop, int(fresh_points or 0)
        self.metrics, self.bench = tuple(metrics or ()), bench

    def data(self, dataset, seed):
        return self.bench.data(dataset, seed) if self.bench else {"dataset": dataset, "seed": int(seed)}

    def spec(self, pheno, dataset, seed, steps=None):
        from nncore.run import EarlyStop
        es = self.early_stop
        if isinstance(es, (list, tuple)):
            es = EarlyStop(int(es[0]), float(es[1]))
        elif isinstance(es, dict):
            es = EarlyStop(int(es["at_step"]), float(es["min_test_acc"]))
        return RunSpec(config={**pheno, **self.data(dataset, seed)}, steps=int(steps or self.steps),
                       eval_every=self.eval_every, max_seconds=self.max_seconds, early_stop=es,
                       fresh_points=self.fresh_points, extra_metrics=self.metrics)


# ---------------------------------------------------------------- fitness (6.5)
def _agg(vals, how):
    if how == "mean":
        return statistics.fmean(vals)
    if how == "median":
        return statistics.median(vals)
    if how == "min":
        return min(vals)
    worst = sorted(vals)[:max(1, math.ceil(len(vals) / 2))]  # cvar50: mean of the worse half
    return statistics.fmean(worst)


def cell_value(cell, metric, n_classes):
    """v_c: the metric of an ok / early_stopped cell (losses negated; test -> train when there
    is no test split); chance (1 / n_classes, or -ln n_classes for losses) otherwise."""
    loss = metric.endswith("_loss")
    chance = -math.log(max(1, n_classes)) if loss else 1.0 / max(1, n_classes)
    if cell is None or cell.get("status") not in ("ok", "early_stopped"):
        return chance
    split, _, m = metric.partition("_")
    v = (cell.get(split) or {}).get(m)
    if v is None and split == "test":
        v = (cell.get("train") or {}).get(m)
    if v is None or not math.isfinite(v):
        return chance
    return -v if loss else v


def fitness(cells, datasets, seeds, s, n_classes_of, n_params=None):
    """(score, fitness dict) for one genome; score None when every cell errored.
    cells: {(dataset, seed): CellResult}."""
    vals, per, sts, params, secs, used = [], {}, [], [], [], []
    for d in datasets:
        vs = []
        for sd in seeds:
            c = cells.get((d, sd))
            used += [c] if c else []
            vs.append(cell_value(c, s.metric, n_classes_of(d, sd)))
            sts.append(c.get("status") if c else "missing")
            if c and c.get("model"):
                params.append(c["model"]["n_params"])
            if c and c.get("steps_done"):
                secs.append(1000.0 * (c.get("seconds") or 0.0) / c["steps_done"])
        per[d] = _agg(vs, s.seed_agg)
        vals += vs
    A = min(per.values()) if s.reduce == "min" else statistics.fmean(per.values())
    S = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    P = max(params) if params else n_params
    pen = s.param_penalty * max(0.0, math.log2(P / s.param_ref)) if P else 0.0
    sec_k = statistics.fmean(secs) if secs else 0.0
    score = A - s.std_weight * S - pen - (s.time_penalty * sec_k if s.time_penalty else 0.0)
    if sts and all(x == "error" for x in sts):
        score = None
    fit = {"acc_agg": A, "seed_agg": s.seed_agg, "reduce": s.reduce, "acc_min": min(vals), "acc_std": S,
           "success_rate": sum(v >= s.success_threshold for v in vals) / len(vals), "n_seeds": len(seeds),
           "penalty": pen, "n_params": P,
           "seconds_mean": statistics.fmean([c.get("seconds") or 0.0 for c in used]) if used else 0.0,
           "per_dataset": per, "statuses": {k: sts.count(k) for k in sorted(set(sts))}}
    return score, fit


def bench_fitness(cells, bench, seeds, n_params=None):
    """(score, fitness) under bench:<id>: lexicographic min-params-subject-to-solve as a scalar."""
    b = dataclasses.replace(bench, seeds=tuple(seeds))
    sm = bench_score(b, cells)
    sts = [st for r in sm["per_dataset"] for st in r["statuses"]]
    score = None if sts and all(x == "error" for x in sts) else bench_scalar(sm)
    fit = {"acc_agg": sm["mean_acc"], "seed_agg": "mean", "reduce": "mean", "acc_min": sm["min_acc"],
           "acc_std": 0.0, "success_rate": sm["n_solved"] / max(1, sm["n_datasets"]), "n_seeds": len(seeds),
           "penalty": 0.0, "n_params": sm["params_max"] or n_params, "n_solved": sm["n_solved"],
           "solved_all": sm["solved_all"], "params_max": sm["params_max"], "params_total": sm["params_total"],
           "per_dataset": {r["dataset"]: r["fresh_acc_mean"] for r in sm["per_dataset"]},
           "statuses": {k: sts.count(k) for k in sorted(set(sts))}}
    return score, fit


def rank_key(gid, score, fit):
    """(score desc, A desc, min v_c desc, n_params asc, genome_id asc); failed last. (Timing is
    left out of the key so the search path never depends on the machine's speed.)"""
    if score is None or fit is None:
        return (1, 0.0, 0.0, 0.0, math.inf, gid)
    return (0, -score, -(fit.get("acc_agg") or 0.0), -(fit.get("acc_min") or 0.0),
            fit.get("n_params") or math.inf, gid)


def escalation_targets(scored, pool, top, margin, n_seeds, max_seeds, pop):
    """Genomes to escalate: score >= T - margin and fewer than max_seeds seeds, where T is the
    score of the ceil(top * pop)-th best in `pool`. scored: {gid: score}."""
    vals = sorted((scored[g] for g in set(pool) if scored.get(g) is not None), reverse=True)
    if not vals:
        return []
    T = vals[min(len(vals), max(1, math.ceil(top * pop))) - 1]
    return [g for g in dict.fromkeys(pool) if scored.get(g) is not None and scored[g] >= T - margin
            and n_seeds[g] < max_seeds]


def escalation_seeds(seeds, max_seeds):
    """The search seeds continued up to max_seeds: [0, 1, 2], 5 -> [0, 1, 2, 3, 4]."""
    out, nxt = list(seeds), max(seeds) + 1
    while len(out) < max_seeds:
        if nxt not in out:
            out.append(nxt)
        nxt += 1
    return out


# ---------------------------------------------------------------- the search
class Evolution:
    """One evolutionary search in a run dir. Evolution(...).run() -> the leaderboard doc;
    Evolution.resume(rundir, evaluator, **overrides) continues a checkpointed one."""

    def __init__(self, settings, space, base, rundir, evaluator, on_event=None, *, task=None, warn=None):
        self.s, self.space, self.base, self.rd, self.ev = settings, space, base, rundir, evaluator
        self.on_event, self.warn = on_event, warn or (lambda *a, **k: None)
        self.bench = get_benchmark(settings.objective[6:]) if settings.objective.startswith("bench:") else None
        self.task = task or Task([base.dataset], settings.steps)
        self.seeds = list(settings.seeds or [0, 1, 2])
        self.max_seeds = max(len(self.seeds), int(settings.max_seeds))
        self.esc_seeds = escalation_seeds(self.seeds, self.max_seeds)
        self.hseeds = list(settings.holdout_seeds or [])
        self.max_params = settings.max_params or space.constraints.get("max_params", 20000)
        self.max_est = settings.max_est_seconds or space.constraints.get("max_est_seconds", 120)
        if settings.gp_activations:
            space.gp = {"rate": settings.gp_rate, "max_depth": settings.gp_max_depth,
                        "max_nodes": settings.gp_max_nodes, "learnable": settings.gp_learnable_consts}
        self.rng = random.Random(settings.search_seed)
        self.records = {}      # gid -> {genome, config, cells, hcells, lcells, n_seeds, lofi, parents, ops, generation}
        self.population, self.hof, self.evaluated = [], [], {}
        self.gen, self.phase, self.stall_n, self.best_prev = -1, "evolve", 0, None
        self.counters = {"evals": 0, "cached": 0, "failed": 0, "rejected": 0, "elapsed_s": 0.0}
        self.holdout_done, self.reason, self.mode, self.reeval = False, None, None, False
        self._valid, self._ms = {}, None
        self.base_gid = space.genome_id(space.encode(base))
        self.t0 = time.time()

    # ---------------------------------------------------------------- helpers
    def emit(self, event, **f):
        if self.on_event is not None:
            try:
                self.on_event(event, **f)
            except Exception:
                pass

    def elapsed(self):
        return self.counters["elapsed_s"] + (time.time() - self.t0)

    def rec(self, gid, genome=None):
        r = self.records.get(gid)
        if r is None:
            r = self.records[gid] = {"genome": None, "config": None, "cells": {}, "hcells": {}, "lcells": {},
                                     "n_seeds": self.evaluated.get(gid, [len(self.seeds)])[0], "lofi": False,
                                     "parents": [], "ops": [], "generation": None}
        if genome is not None and r["genome"] is None:
            r["genome"] = copy.deepcopy(genome)
            r["config"] = self.space.phenotype(genome)
        return r

    def _ncls(self, d, s):
        return n_classes({**self.base_dict(), **self.task.data(d, s)})

    def base_dict(self):
        return self.space.base_dict

    def score(self, gid, holdout=False):
        r = self.records[gid]
        if holdout:
            cells, seeds = r["hcells"], self.hseeds
        elif r["lofi"]:
            cells, seeds = r["lcells"], self.seeds
        else:
            cells, seeds = r["cells"], self.esc_seeds[:r["n_seeds"]]
        if self.bench:
            return bench_fitness(cells, self.bench, seeds, self._valid.get(gid, (None, None, None))[2])
        return fitness(cells, self.task.datasets, seeds, self.s, self._ncls, self._valid.get(gid, (None, None, None))[2])

    def key(self, gid):
        sc, fit = self.score(gid)
        return rank_key(gid, sc, fit)

    def ranked(self, gids):
        return sorted(dict.fromkeys(gids), key=self.key)

    # ---------------------------------------------------------------- validity
    def ms_per_step(self, model):
        if self._ms is None:
            env = rd.read_json(Path(self.ev.store.root) / "env.json", {}) if self.ev.store is not None else {}
            self._ms = {**schema.COST_MS_PER_STEP, **((env or {}).get("ms_per_step") or {})}
        return self._ms.get(model, 2.4)

    def check(self, genome):
        """(ok, reason, n_params) before training: check_config, ui_safe, count_params on the first
        dataset (RNG-safe), max_params, estimated seconds."""
        gid = self.space.genome_id(genome)
        if gid in self._valid:
            return self._valid[gid]
        pheno = self.space.phenotype(genome)
        out = (True, "ok", None)
        try:
            cfg = configio.config_from_dict({**pheno, **self.task.data(self.task.datasets[0], self.seeds[0])})[0]
            bad = self.space.ui_problems(cfg)
            if bad:
                out = (False, f"not UI exact: {bad[0]['message']}", None)
            else:
                n = schema.count_params(cfg)
                est = (self.task.steps * len(self.task.datasets) * len(self.seeds) * self.ms_per_step(cfg.model) / 1000
                       * max(1.0, n / 1000))
                if n > self.max_params:
                    out = (False, f"n_params {n} > max_params {self.max_params}", n)
                elif est > self.max_est:
                    out = (False, f"estimated {est:.0f} s > max_est_seconds {self.max_est:g}", n)
                else:
                    out = (True, "ok", n)
        except Exception as e:  # noqa: BLE001  (invalid or unbuildable phenotype)
            out = (False, f"{type(e).__name__}: {e}", None)
        self._valid[gid] = out
        return out

    # ---------------------------------------------------------------- building
    def _novel(self, genome, ops, taken):
        gid = self.space.genome_id(genome)
        for _ in range(10):
            if gid not in taken:
                break
            genome, more = self.space.mutate(genome, self.rng, n_ops=1)
            ops = ops + more
            gid = self.space.genome_id(genome)
        return genome, ops, gid

    def _add(self, out, genome, parents, ops, g, budget, novelty=True):
        """Validate (and dedupe) a candidate; append its gid to out. False when rejected."""
        taken = set(out) | set(self.hof) | set(self.evaluated)
        if novelty:
            genome, ops, gid = self._novel(genome, ops, taken)
        else:
            gid = self.space.genome_id(genome)
        ok, why, _ = self.check(genome)
        if not ok:
            self.counters["rejected"] += 1
            budget["rejected"] += 1
            self.rd.append("evals.jsonl", {"generation": g, "genome_id": gid, "phase": "search", "status": "rejected",
                                           "reason": why, "t": round(time.time(), 3)})
            return False
        r = self.rec(gid, genome)
        if r["generation"] is None:
            r.update(parents=list(parents), ops=list(ops), generation=g)
        out.append(gid)
        return True

    def _init_population(self, budget):
        s, sp, out = self.s, self.space, []
        base = sp.encode(self.base)
        r = self.rec(self.base_gid, base)
        r.update(parents=[], ops=["base"], generation=0)
        out.append(self.base_gid)
        seeds_in = []
        for f in s.seed_config or ():
            seeds_in.append((sp.encode(configio.load_settings(f).config), [f"seed-config:{Path(f).name}"]))
        for ws in s.warm_start or ():
            name, _, k = str(ws).partition(":")
            lb = rd.read_json(rd.resolve_target(name, self.ev.store.root if self.ev.store else ".") / "leaderboard.json")
            if not lb:
                raise RsiError(f"--warm-start {ws}: no leaderboard.json", "E_NOT_FOUND", value=ws)
            for e in (lb.get("entries") or [])[:int(k or 5)]:
                seeds_in.append((sp.encode(configio.config_from_dict(e["config"])[0]), [f"warm-start:{name}"]))
        for genome, ops in seeds_in:
            if len(out) < s.pop and sp.genome_id(genome) not in out:
                self._add(out, genome, [], ops, 0, budget, novelty=False)
        n_mut = {"mixed": s.pop // 2 - 1, "base": s.pop, "random": 0}[s.init]
        tries = 0
        while len(out) < s.pop and tries < 20 * s.pop and budget["rejected"] <= 3 * s.pop:
            tries += 1
            if n_mut > 0:
                genome, ops = sp.mutate(base, self.rng, n_ops=3)
                ok = self._add(out, genome, [self.base_gid], ["init:mutant"] + ops, 0, budget)
                n_mut -= ok
            else:
                self._add(out, sp.sample(self.rng), [], ["init:random"], 0, budget)
        return out

    def _tournament(self, pop, info):
        k = min(self.s.tournament, len(pop))
        pick = [pop[i] for i in self.rng.sample(range(len(pop)), k)]
        if self.s.objective == "pareto":
            return min(pick, key=lambda g: par.tournament_key(info[g], g))
        return min(pick, key=self.key)

    def _objectives(self, gids):
        out = []
        for g in gids:
            sc, fit = self.score(g)
            out.append(par.objectives(fit.get("acc_agg") if sc is not None else None, fit.get("n_params")))
        return out

    def _pareto_info(self, gids):
        return dict(zip(gids, par.ranks(self._objectives(gids))))

    def _build(self, g):
        s, budget = self.s, {"rejected": 0}
        if g == 0:
            return self._init_population(budget), budget
        out = []
        pop = [x for x in self.population if x in self.records]
        info = self._pareto_info(list(dict.fromkeys(pop))) if s.objective == "pareto" else None
        if s.objective != "pareto":
            for gid in self.ranked(pop + self.hof)[:s.elite]:
                out.append(gid)  # elites: fitness reused
        target = s.pop
        for _ in range(int(round(s.immigrants * s.pop))):
            if len(out) < target:
                self._add(out, self.space.sample(self.rng), [], ["immigrant"], g, budget)
        while len(out) < target and budget["rejected"] <= 3 * s.pop:
            p1 = self._tournament(pop, info)
            if self.rng.random() < s.p_crossover:
                p2 = self._tournament(pop, info)
                child, ops = self.space.crossover(self.records[p1]["genome"], self.records[p2]["genome"], self.rng)
                parents = [p1, p2]
            else:
                child, ops, parents = copy.deepcopy(self.records[p1]["genome"]), ["clone"], [p1]
            child, mops = self.space.mutate(child, self.rng, mutations=s.mutations)
            self._add(out, child, parents, ops + mops, g, budget)
        return out, budget

    # ---------------------------------------------------------------- evaluation
    def _stop_check(self):
        self.rd.heartbeat()
        mode = self.rd.stop_mode()
        if mode:
            self.reason, self.mode = "stopped", mode
            return True
        if self.s.time and self.elapsed() >= self.s.time:
            self.reason = "budget"
            return True
        return False

    def _map(self, jobs, phase, g, *, budgeted=True, steps=None):
        """jobs: [(gid, dataset, seed)]. Runs the missing cells; False if any was not run."""
        specs, keys = [], []
        for gid, d, sd in jobs:
            specs.append(self.task.spec(self.records[gid]["config"], d, sd, steps))
            keys.append((gid, d, sd))
        if not specs:
            return True
        max_new = None
        if budgeted and self.s.max_evals is not None:
            max_new = max(0, int(self.s.max_evals) - self.counters["evals"])
        before = dict(self.ev.stats)
        cells = self.ev.map(specs, stop_check=(self._stop_check if budgeted else
                                               lambda: self.rd.heartbeat() or self.rd.stop_mode() == "now"),
                            max_new=max_new)
        self.counters["evals"] += self.ev.stats["new"] - before["new"]
        self.counters["cached"] += self.ev.stats["cached"] - before["cached"]
        self.counters["failed"] += self.ev.stats["failed"] - before["failed"]
        store = {"search": "cells", "escalate": "cells", "holdout": "hcells", "fidelity": "lcells"}[phase]
        seen = set()
        for (gid, d, sd), cell in zip(keys, cells):
            if cell is None:
                continue
            self.records[gid][store][(d, sd)] = cell
            if (gid, d, sd) not in seen:
                seen.add((gid, d, sd))
                self.rd.append("evals.jsonl", {"generation": g, "genome_id": gid, "seed": sd, "dataset": d,
                                               "key": cell.get("key"), "phase": phase, "cell": cell,
                                               "t": round(time.time(), 3)})
        if any(c is None for c in cells):
            if self.reason is None:
                self.reason = "budget"
            return False
        return True

    def _missing(self, gids, seeds, store="cells"):
        jobs = []
        for gid in dict.fromkeys(gids):
            have = self.records[gid][store]
            for d in self.task.datasets:
                for sd in seeds:
                    if (d, sd) not in have:
                        jobs.append((gid, d, sd))
        return jobs

    def _evaluate(self, built, g):
        new = [x for x in dict.fromkeys(built) if x not in self.evaluated]
        fid = self.s.fidelity
        if fid and len(fid) > 1:
            lo = int(fid[0])
            if not self._map(self._missing(new, self.seeds, "lcells"), "fidelity", g, steps=lo):
                return False
            for x in new:
                self.records[x]["lofi"] = True
            promote = self.ranked(new)[:math.ceil(len(new) / 3)]
            for x in promote:
                self.records[x]["lofi"] = False
            todo = promote + [x for x in built if x not in new]
        else:
            todo = list(dict.fromkeys(built))
        jobs = []
        for x in todo:
            r = self.records[x]
            if not r["lofi"]:
                jobs += self._missing([x], self.esc_seeds[:r["n_seeds"]])
        return self._map(jobs, "search", g)

    def _escalate(self, pop, g):
        n_seeds = {x: (self.records[x]["n_seeds"] if not self.records[x]["lofi"] else self.max_seeds)
                   for x in set(pop) | set(self.hof)}
        scored = {x: self.score(x)[0] for x in n_seeds}
        targets = escalation_targets(scored, list(pop) + list(self.hof), self.s.escalate_top,
                                     self.s.escalate_margin, n_seeds, self.max_seeds, self.s.pop)
        targets = [x for x in targets if x in set(pop)]
        jobs = []
        for x in targets:
            jobs += [j for j in self._missing([x], self.esc_seeds[:self.max_seeds])]
        if not self._map(jobs, "escalate", g):
            return False
        for x in targets:
            self.records[x]["n_seeds"] = self.max_seeds
        return targets

    # ---------------------------------------------------------------- one generation
    def _uncommit(self, snap):
        """A generation cut short (STOP, --time, --max-evals) is not committed: put the GA rng
        and the rejected count back, so the checkpoint (and a resume) rebuild the same generation."""
        self.rng.setstate(snap[0])
        self.counters["rejected"] = snap[1]
        return False

    def _generation(self, g):
        t0 = time.time()
        snap = (self.rng.getstate(), self.counters["rejected"])
        ev0 = (self.counters["evals"], self.counters["cached"])
        built, budget = self._build(g)
        if not self._evaluate(built, g):
            return self._uncommit(snap)
        if self.s.objective == "pareto" and g > 0:
            cand = list(dict.fromkeys(self.population + built))
            pop = [cand[i] for i in par.nsga2_select(self._objectives(cand), cand, self.s.pop)]
        else:
            pop = built
        esc = self._escalate(pop, g)
        if esc is False:
            return self._uncommit(snap)
        for x in dict.fromkeys(built):
            self.evaluated[x] = [self.records[x]["n_seeds"], self.records[x]["lofi"]]
        for x in esc:
            self.evaluated[x] = [self.records[x]["n_seeds"], self.records[x]["lofi"]]
        self.population = list(pop)
        self.hof = self.ranked(self.hof + self.population)[:HOF_SIZE]
        self.gen = g
        scores = [self.score(x)[0] for x in self.population]
        ok = [v for v in scores if v is not None]
        best = self.score(self.hof[0])[0] if self.hof else None
        uniq = list(dict.fromkeys(self.population))
        gens = [self.records[x]["genome"] for x in uniq]
        div = statistics.fmean([self.space.distance(a, b) for i, a in enumerate(gens) for b in gens[i + 1:]]) \
            if len(gens) > 1 else 0.0
        line = {"generation": g, "best": best, "mean": statistics.fmean(ok) if ok else None,
                "median": statistics.median(ok) if ok else None, "unique_frac": len(uniq) / max(1, self.s.pop),
                "diversity": div, "evals": self.counters["evals"] - ev0[0], "cached": self.counters["cached"] - ev0[1],
                "rejected": budget["rejected"], "escalated": len(esc), "seconds": round(time.time() - t0, 3),
                "best_genome_id": self.hof[0] if self.hof else None,
                "best_n_params": self.score(self.hof[0])[1].get("n_params") if self.hof else None}
        self.rd.append("generations.jsonl", line)
        if best is not None and (self.best_prev is None or best >= self.best_prev + 1e-4):
            self.stall_n, self.best_prev = 0, best
        else:
            self.stall_n += 1
        self.checkpoint()
        self.write_outputs()
        self.emit("generation", generation=g, best=best, mean=line["mean"], evals=line["evals"],
                  cached=line["cached"], rejected=line["rejected"], escalated=len(esc), seconds=line["seconds"])
        if self.s.holdout_every and g % self.s.holdout_every == self.s.holdout_every - 1:
            self.holdout()
        return True

    # ---------------------------------------------------------------- holdout
    def holdout(self):
        """Score the top holdout_top of the hall of fame (and the baseline) on the holdout seeds."""
        if not self.hseeds or self.s.holdout_top <= 0 or not self.hof:
            return True
        gids = list(dict.fromkeys(self.hof[:self.s.holdout_top] + [self.base_gid]))
        gids = [x for x in gids if x in self.records and self.records[x]["config"] is not None]
        self.emit("holdout", genomes=len(gids), seeds=self.hseeds)
        self.phase = "holdout"
        self.write_progress(state="running", phase="holdout")
        ok = self._map(self._missing(gids, self.hseeds, "hcells"), "holdout", self.gen, budgeted=False)
        self.phase = "evolve"
        if ok:
            self.holdout_done = True
        self.write_outputs()
        return ok

    def holdout_of(self, gid):
        r = self.records.get(gid)
        if not r or not self.hseeds or any((d, sd) not in r["hcells"] for d in self.task.datasets for sd in self.hseeds):
            return None
        sc, fit = self.score(gid, holdout=True)
        cs = [r["hcells"][(d, sd)] for d in self.task.datasets for sd in self.hseeds]
        accs = [(c.get("test") or {}).get("acc") for c in cs if c.get("status") in ("ok", "early_stopped")]
        accs = [a for a in accs if a is not None]
        return {"seeds": list(self.hseeds), "score": sc, "acc_agg": fit.get("acc_agg"),
                "test_acc_mean": statistics.fmean(accs) if accs else None, "test_acc_min": min(accs) if accs else None}

    # ---------------------------------------------------------------- outputs
    def objective_text(self):
        s = self.s
        if self.bench:
            return s.objective
        if s.objective == "pareto":
            return "pareto(acc_agg, -log2 n_params)"
        t = f"{s.metric}:{s.seed_agg}"
        if s.param_penalty:
            t += f"-{s.param_penalty:g}*log2(P/{s.param_ref:g})"
        if s.std_weight:
            t += f"-{s.std_weight:g}*std"
        if s.time_penalty:
            t += f"-{s.time_penalty:g}*sec_per_1k"
        return t

    def export_config(self, gid):
        """The genome's config; under bench:<id> with the benchmark's data fields (n_points, noise,
        split), but the recipe's own dataset and seed (the benchmark sets those per cell)."""
        cfg = self.records[gid]["config"]
        if not self.bench:
            return dict(cfg)
        return {**cfg, **self.task.data(self.task.datasets[0], self.base.seed), "dataset": cfg["dataset"],
                "seed": cfg["seed"]}

    def entry(self, gid, rank):
        r = self.records[gid]
        sc, fit = self.score(gid)
        cfg = self.export_config(gid)
        cell = next((c for c in r["cells"].values() if c.get("model")), None)
        sf = self.rd.path / "genomes" / f"{gid}.json"
        n = len(self.esc_seeds[:r["n_seeds"]])
        seeds = f"{self.hseeds[0]}-{self.hseeds[-1]}" if self.hseeds else f"0-{n - 1}"
        return {"rank": rank, "genome_id": gid, "trial_id": r.get("trial_id"), "score": sc,
                "fitness": {k: v for k, v in fit.items() if k != "statuses"}, "holdout": self.holdout_of(gid),
                "genome": {n_: r["genome"][n_] for n_ in self.space.active(r["genome"])},
                "config_diff": configio.config_diff(cfg, self.space.base_dict), "config": cfg,
                "describe": (cell or {}).get("model", {}).get("describe"), "generation_found": r["generation"],
                "lineage": {"parents": r["parents"], "ops": r["ops"]}, "settings_file": str(sf),
                "parts": loaded_plugins(),
                "repro": f"python -m rsi run --config {sf} --seeds {seeds} --steps {self.task.steps}"}

    def board(self):
        """Leaderboard order: the hall of fame's top 20, by holdout score when any has one
        (the rest follow in score order). Rank 1 is best.settings.json and status 'best'."""
        top = self.hof[:LEADERBOARD_SIZE]
        ho = {x: (self.holdout_of(x) or {}).get("score") for x in top}
        if any(v is not None for v in ho.values()):
            top = sorted(top, key=lambda x: (ho[x] is None, -(ho[x] or 0.0), top.index(x)))
        return top

    def leaderboard(self, state=None):
        entries = [self.entry(x, i) for i, x in enumerate(self.board(), 1)]
        base = None
        if self.base_gid in self.records:
            sc, _ = self.score(self.base_gid)
            base = {"genome_id": self.base_gid, "score": sc,
                    "holdout_score": (self.holdout_of(self.base_gid) or {}).get("score")}
        rows = []
        for x in self.evaluated:
            if x in self.records:
                sc, fit = self.score(x)
                if sc is not None:
                    rows.append({"genome_id": x, "acc_agg": fit.get("acc_agg"), "n_params": fit.get("n_params")})
        return {"name": self.rd.name, "kind": "evolve", "state": state or "running",
                "sort": "holdout" if any(e["holdout"] for e in entries) else "score", "objective": self.objective_text(),
                "baseline": base, "entries": entries, "pareto": par.front(rows),
                "stats": {"generations_done": self.gen + 1, "evals_done": self.counters["evals"],
                          "evals_cached": self.counters["cached"], "evals_failed": self.counters["failed"],
                          "rejected": self.counters["rejected"], "unique_genomes": len(self.evaluated),
                          "elapsed_s": round(self.elapsed(), 2)}}

    def write_outputs(self, state=None):
        lb = self.leaderboard(state)
        parts = self._parts_block()
        for e in lb["entries"]:
            cfg = configio.config_from_dict(e["config"])[0]
            meta = {"saved_by": "rsi evolve", "created": now_iso(), "run": str(self.rd.path),
                    "genome_id": e["genome_id"], "rank": e["rank"], "objective": lb["objective"], "score": e["score"],
                    "holdout_score": (e["holdout"] or {}).get("score"), "parts": loaded_plugins()}
            c0 = self.records[e["genome_id"]]["cells"].get((e["config"]["dataset"], self.base.seed))
            if c0 and c0.get("status") in ("ok", "early_stopped"):
                meta["expect"] = {"train_acc": (c0.get("train") or {}).get("acc"),
                                  "test_acc": (c0.get("test") or {}).get("acc"), "seed": self.base.seed,
                                  "at_step": self.task.steps, "code_fp": self.ev.code_fp,
                                  "platform": environment()["platform"]}
            kw = {"parts": parts} if parts else {}
            configio.save_settings(self.rd.path / "genomes" / f"{e['genome_id']}.json", cfg,
                                   train={"steps": self.task.steps, "eval_every": max(1, self.task.steps // 20)},
                                   meta=meta, **kw)
            if e["rank"] == 1:
                configio.save_settings(self.rd.path / "best.settings.json", cfg,
                                       train={"steps": self.task.steps, "eval_every": max(1, self.task.steps // 20)},
                                       meta=meta, **kw)
        self.rd.write_json("leaderboard.json", lb)
        return lb

    def _parts_block(self):
        if not self.s.gp_activations:
            return None
        from . import gp
        out = {}
        for x in self.hof[:LEADERBOARD_SIZE]:
            cfg = self.records[x]["config"]
            for f in ("activation", "act_decide", "act_relate", "act_prepare"):
                if gp.is_gp(cfg.get(f)):
                    out[cfg[f]] = {"kind": "gp_activation", "infix": gp.to_infix(gp.parse(cfg[f]))}
        return out or None

    def write_progress(self, **f):
        best = None
        if self.hof:
            x = self.board()[0]  # == leaderboard rank 1 and best.settings.json
            sc, fit = self.score(x)
            r = self.records[x]
            best = {"genome_id": x, "score": sc, "holdout_score": (self.holdout_of(x) or {}).get("score"),
                    "acc_agg": fit.get("acc_agg"), "n_seeds": fit.get("n_seeds"), "n_params": fit.get("n_params"),
                    "describe": next((c["model"]["describe"] for c in r["cells"].values() if c.get("model")), None),
                    "config_diff": configio.config_diff(self.export_config(x), self.space.base_dict),
                    "ranked_by": "holdout" if best_holdout(self, x) else "score"}
            if self.hof[0] != x:
                best["best_by_score"] = {"genome_id": self.hof[0], "score": self.score(self.hof[0])[0]}
        el = self.elapsed()
        done = self.gen + 1
        eta = el / done * (self.s.generations - done) if done else None
        if eta is not None and self.s.time:
            eta = min(eta, max(0.0, self.s.time - el))
        p = self.rd.path
        self.rd.write_progress(kind="evolve", generation=self.gen, generations=self.s.generations,
                               evals_done=self.counters["evals"], evals_cached=self.counters["cached"],
                               evals_failed=self.counters["failed"], rejected=self.counters["rejected"],
                               max_evals=self.s.max_evals, elapsed_s=round(el, 1), time_budget_s=self.s.time,
                               eta_s=round(eta, 1) if eta is not None else None, workers=self.ev.workers, best=best,
                               files={"best_settings": str(p / "best.settings.json"),
                                      "leaderboard": str(p / "leaderboard.json"), "log": str(p / "log.txt")},
                               **f)

    # ---------------------------------------------------------------- checkpoint
    def checkpoint(self):
        keep = set(self.population) | set(self.hof) | {self.base_gid}
        genomes = {x: {"genome": self.records[x]["genome"], "parents": self.records[x]["parents"],
                       "ops": self.records[x]["ops"], "generation": self.records[x]["generation"]}
                   for x in keep if x in self.records and self.records[x]["genome"] is not None}
        st = self.rng.getstate()
        self.rd.write_json("state.json", {
            "format": "rsi/evolve-state", "version": 1, "generation": self.gen, "phase": self.phase,
            "rng": [st[0], list(st[1]), st[2]], "population": self.population, "hof": self.hof,
            "genomes": genomes, "evaluated": self.evaluated, "stall": self.stall_n, "best_prev": self.best_prev,
            "counters": {**self.counters, "elapsed_s": round(self.elapsed(), 3)}, "holdout_done": self.holdout_done,
            "gp_seen": self.space._gp_seen, "code_fp": self.ev.code_fp})

    def restore(self, state, journal):
        self.gen, self.phase = state["generation"], state.get("phase", "evolve")
        r = state["rng"]
        self.rng.setstate((r[0], tuple(r[1]), r[2]))
        self.population, self.hof = list(state["population"]), list(state["hof"])
        self.evaluated = {k: list(v) for k, v in state["evaluated"].items()}
        self.stall_n, self.best_prev = state.get("stall", 0), state.get("best_prev")
        self.counters.update(state.get("counters") or {})
        self.holdout_done = state.get("holdout_done", False)
        self.space._gp_seen = dict(state.get("gp_seen") or {})
        for gid, info in state["genomes"].items():
            rec = self.rec(gid, info["genome"])
            rec.update(parents=info["parents"], ops=info["ops"], generation=info["generation"])
        for gid, (n, lofi) in self.evaluated.items():
            r = self.rec(gid)
            r["n_seeds"], r["lofi"] = n, lofi
        store = {"search": "cells", "escalate": "cells", "holdout": "hcells", "fidelity": "lcells"}
        marks = [i for i, line in enumerate(journal) if "code_change" in line]
        for line in journal[marks[-1] + 1 if marks else 0:]:
            cell, gid = line.get("cell"), line.get("genome_id")
            if not cell or line.get("phase") not in store:
                continue
            if self.ev.cache and cell.get("key") and cell.get("status") in ("ok", "diverged", "early_stopped"):
                self.ev.mem.setdefault(cell["key"], cell)
            if gid in self.records:
                self.records[gid][store[line["phase"]]][(line["dataset"], line["seed"])] = cell

    # ---------------------------------------------------------------- run
    def run(self):
        """Run (or continue) the search; returns the final leaderboard doc (rsi/evolve@1)."""
        state = "crashed"
        self.t0 = time.time()
        try:
            self.write_progress(state="running", phase="evolve")
            if self.gen < 0 and not self.population:
                self.checkpoint()
            if self.reeval:  # code changed: re-score the population and hall of fame first
                self.reeval = False
                if not self._map(self._missing(self.population + self.hof, self.seeds), "search", self.gen):
                    raise RsiError("re-evaluation after the code change was interrupted", "E_INTERRUPTED",
                                   hint=f"python -m rsi evolve --resume {self.rd.name} --allow-code-change")
                self.hof = self.ranked(self.hof + self.population)[:HOF_SIZE]
                self.checkpoint()
            self.emit("start", name=self.rd.name, pop=self.s.pop, generations=self.s.generations,
                      datasets=len(self.task.datasets), seeds=self.seeds, workers=self.ev.workers)
            while True:
                g = self.gen + 1
                if g >= self.s.generations:
                    self.reason = self.reason or "done"
                    break
                if self.stall_n >= self.s.stall:
                    self.reason = "done"
                    self.emit("stall", generations=self.stall_n)
                    break
                if self._stop_check():
                    break
                if self.s.max_evals is not None and self.counters["evals"] >= self.s.max_evals:
                    self.reason = "budget"
                    break
                self.holdout_done = False
                if not self._generation(g):
                    break
                self.write_progress(state="running", phase="evolve")
            if self.mode != "now" and self.s.holdout_top > 0 and not self.holdout_done:
                self.holdout()
            state = {"stopped": "stopped", "budget": "budget"}.get(self.reason, "done")
            self.phase = "done" if state == "done" else "evolve"
            self.checkpoint()
            lb = self.finalize(state)
            self.emit("done", state=state, best=(lb["entries"][0]["score"] if lb["entries"] else None))
            return lb
        except KeyboardInterrupt:
            state = "stopped"
            raise RsiError("interrupted; the last generation is checkpointed", "E_INTERRUPTED",
                           hint=f"python -m rsi evolve --resume {self.rd.name}") from None
        finally:
            self.counters["elapsed_s"] = round(self.elapsed(), 3)
            self.t0 = time.time()
            try:
                self.write_progress(state=state, phase=self.phase)
            except Exception:
                pass
            self.rd.close(state)

    def finalize(self, state):
        """Trial records for the leaderboard genomes, then the final leaderboard."""
        from . import api
        st = self.ev.store
        for x in self.hof[:LEADERBOARD_SIZE]:
            r = self.records[x]
            seeds = self.esc_seeds[:r["n_seeds"]]
            cells = [r["cells"].get((d, sd)) for d in self.task.datasets for sd in seeds]
            if st is None or not all(cells):
                continue
            cfg = configio.config_from_dict(self.export_config(x))[0]
            t = api.make_trial(cfg, cells, budget=api._budget(self.task.steps, self.task.eval_every,
                                                              self.task.max_seconds, self.task.early_stop,
                                                              self.task.fresh_points, self.task.metrics),
                               seeds=seeds, datasets=self.task.datasets, objective=f"{self.s.metric}:{self.s.seed_agg}",
                               tags=list(self.s.tags or ()) + ["evolve"], run=self.rd.name, code_fp=self.ev.code_fp,
                               base=self.base)
            t["genome_id"] = x
            st.put_trial(t)
            r["trial_id"] = t["id"]
        lb = self.write_outputs(state)
        lb.update(dir=str(self.rd.path), files={"leaderboard": str(self.rd.path / "leaderboard.json"),
                                                "best_settings": str(self.rd.path / "best.settings.json"),
                                                "log": str(self.rd.path / "log.txt")},
                  next=f"python -m rsi export {self.rd.name} --rank 1 -o winner.json")
        return lb

    @classmethod
    def resume(cls, rundir, evaluator, on_event=None, warn=None, **overrides):
        """Continue a checkpointed search: manifest settings with the allowed `overrides`
        (RESUMABLE), the frozen space.json, state.json and the evals.jsonl journal."""
        man = rundir.manifest()
        sd = dict(man["settings"])
        sd.update({k: v for k, v in overrides.items() if k in RESUMABLE})
        s = EvolveSettings.from_dict(sd)
        base = configio.config_from_dict(man["base"])[0]
        space = Space.load(rd.read_json(rundir.path / "space.json"), base)
        evo = cls(s, space, base, rundir, evaluator, on_event, task=_task(s, base, man), warn=warn)
        state = rd.read_json(rundir.path / "state.json")
        if state:
            journal = rd.read_jsonl(rundir.path / "evals.jsonl")
            if state.get("code_fp") and state["code_fp"] != evaluator.code_fp:
                # --allow-code-change: older journal lines no longer count; elites are re-evaluated
                rundir.append("evals.jsonl", {"code_change": {"from": state["code_fp"], "to": evaluator.code_fp},
                                              "t": round(time.time(), 3)})
                journal = []
                state = dict(state, evaluated={g: [len(evo.seeds), False] for g in state["evaluated"]})
                evo.reeval = True
            evo.restore(state, journal)
        man["settings"] = s.to_dict()
        rundir.write_json("manifest.json", man)
        return evo


def best_holdout(evo, gid):
    return (evo.holdout_of(gid) or {}).get("score") is not None


def _task(s, base, man=None):
    if s.objective.startswith("bench:"):
        b = get_benchmark(s.objective[6:])
        return Task(b.datasets, b.steps, eval_every=s.eval_every or max(1, b.steps // 20), fresh_points=b.fresh_points,
                    metrics=s.metrics, bench=b)
    datasets = (man or {}).get("datasets") or [base.dataset]
    return Task(datasets, s.steps, eval_every=s.eval_every if s.eval_every is not None else max(1, s.steps // 20),
                max_seconds=s.max_seconds, early_stop=s.early_stop, fresh_points=s.fresh_points, metrics=s.metrics)


# ---------------------------------------------------------------- API entry point
def _norm(v):
    return configio.canonical_json(v)


@closes_stores
def evolve(config=None, overrides=None, *, settings=None, space=None, name=None, out=None, resume=None, force=False,
           allow_code_change=False, workers=None, cache=True, store=None, parts=None, on_event=None, **settings_kw):
    """The `evolve` command (see the module docstring); returns the leaderboard doc."""
    from . import api
    from .pool import Evaluator
    plugins = api.setup_parts(parts)
    st = api.open_store(store)
    if resume:
        return _resume(resume, config, overrides, space, out, settings, settings_kw, allow_code_change, workers, cache,
                       st, on_event)
    s = dataclasses.replace(settings) if settings else EvolveSettings()
    s = EvolveSettings.from_dict({**s.to_dict(), **{k: v for k, v in settings_kw.items() if v is not None}})
    _check_settings(s)
    if s.gp_activations:
        from . import gp  # noqa: F401  (before the config: the base may name gp: activations)
    cfg, _, pinned = api.resolve_config(config, overrides)
    api.inert_warnings(cfg, pinned)
    if s.objective.startswith("bench:"):
        b = get_benchmark(s.objective[6:])
        given = [k for k in ("steps", "seeds", "n_seeds", "datasets", "suite", "fresh_points", "metric")
                 if k in settings_kw and settings_kw[k] not in (None, 0, DEFAULTS.get(k))]
        if given:
            api.warn("W_TASK_OVERRIDDEN", f"the benchmark fixes {', '.join(given)}", fields=given)
        s.steps, s.seeds, s.fresh_points, s.metric, s.max_seeds = b.steps, list(b.seeds), b.fresh_points, "fresh_acc", \
            len(b.seeds)
        datasets = list(b.datasets)
    else:
        datasets = api.resolve_datasets(s.datasets, s.suite, cfg.dataset)
        s.seeds = [0, 1, 2] if s.seeds is None and s.n_seeds is None else api.resolve_seeds(s.seeds, s.n_seeds,
                                                                                               cfg.seed)
        if s.metric == "fresh_acc" and not s.fresh_points:
            s.fresh_points = 2000
            api.warn("W_DEFAULT_FILLED", "--metric fresh_acc: --fresh-points set to 2000", field="fresh_points")
    if s.fidelity:
        s.steps = int(s.fidelity[-1])
    s.holdout_seeds = api.parse_seeds(s.holdout_seeds) or []
    s.seeds = [int(x) for x in s.seeds]
    if s.eval_every is None:
        s.eval_every = max(1, int(s.steps) // 20)
    s.n_seeds, s.datasets, s.suite = None, datasets, None
    try:
        sp = Space.load(space or "default", cfg, pinned=pinned, freeze=s.freeze, unfreeze=s.unfreeze, genes=s.genes,
                        models=s.models)
    except RsiError:
        raise
    except Exception as e:
        raise as_rsi_error(e) from None
    if not sp.genes:
        raise RsiError("no genes left to search (everything is pinned or frozen)", "E_BAD_SPACE",
                       hint="drop some --FIELD flags or pass --unfreeze")
    task = _task(s, cfg, {"datasets": datasets})
    clash = set(s.holdout_seeds) & set(escalation_seeds(s.seeds, max(len(s.seeds), s.max_seeds)))
    if clash:
        raise RsiError(f"--holdout-seeds overlap the search seeds: {sorted(clash)}", "E_USAGE", field="holdout_seeds",
                       hint="holdout seeds must be disjoint, e.g. 1000-1004")
    api._task_warnings(cfg, len(datasets) * len(s.seeds), s.steps * s.pop * max(1, s.generations) // 4, workers)
    path = Path(out) if out else st.root / (name or time.strftime("evolve-%Y%m%d-%H%M%S"))
    with Evaluator(workers, plugins, store=st, cache=cache, gp=s.gp_activations, cache_scope=s.cache_scope,
                   code_fp=_code_fp(plugins, s.gp_activations), log=on_event) as ev:
        if s.cache_scope == "config":
            api.warn("W_CACHE_CONFIG_SCOPE", "cache keys ignore the code fingerprint: results may predate code edits")
        manifest = {"base": configio.config_to_dict(cfg), "settings": s.to_dict(), "space": str(space or "default"),
                    "pinned": list(pinned), "datasets": datasets, "seeds": s.seeds,
                    "budget": {"steps": task.steps, "eval_every": task.eval_every, "max_seconds": task.max_seconds,
                               "fresh_points": task.fresh_points},
                    "objective": None, "code_fp": ev.code_fp, "parts": plugins, "workers": ev.workers,
                    "environment": environment()}
        run_dir = rd.RunDir.create(path, "evolve", manifest, force=force)
        try:
            evo = Evolution(s, sp, cfg, run_dir, ev, on_event, task=task, warn=api.warn)
            manifest["objective"] = evo.objective_text()
            run_dir.write_json("manifest.json", {**run_dir.manifest(), "objective": manifest["objective"]})
            run_dir.write_json("space.json", sp.to_dict())
        except BaseException:
            run_dir.close("crashed")
            raise
        return evo.run()


def _code_fp(plugins, gp):
    return code_fingerprint(list(plugins) + (["rsi.gp"] if gp else []))


def _resume(name, config, overrides, space, out, settings, settings_kw, allow_code_change, workers, cache, st,
            on_event):
    from . import api
    from .pool import Evaluator
    path = Path(out) if out else rd.resolve_target(name, st.root)
    man = rd.read_json(path / "manifest.json")
    if not man or man.get("kind") != "evolve":
        raise RsiError(f"{path} is not an evolve run dir", "E_NOT_FOUND", value=str(path))
    if not (path / "state.json").exists():
        raise RsiError(f"{path} has no state.json to resume from", "E_NOT_FOUND", value=str(path))
    old = man["settings"]
    given = dict(settings.to_dict()) if settings else {}
    given.update({k: v for k, v in settings_kw.items() if v is not None})
    diff = {}
    base_seed = man["base"].get("seed", 0)
    for k, v in given.items():
        if k not in DEFAULTS:
            raise RsiError(f"unknown evolve setting '{k}'", "E_USAGE", field=k)
        if k in RESUMABLE or _norm(v) == _norm(DEFAULTS[k]):
            continue
        if k in ("seeds", "n_seeds"):
            v, k, run = api.resolve_seeds(given.get("seeds"), given.get("n_seeds"), base_seed), "seeds", man.get("seeds")
        elif k in ("datasets", "suite"):
            v, k, run = api.resolve_datasets(given.get("datasets"), given.get("suite")), "datasets", man.get("datasets")
        else:
            run = old.get(k)
        if _norm(v) != _norm(run):
            diff[k] = {"run": run, "given": v}
    if config is not None or overrides:
        cfg, _, _ = api.resolve_config([man["base"]] + ([config] if config is not None else []), overrides)
        if configio.config_key(cfg) != configio.config_key(man["base"]):
            diff["config"] = {"run": configio.config_diff(man["base"]), "given": configio.config_diff(cfg)}
    if space is not None and str(space) != man.get("space"):
        diff["space"] = {"run": man.get("space"), "given": str(space)}
    if diff:
        raise RsiError(f"--resume: only {', '.join('--' + k.replace('_', '-') for k in RESUMABLE)} and --workers may "
                       f"change; got {', '.join(diff)}", "E_RESUME_MISMATCH", details={"diff": diff})
    ov = {k: v for k, v in given.items() if k in RESUMABLE}
    if "holdout_seeds" in ov:
        ov["holdout_seeds"] = api.parse_seeds(ov["holdout_seeds"]) or []
    gp_on = bool(old.get("gp_activations"))
    if gp_on:
        from . import gp  # noqa: F401
    plugins = loaded_plugins()
    missing = [p for p in man.get("parts") or [] if p not in plugins]
    if missing:
        raise RsiError(f"the run used parts {missing} that are not loaded", "E_PARTS_MISSING", value=missing,
                       hint=" ".join(f"--parts {p}" for p in man["parts"]))
    with Evaluator(workers, plugins, store=st, cache=cache, gp=gp_on, cache_scope=old.get("cache_scope", "code"),
                   code_fp=_code_fp(plugins, gp_on), log=on_event) as ev:
        if man.get("code_fp") and man["code_fp"] != ev.code_fp and not allow_code_change:
            raise RsiError(f"the code changed since the run started ({man['code_fp']} -> {ev.code_fp})",
                           "E_CODE_CHANGED", hint=f"python -m rsi evolve --resume {path.name} --allow-code-change")
        run_dir = rd.RunDir.open(path, lock=True)
        try:
            evo = Evolution.resume(run_dir, ev, on_event, warn=api.warn, **ov)
            if man.get("code_fp") != ev.code_fp:
                man = run_dir.manifest()
                run_dir.write_json("manifest.json", {**man, "code_fp": ev.code_fp,
                                                     "code_fp_history": (man.get("code_fp_history") or [])
                                                     + [man.get("code_fp")]})
        except BaseException:
            run_dir.close("crashed")
            raise
        return evo.run()
