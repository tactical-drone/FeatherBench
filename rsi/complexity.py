"""
complexity: an empirical complexity exponent, "discovered" instead of worked out
with big O. For each size of a dataset family (spiral[k], checkerboard[c], ...)
the recipe's capacity knob climbs a ladder until the mean fresh accuracy over the
seeds reaches the threshold; the smallest solving rung and its n_params are kept.
Then log(params) = log(c) + p*log(size) is fitted over the solved sizes.

    python -m rsi complexity --family spiral --sizes 2-6 --model mlp --layers 8,8 --threshold 0.95
    -> "params ~ 12.3 * k^1.42 (r^2 0.97): empirical complexity exponent 1.42"

Knob: custom nn -> width; mlp -> every hidden layer's width (layer count and
activations kept); other models need --knob FIELD (an int field, layers or extra.NAME).
Rungs are tried bottom up (all sizes at once, a few rungs ahead to keep the
workers busy), so the smallest solving rung is exact whatever --workers is.

Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md.
"""
import math
import platform
import statistics
import time

from nncore import configio, schema
from nncore.models import format_layers, parse_layers

from .bench import cell_acc
from .errors import RsiError, as_rsi_error
from .store import closes_stores

DEFAULT_LADDER = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32)
FRESH_POINTS = 2000
DEFAULT_KNOB = {"custom nn": "width", "mlp": "layers"}


def _family(name):
    from nncore import datasets
    fams = getattr(datasets, "FAMILIES", None)
    if fams is None:
        raise RsiError("dataset families are not available in this nncore", "E_UNSUPPORTED")
    if name not in fams:
        raise RsiError(f"unknown family '{name}'", "E_UNKNOWN_PART", field="family", value=name,
                       did_you_mean=fams.suggest(name), allowed=list(fams), hint="python -m rsi describe --section families")
    return fams[name]


def _knob(cfg, knob):
    knob = knob or DEFAULT_KNOB.get(cfg.model)
    if knob is None:
        raise RsiError(f"model '{cfg.model}' has no default capacity knob; pass --knob FIELD (e.g. width)", "E_USAGE",
                       field="knob", allowed=["width", "classes", "layers", "extra.NAME"])
    knob = knob.replace("-", "_") if not knob.startswith("extra.") else knob
    if knob.startswith("extra."):
        return knob
    if knob != "layers" and knob not in configio.INT_FIELDS or knob in ("n_points", "seed"):
        raise RsiError(f"--knob {knob!r}: use an integer model field (width, classes), layers or extra.NAME", "E_USAGE",
                       field="knob", value=knob)
    if knob not in schema.fields_read(cfg.model, cfg):
        raise RsiError(f"--knob {knob}: model '{cfg.model}' does not read it, so the ladder would change nothing",
                       "E_USAGE", field="knob", value=knob)
    if knob == "layers" and not parse_layers(cfg.layers, cfg.activation):
        raise RsiError("mlp with no hidden layers has no width to step; set --layers (e.g. 8,8)", "E_USAGE",
                       field="layers", value=cfg.layers)
    return knob


def apply_knob(cfg_dict, knob, rung, activation=None):
    """cfg_dict with the capacity knob at `rung` (layers: every hidden width = rung)."""
    d = dict(cfg_dict)
    if knob.startswith("extra."):
        d["extra"] = {**d.get("extra", {}), knob[6:]: rung}
    elif knob == "layers":
        act = activation or d["activation"]
        d["layers"] = format_layers([(rung, a) for _, a in parse_layers(d["layers"], act)], act)
    else:
        d[knob] = rung
    return d


def fit_power(points):
    """Least squares of log(y) = log(c) + p*log(x) -> {p, c, r2, n} (None with < 2 distinct x)."""
    pts = [(x, y) for x, y in points if x and y and x > 0 and y > 0]
    if len({x for x, _ in pts}) < 2:
        return None
    xs, ys = [math.log(x) for x, _ in pts], [math.log(y) for _, y in pts]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    p = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    lc = my - p * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - lc - p * x) ** 2 for x, y in zip(xs, ys))
    return {"p": p, "c": math.exp(lc), "r2": 1 - ss_res / ss_tot if ss_tot > 0 else 1.0, "n": len(pts)}


def _slow_warning(ev, cfg, fam, sizes, ladder, rung_cfg, seeds, steps):
    """W_SLOW from the worst case (every rung of every size), counting only uncached cells."""
    from . import api
    from .pool import cell_specs
    specs = [sp for k in sizes for r in ladder
             for sp in cell_specs({**rung_cfg[r], "dataset": fam.dataset_name(k)}, seeds=seeds, steps=steps,
                                  fresh_points=FRESH_POINTS)]
    todo = sum(1 for sp in specs if ev.lookup(ev.key(sp)) is None)
    est = todo * steps * schema.COST_MS_PER_STEP.get(cfg.model, 2.4) / 1000 / max(1, ev.workers)
    if est > 60:
        api.warn("W_SLOW", f"up to {est:.0f} s ({todo} uncached of {len(specs)} cells x {steps} steps; each size "
                           "stops at its first solving rung, so usually much less)", estimated_s=round(est, 1))


@closes_stores
def complexity(config=None, family=None, sizes=None, threshold=0.95, ladder=None, knob=None, seeds=None, steps=3000,
               workers=None, *, overrides=None, cache=True, store=None, parts=None, on_event=None):
    from . import api
    from .pool import Evaluator, cell_specs
    api.setup_parts(parts)
    if not family:
        raise RsiError("complexity needs --family (spiral, checkerboard, rings, stripes, blobs)", "E_USAGE")
    fam = _family(family)
    cfg, _, pinned = api.resolve_config(config, overrides)
    api.inert_warnings(cfg, pinned)
    if "dataset" in pinned:
        api.warn("W_TASK_OVERRIDDEN", "complexity sets the dataset to each family size; --dataset is ignored",
                 fields=["dataset"])
    sizes = list(dict.fromkeys(api.parse_seeds(sizes, "--sizes") or fam.sizes))
    for k in sizes:
        try:
            fam.check_size(k)
        except ValueError as e:
            raise RsiError(str(e), "E_OUT_OF_RANGE", field="sizes", value=k) from None
    ladder = sorted(set(api.parse_seeds(ladder, "--ladder") or DEFAULT_LADDER))
    if not ladder or ladder[0] < 1:
        raise RsiError("--ladder needs positive integers, e.g. 1,2,4,8", "E_USAGE", field="ladder", value=ladder)
    if not 0 < float(threshold) <= 1:
        raise RsiError("--threshold must be in (0, 1]", "E_USAGE", field="threshold", value=threshold)
    seeds = api.parse_seeds(seeds) or [0, 1, 2]
    steps = int(steps)
    if steps < 1:
        raise RsiError("--steps must be >= 1", "E_USAGE", field="steps", value=steps)
    knob = _knob(cfg, knob)
    base = configio.config_to_dict(cfg)
    rung_cfg = {}
    for r in ladder:  # validate every rung up front (bounds, layers grammar)
        try:
            rung_cfg[r] = configio.config_to_dict(configio.config_from_dict(apply_knob(base, knob, r))[0])
        except Exception as e:
            raise as_rsi_error(e) from None
    t0 = time.perf_counter()
    state = {k: {"next": 0, "tried": [], "rung": None, "done": False} for k in sizes}
    n_new = n_cached = n_cells = 0
    st = api.open_store(store)
    with Evaluator(workers, None, store=st, cache=cache) as ev:
        code_fp = ev.code_fp
        _slow_warning(ev, cfg, fam, sizes, ladder, rung_cfg, seeds, steps)
        while not all(s["done"] for s in state.values()):
            active = [k for k in sizes if not state[k]["done"]]
            look = max(1, math.ceil(2 * max(1, ev.workers) / (len(active) * len(seeds))))
            specs, owner = [], []
            for k in active:
                for r in ladder[state[k]["next"]:state[k]["next"] + look]:
                    for sp in cell_specs({**rung_cfg[r], "dataset": fam.dataset_name(k)}, seeds=seeds, steps=steps,
                                         fresh_points=FRESH_POINTS):
                        specs.append(sp)
                        owner.append((k, r))
            cells = ev.map(specs)
            n_cells += len(cells)
            n_new += sum(1 for c in cells if c and not c.get("cached"))
            n_cached += sum(1 for c in cells if c and c.get("cached"))
            got = {}
            for (k, r), c in zip(owner, cells):
                got.setdefault((k, r), []).append(c)
            for k in active:
                s = state[k]
                chance = 1.0 / fam.n_classes(k)
                for r in ladder[s["next"]:s["next"] + look]:
                    cs = got[(k, r)]
                    accs = [cell_acc(c, "fresh_acc", chance) for c in cs]
                    tests = [(c.get("test") or {}).get("acc") for c in cs if c and c.get("status") == "ok"]
                    params = [c["model"]["n_params"] for c in cs if c and c.get("model")]
                    row = {"rung": r, "n_params": max(params) if params else None,
                           "fresh_acc_mean": statistics.fmean(accs), "fresh_acc_min": min(accs),
                           "test_acc_mean": statistics.fmean([t for t in tests if t is not None]) if tests else None,
                           "statuses": sorted({c.get("status") for c in cs if c})}
                    row["solved"] = row["fresh_acc_mean"] >= float(threshold)
                    s["tried"].append(row)
                    s["next"] += 1
                    api._emit(on_event, "rung", size=k, rung=r, fresh_acc=row["fresh_acc_mean"],
                              n_params=row["n_params"], solved=row["solved"])
                    if row["solved"]:
                        s["rung"], s["done"] = row, True
                        break
                if s["next"] >= len(ladder):
                    s["done"] = True
    table = []
    for k in sizes:
        s = state[k]
        best = s["rung"] or {}
        table.append({"size": k, "dataset": fam.dataset_name(k), "solved": bool(s["rung"]), "rung": best.get("rung"),
                      "n_params": best.get("n_params"), "fresh_acc_mean": best.get("fresh_acc_mean"),
                      "max_rung_tried": s["tried"][-1]["rung"] if s["tried"] else None,
                      "best_fresh_acc": max((t["fresh_acc_mean"] for t in s["tried"]), default=None),
                      "tried": s["tried"]})
    solved = [r for r in table if r["solved"]]
    fit = fit_power([(r["size"], r["n_params"]) for r in solved])
    knob_fit = fit_power([(r["size"], r["rung"]) for r in solved])
    sym = fam.symbol
    notes = []
    unsolved = [r["size"] for r in table if not r["solved"]]
    if unsolved:
        notes.append(f"{fam.size_label} {','.join(map(str, unsolved))}: not solved up to {knob}={ladder[-1]} "
                     f"(threshold {threshold}); raise --steps, extend --ladder or lower --threshold")
    if fit:
        fit["formula"] = f"params ~ {fit['c']:.3g} * {sym}^{fit['p']:.2f}"
        summary = f"{fit['formula']} (r^2 {fit['r2']:.2f}): empirical complexity exponent {fit['p']:.2f}"
        if fit["n"] == 2:
            notes.append("only 2 solved sizes: the fit is exact (r^2 = 1 says nothing)")
        if knob_fit:
            knob_fit["formula"] = f"{knob} ~ {knob_fit['c']:.3g} * {sym}^{knob_fit['p']:.2f}"
    else:
        summary = f"not enough solved sizes for a fit (need 2, got {len(solved)})"
    return {"family": fam.to_dict(), "sizes": sizes, "knob": knob, "ladder": ladder, "threshold": float(threshold),
            "metric": "fresh_acc", "seeds": seeds, "steps": steps, "fresh_points": FRESH_POINTS,
            "config_diff": configio.config_diff(cfg), "config": base, "config_key": configio.config_key(cfg),
            "table": table, "fit": fit, "knob_fit": knob_fit if fit else None, "summary": summary, "notes": notes,
            "n_cells": n_cells, "n_new": n_new, "n_cached": n_cached,
            "seconds": round(time.perf_counter() - t0, 2),
            "environment": {"platform": platform.platform(), "code_fp": code_fp}}
