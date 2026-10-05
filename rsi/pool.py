"""
pool: Evaluator, the one way the console trains cells. Cache first (memory,
then the store), then a pool of spawned worker processes (workers=0: serial,
in-process). Results come back in input order; duplicate specs run once.

    with Evaluator(workers=4, parts=["my_parts"], store=Store("runs")) as ev:
        specs = cell_specs(cfg, datasets=["Moons", "Circles"], seeds=[0, 1, 2], steps=3000, fresh_points=2000)
        cells = ev.map(specs, on_result=lambda i, cell: ..., stop_check=lambda: False)

A worker crash (BrokenProcessPool) re-queues the affected cells once on a new
pool, one at a time; a second crash marks that cell status 'error'
("worker crashed"). Only ok | diverged | early_stopped cells are cached.

Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md.
"""
import collections
import concurrent.futures as cf
import multiprocessing as mp
import os
import signal
import sys
from concurrent.futures.process import BrokenProcessPool

from nncore.activations import ACTIVATIONS
from nncore.configio import config_key, config_to_dict
from nncore.registry import disable_lazy_plugins, load_plugins, loaded_plugins
from nncore.run import EarlyStop, RunSpec, code_fingerprint, run_unit

CACHEABLE = ("ok", "diverged", "early_stopped")
POLL_S = 0.5
GP_FIELDS = ("activation", "act_decide", "act_relate", "act_prepare", "layers")


def uses_gp(config):
    """True if a config dict names a GP expression activation (gp:... / gpl:...)."""
    return any(isinstance(config.get(f), str) and ("gp:" in config[f] or "gpl:" in config[f]) for f in GP_FIELDS)


def _gp_lazy(name):
    """ACTIVATIONS resolver: a 'gp:' / 'gpl:' name imports rsi.gp on first use, so GP winners
    run, check, replay and export without --gp-activations (here and in every worker)."""
    if isinstance(name, str) and name.startswith(("gp:", "gpl:")):
        import importlib
        return importlib.import_module("rsi.gp")._resolve(name)
    return None


if _gp_lazy not in ACTIVATIONS._resolvers:
    ACTIVATIONS.add_resolver(_gp_lazy)


def default_workers():
    """max(1, min(cpu_count - 1, 8))."""
    return max(1, min((os.cpu_count() or 2) - 1, 8))


# ---------------------------------------------------------------- worker side
def _init(parts, gp, path):
    sys.stdout = sys.stderr  # parts may print; the parent's stdout carries JSON
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # the coordinator handles Ctrl-C
    except (ValueError, OSError):
        pass
    for p in path:
        if p not in sys.path:
            sys.path.append(p)
    import nncore  # noqa: F401  (sets torch threads)
    if parts:
        load_plugins(*parts)
    else:
        disable_lazy_plugins()
    if gp:
        import importlib
        importlib.import_module("rsi.gp").install_resolver()


def _work(spec):
    return run_unit(spec)


# ---------------------------------------------------------------- helpers
def cell_specs(config, *, datasets=None, seeds=None, steps=2000, eval_every=0, max_seconds=None, early_stop=None,
               fresh_points=0, metrics=(), stop_on_nonfinite=True):
    """RunSpecs for config x datasets x seeds (dataset-major). config: Config or config dict;
    datasets / seeds default to the config's own. early_stop: EarlyStop, (at_step, min_acc) or None."""
    base = config_to_dict(config) if not isinstance(config, dict) else dict(config)
    if isinstance(early_stop, (tuple, list)):
        early_stop = EarlyStop(int(early_stop[0]), float(early_stop[1]))
    out = []
    for d in (datasets or [base["dataset"]]):
        for s in (seeds if seeds is not None else [base["seed"]]):
            out.append(RunSpec(config={**base, "dataset": d, "seed": int(s)}, steps=int(steps),
                               eval_every=int(eval_every or 0), max_seconds=max_seconds, early_stop=early_stop,
                               stop_on_nonfinite=stop_on_nonfinite, fresh_points=int(fresh_points or 0),
                               extra_metrics=tuple(metrics or ())))
    return out


def error_cell(spec, key, message, kind="WorkerCrashed", phase="worker"):
    """A CellResult with status 'error' for a cell that never returned."""
    cfg = spec.config or {}
    try:
        ck = config_key(cfg)
    except Exception:
        ck = None
    return {"key": key, "config_key": ck, "dataset": cfg.get("dataset"), "seed": cfg.get("seed"),
            "steps": int(spec.steps), "steps_done": 0, "status": "error", "diverged_at": None, "train": None,
            "test": None, "fresh": None, "model": None, "seconds": 0.0, "cached": False, "curve": None,
            "error": {"type": kind, "message": message, "phase": phase}}


# ---------------------------------------------------------------- the evaluator
class Evaluator:
    """Context manager. workers=0 (and 1 unless isolate=True: one worker process buys nothing
    but start-up time) runs cells serially in this process; N > 1 uses a spawned
    ProcessPoolExecutor (max_tasks_per_child=200), created on first use and reused.
    cache=False neither reads nor writes the cache (replay). cache_scope='config' drops
    code_fp from the key (results survive code edits; W_CACHE_CONFIG_SCOPE)."""

    def __init__(self, workers=None, parts=None, *, store=None, code_fp=None, gp=False, cache=True,
                 cache_scope="code", log=None, isolate=False):
        self.workers = default_workers() if workers is None else max(0, int(workers))
        self.serial = self.workers == 0 or (self.workers == 1 and not isolate)
        self.parts = list(loaded_plugins() if parts is None else parts)
        self.store, self.gp, self.cache, self.cache_scope = store, gp, cache, cache_scope
        self._fixed_fp = code_fp is not None
        self.code_fp = code_fp or code_fingerprint(self.parts)
        self.log = log
        self.mem = {}
        self.stats = {"new": 0, "cached": 0, "failed": 0, "crashed": 0, "skipped": 0}
        self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def abort(self):
        """Stop now: cancel queued cells and terminate the workers."""
        pool, self._pool = self._pool, None
        if pool is not None:
            procs = list(getattr(pool, "_processes", {}).values())
            pool.shutdown(wait=False, cancel_futures=True)
            for p in procs:
                try:
                    p.terminate()
                except Exception:
                    pass

    # ---------------------------------------------------------------- cache
    def fp_for(self, config):
        """The code fingerprint for a config: code_fp, plus rsi/gp.py when it names a gp: activation
        (so edits to the GP primitives invalidate those cells). A code_fp given at init is used as is."""
        if self._fixed_fp or not uses_gp(config_to_dict(config) if not isinstance(config, dict) else config):
            return self.code_fp
        return code_fingerprint(self.parts + ["rsi.gp"])

    def key(self, spec):
        """The cell's cache key: RunSpec.key(fp_for(config)), or RunSpec.key('') with cache_scope='config'."""
        return spec.key("" if self.cache_scope == "config" else self.fp_for(spec.config))

    def lookup(self, key):
        if not self.cache:
            return None
        cell = self.mem.get(key)
        if cell is None and self.store is not None:
            cell = self.store.get_cell(key)
            if cell is not None:
                self.mem[key] = cell
        return cell

    def _finish(self, key, cell, first, on_result, results):
        cell = dict(cell, key=key, cached=False)
        if self.cache_scope == "config":
            cell["cache_scope"] = "config"
        results[key] = cell
        self.stats["new"] += 1
        if cell.get("status") == "error":
            self.stats["failed"] += 1
        if self.cache and cell.get("status") in CACHEABLE:
            self.mem[key] = cell
            if self.store is not None:
                self.store.put_cell(key, cell)
        if on_result is not None:
            on_result(first[key], cell)

    # ---------------------------------------------------------------- map
    def map(self, specs, *, on_result=None, stop_check=None, max_new=None):
        """CellResults in input order; None for cells not run (stop_check() turned true, or
        beyond max_new new trainings). Duplicate keys run once. on_result(i, cell) fires once
        per distinct key (i = first input index), cache hits included (cell['cached']).
        stop_check() is polled between cells and at least every 0.5 s while waiting."""
        specs = list(specs)
        keys = [self.key(s) for s in specs]
        first, order = {}, []
        for i, k in enumerate(keys):
            if k not in first:
                first[k] = i
                order.append(k)
        results, todo = {}, []
        for k in order:
            hit = self.lookup(k)
            if hit is not None:
                cell = dict(hit, key=k, cached=True)
                results[k] = cell
                self.stats["cached"] += 1
                if on_result is not None:
                    on_result(first[k], cell)
            else:
                todo.append(k)
        if max_new is not None:
            self.stats["skipped"] += max(0, len(todo) - int(max_new))
            todo = todo[:max(0, int(max_new))]
        if todo:
            run = self._serial if self.serial else self._parallel
            run(todo, specs, first, on_result, stop_check, results)
        return [dict(results[k]) if k in results else None for k in keys]

    def evaluate(self, spec):
        """One cell (cached when possible)."""
        return self.map([spec])[0]

    def run_cells(self, config, **kw):
        """map(cell_specs(config, **kw)): config x datasets x seeds."""
        return self.map(cell_specs(config, **kw))

    def _serial(self, todo, specs, first, on_result, stop_check, results):
        for k in todo:
            if stop_check is not None and stop_check():
                break
            self._finish(k, run_unit(specs[first[k]]), first, on_result, results)

    def _new_pool(self):
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        kw = {"max_tasks_per_child": 200} if sys.version_info >= (3, 11) else {}  # 3.10 lacks it
        self._pool = cf.ProcessPoolExecutor(max_workers=self.workers, mp_context=mp.get_context("spawn"),
                                            initializer=_init, initargs=(self.parts, self.gp, list(sys.path)), **kw)
        return self._pool

    def _parallel(self, todo, specs, first, on_result, stop_check, results):
        pending, inflight = collections.deque(todo), {}
        attempts = collections.Counter()
        stopped = False
        try:
            while pending or inflight:
                retrying = any(attempts[k] for k in list(pending)[:1]) or any(attempts[k] for k in inflight.values())
                window = 1 if retrying else 2 * self.workers
                while pending and not stopped and len(inflight) < window:
                    k = pending.popleft()
                    pool = self._pool or self._new_pool()
                    inflight[pool.submit(_work, specs[first[k]])] = k
                    if attempts[k]:
                        break  # a retried cell runs alone, so a second crash is its own
                if not inflight:
                    break
                done, _ = cf.wait(list(inflight), timeout=POLL_S, return_when=cf.FIRST_COMPLETED)
                if not stopped and stop_check is not None and stop_check():
                    stopped = True
                    pending.clear()
                    for f in list(inflight):
                        if f.cancel():
                            inflight.pop(f)
                broken = []
                for f in done:
                    k = inflight.pop(f, None)
                    if k is None or f.cancelled():
                        continue
                    try:
                        cell = f.result()
                    except BrokenProcessPool:
                        broken.append(k)
                        continue
                    except Exception as e:  # e.g. an unpicklable result
                        cell = error_cell(specs[first[k]], k, f"{type(e).__name__}: {e}", type(e).__name__)
                    self._finish(k, cell, first, on_result, results)
                if broken:
                    rest, _ = cf.wait(list(inflight), timeout=5)
                    for f in list(inflight):
                        k = inflight.pop(f)
                        try:
                            if f in rest and not f.cancelled():
                                self._finish(k, f.result(), first, on_result, results)
                                continue
                        except Exception:
                            pass
                        broken.append(k)
                    self.stats["crashed"] += 1
                    if self.log:
                        self.log("worker_crashed", cells=len(broken))
                    self.abort()  # the next submit starts a fresh pool
                    for k in broken:
                        attempts[k] += 1
                        if attempts[k] >= 2:
                            self._finish(k, error_cell(specs[first[k]], k, "worker crashed"), first, on_result,
                                         results)
                        elif not stopped:
                            pending.appendleft(k)
        except KeyboardInterrupt:
            self.abort()
            raise
