"""
doctor: is this machine ready? Python and package versions, my_parts, the
store, a spawned worker, and the numeric fingerprint of the default config.

    python -m rsi doctor                 quick checks (~5 s)
    python -m rsi doctor --golden        + default config 7000 steps vs tests/golden.json
    python -m rsi doctor --record-golden + write this machine's numbers into tests/golden.json
    python -m rsi doctor --calibrate     + measure ms/step per model into <store>/env.json

Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md.
"""
import contextlib
import importlib
import json
import platform
import sys
import time

FINGERPRINT = {"init_param_sum": 17.098907, "data_sum": 0.786433,
               "loss": {"1": 1.873401, "10": 1.620902, "100": 1.452731}}  # Linux x86_64, torch 2.14 (A6)
V0 = {"skip_decide": "none"}  # "fourier-gauss-v0": the recipe FINGERPRINT was taken with


def _check(name, fn):
    t0 = time.perf_counter()
    try:
        info = fn()
        ok = True if not isinstance(info, dict) or "ok" not in info else bool(info.pop("ok"))
        return {"name": name, "ok": ok, **(info if isinstance(info, dict) else {"info": info}),
                "seconds": round(time.perf_counter() - t0, 3)}
    except Exception as e:
        return {"name": name, "ok": False, "error": f"{type(e).__name__}: {e}",
                "seconds": round(time.perf_counter() - t0, 3)}


def _packages():
    out = {}
    for mod in ("torch", "numpy", "PyQt6", "pyqtgraph"):
        try:
            m = importlib.import_module(mod)
            v = getattr(m, "__version__", None)
            if mod == "PyQt6":
                from PyQt6.QtCore import PYQT_VERSION_STR
                v = PYQT_VERSION_STR
            out[mod] = v
        except Exception as e:
            out[mod] = None
            out[f"{mod}_error"] = str(e)
    return {"ok": out.get("torch") is not None and out.get("numpy") is not None, "versions": out,
            "note": "PyQt6/pyqtgraph are only needed for the window and --png" if not out.get("PyQt6") else None}


def fingerprint():
    """Init param sum, data sum and losses at steps 1/10/100 of the default config, seed 0."""
    from nncore import Config, Session
    with contextlib.redirect_stdout(sys.stderr):
        s = Session(Config(**V0))
        init = float(sum(float(p.detach().sum()) for p in s.model.parameters()))
        data = float(s.X_all.sum())
        losses, done = {}, 0
        for k in (1, 10, 100):
            losses[str(k)] = s.train(k - done)
            done = k
    return {"init_param_sum": round(init, 6), "data_sum": round(data, 6),
            "loss": {k: round(v, 6) for k, v in losses.items()}}


def _fingerprint_check():
    fp = fingerprint()
    ref = platform.system() == "Linux" and platform.machine() == "x86_64"
    import torch
    same = (abs(fp["init_param_sum"] - FINGERPRINT["init_param_sum"]) < 1e-5
            and abs(fp["data_sum"] - FINGERPRINT["data_sum"]) < 1e-5
            and all(abs(fp["loss"][k] - FINGERPRINT["loss"][k]) < 1e-5 for k in FINGERPRINT["loss"]))
    return {"ok": same or not ref, "values": fp, "reference": FINGERPRINT, "matches_reference": same,
            "note": None if same else ("init/data sums should match everywhere; losses may differ across "
                                       f"platforms (this is {platform.system()}/torch {torch.__version__})")}


def _store(store):
    from .store import Store
    with Store(store) as st:
        return {"path": str(st.path), "cells": st.n_cells(), "trials": st.count()}


def _worker():
    from nncore import Config
    from nncore.configio import config_to_dict
    from nncore.registry import loaded_plugins
    from nncore.run import RunSpec

    from .pool import Evaluator
    spec = RunSpec(config=config_to_dict(Config(model="mlp", dataset="Moons", n_points=100)), steps=5)
    with Evaluator(1, loaded_plugins(), store=None, cache=False, isolate=True) as ev:
        cell = ev.evaluate(spec)
    return {"ok": cell["status"] == "ok", "status": cell["status"], "error": cell.get("error")}


def _golden_file():
    from nncore.registry import REPO_ROOT
    return REPO_ROOT / "tests" / "golden.json"


def golden_entry(doc):
    import torch
    for e in doc.get("entries", []):
        if e["system"] != platform.system() or e.get("machine") not in (None, platform.machine()):
            continue
        if e.get("torch") is None or torch.__version__.startswith(e["torch"]):
            return e
    return None


def _golden(record):
    import torch
    from nncore import Config
    from nncore.configio import config_to_dict
    from nncore.run import RunSpec, run_unit
    path = _golden_file()
    doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "format": "nn-playground/golden", "version": 1, "config": "default Config(), seed 0", "steps": 7000,
        "places": 3, "entries": []}
    cell = run_unit(RunSpec(config=config_to_dict(Config()), steps=int(doc.get("steps", 7000))))
    got = {"train_acc": cell["train"]["acc"], "test_acc": cell["test"]["acc"]}
    e = golden_entry(doc)
    places = int(doc.get("places", 3))
    out = {"steps": doc.get("steps", 7000), "got": got, "expected": e}
    if e is not None:
        out["ok"] = all(round(abs(got[k] - e[k]), places) == 0 for k in ("train_acc", "test_acc"))
    else:
        out["ok"] = record
        out["note"] = f"no golden entry for {platform.system()}-{platform.machine()}/torch {torch.__version__}"
    if record:
        ver = ".".join(torch.__version__.split("+")[0].split(".")[:2])
        doc["entries"] = [x for x in doc["entries"] if not (x["system"] == platform.system()
                                                             and x.get("machine") == platform.machine()
                                                             and x.get("torch") == ver)]
        doc["entries"].append({"system": platform.system(), "machine": platform.machine(), "torch": ver,
                               "train_acc": round(got["train_acc"], 5), "test_acc": round(got["test_acc"], 5),
                               "note": f"recorded by rsi doctor --record-golden, Python {platform.python_version()}, "
                                       f"torch {torch.__version__}"})
        from nncore.configio import atomic_write_text
        atomic_write_text(path, json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
        out.update(recorded=str(path), ok=True)
    return out


def calibrate(store=None, steps=300):
    """ms per training step for mlp and custom nn (default sizes, one core) -> <store>/env.json."""
    from nncore import Config, Session
    from nncore.run import environment

    from .rundir import atomic_write_json
    from .store import Store
    ms = {}
    for model in ("mlp", "custom nn"):
        with contextlib.redirect_stdout(sys.stderr):
            s = Session(Config(model=model))
            s.train(20)
            t0 = time.perf_counter()
            s.train(steps)
        ms[model] = round((time.perf_counter() - t0) * 1000 / steps, 3)
    st = Store(store)
    doc = {"ms_per_step": ms, "measured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "env": environment()}
    atomic_write_json(st.root / "env.json", doc)
    st.close()
    return {"ms_per_step": ms, "file": str(st.root / "env.json")}


def run_doctor(*, golden=False, record_golden=False, calibrate=False, store=None, parts=None):
    from nncore.run import environment

    from .api import setup_parts
    checks = [_check("python", lambda: {"ok": sys.version_info >= (3, 10), "version": platform.python_version(),
                                        "need": ">=3.10"}),
              _check("packages", _packages)]
    checks.append(_check("parts", lambda: {"loaded": setup_parts(parts)}))
    checks.append(_check("store", lambda: _store(store)))
    checks.append(_check("fingerprint", _fingerprint_check))
    checks.append(_check("spawn_worker", _worker))
    if golden or record_golden:
        checks.append(_check("golden", lambda: _golden(record_golden)))
    if calibrate:
        checks.append(_check("calibrate", lambda: globals()["calibrate"](store)))
    return {"ok": all(c["ok"] for c in checks), "checks": checks, "env": environment()}
