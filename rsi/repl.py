"""
repl: one in-process Session driven by JSONL (rsi/repl@1). Each stdin line is
a request, each answer one stdout line; an error answers ok=false and the
session goes on (configure is transactional, so the state is unchanged).

    {"cmd":"set","set":{"lr":0.003}}       {"cmd":"train","steps":500,"eval_every":100}
    {"cmd":"eval"}  {"cmd":"map","size":"48x24"}  {"cmd":"predict","points":[[0.1,0.2]]}
    {"cmd":"reset","what":"weights"|"data","seed":5}  {"cmd":"describe"}  {"cmd":"save","path":"x.json"}
    {"cmd":"quit"}

Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md.
"""
import contextlib
import json
import sys

from nncore import configio
from nncore.registry import loaded_plugins

from .errors import RsiError, as_rsi_error
from .io import now_iso, to_json

COMMANDS = ("set", "train", "eval", "map", "predict", "reset", "describe", "save", "quit")


def _train(s, req):
    steps = int(req.get("steps", 1))
    every = int(req.get("eval_every") or 0)
    if steps < 0:
        raise RsiError("steps must be >= 0", "E_BAD_VALUE", field="steps", value=steps)
    curve = {"step": [], "train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []} if every else None
    done, loss = 0, float("nan")
    while done < steps:
        n = min(every or steps, steps - done)
        loss = s.train(n, stop_on_nonfinite=bool(req.get("stop_on_nonfinite", True)))
        done += n
        if s.diverged_at is not None and req.get("stop_on_nonfinite", True):
            break
        if every:
            ev = s.evaluate()
            curve["step"].append(s.step_count)
            for k in ("train", "test"):
                curve[f"{k}_loss"].append(ev[k]["loss"])
                curve[f"{k}_acc"].append(ev[k]["acc"])
    return {"step": s.step_count, "loss": loss, "diverged_at": s.diverged_at, "curve": curve}


def handle(s, req):
    """One request -> result dict (raises on error)."""
    cmd = req.get("cmd")
    if cmd == "set":
        changes = req.get("set") or {}
        if not isinstance(changes, dict):
            raise RsiError("'set' must be an object of field: value", "E_BAD_VALUE")
        resolved = {}
        for k, v in changes.items():
            v = configio.coerce_value(k, v)
            if k == "features":
                v = tuple(configio.resolve_name(k, x)[0] for x in v)
            elif k in configio.FIELD_REGISTRY:
                v = configio.resolve_name(k, v)[0]
            resolved[k] = v
        rebuilt = s.configure(skip_irrelevant=bool(req.get("skip_irrelevant", False)),
                              keep_optimizer_state=bool(req.get("keep_optimizer_state", False)), **resolved)
        return {"rebuilt": rebuilt, "config_diff": configio.config_diff(s.cfg), "step": s.step_count}
    if cmd == "train":
        return _train(s, req)
    if cmd == "eval":
        ev = s.evaluate(extra_metrics=tuple(req.get("metrics") or ()))
        out = {"step": s.step_count, **ev}
        if req.get("fresh_points"):
            out["fresh"] = s.evaluate_fresh(int(req["fresh_points"]))
        return out
    if cmd == "map":
        from .render import ascii_map, parse_size
        w, h = parse_size(req.get("size"))
        return {"map": ascii_map(s, w, h, points=bool(req.get("points", True))), "step": s.step_count}
    if cmd == "predict":
        logits = s.predict(req.get("points") or [])
        return {"logits": logits.tolist(), "pred": logits.argmax(1).tolist()}
    if cmd == "reset":
        what = req.get("what", "weights")
        if what == "weights":
            s.reset_model()
            return {"rebuilt": "model", "step": s.step_count}
        if what == "data":
            s.new_data(req.get("seed"))
            return {"rebuilt": "data", "seed": s.cfg.seed, "step": s.step_count}
        raise RsiError(f"reset what={what!r}: use 'weights' or 'data'", "E_BAD_VALUE", field="what", value=what)
    if cmd == "describe":
        return {"describe": s.describe(), "n_params": s.n_params, "hidden_sizes": s.hidden_sizes,
                "n_classes": s.n_classes, "step": s.step_count, "config": configio.config_to_dict(s.cfg)}
    if cmd == "save":
        path = req.get("path")
        if not path:
            raise RsiError("save needs a path", "E_BAD_VALUE", field="path")
        configio.save_settings(path, s.cfg, train={"steps": s.step_count},
                               meta={"saved_by": "rsi repl", "created": now_iso(), "parts": loaded_plugins()})
        return {"settings_file": str(path)}
    raise RsiError(f"unknown cmd {cmd!r}", "E_USAGE", value=cmd, allowed=list(COMMANDS))


def serve(cfg, stdin, stdout):
    """Run the JSONL loop until quit / EOF; returns the exit code (0)."""
    from nncore.session import Session

    def say(doc):
        try:
            stdout.write(to_json(doc) + "\n")
            stdout.flush()
        except BrokenPipeError:
            raise SystemExit(0)
    with contextlib.redirect_stdout(sys.stderr):
        s = Session(cfg)
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        cmd = None
        try:
            req = json.loads(line)
            if not isinstance(req, dict):
                raise RsiError("a request is a JSON object with 'cmd'", "E_USAGE")
            cmd = req.get("cmd")
            if cmd == "quit":
                break
            with contextlib.redirect_stdout(sys.stderr):
                result = handle(s, req)
            say({"ok": True, "cmd": cmd, "schema": "rsi/repl@1", "result": result, **({"id": req["id"]} if "id" in req else {})})
        except ValueError as e:
            err = as_rsi_error(e) if not isinstance(e, json.JSONDecodeError) else RsiError(f"bad JSON: {e}", "E_USAGE")
            if err.code == "E_INTERNAL":
                err = RsiError(str(e), "E_BAD_VALUE")
            say({"ok": False, "cmd": cmd, "schema": "rsi/repl@1", "error": err.to_dict()})
        except Exception as e:
            say({"ok": False, "cmd": cmd, "schema": "rsi/repl@1", "error": as_rsi_error(e).to_dict()})
    return 0
