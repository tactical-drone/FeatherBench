"""
schemas: JSON Schema (2020-12) documents for `python -m rsi schema NAME`.
They describe the stable fields; documents may carry more (adding a field is
not a breaking change; a breaking change bumps @N).

Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md.
"""
D = "https://json-schema.org/draft/2020-12/schema"
NUM = {"type": ["number", "null"]}
STR = {"type": "string"}
INT = {"type": "integer"}
OBJ = {"type": "object"}
STATS = {"type": ["object", "null"], "properties": {k: NUM for k in ("mean", "median", "min", "max", "std")} | {"n": INT}}
SPLIT = {"type": ["object", "null"], "properties": {"loss": NUM, "acc": NUM}, "additionalProperties": NUM}
WARNING = {"type": "object", "required": ["code", "message"],
           "properties": {"code": {"type": "string", "pattern": "^W_"}, "message": STR, "field": {"type": ["string", "null"]}}}
ERROR = {"type": "object", "required": ["code", "exit", "message"],
         "properties": {"code": {"type": "string", "pattern": "^E_"}, "exit": INT, "message": STR,
                        "field": {"type": ["string", "null"]}, "value": {},
                        "did_you_mean": {"type": "array"}, "allowed": {"type": ["array", "null"]},
                        "hint": {"type": ["string", "null"]}, "traceback": {"type": ["string", "null"]},
                        "details": OBJ}}


def _config():
    from nncore.configio import FIELDS
    return {"type": "object", "required": FIELDS, "properties": {f: {} for f in FIELDS} | {
        "features": {"type": "array", "items": STR, "minItems": 1}, "batch_size": {"type": ["integer", "null"]},
        "layers": STR, "extra": OBJ}, "additionalProperties": False}


CELL = {"type": "object", "required": ["key", "dataset", "seed", "steps", "steps_done", "status", "train", "test"],
        "properties": {"key": {"type": ["string", "null"], "pattern": "^k_[0-9a-f]{16}$"}, "config_key": {"type": ["string", "null"]},
                       "dataset": STR, "seed": INT, "steps": INT, "steps_done": INT,
                       "status": {"enum": ["ok", "diverged", "early_stopped", "timeout", "error"]},
                       "diverged_at": {"type": ["integer", "null"]}, "train": SPLIT, "test": SPLIT, "fresh": SPLIT,
                       "model": {"type": ["object", "null"], "properties": {"describe": STR, "n_params": INT,
                                                                            "hidden_sizes": {"type": "array"}, "n_classes": INT}},
                       "seconds": {"type": "number"}, "cached": {"type": "boolean"},
                       "curve": {"type": ["object", "null"]},
                       "error": {"type": ["object", "null"], "properties": {"type": STR, "message": STR,
                                                                            "phase": {"enum": ["build", "train", "eval", "worker"]}}}}}


def _trial():
    return {"type": "object", "required": ["id", "created", "status", "config", "config_key", "budget", "cells", "summary"],
            "properties": {"id": {"type": "string", "pattern": "^t_[0-9a-f]{12}$"}, "created": STR,
                           "status": {"enum": ["ok", "partial", "failed"]}, "cached": {"type": "boolean"},
                           "config": _config(), "config_diff": OBJ, "config_key": STR,
                           "budget": {"type": "object", "properties": {"steps": INT, "eval_every": INT,
                                                                       "max_seconds": NUM, "early_stop": {"type": ["object", "null"]},
                                                                       "fresh_points": INT}},
                           "seeds": {"type": "array", "items": INT}, "datasets": {"type": "array", "items": STR},
                           "cells": {"type": "array", "items": {"anyOf": [CELL, {"type": "null"}]}},
                           "summary": {"type": "object", "properties": {"n_cells": INT, "n_ok": INT, "test_acc": STATS,
                                                                        "train_acc": STATS, "n_params": {"type": ["integer", "null"]}}},
                           "objective": STR, "fitness": NUM, "map": {"type": ["array", "null"], "items": STR},
                           "png": {"type": ["string", "null"]}, "settings_file": {"type": ["string", "null"]},
                           "tags": {"type": "array", "items": STR}, "run": {"type": ["string", "null"]}}}


def _entry():
    return {"type": "object", "required": ["rank", "score", "config"],
            "properties": {"rank": INT, "genome_id": STR, "trial_id": {"type": ["string", "null"]}, "score": NUM,
                           "fitness": OBJ, "holdout": {"type": ["object", "null"]}, "genome": OBJ, "config_diff": OBJ,
                           "config": _config(), "describe": {"type": ["string", "null"]},
                           "settings_file": {"type": ["string", "null"]}, "repro": STR}}


def _space():
    gene = {"type": "object", "required": ["kind"], "properties": {
        "kind": {"enum": ["choice", "part", "activation", "ordinal", "int", "float", "subset", "layers"]},
        "choices": {"anyOf": [{"type": "array"}, {"const": "*"}]}, "registry": STR, "exclude": {"type": "array"},
        "extra_choices": {"type": "array"}, "low": {"type": "number"}, "high": {"type": "number"},
        "log": {"type": "boolean"}, "quantum": {"type": "number"}, "sig": INT, "zero_prob": {"type": "number"},
        "min": INT, "must_include": {"type": "array"}, "min_depth": INT, "max_depth": INT, "widths": {"type": "array"},
        "act": OBJ, "when": {"type": "object", "additionalProperties": {"type": "array"}}}}
    return {"type": "object", "required": ["format", "version", "genes"],
            "properties": {"format": {"const": "rsi/space"}, "version": {"const": 1}, "ui_safe": {"type": "boolean"},
                           "genes": {"type": "object", "additionalProperties": gene},
                           "inert": {"type": "object", "additionalProperties": {"type": "array", "items": STR}},
                           "frozen_by_default": {"type": "array", "items": STR},
                           "mutation_weights": {"type": "object", "additionalProperties": {"type": "number"}},
                           "constraints": {"type": "object", "properties": {"max_params": INT, "max_est_seconds": {"type": "number"}}}}}


def get(name):
    """The JSON Schema for one document name (see NAMES)."""
    meta = {"type": "object", "properties": {"rsi": STR, "nncore": STR, "torch": STR, "code_fp": STR, "parts": {"type": "array"}}}
    docs = {
        "envelope": {"type": "object", "required": ["ok", "command", "schema"],
                     "properties": {"ok": {"type": "boolean"}, "command": STR, "schema": {"type": "string", "pattern": "^rsi/[a-z-]+@\\d+$"},
                                    "result": {}, "error": ERROR, "warnings": {"type": "array", "items": WARNING},
                                    "meta": meta, "elapsed_s": {"type": "number"}}},
        "error": {"type": "object", "required": ["ok", "error"],
                  "properties": {"ok": {"const": False}, "schema": {"const": "rsi/error@1"}, "error": ERROR, "result": {}}},
        "settings": {"type": "object", "required": ["format", "version", "config"],
                     "properties": {"format": {"const": "nn-playground/settings"}, "version": {"const": 1},
                                    "config": _config(), "train": {"type": "object", "properties": {"steps": INT, "eval_every": INT}},
                                    "ui": OBJ, "meta": {"type": "object", "properties": {
                                        "saved_by": STR, "created": STR, "run": STR, "genome_id": STR, "trial_id": STR,
                                        "rank": INT, "objective": STR, "score": NUM, "holdout_score": NUM,
                                        "expect": {"type": "object", "properties": {"train_acc": NUM, "test_acc": NUM, "seed": INT,
                                                                                    "at_step": INT, "code_fp": STR, "platform": STR}},
                                        "parts": {"type": "array", "items": STR}, "note": STR}},
                                    "parts": OBJ}},
        "cell": CELL,
        "trial": _trial(),
        "sweep": {"type": "object", "required": ["kind", "name", "status", "trials"],
                  "properties": {"kind": {"const": "sweep"}, "name": STR, "dir": STR, "status": {"enum": ["done", "stopped"]},
                                 "axes": OBJ, "mode": {"enum": ["grid", "oat", "random"]}, "n_trials": INT, "n_cells": INT,
                                 "n_cached": INT, "n_failed": INT, "ranked_by": STR,
                                 "trials": {"type": "array", "items": {"type": "object", "properties": {
                                     "rank": INT, "id": STR, "fitness": NUM, "config_diff": OBJ, "summary": OBJ}}},
                                 "groups": {"type": "array"}, "best": {"type": ["object", "null"]}}},
        "space": _space(),
        "evolve-status": {"type": "object", "required": ["name", "dir", "state"],
                          "properties": {"name": STR, "dir": STR, "kind": STR,
                                         "state": {"enum": ["running", "stopping", "stopped", "budget", "done", "crashed", "stale"]},
                                         "pid": {"type": ["integer", "null"]}, "alive": {"type": "boolean"},
                                         "heartbeat_age_s": NUM, "generation": INT, "generations": INT, "evals_done": INT,
                                         "best": {"type": ["object", "null"]}, "history": {"type": "array"}, "files": OBJ,
                                         "next": STR}},
        "leaderboard": {"type": "object", "required": ["entries"],
                        "properties": {"name": STR, "kind": STR, "state": {"type": ["string", "null"]}, "sort": STR,
                                       "baseline": {"type": ["object", "null"]}, "entries": {"type": "array", "items": _entry()},
                                       "pareto": {"type": "array"}, "stats": {"type": ["object", "null"]}}},
        "repl": {"type": "object", "required": ["ok"],
                 "properties": {"ok": {"type": "boolean"}, "cmd": {"type": ["string", "null"]}, "schema": {"const": "rsi/repl@1"},
                                "result": OBJ, "error": ERROR},
                 "$comment": "requests: {cmd: set|train|eval|map|predict|reset|describe|save|quit, ...}"},
    }
    if name not in docs:
        return None
    return {"$schema": D, "$id": f"rsi/{name}@1", "title": f"rsi {name}", **docs[name]}


NAMES = ("envelope", "error", "settings", "cell", "trial", "sweep", "space", "evolve-status", "leaderboard", "repl")
