"""
configio: Config <-> JSON, value coercion, part-name resolution, bounds,
validation, settings files and hashing. No torch randomness, no UI.

    cfg, warnings = config_from_dict({"model": "mlp", "features": "x,y,x^2", "lr": "0.01"})
    check_config(cfg)                       # ConfigError with .code / .did_you_mean
    save_settings("winner.json", cfg, train={"steps": 3000})
    s = load_settings("winner.json")        # s.config, s.ui, s.train, s.meta, s.warnings
    config_key(cfg)                         # 16-hex hash, stable across key order and 1 == 1.0

Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md.
"""
import copy
import dataclasses
import difflib
import hashlib
import json
import math
import os
import re
import tempfile
import time
import unicodedata
from typing import NamedTuple

from .activations import ACTIVATIONS
from .config import DATA_KEYS, LIVE_KEYS, MODEL_KEYS, OPT_KEYS, SAMPLER_KEYS, Config
from .datasets import DATASETS, SPLITTERS
from .features import FEATURES
from .initializers import INITIALIZERS
from .layers import EXPANSIONS, LAYERS, SKIPS
from .losses import LOSSES
from .models import MODELS, parse_layers
from .optimizers import OPTIMIZERS, SCHEDULES
from .registry import ensure_plugins, loaded_plugins
from .samplers import SAMPLERS
from .training import TRAIN_STEPS

SETTINGS_FORMAT = "nn-playground/settings"
SETTINGS_VERSION = 1

PART_REGISTRIES = {
    "dataset": DATASETS, "splitter": SPLITTERS, "features": FEATURES, "model": MODELS,
    "layer": LAYERS, "skip": SKIPS, "expand": EXPANSIONS, "activation": ACTIVATIONS, "init": INITIALIZERS,
    "loss": LOSSES, "optimizer": OPTIMIZERS, "schedule": SCHEDULES, "train_step": TRAIN_STEPS, "sampler": SAMPLERS,
}  # registry key -> Registry (nncore.REGISTRIES is the older, UI-facing map)
FIELD_REGISTRY = {
    "dataset": "dataset", "splitter": "splitter", "features": "features", "model": "model",
    "expand": "expand", "activation": "activation", "act_decide": "activation", "act_relate": "activation",
    "act_prepare": "activation", "layer": "layer", "skip": "skip", "skip_decide": "skip", "skip_relate": "skip",
    "skip_prepare": "skip", "init": "init", "loss": "loss",
    "optimizer": "optimizer", "schedule": "schedule", "train_step": "train_step", "sampler": "sampler",
}  # Config field -> PART_REGISTRIES key ("features" holds several names)
SPECIAL_VALUES = {"act_decide": {"same"}, "act_relate": {"same"}, "act_prepare": {"same"},
                  "skip_decide": {"same"}, "skip_relate": {"same"}, "skip_prepare": {"same"}}
FIELD_BOUNDS = {"n_points": (1, 100000), "noise": (0.0, 1.0), "test_frac": (0.0, 0.999999), "seed": (0, 2**32 - 1),
                "width": (0, 512), "classes": (1, 512), "fourier_freq": (1e-6, 100.0), "lr": (1e-6, 100.0),
                "weight_decay": (0.0, 1.0), "batch_size": (1, 100000)}  # hard validity, inclusive
UI_LIMITS = {"n_points": (20, 5000), "noise": {"min": 0.0, "max": 0.5, "step": 0.01}, "seed": (0, 99999),
             "width": (0, 128), "classes": (1, 128),
             "weight_decay": {"choices": [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]},
             "batch_size": {"choices": [None, 8, 16, 32, 64, 128, 256]}}  # what the UI widgets show exactly
FIELD_HELP = {
    "dataset": "2-D problem; classes = max(y)+1",
    "n_points": "points generated before the split (some datasets round it)",
    "noise": "dataset-specific noise level (UI slider 0..0.5)",
    "splitter": "how points split into train / test",
    "test_frac": "fraction held out for test, 0 <= f < 1",
    "seed": "drives data, split, init AND batch order; use --seeds for replicates",
    "features": "input columns, in the order given (order changes the numbers)",
    "model": "architecture; 'custom nn' lives in my_parts.py",
    "width": "custom nn: hidden width",
    "classes": "custom nn: width of the head/decide/perc stage",
    "expand": "custom nn: input expansion before the hidden layer",
    "fourier_freq": "custom nn: spread of starting frequencies for 'fourier (sin)'",
    "layers": "mlp: '8,8' or '8:sin,4:tanh'; empty = linear",
    "activation": "default activation (mlp layers without :act; custom nn hidden)",
    "act_decide": "custom nn: activation of decide; 'same' = activation",
    "act_relate": "custom nn: activation of relate; 'same' = activation",
    "act_prepare": "custom nn: activation of prepare; 'same' = activation",
    "layer": "weighted transform per mlp layer",
    "skip": "how a layer output combines with its input (custom nn: the skip around perc)",
    "skip_decide": "custom nn: skip around decide (input head); 'same' = skip",
    "skip_relate": "custom nn: skip around relate (input perc); 'same' = skip",
    "skip_prepare": "custom nn: skip around prepare (input relate); 'same' = skip",
    "init": "weight initialisation, run right after the build",
    "loss": "training loss (also the reported loss)",
    "optimizer": "torch optimizer",
    "lr": "learning rate",
    "weight_decay": "L2 weight decay passed to the optimizer",
    "schedule": "learning-rate schedule, stepped once per train step",
    "train_step": "what one optimisation step does",
    "sampler": "which training rows each step sees",
    "batch_size": "rows per step; null = full batch",
    "extra": "free-form knobs for your own parts",
}

FIELDS = [f.name for f in dataclasses.fields(Config)]
_DEFAULT = Config()
INT_FIELDS = {"n_points", "seed", "width", "classes"}
FLOAT_FIELDS = {"noise", "test_frac", "fourier_freq", "lr", "weight_decay"}
LEVEL = {**{k: "data" for k in DATA_KEYS}, **{k: "model" for k in MODEL_KEYS}, **{k: "optimizer" for k in OPT_KEYS},
         **{k: "sampler" for k in SAMPLER_KEYS}, **{k: "live" for k in LIVE_KEYS}}


class ConfigError(ValueError):
    """A bad config value. .code is one of E_UNKNOWN_FIELD, E_UNKNOWN_PART, E_AMBIGUOUS,
    E_BAD_VALUE, E_OUT_OF_RANGE, E_BAD_LAYERS, E_BAD_SETTINGS, E_PARTS_MISSING, or a resolver's
    ResolveError code (E_BAD_EXPR for a malformed gp: activation)."""
    def __init__(self, message, code="E_BAD_VALUE", field=None, value=None, did_you_mean=None, allowed=None,
                 hint=None):
        super().__init__(message)
        self.code, self.field, self.value, self.hint = code, field, value, hint
        self.did_you_mean, self.allowed = list(did_you_mean or []), allowed

    def to_dict(self):
        return json_safe({"code": self.code, "message": str(self), "field": self.field, "value": self.value,
                          "did_you_mean": self.did_you_mean, "allowed": self.allowed, "hint": self.hint})


def _issue(err):
    return {"code": err.code, "field": err.field, "value": json_safe(err.value), "message": str(err),
            "did_you_mean": err.did_you_mean}


# ---------------------------------------------------------------- names
_ASCII = {"²": "^2", "³": "^3", "·": "*", "÷": "/", "×": "x", "→": "->", "θ": "theta", "γ": "gamma",
          "σ": "sigma", "π": "pi", "√": "sqrt", "≥": ">=", "≤": "<=", "–": "-", "—": "-"}


def ascii_name(name):
    """Typeable spelling of a part name: 'x²' -> 'x^2', 'step (÷10 every 2000)' -> 'step (/10 every 2000)'."""
    s = "".join(_ASCII.get(ch, ch) for ch in name)
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def _fold(text):
    s = ascii_name(" ".join(str(text).split())).lower()
    return s.replace("^", "").replace("->", ">")


def _allowed(field):
    reg = PART_REGISTRIES[FIELD_REGISTRY[field]]
    return list(reg) + sorted(SPECIAL_VALUES.get(field, ()))


def _suggest(text, names):
    low = {n.lower(): n for n in names}
    out = [low[m] for m in difflib.get_close_matches(str(text).lower(), list(low), n=3, cutoff=0.5)]
    out += [n for n in names if _fold(n) == _fold(text)]
    return list(dict.fromkeys(out))[:3]


_FAMILY_LIKE = re.compile(r"\s*([A-Za-z][\w -]*?)\s*\[\s*(\d+)\s*\]\s*\Z")


def _family_issue(text):
    """For a 'family[size]'-looking dataset name that does not resolve: (canonical name or None,
    ConfigError) -- a known family with the case / spacing off resolves to its canonical name; a
    size out of range is E_OUT_OF_RANGE; a misspelt family gets close family names. None if the
    text does not look like family[size]."""
    from . import datasets
    fams = getattr(datasets, "FAMILIES", None)
    m = _FAMILY_LIKE.match(text) if isinstance(text, str) and fams is not None else None
    if m is None:
        return None
    name, size = m.group(1).strip().lower(), int(m.group(2))
    hint = "python -m rsi describe --section families"
    if name not in fams:
        close = difflib.get_close_matches(name, list(fams), n=3, cutoff=0.5)
        return None, ConfigError(f"unknown dataset family '{m.group(1).strip()}' in '{text}'", "E_UNKNOWN_PART",
                                 "dataset", text, [f"{c}[{size}]" for c in close], list(fams), hint)
    try:
        return fams[name].dataset_name(size), None
    except ValueError as e:
        return None, ConfigError(str(e), "E_OUT_OF_RANGE", "dataset", text, None, None, hint)


def resolve_name(field, text):
    """Part name for `field` -> (canonical, warning | None). Order: exact, case-insensitive,
    ASCII-folded ('x^2' -> 'x²', '/' -> '÷', '*' -> '·'), unique prefix. Raises ConfigError
    E_UNKNOWN_PART / E_AMBIGUOUS with did_you_mean."""
    if field not in FIELD_REGISTRY:
        raise ConfigError(f"'{field}' does not name a part", "E_UNKNOWN_FIELD", field, text)
    reg = PART_REGISTRIES[FIELD_REGISTRY[field]]
    special = SPECIAL_VALUES.get(field, set())
    if not isinstance(text, str):
        raise ConfigError(f"{field} must be a name (string), got {text!r}", "E_BAD_VALUE", field, text)
    if text in special or reg.has(text):
        return text, None
    ensure_plugins()
    if reg.has(text):
        return text, None
    fam = _family_issue(text) if field == "dataset" else None
    if fam is not None:
        if fam[1] is not None:
            raise fam[1]
        return fam[0], {"code": "W_ALIAS_USED", "field": field, "value": text, "canonical": fam[0],
                        "message": f"{field} '{text}' resolved to '{fam[0]}'"}
    names = _allowed(field)
    stripped = " ".join(text.split())
    for how, match in (("case", lambda n: n.lower() == stripped.lower()),
                       ("ascii", lambda n: _fold(n) == _fold(stripped)),
                       ("prefix", lambda n: stripped and _fold(n).startswith(_fold(stripped)))):
        hits = [n for n in names if match(n)]
        if len(hits) == 1:
            warn = {"code": "W_ALIAS_USED", "field": field, "value": text, "canonical": hits[0],
                    "message": f"{field} '{text}' resolved to '{hits[0]}'"}
            return hits[0], warn
        if len(hits) > 1:
            raise ConfigError(f"{field} '{text}' is ambiguous: {', '.join(hits)}", "E_AMBIGUOUS", field, text,
                              hits[:8], names, f"python -m rsi describe --lever {field}")
    rej = reg.rejection(text)
    if rej is not None:  # a resolver owns the name but rejected it ('gp:foo(x)'): say why
        raise ConfigError(str(rej), rej.code, field, text, hint=rej.hint)
    raise ConfigError(f"unknown {reg.kind} '{text}'", "E_UNKNOWN_PART", field, text, _suggest(text, names), names,
                      f"python -m rsi describe --lever {field}")


# ---------------------------------------------------------------- values
def _num(field, value):
    if isinstance(value, bool):
        raise ConfigError(f"{field} must be a number, got {value!r}", "E_BAD_VALUE", field, value)
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            raise ConfigError(f"{field} must be a number, got {value!r}", "E_BAD_VALUE", field, value) from None
    if hasattr(value, "item") and not isinstance(value, (int, float)):
        value = value.item()  # numpy / torch scalar
    if not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be a number, got {value!r}", "E_BAD_VALUE", field, value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ConfigError(f"{field} must be finite, got {value!r}", "E_BAD_VALUE", field, value)
    return value


def _int(field, value):
    v = _num(field, value)
    if isinstance(v, float):
        if not v.is_integer():
            raise ConfigError(f"{field} must be a whole number, got {value!r}", "E_BAD_VALUE", field, value)
        v = int(v)
    return v


def coerce_value(field, value):
    """str | JSON value -> the field's type: numeric strings -> numbers, integral floats -> int
    for int fields, 'full'/'none'/'null'/None -> None for batch_size, 'x,y' -> ('x', 'y').
    Names are not resolved here (see resolve_name); bounds are checked by validate_config."""
    if field not in FIELDS:
        raise ConfigError(f"unknown config field '{field}'", "E_UNKNOWN_FIELD", field, value,
                          difflib.get_close_matches(str(field), FIELDS, n=3), FIELDS)
    if field in INT_FIELDS:
        return _int(field, value)
    if field in FLOAT_FIELDS:
        return float(_num(field, value))
    if field == "batch_size":
        if value is None or (isinstance(value, str) and value.strip().lower() in ("full", "none", "null")):
            return None
        return _int(field, value)
    if field == "features":
        if isinstance(value, str):  # 'x,y,sin x' (';' works too) or a JSON list '["x", "y"]'
            text = value.strip()
            if text.startswith("["):
                try:
                    value = json.loads(text)
                except ValueError:
                    value = text.strip("[]").replace('"', "").split(",")
            else:
                value = re.split(r"[,;]", text)
        if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
            raise ConfigError(f"features must be a list of names, got {value!r}", "E_BAD_VALUE", field, value)
        return tuple(v.strip() for v in value if v.strip())
    if field == "layers":
        if isinstance(value, bool) or not isinstance(value, (str, int, list, tuple)):
            raise ConfigError(f"layers must be a string like '8,8', got {value!r}", "E_BAD_VALUE", field, value)
        return ",".join(map(str, value)) if isinstance(value, (list, tuple)) else str(value)
    if field == "extra":
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                raise ConfigError(f"extra must be a JSON object, got {value!r}", "E_BAD_VALUE", field,
                                  value) from None
        if not isinstance(value, dict):
            raise ConfigError(f"extra must be an object, got {value!r}", "E_BAD_VALUE", field, value)
        return copy.deepcopy(value)
    if not isinstance(value, str):
        raise ConfigError(f"{field} must be a name (string), got {value!r}", "E_BAD_VALUE", field, value)
    return value.strip()


def normalise(cfg):
    """A new Config with every value coerced to its type (features tuple, order kept).
    No name resolution, no validation; normalise(Config()) == Config()."""
    return Config(**{k: coerce_value(k, getattr(cfg, k)) for k in FIELDS})


def config_to_dict(cfg):
    """Every field in dataclass order, JSON-ready (features as a list)."""
    d = {k: copy.deepcopy(getattr(cfg, k)) for k in FIELDS}
    d["features"] = list(d["features"])
    return json_safe(d)


def _from_raw(d):
    unknown = [k for k in d if k not in FIELDS]
    if unknown:
        raise ConfigError(f"unknown config field '{unknown[0]}'", "E_UNKNOWN_FIELD", unknown[0], d[unknown[0]],
                          difflib.get_close_matches(unknown[0], FIELDS, n=3), FIELDS)
    return normalise(dataclasses.replace(Config(), **d))


def config_from_dict(d, base=None, *, strict=True):
    """(Config, warnings) from a dict. Missing fields come from `base` (default Config(),
    then with W_DEFAULT_FILLED). strict: unknown fields / parts / bad values raise ConfigError;
    lenient: W_UNKNOWN_KEY, and a bad value falls back to the default with a warning."""
    if not isinstance(d, dict):
        raise ConfigError(f"config must be an object, got {type(d).__name__}", "E_BAD_SETTINGS")
    warns = []
    cfg = copy.deepcopy(base) if base is not None else Config()
    if base is None:
        missing = [k for k in FIELDS if k not in d]
        if missing:
            warns.append({"code": "W_DEFAULT_FILLED", "field": None, "fields": missing,
                          "message": f"defaults used for: {', '.join(missing)}"})
    values = {}
    for k, v in d.items():
        try:
            v = coerce_value(k, v)
            if k == "features":
                items = []
                for item in v:
                    name, w = resolve_name(k, item)
                    items.append(name)
                    if w:
                        warns.append(w)
                v = tuple(items)
            elif k in FIELD_REGISTRY:
                v, w = resolve_name(k, v)
                if w:
                    warns.append(w)
            values[k] = v
        except ConfigError as e:
            if strict:
                raise
            if e.code == "E_UNKNOWN_FIELD":
                warns.append({"code": "W_UNKNOWN_KEY", "field": k, "value": json_safe(v),
                              "message": f"ignored unknown key '{k}'", "did_you_mean": e.did_you_mean})
            else:
                warns.append({"code": "W_UNKNOWN_PART" if e.code == "E_UNKNOWN_PART" else "W_DEFAULT_FILLED",
                              "field": k, "value": json_safe(v), "did_you_mean": e.did_you_mean,
                              "message": f"{e}; using {getattr(cfg, k)!r}"})
    cfg = normalise(dataclasses.replace(cfg, **values))
    for issue in validate_config(cfg):
        if strict:
            raise ConfigError(issue["message"], issue["code"], issue["field"], issue["value"],
                              issue.get("did_you_mean"))
        f = issue["field"]
        fallback = getattr(base, f) if base is not None else getattr(_DEFAULT, f)
        cfg = dataclasses.replace(cfg, **{f: copy.deepcopy(fallback)})
        warns.append({**issue, "code": "W_DEFAULT_FILLED", "message": f"{issue['message']}; using {fallback!r}"})
    return cfg, warns


# ---------------------------------------------------------------- validation
def validate_config(cfg):
    """Issues [{code, field, value, message, did_you_mean}] for a Config: types, registry
    names (exact), FIELD_BOUNDS, and parse_layers when model == 'mlp'. Draws no random numbers."""
    issues = []
    vals = {}
    for k in FIELDS:
        try:
            vals[k] = coerce_value(k, getattr(cfg, k))
        except ConfigError as e:
            issues.append(_issue(e))
    for k, reg_key in FIELD_REGISTRY.items():
        if k not in vals:
            continue
        for name in (vals[k] if k == "features" else (vals[k],)):
            if name in SPECIAL_VALUES.get(k, ()) or PART_REGISTRIES[reg_key].has(name):
                continue
            ensure_plugins()
            if not PART_REGISTRIES[reg_key].has(name):
                fam = _family_issue(name) if k == "dataset" else None
                if fam is not None:  # family[size]: out of range, misspelt family, or not canonical
                    err = fam[1] or ConfigError(f"unknown dataset '{name}'", "E_UNKNOWN_PART", k, name, [fam[0]])
                    issues.append(_issue(err))
                    continue
                rej = PART_REGISTRIES[reg_key].rejection(name)
                if rej is not None:
                    issues.append({"code": rej.code, "field": k, "value": name, "message": str(rej),
                                   "did_you_mean": [], "hint": rej.hint})
                    continue
                names = _allowed(k)
                issues.append({"code": "E_UNKNOWN_PART", "field": k, "value": name,
                               "message": f"unknown {PART_REGISTRIES[reg_key].kind} '{name}'",
                               "did_you_mean": _suggest(name, names)})
    for k, (lo, hi) in FIELD_BOUNDS.items():
        v = vals.get(k)
        if v is not None and not lo <= v <= hi:
            rng = "[0, 1)" if k == "test_frac" else f"[{lo}, {hi}]"
            issues.append({"code": "E_OUT_OF_RANGE", "field": k, "value": v,
                           "message": f"{k} must be in {rng}, got {v!r}", "did_you_mean": []})
    if vals.get("model") == "mlp" and "layers" in vals and not any(i["field"] in ("layers", "activation")
                                                                   for i in issues):
        try:
            parse_layers(vals["layers"], vals["activation"])
        except ValueError as e:
            acts = PART_REGISTRIES["activation"]
            bad = next((a for a in (t.partition(":")[2].strip() for t in str(vals["layers"]).split(",") if ":" in t)
                        if a and not acts.has(a) and not acts.has("".join(a.split()))), None)
            rej = acts.rejection(bad) if bad is not None else None
            if rej is not None:  # e.g. a malformed gp: expression inside layers
                issues.append({"code": rej.code, "field": "layers", "value": bad,
                               "message": f"layers {vals['layers']!r}: {rej}", "did_you_mean": [], "hint": rej.hint})
            elif bad is not None:  # an unknown activation inside layers: name it, with suggestions
                issues.append({"code": "E_BAD_LAYERS", "field": "layers", "value": bad,
                               "message": f"layers {vals['layers']!r}: unknown activation '{bad}'",
                               "did_you_mean": _suggest(bad, _allowed("activation"))})
            else:
                issues.append({"code": "E_BAD_LAYERS", "field": "layers", "value": vals["layers"],
                               "message": f"layers {vals['layers']!r}: {e}", "did_you_mean": []})
    return issues


def check_config(cfg):
    """Raise ConfigError for the first validate_config issue."""
    issues = validate_config(cfg)
    if issues:
        i = issues[0]
        if i.get("hint"):  # a resolver's reason (E_BAD_EXPR ...): its own hint, no part list
            raise ConfigError(i["message"], i["code"], i["field"], i["value"], i.get("did_you_mean"), hint=i["hint"])
        raise ConfigError(i["message"], i["code"], i["field"], i["value"], i.get("did_you_mean"),
                          _allowed(i["field"]) if i["field"] in FIELD_REGISTRY else None,
                          f"python -m rsi describe --lever {i['field']}")


def ui_issues(cfg):
    """W_NOT_UI_EXACT for values the UI widgets cannot show exactly; W_FEATURE_ORDER when
    features are not in registry order (the UI must keep the given order)."""
    out = []

    def warn(f, v, why):
        out.append({"code": "W_NOT_UI_EXACT", "field": f, "value": json_safe(v), "message": f"{f}={v!r}: {why}"})
    for f, lim in UI_LIMITS.items():
        v = getattr(cfg, f)
        if isinstance(lim, tuple) and not lim[0] <= v <= lim[1]:
            warn(f, v, f"UI range is {lim[0]}..{lim[1]}")
        elif isinstance(lim, dict) and "choices" in lim and v not in lim["choices"]:
            warn(f, v, f"UI offers {lim['choices']}")
        elif isinstance(lim, dict) and "step" in lim and (not lim["min"] <= v <= lim["max"]
                                                         or abs(round(v / lim["step"]) * lim["step"] - v) > 1e-9):
            warn(f, v, f"UI slider is {lim['min']}..{lim['max']} in steps of {lim['step']}")
    order = list(FEATURES)
    feats = [f for f in cfg.features if f in order]
    if feats != sorted(feats, key=order.index):
        out.append({"code": "W_FEATURE_ORDER", "field": "features", "value": list(cfg.features),
                    "message": "features are not in registry order; keep this order when loading in the UI"})
    return out


# ---------------------------------------------------------------- JSON
def json_safe(obj, *, nonfinite=None, _path=""):
    """NaN/inf -> None, tuples/sets -> lists, numpy/torch scalars and arrays -> python.
    Paths of replaced non-finite floats are appended to `nonfinite` if given."""
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        if nonfinite is not None:
            nonfinite.append(_path or "$")
        return None
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, dict):
        return {str(k): json_safe(v, nonfinite=nonfinite, _path=f"{_path}.{k}" if _path else str(k))
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        items = sorted(obj, key=repr) if isinstance(obj, (set, frozenset)) else obj
        return [json_safe(v, nonfinite=nonfinite, _path=f"{_path}[{i}]") for i, v in enumerate(items)]
    if isinstance(obj, Config):
        return config_to_dict(obj)
    if hasattr(obj, "tolist"):  # numpy / torch
        return json_safe(obj.tolist(), nonfinite=nonfinite, _path=_path)
    if hasattr(obj, "item"):
        return json_safe(obj.item(), nonfinite=nonfinite, _path=_path)
    return repr(obj)


def canonical_json(obj):
    """Stable text for hashing: sorted keys, no spaces, ASCII, repr floats, NaN -> null."""
    return json.dumps(json_safe(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def dumps(obj, *, pretty=False, ascii=True):
    return json.dumps(json_safe(obj), ensure_ascii=ascii, allow_nan=False, indent=2 if pretty else None)


def _as_config(cfg_or_dict):
    return _from_raw(cfg_or_dict) if isinstance(cfg_or_dict, dict) else normalise(cfg_or_dict)


def config_key(cfg_or_dict):
    """sha256 of the canonical JSON of the normalised config, first 16 hex."""
    return hashlib.sha256(canonical_json(config_to_dict(_as_config(cfg_or_dict))).encode()).hexdigest()[:16]


def config_diff(cfg, base=None):
    """{field: value} where cfg differs from base (default Config()), JSON-ready."""
    a = config_to_dict(_as_config(cfg))
    b = config_to_dict(_as_config(base) if base is not None else _DEFAULT)
    return {k: v for k, v in a.items() if v != b[k]}


# ---------------------------------------------------------------- settings files
class Settings(NamedTuple):
    config: Config
    ui: dict
    train: dict
    meta: dict
    parts: dict
    warnings: list


def _read_doc(path_or_obj):
    if isinstance(path_or_obj, dict):
        return copy.deepcopy(path_or_obj)
    try:
        with open(path_or_obj, encoding="utf-8-sig") as f:
            return json.load(f)
    except ValueError as e:
        raise ConfigError(f"{path_or_obj}: not valid JSON ({e})", "E_BAD_SETTINGS", value=str(path_or_obj)) from None


def load_settings(path_or_obj, *, strict=True):
    """Settings from a settings doc, a bare config dict, or a trial record (anything with a
    "config" object). Never imports code. strict (console) raises ConfigError; lenient (UI)
    falls back to defaults with warnings."""
    doc = _read_doc(path_or_obj)
    if not isinstance(doc, dict):
        raise ConfigError("settings must be a JSON object", "E_BAD_SETTINGS")
    ui, train, meta, parts = {}, {}, {}, {}
    if "format" in doc:
        if doc["format"] != SETTINGS_FORMAT:
            raise ConfigError(f"unknown settings format {doc['format']!r}", "E_BAD_SETTINGS", "format", doc["format"])
        version = doc.get("version", 1)
        if not isinstance(version, int) or version > SETTINGS_VERSION:
            raise ConfigError(f"settings version {version!r} is newer than this code ({SETTINGS_VERSION})",
                              "E_BAD_SETTINGS", "version", version)
        raw = doc.get("config", {})
        ui, train, meta, parts = (doc.get(k) or {} for k in ("ui", "train", "meta", "parts"))
    elif isinstance(doc.get("config"), dict):  # trial record / leaderboard entry
        raw = doc["config"]
        meta = {k: doc[k] for k in ("id", "trial_id", "genome_id", "run") if k in doc}
        if isinstance(doc.get("parts"), list):  # the part modules it was trained with
            meta["parts"] = list(doc["parts"])
    else:
        raw = doc
    try:
        cfg, warns = config_from_dict(raw, strict=strict)
    except ConfigError as e:
        missing = [m for m in (meta.get("parts") or []) if m not in loaded_plugins()]
        if e.code == "E_UNKNOWN_PART" and missing:
            raise ConfigError(f"{e} (the file was saved with parts {', '.join(missing)}, not loaded here; "
                              f"pass --parts)", "E_PARTS_MISSING", e.field, e.value, e.did_you_mean,
                              hint=f"--parts {' --parts '.join(missing)}") from None
        raise
    return Settings(cfg, dict(ui), dict(train), dict(meta), dict(parts), warns)


def _umask():
    old = os.umask(0)
    os.umask(old)
    return old


_UMASK = _umask()


def atomic_write_text(path, text):
    """Write via tmp file + fsync + os.replace (retried: Windows scanners hold files briefly).
    The file gets the old file's mode, or the usual 0666 & ~umask (mkstemp would make it 0600)."""
    path = os.fspath(path)
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    try:
        mode = os.stat(path).st_mode & 0o7777
    except OSError:
        mode = 0o666 & ~_UMASK
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.chmod(tmp, mode)
        except OSError:
            pass
        for i in range(5):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if i == 4:
                    raise
                time.sleep(0.05 * 2 ** i)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def save_settings(path, cfg, *, ui=None, train=None, meta=None, parts=None):
    """Write a settings doc atomically (UTF-8, ensure_ascii=False). An existing file's ui block
    is kept when ui is None."""
    if ui is None and os.path.exists(path):
        try:
            old = _read_doc(path)
            ui = old.get("ui") if isinstance(old, dict) and old.get("format") == SETTINGS_FORMAT else None
        except (OSError, ConfigError):
            ui = None
    doc = {"format": SETTINGS_FORMAT, "version": SETTINGS_VERSION, "config": config_to_dict(cfg),
           "train": train or {}, "ui": ui or {}, "meta": meta or {}, "parts": parts or {}}
    atomic_write_text(path, json.dumps(json_safe(doc), ensure_ascii=False, indent=1, allow_nan=False) + "\n")


_SAFE = re.compile(r"^[A-Za-z0-9_.,:/+=@%-]+$")


def _quote(s):
    s = str(s)
    if _SAFE.match(s):
        return s
    return '"' + re.sub(r'(["\\$`])', r"\\\1", s) + '"'


def config_to_command(cfg, base=None, *, prog="python -m rsi run"):
    """Copy-as-command: prog plus one flag per field that differs from base."""
    parts = [prog]
    for k, v in config_diff(cfg, base).items():
        flag = "--" + k.replace("_", "-")
        if k == "extra":
            for ek, ev in v.items():
                parts += ["--set", _quote(f"extra.{ek}={json.dumps(ev, ensure_ascii=False)}")]
        elif k == "features":
            parts += [flag, _quote(",".join(v))]
        elif k == "batch_size" and v is None:
            parts += [flag, "full"]
        elif k == "layers" and v == "":
            parts += ["--set", '"layers="']
        else:
            parts += [flag, _quote(repr(v) if isinstance(v, float) else v)]
    return " ".join(parts)
