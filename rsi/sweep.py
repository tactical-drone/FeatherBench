"""
sweep: turning --vary / --vary-json / --oat into lists of config overrides.

    expand_values("activation", "tanh,relu")       -> ['tanh', 'relu']   (';' splits instead when present)
    expand_values("act_decide", "@all+same")       -> every activation + 'same'
    expand_values("dataset", "@suite:general")     -> that suite's datasets
    grid({"activation": [...], "lr": [...]})       -> [{"activation": .., "lr": ..}, ...]  (product, in order)
    one_at_a_time(base, ["lr", "act_decide"])      -> [(field, value, overrides), ...]

Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md.
"""
import difflib
import itertools
import json

from nncore import configio
from nncore.config import Config

from .errors import RsiError


def coerce_override(field, value):
    """One override (field or 'extra.NAME') -> (field, canonical value, warnings)."""
    if field.startswith("extra."):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                pass
        return field, value, []
    v = configio.coerce_value(field, value)
    warns = []
    if field == "features":
        items = []
        for item in v:
            name, w = configio.resolve_name(field, item)
            items.append(name)
            warns += [w] if w else []
        v = tuple(items)
    elif field in configio.FIELD_REGISTRY:
        v, w = configio.resolve_name(field, v)
        warns += [w] if w else []
    return field, v, warns


def _split(spec):
    sep = ";" if ";" in spec else ","
    return [s.strip() for s in spec.split(sep) if s.strip() != ""] or ([""] if spec.strip() == "" else [])


def expand_values(field, spec):
    """Values for one axis: a list, or a ','/';'-separated string; '@all' (every registered
    name), '@all+same' (plus special values such as 'same'), '@suite:NAME' (datasets).
    Names are resolved, numbers coerced; layers values need ';' between them."""
    if field != "extra" and not field.startswith("extra.") and field not in configio.FIELDS:
        raise RsiError(f"unknown config field '{field}'", "E_UNKNOWN_FIELD", field=field,
                       did_you_mean=difflib.get_close_matches(field, configio.FIELDS, n=3))
    if isinstance(spec, (list, tuple)):
        raw = list(spec)
    else:
        spec = str(spec)
        if spec.startswith("@"):
            raw = _special(field, spec)
        elif field == "layers" and ";" not in spec:
            raw = [spec]  # one layers string; separate several with ';'
        else:
            raw = _split(spec)
    out = []
    for v in raw:
        _, val, _ = coerce_override(field, v)
        if val not in out:
            out.append(val)
    if not out:
        raise RsiError(f"--vary {field}: no values", "E_USAGE", field=field)
    return out


def _special(field, spec):
    if spec.startswith("@suite:"):
        if field != "dataset":
            raise RsiError("@suite:NAME only works for dataset", "E_USAGE", field=field)
        from .api import suite_names
        return suite_names(spec[7:])
    if spec in ("@all", "@all+same"):
        if field not in configio.FIELD_REGISTRY:
            raise RsiError(f"@all needs a part field, not '{field}'", "E_USAGE", field=field)
        if field == "dataset":
            from .api import suite_names
            return suite_names("all")
        names = list(configio.PART_REGISTRIES[configio.FIELD_REGISTRY[field]])
        if spec == "@all+same":
            names += sorted(configio.SPECIAL_VALUES.get(field, ()))
        return names
    raise RsiError(f"unknown value spec {spec!r} (use @all, @all+same, @suite:NAME)", "E_USAGE", field=field)


def grid(vary):
    """Cartesian product of {field: [values]} as override dicts, first axis slowest."""
    if not vary:
        return [{}]
    fields = list(vary)
    return [dict(zip(fields, combo)) for combo in itertools.product(*(vary[f] for f in fields))]


def oat_values(base, field):
    """The values --oat tries for one field: every other choice, or the lever's suggested values."""
    from nncore import schema
    cur = getattr(base, field)
    if field == "dataset":  # every dataset, as --vary dataset=@all (each trial trains its own)
        return [n for n in _special(field, "@all") if n != cur]
    if field in configio.FIELD_REGISTRY and field != "features":
        names = list(configio.PART_REGISTRIES[configio.FIELD_REGISTRY[field]])
        names += sorted(configio.SPECIAL_VALUES.get(field, ()))
        return [n for n in names if n != cur]
    if field == "features":  # toggle each feature (order kept, at least one left)
        out = []
        for f in configio.PART_REGISTRIES["features"]:
            new = tuple(x for x in cur if x != f) if f in cur else tuple(cur) + (f,)
            if new:
                out.append(new)
        return out
    sug = schema.SUGGESTED.get(field)
    if not sug:
        raise RsiError(f"--oat {field}: no suggested values; use --vary {field}=a,b", "E_USAGE", field=field)
    return [configio.coerce_value(field, v) for v in sug if configio.coerce_value(field, v) != cur]


def one_at_a_time(base, fields):
    """[(field, value, overrides)] changing one field at a time around base (a Config)."""
    base = base if isinstance(base, Config) else configio.config_from_dict(base)[0]
    out = []
    for f in fields:
        for v in oat_values(base, f):
            out.append((f, v, {f: v}))
    return out
