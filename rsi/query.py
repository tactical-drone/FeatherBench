"""
query: the --where / --sort mini language over trial records.

    parse_where("test_acc>=0.95")(rec)      PATH OP VALUE, OP in == != < <= > >= ~ in
    parse_where("config.model==mlp")        VALUE is JSON when it parses, else a string
    parse_where("tag~ab")                   ~ = substring, or list-contains; `in` takes a ;-list
    parse_sort("-fitness") -> ("fitness", True)

Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md.
"""
import json
import re

from .errors import RsiError

ALIASES = {"test_acc": "summary.test_acc.mean", "train_acc": "summary.train_acc.mean",
           "fresh_acc": "summary.fresh_acc.mean", "test_loss": "summary.test_loss.mean",
           "train_loss": "summary.train_loss.mean", "params": "summary.n_params", "seconds": "summary.seconds",
           "tag": "tags", "dataset": "config.dataset", "model": "config.model"}
_WHERE = re.compile(r"^\s*([A-Za-z_][\w.\[\]\-]*)\s*(==|!=|<=|>=|<|>|~|\s+in\s+|=)\s*(.*?)\s*$", re.S)


def expand_path(path):
    return ALIASES.get(path, path)


def get_path(rec, path):
    """Dotted lookup with aliases ('test_acc' = 'summary.test_acc.mean'); list[i] works; None if absent."""
    cur = rec
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", expand_path(path)):
        if part.startswith("[") and isinstance(cur, list):
            i = int(part[1:-1])
            cur = cur[i] if i < len(cur) else None
        elif isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.isdigit():
            cur = cur[int(part)] if int(part) < len(cur) else None
        else:
            return None
        if cur is None:
            return None
    return cur


def _value(text):
    try:
        return json.loads(text)
    except ValueError:
        return text.strip("'\"") if len(text) > 1 and text[0] == text[-1] and text[0] in "'\"" else text


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _comparable(a, b):
    return (_num(a) and _num(b)) or (isinstance(a, str) and isinstance(b, str))


def _cmp(a, op, b):
    if op == "in":
        return any(x in b for x in a) if isinstance(a, list) else a in b
    if a is None:
        return (op == "==" and b is None) or (op == "!=" and b is not None)
    if op == "~":
        return b in a if isinstance(a, list) else str(b).lower() in str(a).lower()
    if isinstance(a, list) and op in ("==", "!="):
        return (b in a) == (op == "==")
    if op in ("==", "!="):
        eq = a == b if _comparable(a, b) or b is None else str(a) == str(b)
        return eq == (op == "==")
    if isinstance(a, str) and _num(b):  # 'created>=2026': compare as text
        b = str(b)
    if not _comparable(a, b):
        if _num(a) and isinstance(b, str):  # 'test_acc>abc': a typo, not an empty result
            raise RsiError(f"bad --where: {b!r} is not a number, but the field holds numbers ({a!r})", "E_BAD_QUERY",
                           value=b, hint='e.g. --where "test_acc>=0.95"')
        return False
    return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]


def parse_where(expr):
    """'PATH OP VALUE' -> predicate(rec) -> bool. Raises RsiError E_BAD_QUERY."""
    m = _WHERE.match(expr or "")
    if not m or m.group(3) == "":
        raise RsiError(f"bad --where {expr!r}: expected PATH OP VALUE with OP in == != < <= > >= ~ in",
                       "E_BAD_QUERY", value=expr, hint="e.g. --where \"test_acc>=0.95\" --where \"config.model==mlp\"")
    path, op, raw = m.group(1), m.group(2).strip(), m.group(3)
    if op == "=":
        if raw[0] in "<>!":  # '=>' / '=<' / '=!': the operator written backwards
            raise RsiError(f"bad --where {expr!r}: '={raw[0]}' is not an operator; did you mean '{raw[0]}='?",
                           "E_BAD_QUERY", value=expr, did_you_mean=[f"{path}{raw[0]}={raw[1:].lstrip()}"],
                           hint="OP is one of == != < <= > >= ~ in")
        raise RsiError(f"bad --where {expr!r}: '=' is not an operator; equality is '=='", "E_BAD_QUERY", value=expr,
                       did_you_mean=[f"{path}=={raw}"], hint="OP is one of == != < <= > >= ~ in")
    if raw[0] in "<>=!~":
        raise RsiError(f"bad --where {expr!r}: the value starts with {raw[0]!r} (a doubled operator?)", "E_BAD_QUERY",
                       value=expr, did_you_mean=[f"{path}{op}{raw.lstrip('<>=!~')}"],
                       hint="OP is one of == != < <= > >= ~ in; quote a value that really starts with it")
    if op == "in":
        val = [_value(v.strip()) for v in raw.split(";") if v.strip()]
    else:
        val = _value(raw)
    return lambda rec: _cmp(get_path(rec, path), op, val)


def parse_sort(key):
    """'-fitness' -> ('fitness', True); 'params' -> ('params', False)."""
    key = (key or "").strip()
    if not key or key in ("-", "+"):
        raise RsiError("empty --sort key", "E_BAD_QUERY")
    return (key[1:], True) if key[0] == "-" else (key.lstrip("+"), False)


def sort_records(recs, sort=()):
    """Stable multi-key sort; missing values go last whatever the direction."""
    out = list(recs)
    for key in reversed([parse_sort(k) for k in sort]):
        path, desc = key
        present = [r for r in out if get_path(r, path) is not None]
        absent = [r for r in out if get_path(r, path) is None]
        try:
            present.sort(key=lambda r: get_path(r, path), reverse=desc)
        except TypeError:
            present.sort(key=lambda r: str(get_path(r, path)), reverse=desc)
        out = present + absent
    return out


def select_fields(rec, fields):
    return {f: get_path(rec, f) for f in fields}
