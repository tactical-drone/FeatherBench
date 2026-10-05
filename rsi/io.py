"""
io: the stdout contract. install_guard() sends every print (parts, torch,
libraries) to stderr; only emit() writes the one JSON document to the real
stdout. Progress goes to stderr through Logger (text | jsonl | quiet).

Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md.
"""
import json
import math
import sys
import time

REAL_OUT = None


def _reconf(stream, **kw):
    try:
        stream.reconfigure(**kw)
    except (AttributeError, ValueError, OSError):
        pass


def install_guard():
    """sys.stdout -> sys.stderr for the rest of the process; returns the real stdout."""
    global REAL_OUT
    if REAL_OUT is None:
        REAL_OUT = sys.stdout
        _reconf(sys.stderr, encoding="utf-8", errors="replace", line_buffering=True)
        sys.stdout = sys.stderr
    return REAL_OUT


def set_unicode(stream=None):
    """--unicode: raw UTF-8 JSON on stdout."""
    _reconf(stream or REAL_OUT or sys.stdout, encoding="utf-8", errors="strict")


def to_json(obj, *, pretty=False, ascii=True, nonfinite=None):
    from nncore.configio import json_safe
    return json.dumps(json_safe(obj, nonfinite=nonfinite), ensure_ascii=ascii, allow_nan=False,
                      indent=2 if pretty else None)


def fit_encoding(text, encoding):
    """text with each character `encoding` cannot write folded to ASCII ('θ' -> 'theta',
    '→' -> '->'), or '?' (a cp1252 console would otherwise raise UnicodeEncodeError)."""
    if not encoding:
        return text
    try:
        text.encode(encoding)
        return text
    except UnicodeEncodeError:
        pass
    except LookupError:
        return text
    from nncore.configio import ascii_name

    def one(ch):
        try:
            ch.encode(encoding)
            return ch
        except UnicodeEncodeError:
            return ascii_name(ch) or "?"
    return "".join(one(ch) for ch in text)


def emit(text, out=None):
    """Write one document (a str ending without newline) to the real stdout."""
    out = out or REAL_OUT or sys.stdout
    text = fit_encoding(text, getattr(out, "encoding", None))
    try:
        out.write(text + "\n")
        out.flush()
    except BrokenPipeError:
        pass


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------- progress
class Logger:
    """Progress events on stderr. log(event, **fields); text mode prints one short line."""
    def __init__(self, mode="text", stream=None):
        self.mode, self.stream = mode, stream

    def __call__(self, event, **fields):
        if self.mode == "quiet":
            return
        stream = self.stream or sys.stderr
        try:
            if self.mode == "jsonl":
                stream.write(to_json({"event": event, "t": round(time.time(), 3), **fields}) + "\n")
            else:
                msg = fields.pop("message", None)
                bits = " ".join(f"{k}={_short(v)}" for k, v in fields.items() if v is not None)
                stream.write(f"[rsi] {event}" + (f": {msg}" if msg else "") + (f"  {bits}" if bits else "") + "\n")
            stream.flush()
        except (BrokenPipeError, ValueError):
            pass


def _short(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (dict, list)):
        s = json.dumps(v, ensure_ascii=False, default=str)
        return s if len(s) < 80 else s[:77] + "..."
    return str(v)


# ---------------------------------------------------------------- --format text
def _num(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return "nan" if not math.isfinite(v) else f"{v:.4f}"
    return str(v)


def _table(rows, cols):
    cells = [[_num(r.get(c)) if not isinstance(r.get(c), str) else r.get(c) for c in cols] for r in rows]
    widths = [max([len(c)] + [len(x[i]) for x in cells]) for i, c in enumerate(cols)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(cols, widths))]
    lines += ["  ".join(x.ljust(w) for x, w in zip(row, widths)) for row in cells]
    return "\n".join(lines)


def _mean(summary, key):
    return ((summary or {}).get(key) or {}).get("mean")


def render_text(doc):
    """A human rendering of an envelope (tables for the common schemas)."""
    if not doc.get("ok"):
        e = doc["error"]
        s = f"error {e['code']} (exit {e['exit']}): {e['message']}"
        if e.get("did_you_mean"):
            s += f"\n  did you mean: {', '.join(map(str, e['did_you_mean']))}"
        if e.get("hint"):
            s += f"\n  hint: {e['hint']}"
        return s
    r, schema = doc.get("result") or {}, doc.get("schema", "")
    out = []
    if schema == "rsi/trial@1":
        sm = r.get("summary") or {}
        out.append(f"trial {r.get('id')}  status {r.get('status')}  fitness {_num(r.get('fitness'))}  "
                   f"params {sm.get('n_params')}  cached {r.get('cached')}")
        out.append(f"config diff: {json.dumps(r.get('config_diff'), ensure_ascii=False)}")
        rows = [{"dataset": c.get("dataset"), "seed": c.get("seed"), "status": c.get("status"),
                 "train_acc": (c.get("train") or {}).get("acc"), "test_acc": (c.get("test") or {}).get("acc"),
                 "fresh_acc": (c.get("fresh") or {}).get("acc"), "secs": c.get("seconds")} for c in r.get("cells", [])]
        out.append(_table(rows, ["dataset", "seed", "status", "train_acc", "test_acc", "fresh_acc", "secs"]))
        if r.get("map"):
            out += r["map"]
    elif schema == "rsi/sweep@1":
        out.append(f"sweep {r.get('name')}  {r.get('status')}  trials {r.get('n_trials')}  cells {r.get('n_cells')}"
                   f"  cached {r.get('n_cached')}  failed {r.get('n_failed')}")
        rows = [{"rank": t["rank"], "fitness": t.get("fitness"), "test_acc": _mean(t.get("summary"), "test_acc"),
                 "params": (t.get("summary") or {}).get("n_params"), "id": t.get("id"),
                 "diff": json.dumps(t.get("config_diff"), ensure_ascii=False)} for t in r.get("trials", [])]
        out.append(_table(rows, ["rank", "fitness", "test_acc", "params", "id", "diff"]))
    elif schema == "rsi/leaderboard@1":
        rows = [{"rank": e.get("rank"), "score": e.get("score"), "holdout": (e.get("holdout") or {}).get("score"),
                 "params": (e.get("fitness") or {}).get("n_params"), "id": e.get("genome_id") or e.get("trial_id"),
                 "diff": json.dumps(e.get("config_diff"), ensure_ascii=False)} for e in r.get("entries", [])]
        out.append(_table(rows, ["rank", "score", "holdout", "params", "id", "diff"]))
    elif schema == "rsi/query@1":
        rows = r.get("rows", [])
        cols = list(dict.fromkeys(k for row in rows for k in row)) or ["id"]
        out.append(_table(rows, cols))
    elif schema == "rsi/describe@1" and "levers" in r:
        rows = [{"name": lv["name"], "kind": lv["kind"], "default": json.dumps(lv["default"], ensure_ascii=False),
                 "read_by": lv["read_by"] if isinstance(lv["read_by"], str) else ",".join(lv["read_by"]),
                 "help": lv.get("help", "")} for lv in r["levers"]]
        out.append(_table(rows, ["name", "kind", "default", "read_by", "help"]))
    else:
        out.append(json.dumps(r, ensure_ascii=False, indent=2, default=str))
    for w in doc.get("warnings") or []:
        out.append(f"warning {w.get('code')}: {w.get('message')}")
    return "\n".join(out)
