"""
store: <store>/rsi.sqlite, the cell cache (one row per RunSpec key) and the
trial index behind `runs query/show/stats`. WAL mode, busy_timeout 30 s;
only coordinator processes write (workers never open it).

    st = Store("runs")
    st.get_cell(key) / st.put_cell(key, cell)      only ok | diverged | early_stopped are cached
    st.put_trial(rec); st.trial("t_3f9a")          prefix lookup, E_NOT_FOUND / E_AMBIGUOUS_ID
    st.query(where=["test_acc>=0.9"], sort=["-fitness"], limit=20)

Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md.
"""
import contextvars
import functools
import json
import os
import sqlite3
import statistics
import time
from pathlib import Path

from .errors import RsiError
from .query import get_path, parse_where, select_fields, sort_records

CACHEABLE = ("ok", "diverged", "early_stopped")
DEFAULT_FIELDS = ("id", "created", "run", "status", "fitness", "test_acc", "params", "config_diff")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS cells (key TEXT PRIMARY KEY, config_key TEXT, dataset TEXT, seed INTEGER,
                                  status TEXT, steps INTEGER, cell TEXT NOT NULL, created REAL);
CREATE TABLE IF NOT EXISTS trials (id TEXT PRIMARY KEY, created TEXT, run TEXT, status TEXT, fitness REAL,
                                   config_key TEXT, tags TEXT, rec TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS trials_run ON trials(run);
CREATE INDEX IF NOT EXISTS cells_config ON cells(config_key);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""
_opened = contextvars.ContextVar("rsi_opened_stores", default=None)


def closes_stores(fn):
    """Close every Store opened during the call when the outermost decorated call returns.
    Windows cannot delete or move a run dir while rsi.sqlite is still open, and the API
    functions open stores from paths; a Store passed in by the caller stays open."""
    @functools.wraps(fn)
    def wrap(*args, **kwargs):
        if _opened.get() is not None:  # nested: the outermost call closes
            return fn(*args, **kwargs)
        token = _opened.set([])
        try:
            return fn(*args, **kwargs)
        finally:
            for st in _opened.get():
                st.close()
            _opened.reset(token)
    return wrap


def default_root():
    from nncore.registry import REPO_ROOT
    return Path(os.environ.get("RSI_STORE") or REPO_ROOT / "runs")


class Store:
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else default_root()
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self.path = self.root / "rsi.sqlite"
            self.db = sqlite3.connect(str(self.path), timeout=30, isolation_level=None, check_same_thread=False)
            self.db.execute("PRAGMA busy_timeout=30000")
            try:
                self.db.execute("PRAGMA journal_mode=WAL")
            except sqlite3.OperationalError:
                pass  # network drives: fall back to the default journal
            self.db.executescript(_SCHEMA)
            self.db.execute("INSERT OR IGNORE INTO meta VALUES ('schema', '1')")
            opened = _opened.get()
            if opened is not None:
                opened.append(self)
        except (OSError, sqlite3.Error) as e:
            raise RsiError(f"cannot open the store at {self.root}: {e}", "E_IO") from None

    def close(self):
        try:
            self.db.close()
        except sqlite3.Error:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _write(self, sql, args):
        for i in range(5):
            try:
                self.db.execute(sql, args)
                return
            except sqlite3.OperationalError as e:
                if "locked" not in str(e) or i == 4:
                    raise RsiError(f"store write failed: {e}", "E_IO") from None
                time.sleep(0.2 * (i + 1))

    # ---------------------------------------------------------------- cells
    def get_cell(self, key):
        row = self.db.execute("SELECT cell FROM cells WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put_cell(self, key, cell):
        if cell.get("status") not in CACHEABLE:
            return
        self._write("INSERT OR REPLACE INTO cells VALUES (?,?,?,?,?,?,?,?)",
                    (key, cell.get("config_key"), cell.get("dataset"), cell.get("seed"), cell.get("status"),
                     cell.get("steps"), json.dumps({**cell, "cached": False}, ensure_ascii=True), time.time()))

    def n_cells(self):
        return self.db.execute("SELECT COUNT(*) FROM cells").fetchone()[0]

    # ---------------------------------------------------------------- trials
    def put_trial(self, rec):
        self._write("INSERT OR REPLACE INTO trials VALUES (?,?,?,?,?,?,?,?)",
                    (rec["id"], rec.get("created"), rec.get("run"), rec.get("status"), rec.get("fitness"),
                     rec.get("config_key"), json.dumps(rec.get("tags") or []), json.dumps(rec, ensure_ascii=True)))

    def trial(self, id_or_prefix):
        """The trial with this id, or the only one whose id starts with it ('t_' optional)."""
        p = str(id_or_prefix).strip()
        row = self.db.execute("SELECT rec FROM trials WHERE id=?", (p,)).fetchone()
        if row:
            return json.loads(row[0])
        pre = p if p.startswith("t_") else "t_" + p
        esc = pre.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        rows = self.db.execute("SELECT id, rec FROM trials WHERE id LIKE ? ESCAPE '\\' LIMIT 11",
                               (esc + "%",)).fetchall()
        if not rows:
            raise RsiError(f"no trial '{p}' in {self.path}", "E_NOT_FOUND", value=p,
                           hint="python -m rsi runs query --limit 20")
        if len(rows) > 1:
            raise RsiError(f"trial id '{p}' is ambiguous ({len(rows)}+ matches)", "E_AMBIGUOUS_ID", value=p,
                           did_you_mean=[r[0] for r in rows[:10]])
        return json.loads(rows[0][1])

    def iter_trials(self, run=None, tag=None):
        sql, args = "SELECT rec FROM trials", []
        if run is not None:
            sql, args = sql + " WHERE run=?", [run]
        for (rec,) in self.db.execute(sql + " ORDER BY created, rowid", args).fetchall():
            r = json.loads(rec)
            if tag is None or tag in (r.get("tags") or []):
                yield r

    def query(self, where=(), sort=(), limit=20, fields=None, run=None, tag=None, objective=None):
        """Matching trials (newest first unless sorted), projected onto `fields`."""
        preds = [parse_where(w) for w in where]
        recs = [r for r in self.iter_trials(run, tag) if all(p(r) for p in preds)]
        if objective:
            for r in recs:
                r["objective_value"] = objective_value(r, objective)
            sort = list(sort) or ["-objective_value"]
        recs = sort_records(recs[::-1], sort) if sort else recs[::-1]
        if limit:
            recs = recs[:int(limit)]
        cols = list(fields) if fields else list(DEFAULT_FIELDS) + (["objective_value"] if objective else [])
        return [select_fields(r, cols) for r in recs]

    def count(self):
        return self.db.execute("SELECT COUNT(*) FROM trials").fetchone()[0]

    def stats(self, group_by, where=(), metric="summary.test_acc.mean"):
        """[{group, n, mean, min, max, std}] of `metric` per value of `group_by`, best mean first."""
        preds = [parse_where(w) for w in where]
        groups = {}
        for r in self.iter_trials():
            if all(p(r) for p in preds):
                g = get_path(r, group_by)
                k = json.dumps(g, sort_keys=True, ensure_ascii=False)
                groups.setdefault(k, (g, []))[1].append(get_path(r, metric))
        out = []
        for g, vals in groups.values():
            v = [x for x in vals if isinstance(x, (int, float)) and not isinstance(x, bool)]
            out.append({"group": g, "n_trials": len(vals), "n": len(v),
                        "mean": statistics.fmean(v) if v else None, "min": min(v) if v else None,
                        "max": max(v) if v else None, "std": statistics.pstdev(v) if v else None})
        return sorted(out, key=lambda r: (r["mean"] is None, -(r["mean"] or 0)))


def objective_value(rec, spec):
    """'METRIC[:AGG]' over a trial's ok/early_stopped cells, e.g. 'fresh_acc:min'; losses are not negated."""
    metric, _, agg = spec.partition(":")
    split, _, m = metric.partition("_")
    if split not in ("train", "test", "fresh") or not m:
        raise RsiError(f"bad --objective {spec!r}: expected METRIC[:mean|median|min|max], e.g. test_acc:min",
                       "E_BAD_QUERY", value=spec)
    vals = [(c.get(split) or {}).get(m) for c in rec.get("cells") or [] if c.get("status") in ("ok", "early_stopped")]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    agg = agg or "mean"
    fn = {"mean": statistics.fmean, "median": statistics.median, "min": min, "max": max}.get(agg)
    if fn is None:
        raise RsiError(f"bad --objective aggregate {agg!r} (mean|median|min|max)", "E_BAD_QUERY", value=spec)
    return fn(vals)
