"""
The setup garage: what a sim racer does between laps, for nets.

  Job                 runs a console call (rsi.run / evolve / bench) off the UI thread
  SetupSheet          one row per stint: the change, how far it trained, what it scored
  dialogs             laps result, the machine's suggestions, the FeatherBench top 10
  identicon(name)     a GitHub-style 5x5 badge for a username, drawn locally (no network)

Nothing here trains in the window's session; the console runs its own cells, so the
live view keeps going while a job runs.
"""
import hashlib
import json
import time
import traceback

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from nncore import configio
from nncore.registry import REPO_ROOT

from . import theme

LEADERBOARD = REPO_ROOT / "featherbench" / "leaderboard.json"


# ---------------------------------------------------------------- background jobs
class Job(QtCore.QThread):
    """fn(progress) in a worker thread. progress(text) is safe to call from fn; the result
    or the error arrives on the UI thread through finished_ok / failed."""
    progress = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, title, fn, parent=None):
        super().__init__(parent)
        self.title, self.fn = title, fn

    def run(self):
        try:
            self.finished_ok.emit(self.fn(self.progress.emit))
        except Exception as e:  # shown in the status bar; the details go to the console
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------- setup sheet
def describe_change(new, old):
    """'lr 0.03 -> 0.01, skip_decide none -> residual (same width)' (settings dicts)."""
    if old is None:
        return "start"
    parts = []
    for k, v in configio.config_diff(configio.config_from_dict(new)[0], configio.config_from_dict(old)[0]).items():
        before = old.get(k)
        fmt = lambda x: ",".join(x) if isinstance(x, (list, tuple)) else ("full" if x is None and k == "batch_size" else x)  # noqa: E731
        parts.append(f"{k} {fmt(before)} → {fmt(v)}")
    return ", ".join(parts) or "same setup"


class SetupSheet:
    """Stints in order. The open stint (the last row) is updated as training goes."""
    COLUMNS = ("#", "change", "steps", "train %", "test %", "best test %", "5 laps mean / worst %", "time")

    def __init__(self):
        self.rows = []

    def start(self, config, change):
        self.rows.append({"n": len(self.rows) + 1, "change": change, "config": config, "steps": 0,
                          "train": None, "test": None, "best": None, "laps": None,
                          "time": time.strftime("%H:%M:%S")})

    def update(self, steps, train, test, best):
        if self.rows:
            r = self.rows[-1]
            r.update(steps=steps, train=train, test=test, best=best)

    def cells(self, r):
        pct = lambda x: "" if x is None or x != x else f"{x * 100:.1f}"  # noqa: E731
        laps = r["laps"]
        lap = f"{laps['mean'] * 100:.1f} / {laps['min'] * 100:.1f}" if laps else ""
        return [str(r["n"]), r["change"], str(r["steps"]), pct(r["train"]), pct(r["test"]), pct(r["best"]),
                lap, r["time"]]

    def to_csv(self):
        import csv
        import io
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(self.COLUMNS + ("settings",))
        for r in self.rows:
            w.writerow(self.cells(r) + [json.dumps(r["config"], ensure_ascii=False)])
        return out.getvalue()


class SetupSheetDialog(QtWidgets.QDialog):
    def __init__(self, sheet, on_load, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Setup sheet")
        self.resize(980, 420)
        self.sheet, self.on_load = sheet, on_load
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel("One row per stint: what you changed, how far it trained and what it "
                                       "scored. Select a row and load it to go back to that setup."))
        self.table = QtWidgets.QTableWidget(0, len(SetupSheet.COLUMNS))
        self.table.setHorizontalHeaderLabels(SetupSheet.COLUMNS)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.doubleClicked.connect(lambda _: self._load())
        lay.addWidget(self.table)
        btns = QtWidgets.QHBoxLayout()
        for text, fn in (("Load selected setup", self._load), ("Export CSV…", self._export),
                         ("Refresh", self.refresh), ("Close", self.close)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            btns.addWidget(b)
        lay.addLayout(btns)
        self.refresh()

    def refresh(self):
        rows = self.sheet.rows
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, text in enumerate(self.sheet.cells(r)):
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(text))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        if rows:
            self.table.scrollToBottom()

    def _load(self):
        i = self.table.currentRow()
        if 0 <= i < len(self.sheet.rows):
            self.on_load(self.sheet.rows[i]["config"], f"setup #{self.sheet.rows[i]['n']}")

    def _export(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export setup sheet", "setup_sheet.csv",
                                                        "CSV table (*.csv)")
        if path:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(self.sheet.to_csv())


# ---------------------------------------------------------------- results dialogs
def _table(headers, rows):
    t = QtWidgets.QTableWidget(len(rows), len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    t.verticalHeader().setVisible(False)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            item = v if isinstance(v, QtWidgets.QTableWidgetItem) else QtWidgets.QTableWidgetItem(str(v))
            t.setItem(i, j, item)
    t.resizeColumnsToContents()
    return t


def laps_dialog(parent, result, steps):
    """rsi run result over seeds 0-4 -> per-lap table plus mean / worst."""
    cells = result.get("cells") or []
    pct = lambda x: "" if x is None else f"{x * 100:.1f}"  # noqa: E731
    rows = [[c.get("seed"), c.get("status"), pct((c.get("train") or {}).get("acc")),
             pct((c.get("test") or {}).get("acc")), pct((c.get("fresh") or {}).get("acc"))] for c in cells]
    sm = result.get("summary") or {}
    d = QtWidgets.QDialog(parent)
    d.setWindowTitle("5 laps")
    d.resize(560, 330)
    lay = QtWidgets.QVBoxLayout(d)
    ta, fa = sm.get("test_acc") or {}, sm.get("fresh_acc") or {}
    lay.addWidget(QtWidgets.QLabel(
        f"<b>{steps} steps per lap, seeds 0-4.</b>  Test {pct(ta.get('mean'))}% mean, worst {pct(ta.get('min'))}%."
        f"  On 2000 fresh points {pct(fa.get('mean'))}% mean, worst {pct(fa.get('min'))}%."
        f"  {sm.get('n_params', '?')} params."))
    lay.addWidget(_table(["seed", "status", "train %", "test %", "fresh %"], rows))
    lay.addWidget(QtWidgets.QLabel("The worst lap is the honest number: a setup that only shines on one seed "
                                   "got lucky."))
    b = QtWidgets.QPushButton("Close")
    b.clicked.connect(d.accept)
    lay.addWidget(b)
    d.show()
    return d


def suggestions_dialog(parent, lb, current, on_apply):
    """The machine's best setups from an evolve leaderboard, as changes to `current`."""
    cur_cfg = configio.config_from_dict(current)[0]
    rows, configs = [], []
    for e in (lb.get("entries") or [])[:8]:
        cfg = configio.config_from_dict(e["config"])[0]
        diff = configio.config_diff(cfg, cur_cfg)
        if not diff:
            continue
        ho = (e.get("holdout") or {}).get("test_acc_mean")
        fit = e.get("fitness") or {}
        rows.append([len(rows) + 1, ", ".join(f"{k}={v}" for k, v in diff.items()),
                     "" if ho is None else f"{ho * 100:.1f}", f"{(fit.get('acc_agg') or 0) * 100:.1f}",
                     fit.get("n_params", "")])
        configs.append(e["config"])
    d = QtWidgets.QDialog(parent)
    d.setWindowTitle("The machine suggests")
    d.resize(860, 380)
    lay = QtWidgets.QVBoxLayout(d)
    if not rows:
        lay.addWidget(QtWidgets.QLabel("The machine found nothing better than your setup this time. "
                                       "Man wins this round."))
    else:
        lay.addWidget(QtWidgets.QLabel("Changes from your current setup, best first. Holdout = unseen seeds, "
                                       "the honest score. Apply one, then train it yourself."))
        t = _table(["#", "change", "holdout test %", "search acc %", "params"], rows)
        t.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        lay.addWidget(t)
        b = QtWidgets.QPushButton("Apply selected change")

        def apply():
            i = t.currentRow()
            if 0 <= i < len(configs):
                on_apply(configs[i], f"machine suggestion #{i + 1}")
        b.clicked.connect(apply)
        t.doubleClicked.connect(lambda _: apply())
        lay.addWidget(b)
    c = QtWidgets.QPushButton("Close")
    c.clicked.connect(d.accept)
    lay.addWidget(c)
    d.show()
    return d


def bench_dialog(parent, result):
    """rsi bench result -> solved per dataset, params_max, the attempt command."""
    per = result.get("per_dataset") or []
    sm = result.get("summary") or result
    pct = lambda x: "" if x is None else f"{x * 100:.1f}"  # noqa: E731
    rows = [[r.get("dataset"), "yes" if r.get("solved") else "no", pct(r.get("fresh_acc_mean")), r.get("n_params", "")]
            for r in per]
    d = QtWidgets.QDialog(parent)
    d.setWindowTitle("FeatherBench")
    d.resize(640, 460)
    lay = QtWidgets.QVBoxLayout(d)
    solved_all = sm.get("solved_all")
    lay.addWidget(QtWidgets.QLabel(
        f"<b>{'Solved every pattern!' if solved_all else 'Not every pattern solved yet.'}</b>  "
        f"{sm.get('n_solved', '?')}/{sm.get('n_datasets', len(rows))} solved, params_max "
        f"<b>{sm.get('params_max', '?')}</b>, mean accuracy {pct(sm.get('mean_acc'))}%."))
    t = _table(["dataset", "solved", "fresh acc %", "params"], rows)
    t.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
    lay.addWidget(t)
    lay.addWidget(QtWidgets.QLabel("Fewest params that solves every pattern wins. To enter, see "
                                   "FeatherBench > How to make an attempt."))
    b = QtWidgets.QPushButton("Close")
    b.clicked.connect(d.accept)
    lay.addWidget(b)
    d.show()
    return d


# ---------------------------------------------------------------- leaderboard
def identicon(name, size=28):
    """A GitHub-style 5x5 mirrored badge for `name`, coloured from its hash (no network)."""
    h = hashlib.md5(str(name).lower().encode()).digest()
    color = QtGui.QColor.fromHsv(h[0] * 360 // 256, 150 + h[1] % 90, 200 + h[2] % 55)
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtGui.QColor(theme.WINDOW_BG))
    p = QtGui.QPainter(pm)
    cell = size / 6
    for row in range(5):
        for col in range(3):
            if (h[3 + row * 3 + col] if 3 + row * 3 + col < len(h) else h[row]) % 2 == 0:
                for c in {col, 4 - col}:
                    p.fillRect(QtCore.QRectF(cell * (c + 0.5), cell * (row + 0.5), cell, cell), color)
    p.end()
    return pm


def load_leaderboard(path=LEADERBOARD):
    """(entries, note). Entries are dicts with at least github and config; missing file -> []."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [], "No FeatherBench entries yet. Be the first: FeatherBench > How to make an attempt."
    except (OSError, ValueError) as e:
        return [], f"Could not read {path}: {e}"
    entries = [e for e in (doc.get("entries") or []) if isinstance(e, dict) and e.get("github") and
               isinstance(e.get("config"), dict)]
    return entries, None


def leaderboard_dialog(parent, on_load, path=LEADERBOARD):
    entries, note = load_leaderboard(path)
    d = QtWidgets.QDialog(parent)
    d.setWindowTitle("FeatherBench top 10")
    d.resize(760, 440)
    lay = QtWidgets.QVBoxLayout(d)
    lay.addWidget(QtWidgets.QLabel("<b>FeatherBench</b>: fewest params that solves every pattern wins. "
                                   "Load a racer's setup and try to beat it."))
    if note:
        lay.addWidget(QtWidgets.QLabel(note))
    top = entries[:10]
    rows = []
    for i, e in enumerate(top, 1):
        who = QtWidgets.QTableWidgetItem(f"  {e['github']}")
        who.setIcon(QtGui.QIcon(identicon(e["github"])))
        acc, att = e.get("mean_acc"), e.get("attempt") or {}
        tok, cost = att.get("tokens"), att.get("cost_usd")
        rows.append([e.get("rank", i), who, e.get("feather_score", ""), e.get("params_max", ""),
                     f"{e.get('n_solved', '?')}/{e.get('n_datasets', 12)}", "" if acc is None else f"{acc * 100:.1f}",
                     att.get("model") or "?", "" if tok is None else f"{tok:,}",
                     "" if cost is None else f"${cost:,.2f}", (e.get("submitted") or "")[:10]])
    if rows:
        t = _table(["rank", "racer", "score", "params_max", "solved", "mean acc %", "AI model", "tokens", "cost",
                    "date"], rows)
        t.setToolTip("AI model, tokens and cost are self-reported by the racer; scores are CI's own runs.")
        t.setIconSize(QtCore.QSize(24, 24))
        t.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        lay.addWidget(t)
        b = QtWidgets.QPushButton("Load selected racer's setup")

        def load():
            i = t.currentRow()
            if 0 <= i < len(top):
                on_load(top[i]["config"], f"{top[i]['github']}'s FeatherBench setup")
        b.clicked.connect(load)
        t.doubleClicked.connect(lambda _: load())
        lay.addWidget(b)
    c = QtWidgets.QPushButton("Close")
    c.clicked.connect(d.accept)
    lay.addWidget(c)
    d.show()
    return d


ATTEMPT_HTML = """
<h3>Make a FeatherBench attempt</h3>
<p><b>Fewest params that solves every pattern wins.</b> Tune a setup here (or let an AI
agent search for one), then:</p>
<ol>
<li>File &gt; Save settings, e.g. <code>my_setup.json</code></li>
<li>Run the benchmark on it:<br>
<code>python -m rsi featherbench --config my_setup.json --workers 4 --submit featherbench/submissions/YOUR_GITHUB.json</code></li>
<li>Open a pull request that adds that one file. CI re-runs it; if it holds up, you are on the
leaderboard, and the top 10 can be loaded right here (FeatherBench &gt; Top 10).</li>
</ol>
<p>The full rules are in the README, section <i>FeatherBench</i>.</p>
"""
