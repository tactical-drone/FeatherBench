"""rsi sweep: value expansion, grids, one-at-a-time, the sweep run dir, stop, rundir locks.
Part of nn-playground (AGPL-3.0; see COMMERCIAL.md)."""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from tests._util import quiet  # noqa: F401  (loads my_parts)
from nncore import ACTIVATIONS, Config, configio
from nncore.configio import load_settings
import rsi
from rsi import rundir
from rsi.errors import RsiError
from rsi.store import Store
from rsi.sweep import expand_values, grid, one_at_a_time

BASE = {"model": "mlp", "dataset": "Moons", "n_points": 120}


class Expand(unittest.TestCase):
    def test_values(self):
        self.assertEqual(expand_values("activation", "tanh, relu"), ["tanh", "relu"])
        self.assertEqual(expand_values("lr", "0.01,0.1"), [0.01, 0.1])
        self.assertEqual(expand_values("width", ["8", 16.0]), [8, 16])
        self.assertEqual(expand_values("layers", "8,8"), ["8,8"])  # one layers string
        self.assertEqual(expand_values("layers", "8,8;16"), ["8,8", "16"])
        self.assertEqual(expand_values("batch_size", "full;32"), [None, 32])
        self.assertEqual(expand_values("features", "x,y;x,y,x^2"), [("x", "y"), ("x", "y", "x²")])
        self.assertEqual(expand_values("activation", "@all"), list(ACTIVATIONS))
        self.assertEqual(expand_values("act_decide", "@all+same"), list(ACTIVATIONS) + ["same"])
        self.assertEqual(expand_values("dataset", "@suite:classic")[0], "XOR quadrants")
        self.assertEqual(expand_values("extra.clip", "0.5,1"), [0.5, 1])
        with self.assertRaises(configio.ConfigError) as cm:
            expand_values("activation", "relux")
        self.assertIn("relu", cm.exception.did_you_mean)
        with self.assertRaises(RsiError) as cm:
            expand_values("bogus", "1")
        self.assertEqual(cm.exception.code, "E_UNKNOWN_FIELD")

    def test_grid_and_oat(self):
        g = grid({"a": [1, 2], "b": ["x", "y"]})
        self.assertEqual(g, [{"a": 1, "b": "x"}, {"a": 1, "b": "y"}, {"a": 2, "b": "x"}, {"a": 2, "b": "y"}])
        self.assertEqual(grid({}), [{}])
        oat = one_at_a_time(Config(), ["lr", "skip"])
        lrs = [v for f, v, _ in oat if f == "lr"]
        self.assertNotIn(0.03, lrs)
        self.assertIn(0.003, lrs)
        self.assertTrue(all(o == {f: v} for f, v, o in oat))
        self.assertIn("same", [v for f, v, _ in one_at_a_time(Config(), ["act_relate"])] + ["same"])
        with self.assertRaises(RsiError):
            one_at_a_time(Config(), ["layers"])  # no suggested values
        ds = [v for _, v, _ in one_at_a_time(Config(), ["dataset"])]  # every other dataset, as @all
        self.assertEqual(ds, [d for d in expand_values("dataset", "@all") if d != Config().dataset])


class SweepRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.store = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def sweep(self, **kw):
        kw = {"overrides": BASE, "steps": 30, "workers": 0, "store": str(self.store), **kw}
        with quiet():
            return rsi.sweep(**kw)

    def test_grid_sweep(self):
        res = self.sweep(vary=["activation=tanh,relu"], name="g1", average_over=["activation"])
        self.assertEqual((res["status"], res["mode"], res["n_trials"], res["n_cells"]), ("done", "grid", 2, 2))
        self.assertEqual(res["axes"], {"activation": ["tanh", "relu"]})
        self.assertEqual({t["config_diff"]["activation"] for t in res["trials"]}, {"tanh", "relu"})
        self.assertEqual(res["groups"][0]["test_acc"]["n"], 2)
        d = self.store / "g1"
        cfg = load_settings(d / "best.settings.json").config
        self.assertEqual((cfg.model, cfg.dataset), ("mlp", "Moons"))
        lb = json.loads((d / "leaderboard.json").read_text())
        self.assertEqual([e["rank"] for e in lb["entries"]], [1, 2])
        self.assertEqual(rsi.status(d)["state"], "done")
        with Store(self.store) as st:
            self.assertEqual(len(list(st.iter_trials(run="g1"))), 2)
        with self.assertRaises(RsiError) as cm:
            self.sweep(vary=["activation=tanh,relu"], name="g1")
        self.assertEqual(cm.exception.code, "E_EXISTS")
        again = self.sweep(vary=["activation=tanh,relu"], name="g1", force=True)
        self.assertEqual(again["n_cached"], again["n_cells"])
        self.assertEqual([t["fitness"] for t in again["trials"]], [t["fitness"] for t in res["trials"]])
        lb2 = rsi.leaderboard(["g1"], sort="params", pareto=True, store=self.store)
        self.assertTrue(lb2["pareto"])

    def test_oat_and_limits(self):
        res = self.sweep(oat="skip", name="o1", seeds="0-1")
        self.assertEqual(res["mode"], "oat")
        self.assertEqual(res["n_trials"], 1 + len([s for s in configio.PART_REGISTRIES["skip"] if s != "none"]))
        self.assertEqual(res["n_cells"], 2 * res["n_trials"])
        with self.assertRaises(RsiError) as cm:
            self.sweep(vary=["lr=0.01,0.02,0.03"], max_runs=2, name="o2")
        self.assertEqual((cm.exception.code, cm.exception.exit), ("E_USAGE", 2))
        with self.assertRaises(RsiError) as cm:
            self.sweep(name="o3")
        self.assertEqual(cm.exception.code, "E_USAGE")

    def test_task_axes(self):
        """--vary dataset / seed trains each trial on its own dataset / seed (not the base's)."""
        res = self.sweep(vary=["dataset=Moons;Circles"], name="ta1", top=1, overrides={**BASE, "n_points": 80})
        recs = [json.loads(ln) for ln in (self.store / "ta1" / "trials.jsonl").read_text().splitlines()]
        self.assertEqual(sorted((r["config"]["dataset"], r["datasets"], r["cells"][0]["dataset"]) for r in recs),
                         [("Circles", ["Circles"], "Circles"), ("Moons", ["Moons"], "Moons")])
        self.assertEqual((res["n_trials"], len(res["trials"]), res["top"], res["truncated"]), (2, 1, 1, True))
        res = self.sweep(vary=["seed=0,1"], name="ta2", overrides={**BASE, "n_points": 80})
        recs = [json.loads(ln) for ln in (self.store / "ta2" / "trials.jsonl").read_text().splitlines()]
        self.assertEqual(sorted(r["cells"][0]["seed"] for r in recs), [0, 1])
        self.assertFalse(res["truncated"])
        with self.assertRaises(RsiError) as cm:
            self.sweep(vary=["seed=0,1"], seeds="0-2", name="ta3")
        self.assertEqual(cm.exception.code, "E_USAGE")

    def test_stop_file(self):
        def on_event(event, **f):  # ask to stop after the first cell
            if event == "cell":
                rundir.request_stop(self.store / "s1")
        res = self.sweep(vary=["lr=0.01,0.02,0.03"], name="s1", seeds="0-1", on_event=on_event)
        self.assertEqual(res["status"], "stopped")
        self.assertLess(res["n_cells"], 6)
        st = rsi.status(self.store / "s1")
        self.assertEqual(st["state"], "stopped")
        self.assertFalse((self.store / "s1" / "STOP").exists())


class RunDirs(unittest.TestCase):
    def test_lock_and_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r"
            rd = rundir.RunDir.create(p, "sweep", {})
            self.assertTrue((p / "lock").exists())
            with self.assertRaises(RsiError) as cm:  # same pid holds it: a live, fresh owner elsewhere
                info = json.loads((p / "lock").read_text())
                info["pid"] = os.getppid()
                (p / "lock").write_text(json.dumps(info))
                rundir.RunDir.open(p, lock=True)
            self.assertEqual(cm.exception.code, "E_LOCKED")
            info["pid"] = 2 ** 22 + 12345  # dead pid: the lock is stale and taken over
            (p / "lock").write_text(json.dumps(info))
            rundir.RunDir.open(p, lock=True).close("done")
            self.assertEqual(rundir.read_status(p)["state"], "done")
            prog = json.loads((p / "progress.json").read_text())
            prog.update(state="running", pid=2 ** 22 + 12345)
            (p / "progress.json").write_text(json.dumps(prog))
            self.assertEqual(rundir.read_status(p)["state"], "stale")
            with open(p / "x.jsonl", "w") as f:
                f.write('{"a": 1}\n{"a": 2}\n{"a": ')
            self.assertEqual(rundir.read_jsonl(p / "x.jsonl"), [{"a": 1}, {"a": 2}])
            self.assertEqual(rd.name, "r")

    def test_wait_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "w"
            rundir.RunDir.create(p, "evolve", {})  # running, owned by this live process
            t0 = time.time()
            with self.assertRaises(RsiError) as cm:
                rsi.wait(p, timeout=0.3)
            self.assertEqual((cm.exception.code, cm.exception.exit), ("E_TIMEOUT", 7))
            self.assertEqual(cm.exception.result["state"], "running")
            self.assertLess(time.time() - t0, 3)
            res = rsi.stop(p)
            self.assertEqual(res["state"], "stopping")
            self.assertTrue((p / "STOP").exists())


if __name__ == "__main__":
    unittest.main()
