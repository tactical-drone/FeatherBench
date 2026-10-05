"""rsi store, query language and Evaluator (cache, dedupe, order, workers, crash recovery).
Part of nn-playground (AGPL-3.0; see COMMERCIAL.md)."""
import json
import tempfile
import unittest

from tests._util import quiet  # noqa: F401  (loads my_parts)
from nncore import Config
from nncore.configio import config_to_dict
from rsi.errors import RsiError
from rsi.pool import Evaluator, cell_specs
from rsi.query import get_path, parse_sort, parse_where, sort_records
from rsi.store import Store

MLP = Config(model="mlp", dataset="Moons", n_points=120)


def _rec(i, acc, model="mlp", tags=(), run=None):
    return {"id": f"t_{i:012x}", "created": f"2026-10-05T00:00:{i:02d}Z", "status": "ok", "run": run,
            "fitness": acc, "config_key": f"k{i}", "tags": list(tags), "config": {"model": model, "lr": 0.01 * i},
            "config_diff": {"lr": 0.01 * i}, "summary": {"test_acc": {"mean": acc}, "n_params": 100 - i},
            "cells": [{"status": "ok", "test": {"acc": acc}, "fresh": {"acc": None if acc is None else acc - 0.1}}]}


class Query(unittest.TestCase):
    def test_where(self):
        r = _rec(3, 0.9, tags=["ab"])
        for expr, want in [("test_acc>=0.9", True), ("test_acc>0.9", False), ("params<100", True),
                           ("config.model==mlp", True), ("config.model!=mlp", False), ("config.model~ML", True),
                           ("tag~ab", True), ("tag==ab", True), ("config.model in mlp;custom nn", True),
                           ("status in failed;partial", False), ("cells[0].test.acc==0.9", True),
                           ("missing.path==null", True), ("fitness<0.5", False), ("created>=2026", True)]:
            self.assertEqual(parse_where(expr)(r), want, expr)

    def test_bad_query(self):
        for bad in ("", "test_acc", ">= 3", "a.b >=", "test_acc=0.9", "test_acc>>0.9", "test_acc=>0.9"):
            with self.assertRaises(RsiError) as cm:
                parse_where(bad)
            self.assertEqual(cm.exception.code, "E_BAD_QUERY")
            self.assertEqual(cm.exception.exit, 3)
        with self.assertRaises(RsiError) as cm:
            parse_where("config.model=mlp")
        self.assertEqual(cm.exception.did_you_mean, ["config.model==mlp"])
        for bad, fix in (("test_acc=>0.8", "test_acc>=0.8"), ("test_acc=<0.8", "test_acc<=0.8"),
                         ("config.model=!mlp", "config.model!=mlp")):  # operator written backwards
            with self.assertRaises(RsiError) as cm:
                parse_where(bad)
            self.assertEqual(cm.exception.did_you_mean, [fix])
            parse_where(fix)  # the suggestion itself parses
        with self.assertRaises(RsiError) as cm:  # a number field against a non-number: a typo, not 0 rows
            parse_where("test_acc>abc")(_rec(3, 0.9))
        self.assertEqual(cm.exception.code, "E_BAD_QUERY")

    def test_sort_and_paths(self):
        self.assertEqual(parse_sort("-fitness"), ("fitness", True))
        self.assertEqual(parse_sort("params"), ("params", False))
        recs = [_rec(1, 0.5), _rec(2, None), _rec(3, 0.9)]
        self.assertEqual([r["id"][-1] for r in sort_records(recs, ["-fitness"])], ["3", "1", "2"])  # None last
        self.assertEqual([r["id"][-1] for r in sort_records(recs, ["fitness"])], ["1", "3", "2"])
        self.assertEqual(get_path({"a": [{"b": 2}]}, "a[0].b"), 2)
        self.assertIsNone(get_path({"a": 1}, "a.b"))


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.st = Store(self.tmp.name)

    def tearDown(self):
        self.st.close()
        self.tmp.cleanup()

    def test_cells_cache_only_good_statuses(self):
        for st in ("ok", "diverged", "early_stopped", "timeout", "error"):
            self.st.put_cell(f"k_{st}", {"status": st, "dataset": "Moons", "seed": 0})
        self.assertEqual({k for k in ("k_ok", "k_diverged", "k_early_stopped", "k_timeout", "k_error")
                          if self.st.get_cell(k)}, {"k_ok", "k_diverged", "k_early_stopped"})
        self.assertIsNone(self.st.get_cell("nope"))

    def test_trials(self):
        for i, acc in enumerate([0.5, 0.9, 0.7], 1):
            self.st.put_trial(_rec(i, acc, tags=["x"] if i == 2 else [], run="r1" if i < 3 else None))
        self.assertEqual(self.st.trial("t_000000000002")["fitness"], 0.9)
        with self.assertRaises(RsiError) as cm:
            self.st.trial("t_0000")
        self.assertEqual((cm.exception.code, cm.exception.exit), ("E_AMBIGUOUS_ID", 5))
        with self.assertRaises(RsiError) as cm:
            self.st.trial("t_ffff")
        self.assertEqual(cm.exception.code, "E_NOT_FOUND")
        rows = self.st.query(where=["test_acc>=0.6"], sort=["-fitness"], fields=["id", "fitness"])
        self.assertEqual([r["fitness"] for r in rows], [0.9, 0.7])
        self.assertEqual(len(self.st.query(run="r1")), 2)
        self.assertEqual(len(self.st.query(tag="x")), 1)
        self.assertEqual(self.st.query(limit=1)[0]["id"], "t_000000000003")  # newest first
        rows = self.st.query(objective="fresh_acc:min", fields=["id", "objective_value"])
        self.assertAlmostEqual(rows[0]["objective_value"], 0.8)
        stats = self.st.stats("config.model", metric="summary.test_acc.mean")
        self.assertEqual(stats[0]["group"], "mlp")
        self.assertEqual(stats[0]["n"], 3)
        self.assertAlmostEqual(stats[0]["mean"], 0.7)


class EvaluatorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.st = Store(self.tmp.name)

    def tearDown(self):
        self.st.close()
        self.tmp.cleanup()

    def test_serial_cache_dedupe_order(self):
        specs = cell_specs(MLP, seeds=[0, 1], steps=20)
        seen = []
        with Evaluator(0, ["my_parts"], store=self.st) as ev:
            a = ev.map([specs[1], specs[0], specs[1]], on_result=lambda i, c: seen.append((i, c["cached"])))
            self.assertEqual([c["seed"] for c in a], [1, 0, 1])
            self.assertEqual(seen, [(0, False), (1, False)])  # the duplicate ran once
            self.assertEqual(ev.stats["new"], 2)
        with Evaluator(0, ["my_parts"], store=self.st) as ev2:  # a new evaluator: the store answers
            b = ev2.map(specs)
            self.assertTrue(all(c["cached"] for c in b))
            self.assertEqual([c["train"] for c in b], [a[1]["train"], a[0]["train"]])
        with Evaluator(0, ["my_parts"], store=self.st, cache=False) as ev3:
            self.assertFalse(ev3.map(specs[:1])[0]["cached"])

    def test_max_new_stop_and_errors(self):
        specs = cell_specs(MLP, seeds=[0, 1, 2], steps=10)
        with Evaluator(0, ["my_parts"], store=self.st) as ev:
            out = ev.map(specs, max_new=1)
            self.assertEqual([c is not None for c in out], [True, False, False])
            out = ev.map(specs, stop_check=lambda: True)
            self.assertEqual([c is not None for c in out], [True, False, False])  # only the cached one
            bad = cell_specs(Config(model="mlp", lr=1.0, optimizer="SGD", layers="16:square,16:square"), steps=5,
                             max_seconds=1e-9)[0]
            cell = ev.evaluate(bad)
            self.assertIn(cell["status"], ("timeout", "diverged", "ok"))
            if cell["status"] == "timeout":
                self.assertIsNone(self.st.get_cell(cell["key"]))
        scoped = Evaluator(0, ["my_parts"], store=self.st, cache_scope="config")
        self.assertNotEqual(scoped.key(specs[0]), Evaluator(0, ["my_parts"], store=self.st).key(specs[0]))

    def test_workers_match_serial_and_survive_a_crash(self):
        cfg = config_to_dict(Config(model="_test mlp", dataset="Moons", n_points=120))
        specs = [*cell_specs(cfg, seeds=[0, 1, 2], steps=30), *cell_specs({**cfg, "model": "_test crash"}, steps=5)]
        from nncore import MODELS
        with quiet():
            from tests import _crash_parts  # noqa: F401
        MODELS.register("_test crash", _crash_parts._crash)
        MODELS.register("_test mlp", _crash_parts.build_mlp)
        try:
            with Evaluator(0, ["my_parts", "tests._crash_parts"], store=None) as ev:
                serial = ev.map(specs[:3])
            with Evaluator(2, ["my_parts", "tests._crash_parts"], store=None) as ev:
                par = ev.map(specs)
                self.assertGreaterEqual(ev.stats["crashed"], 1)
        finally:
            MODELS.pop("_test crash", None)
            MODELS.pop("_test mlp", None)
        self.assertEqual([c["train"] for c in par[:3]], [c["train"] for c in serial])
        self.assertEqual([c["test"] for c in par[:3]], [c["test"] for c in serial])
        self.assertEqual(par[3]["status"], "error")
        self.assertEqual(par[3]["error"]["message"], "worker crashed")
        json.dumps(par, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
