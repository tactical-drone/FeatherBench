"""run_unit: same numbers as a Session, JSON-safe, never raises. Part of nn-playground (AGPL-3.0)."""
import contextlib
import io
import json
import unittest

import torch

from tests._util import quiet
from nncore import MODELS, Config, Session
from nncore.configio import config_to_dict
from nncore.run import EarlyStop, RunSpec, code_fingerprint, environment, run_unit, summarize

MLP = config_to_dict(Config(model="mlp"))


class RunUnit(unittest.TestCase):
    def test_equals_session(self):
        with quiet():
            s = Session(Config())
            s.train(300)
            ev = s.evaluate()
            cell = run_unit(RunSpec(config=config_to_dict(Config()), steps=300))
        self.assertEqual(cell["status"], "ok")
        self.assertEqual(cell["train"], ev["train"])
        self.assertEqual(cell["test"], ev["test"])
        self.assertEqual(cell["model"]["n_params"], s.n_params)
        self.assertEqual(cell["steps_done"], 300)
        self.assertRegex(cell["key"], r"^k_[0-9a-f]{16}$")
        json.dumps(cell, allow_nan=False)

    def test_eval_every_does_not_change_numbers(self):
        a = run_unit(RunSpec(config=MLP, steps=300))
        b = run_unit(RunSpec(config=MLP, steps=300, eval_every=7))
        self.assertEqual((a["train"], a["test"]), (b["train"], b["test"]))
        self.assertLessEqual(len(b["curve"]["step"]), 50)
        self.assertEqual(b["curve"]["step"][-1], 300)
        self.assertIsNone(a["curve"])

    def test_stdout_clean_and_rng_restored(self):
        torch.manual_seed(9)
        before = torch.get_rng_state()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            run_unit(RunSpec(config=config_to_dict(Config()), steps=20, fresh_points=100))
        self.assertEqual(out.getvalue(), "")
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_errors_do_not_raise(self):
        def broken(n_in, n_out, cfg):
            raise RuntimeError("part exploded")
        MODELS.register("_test broken", broken)
        try:
            cell = run_unit(RunSpec(config={**MLP, "model": "_test broken"}, steps=10))
        finally:
            MODELS.pop("_test broken")
        self.assertEqual(cell["status"], "error")
        self.assertEqual(cell["error"]["phase"], "build")
        self.assertIn("part exploded", cell["error"]["message"])
        bad = run_unit(RunSpec(config={**MLP, "activation": "relux"}, steps=10))
        self.assertEqual(bad["status"], "error")

    def test_timeout(self):
        cell = run_unit(RunSpec(config=MLP, steps=5000, max_seconds=1e-6))
        self.assertEqual(cell["status"], "timeout")
        self.assertLess(cell["steps_done"], 5000)

    def test_diverged(self):
        cfg = config_to_dict(Config(model="mlp", layers="16:square,16:square,16:square", optimizer="SGD", lr=1.0,
                                    dataset="Moons"))
        cell = run_unit(RunSpec(config=cfg, steps=500))
        self.assertEqual(cell["status"], "diverged")
        self.assertIsNone(cell["test"]["acc"])
        self.assertLess(cell["diverged_at"], 500)
        json.dumps(cell, allow_nan=False)

    def test_early_stop_and_fresh(self):
        cell = run_unit(RunSpec(config=MLP, steps=600, early_stop=EarlyStop(100, 0.99), fresh_points=300))
        self.assertEqual(cell["status"], "early_stopped")
        self.assertEqual(cell["steps_done"], 100)
        self.assertEqual(set(cell["fresh"]), {"loss", "acc"})
        no_test = run_unit(RunSpec(config={**MLP, "splitter": "none (train on all)"}, steps=20))
        self.assertIsNone(no_test["test"]["acc"])

    def test_key(self):
        a, b = RunSpec(config=MLP, steps=10), RunSpec(config=dict(reversed(list(MLP.items()))), steps=10)
        self.assertEqual(a.key("c_x"), b.key("c_x"))
        self.assertNotEqual(a.key("c_x"), a.key("c_y"))
        self.assertNotEqual(a.key("c_x"), RunSpec(config=MLP, steps=11).key("c_x"))
        self.assertEqual(a.key("c_x"), RunSpec(config=MLP, steps=10, max_seconds=5).key("c_x"))


class Aggregates(unittest.TestCase):
    def test_summarize(self):
        cells = [run_unit(RunSpec(config={**MLP, "seed": s}, steps=50)) for s in (0, 1)]
        cells.append({"status": "error", "dataset": MLP["dataset"], "model": None, "seconds": 0.1})
        sm = summarize(cells)
        self.assertEqual((sm["n_cells"], sm["n_ok"], sm["n_error"]), (3, 2, 1))
        self.assertEqual(sm["test_acc"]["n"], 2)
        self.assertAlmostEqual(sm["test_acc"]["mean"], (cells[0]["test"]["acc"] + cells[1]["test"]["acc"]) / 2)
        self.assertIn(MLP["dataset"], sm["per_dataset"])
        self.assertEqual(sm["n_params"], cells[0]["model"]["n_params"])

    def test_provenance(self):
        fp = code_fingerprint()
        self.assertRegex(fp, r"^c_[0-9a-f]{12}$")
        self.assertEqual(fp, code_fingerprint())
        self.assertNotEqual(code_fingerprint([]), code_fingerprint(["my_parts"]))
        env = environment()
        self.assertEqual(env["nncore"], "0.3.0")
        self.assertIn("torch", env)


if __name__ == "__main__":
    unittest.main()
