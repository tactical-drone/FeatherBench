"""rsi complexity: the power-law fit, the capacity knob, a tiny ladder run and the CLI document.
Part of nn-playground (AGPL-3.0; see COMMERCIAL.md)."""
import math
import tempfile
import unittest

from tests._util import quiet  # noqa: F401  (loads my_parts)
import rsi
from nncore import Config, configio
from rsi.cli import invoke
from rsi.complexity import apply_knob, fit_power
from rsi.errors import RsiError


class ComplexityTests(unittest.TestCase):
    def test_fit_and_knob(self):
        f = fit_power([(k, 3 * k ** 2) for k in (2, 3, 5, 8)])
        self.assertAlmostEqual(f["p"], 2.0)
        self.assertAlmostEqual(f["c"], 3.0)
        self.assertAlmostEqual(f["r2"], 1.0)
        self.assertIsNone(fit_power([(2, 10)]))
        noisy = fit_power([(2, 10), (3, 30), (4, 35), (6, 90)])
        self.assertTrue(0 < noisy["r2"] < 1 and 1 < noisy["p"] < 2.5)
        d = configio.config_to_dict(Config(model="mlp", layers="8:sin,4", activation="tanh"))
        self.assertEqual(apply_knob(d, "layers", 3)["layers"], "3:sin,3")
        self.assertEqual(apply_knob(d, "width", 5)["width"], 5)
        self.assertEqual(apply_knob(d, "extra.k", 2)["extra"], {"k": 2})

    def test_run_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp, quiet():
            r = rsi.complexity(family="stripes", sizes=[2, 3], threshold=0.6, ladder=[1, 2], seeds=[0], steps=30,
                               workers=0, overrides={"model": "mlp", "layers": "4", "n_points": 200}, store=tmp)
            self.assertEqual((r["knob"], r["ladder"], [t["dataset"] for t in r["table"]]),
                             ("layers", [1, 2], ["stripes[2]", "stripes[3]"]))
            for t in r["table"]:
                self.assertTrue(t["tried"])
                if t["solved"]:
                    self.assertGreaterEqual(t["fresh_acc_mean"], 0.6)
                    self.assertTrue(all(not x["solved"] for x in t["tried"][:-1]))
            if r["fit"]:
                self.assertIn("empirical complexity exponent", r["summary"])
                self.assertTrue(math.isfinite(r["fit"]["p"]))
            else:
                self.assertIn("not enough solved sizes", r["summary"])
            again = rsi.complexity(family="stripes", sizes=[2, 3], threshold=0.6, ladder=[1, 2], seeds=[0], steps=30,
                                   workers=0, overrides={"model": "mlp", "layers": "4", "n_points": 200}, store=tmp)
            self.assertEqual((again["n_new"], again["table"]), (0, r["table"]))  # every cell cached
        for kw, code in (({"family": "spirall"}, "E_UNKNOWN_PART"), ({"family": "spiral", "sizes": [1]}, "E_OUT_OF_RANGE"),
                         ({"family": "spiral", "overrides": {"model": "mlp", "layers": ""}}, "E_USAGE"),
                         ({"family": "spiral", "knob": "lr"}, "E_USAGE")):
            with self.assertRaises(RsiError) as cm, quiet():
                rsi.complexity(**kw)
            self.assertEqual(cm.exception.code, code, kw)
        code, d = invoke(["complexity", "--family", "nope", "--sizes", "2-3"])
        self.assertEqual((code, d["error"]["code"]), (3, "E_UNKNOWN_PART"))
        self.assertIn("spiral", d["error"]["allowed"])
        from nncore import Config, configio
        from nncore.datasets import FAMILIES
        from rsi.complexity import _slow_warning, apply_knob
        from rsi.pool import Evaluator, cell_specs
        cfg = Config(model="mlp", layers="4")
        rung = {r: apply_knob(configio.config_to_dict(cfg), "layers", r) for r in (1, 2)}
        fam = FAMILIES.get("spiral")
        with Evaluator(0, [], store=None) as ev, rsi.collect_warnings() as ws:
            _slow_warning(ev, cfg, fam, [2, 3], [1, 2], rung, [0, 1], 100000)  # worst case: every rung
            self.assertEqual([w["code"] for w in ws], ["W_SLOW"])
            self.assertIn("8 uncached of 8 cells", ws[0]["message"])
            for r in (1, 2):
                for k in (2, 3):
                    for sp in cell_specs({**rung[r], "dataset": fam.dataset_name(k)}, seeds=[0, 1], steps=100000,
                                         fresh_points=2000):
                        ev.mem[ev.key(sp)] = {"status": "ok"}
            ws.clear()
            _slow_warning(ev, cfg, fam, [2, 3], [1, 2], rung, [0, 1], 100000)  # all cached: no warning
            self.assertEqual(ws, [])
        code, d = invoke(["complexity", "--family", "spiral", "--sizes", "3-2"])
        self.assertEqual((code, d["error"]["code"]), (2, "E_USAGE"))
        self.assertIn("--sizes", d["error"]["message"])  # names the flag, not 'seed'


if __name__ == "__main__":
    unittest.main()
