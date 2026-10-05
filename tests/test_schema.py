"""schema: RNG-safe introspection and the lever table. Part of nn-playground (AGPL-3.0; see COMMERCIAL.md)."""
import contextlib
import io
import json
import unittest

import torch

from tests._util import SLOW, quiet, same_params
from nncore import Config, Session, schema
from nncore.configio import FIELDS


class RngSafety(unittest.TestCase):
    def test_rng_untouched(self):
        torch.manual_seed(5)
        before = torch.get_rng_state()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            schema.fields_read("custom nn", Config())
            schema.build_info(Config())
            schema.count_params(Config(model="mlp"))
            schema.describe()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(out.getvalue(), "")  # Wide's print went to stderr

    def test_count_params_mid_run(self):
        """450 + 450 with RSI_SLOW=1; 150 + 150 otherwise (any shift of the RNG shows within a few steps)."""
        k = 450 if SLOW else 150
        with quiet():
            ref = Session(Config())
            ref.train(2 * k)
            s = Session(Config())
            s.train(k)
            n = schema.count_params(Config(width=16))
            s.train(k)
        self.assertGreater(n, 0)
        self.assertTrue(same_params(s.model, ref.model))
        self.assertEqual(s.evaluate(), ref.evaluate())


class Introspection(unittest.TestCase):
    def test_fields_read(self):
        self.assertTrue({"layers", "activation", "layer", "skip"} <= schema.fields_read("mlp"))
        cn = schema.fields_read("custom nn")
        self.assertTrue({"width", "fourier_freq", "classes", "expand"} <= cn)
        self.assertNotIn("layers", cn)

    def test_inert_fields(self):
        self.assertEqual(schema.inert_fields(Config()), {"layers", "layer"})
        self.assertIn("width", schema.inert_fields(Config(model="mlp")))
        self.assertNotIn("features", schema.inert_fields(Config(model="mlp")))

    def test_build_info_matches_session(self):
        with quiet():
            for cfg in (Config(), Config(model="mlp", layers="5:tanh,3"), Config(dataset="Moons")):
                info, s = schema.build_info(cfg), Session(cfg)
                self.assertEqual(info["n_params"], s.n_params)
                self.assertEqual(info["hidden_sizes"], s.hidden_sizes)
                self.assertEqual(info["describe"], s.describe())
                self.assertEqual(info["n_classes"], s.n_classes)

    def test_field_specs(self):
        specs = schema.field_specs()
        self.assertEqual([s["name"] for s in specs], FIELDS)
        by = {s["name"]: s for s in specs}
        self.assertEqual(by["layers"]["read_by"], ["mlp"])
        self.assertIn("custom nn", by["width"]["read_by"])
        self.assertEqual(by["lr"]["read_by"], "*")
        self.assertEqual(by["features"]["ascii_aliases"]["x^2"], "x²")
        self.assertEqual(by["act_decide"]["special"], ["same"])
        self.assertEqual(by["batch_size"]["kind"], "int_or_null")
        self.assertFalse(by["seed"]["search"])
        mlp_only = {s["name"] for s in schema.field_specs("mlp")}
        self.assertNotIn("width", mlp_only)
        self.assertIn("layers", mlp_only)

    def test_describe(self):
        with quiet():
            d = schema.describe()
        json.dumps(d, allow_nan=False)
        for k in ("code_fp", "levers", "models", "registries", "metrics", "objectives"):
            self.assertIn(k, d)
        self.assertIn("general", d["suites"])
        self.assertEqual(set(schema.describe(section="models")), {"models", "code_fp", "parts"})
        self.assertIn("activation", schema.catalog()["registries"])


if __name__ == "__main__":
    unittest.main()
