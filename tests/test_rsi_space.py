"""rsi space: loading and freezing, phenotype identity, validity of samples / mutations /
crossovers, inactive genes, ui_safe. Part of nn-playground (AGPL-3.0; see COMMERCIAL.md)."""
import json
import random
import tempfile
import unittest
from pathlib import Path

from tests._util import quiet  # noqa: F401  (loads my_parts)
import rsi
from nncore import Config, configio
from rsi.errors import RsiError
from rsi.space import Space, poisson

ROOT = Path(__file__).resolve().parent.parent


class SpaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sp = Space.load("default", Config())

    def test_identity_and_twins(self):
        for base in (Config(), Config(model="mlp", layers="8:sin,4", activation="tanh"),
                     Config(model="mlp", features=("y", "x", "x²"), batch_size=None, lr=0.0123456),
                     Config(fourier_freq=3.333, width=7)):
            sp = Space.load("default", base)
            self.assertEqual(sp.phenotype(sp.encode(base)), configio.config_to_dict(base))
        g = self.sp.encode(Config())
        twin = dict(g, layers=[[16, "relu"]], skip="add")  # mlp-only genes on a custom nn genome
        self.assertEqual(self.sp.genome_id(g), self.sp.genome_id(twin))
        self.assertNotIn("layers", self.sp.active(g))
        self.assertIn("layers", self.sp.active(dict(g, model="mlp")))
        mlp = self.sp.phenotype(dict(g, model="mlp", layers=[[8, "softplus"], [8, "softplus"]]))
        self.assertEqual(mlp["layers"], "8,8")  # base string kept verbatim when it means the same
        self.assertEqual(self.sp.phenotype(dict(g, activation="tanh", model="mlp",
                                                layers=[[8, "softplus"]]))["layers"], "8:softplus")

    def test_operators_are_valid(self):
        sp, rng = self.sp, random.Random(1)
        pool = [sp.encode(Config()), sp.encode(Config(model="mlp"))]
        for i in range(1000):
            op = i % 3
            if op == 0:
                g = sp.sample(rng)
            elif op == 1:
                parent = rng.choice(pool)
                g, ops = sp.mutate(parent, rng)
                self.assertTrue(ops)
                self.assertTrue(any(json.dumps(g[n]) != json.dumps(parent[n]) for n in sp.active(parent)), ops)
            else:
                g, _ = sp.crossover(rng.choice(pool), rng.choice(pool), rng)
            configio.check_config(configio.config_from_dict(sp.phenotype(g))[0])
            pool.append(g)
            pool = pool[-30:]
        self.assertEqual([poisson(random.Random(0), 0) for _ in range(3)], [0, 0, 0])

    def test_load_rules(self):
        sp = Space.load("default", Config(model="mlp"), pinned=["model", "lr"], freeze=["init"], unfreeze=["loss"])
        self.assertTrue({"model", "lr", "init", "width", "act_decide"}.isdisjoint(sp.genes))
        self.assertIn("loss", sp.genes)
        self.assertEqual(sp.doc["dropped"]["lr"], "pinned")
        self.assertEqual(list(Space.load("default", Config(), genes=["lr", "width"]).genes), ["width", "lr"])
        self.assertEqual(Space.load("default", Config(), models=["mlp", "custom nn"]).genes["model"]["choices"],
                         ["custom nn", "mlp"])
        again = Space.load(self.sp.to_dict(), Config())  # the frozen space.json loads as is
        self.assertEqual(again.genes, self.sp.genes)
        doc = json.loads((ROOT / "rsi" / "spaces" / "default.json").read_text())
        bad = [({"dataset": {"kind": "choice", "choices": ["Moons"]}}, "task"),
               ({"width": {"kind": "ordinal", "choices": [8, 200]}}, "UI"),
               ({"lr": {"kind": "float", "low": 0.0, "high": 1000.0}}, "bounds"),
               ({"act_decide": {"kind": "activation", "choices": ["relux"]}}, "relux"),
               ({"nope": {"kind": "choice", "choices": [1]}}, "Config field")]
        for genes, word in bad:
            with self.assertRaises(RsiError) as cm:
                Space.load({**doc, "genes": genes}, Config())
            self.assertEqual(cm.exception.code, "E_BAD_SPACE")
            self.assertIn(word, str(cm.exception))
        ok = Space.load({**doc, "ui_safe": False, "genes": {"width": {"kind": "ordinal", "choices": [8, 200]}}}, Config())
        self.assertEqual(ok.genes["width"]["choices"], [8, 200])

    def test_examples_and_random_sweep(self):
        for f in sorted((ROOT / "examples" / "spaces").glob("*.json")):
            base = Config(model="mlp") if "mlp" in f.name else Config()
            sp = Space.load(f, base)
            self.assertTrue(sp.genes, f.name)
        with tempfile.TemporaryDirectory() as tmp, quiet():
            res = rsi.sweep(overrides={"model": "mlp", "dataset": "Moons", "n_points": 120, "lr": 0.01}, random_n=3,
                            space="default", steps=10, workers=0, store=tmp, name="rnd")
        self.assertEqual((res["mode"], res["n_trials"]), ("random", 4))
        for t in res["trials"]:  # fields given on the command line stay pinned
            self.assertTrue({"model", "lr"}.isdisjoint(t["config_diff"]), t["config_diff"])

    def test_extra_genes(self):
        base = Config(model="mlp", train_step="clip grad norm (extra)", extra={"k": 1})
        sp = Space.load({"format": "rsi/space", "version": 1, "genes": {
            "extra.k": {"kind": "choice", "choices": [1, 2]}, "extra.clip": {"kind": "float", "low": 0.1, "high": 5},
            "lr": {"kind": "float", "low": 0.001, "high": 0.1, "log": True}}}, base)
        g = sp.encode(base)
        self.assertEqual((g["extra.k"], g["extra.clip"]), (1, None))  # None: not in the base's extra
        self.assertEqual(sp.phenotype(g), configio.config_to_dict(base))
        rng = random.Random(0)
        for _ in range(100):
            g2, _ = sp.mutate(g, rng)
            ex = sp.phenotype(g2)["extra"]
            self.assertNotIn(None, ex.values())
            configio.check_config(configio.config_from_dict(sp.phenotype(g2))[0])


if __name__ == "__main__":
    unittest.main()
