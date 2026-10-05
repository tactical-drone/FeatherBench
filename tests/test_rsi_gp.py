"""rsi gp (P2): expression syntax, simplify, sanity, no eval/exec, the ACTIVATIONS resolver, learnable
constants, variation operators, `gp check`. Part of nn-playground (AGPL-3.0; see COMMERCIAL.md)."""
import ast
import random
import unittest
from pathlib import Path

import torch

from tests._util import quiet  # noqa: F401  (loads my_parts)
from nncore import ACTIVATIONS, Config, configio
from nncore.models import parse_layers
from nncore.run import RunSpec, run_unit
from rsi import gp
from rsi.cli import invoke
from rsi.errors import RsiError

ROOT = Path(__file__).resolve().parent.parent


class GPTests(unittest.TestCase):
    def test_syntax(self):
        t = gp.parse("gp:mul(x;sin(mul(3;x)))")
        self.assertEqual(gp.to_name(t), "gp:mul(x;sin(mul(3;x)))")
        self.assertEqual(gp.to_infix(t), "x*sin(3*x)")
        self.assertEqual((gp.size(t), gp.depth(t)), (6, 4))
        self.assertEqual(gp.to_name(gp.parse("add(x;pi)")), "gp:add(x;pi)")
        for bad in ("gp:mul(x)", "gp:foo(x)", "gp:add(x,1)", "gp:sin(x", "gp:", "gp:sin(x) "):
            with self.assertRaises(gp.GPError, msg=bad):
                gp.parse(bad)
        self.assertEqual(parse_layers("8:gp:sin(x),4", "tanh"), [(8, "gp:sin(x)"), (4, "tanh")])

    def test_simplify_and_sanity(self):
        s = lambda e: gp.to_name(gp.simplify(gp.parse(e)))  # noqa: E731
        self.assertEqual(s("gp:add(mul(x;1);0)"), "gp:x")
        self.assertEqual(s("gp:neg(neg(sin(x)))"), "gp:sin(x)")
        self.assertEqual(s("gp:mul(add(1;2);x)"), "gp:mul(3;x)")
        self.assertEqual(s("gp:sub(x;x)"), "gp:0")
        for e, ok in (("gp:sin(x)", True), ("gp:mul(2;3)", False), ("gp:mul(x;mul(x;mul(x;mul(x;mul(x;x)))))", False),
                      ("gp:relu(sub(0;sq(x)))", False)):
            self.assertEqual(gp.sanity(gp.parse(e))[0], ok, e)
        self.assertEqual(gp.semantic_hash(gp.parse("gp:add(x;x)")), gp.semantic_hash(gp.parse("gp:mul(2;x)")))

    def test_no_eval_or_exec(self):
        tree = ast.parse((ROOT / "rsi" / "gp.py").read_text(encoding="utf-8"))
        calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertFalse(calls & {"eval", "exec", "compile", "__import__"})

    def test_resolver_training_and_learnable(self):
        name = "gp:mul(x;sin(mul(3;x)))"
        self.assertTrue(ACTIVATIONS.has(name))
        cfg = Config(model="mlp", layers=f"8:{name},4", dataset="Moons", n_points=120)
        configio.check_config(cfg)
        cell = run_unit(RunSpec(config=configio.config_to_dict(cfg), steps=20))
        self.assertEqual(cell["status"], "ok", cell.get("error"))
        m = ACTIVATIONS.get("gpl:add(x;mul(0.5;sin(x)))")()
        self.assertEqual(len(list(m.parameters())), 1)
        y = m(torch.linspace(-1, 1, 5))
        y.sum().backward()
        self.assertIsNotNone(next(m.parameters()).grad)
        self.assertEqual(len(list(ACTIVATIONS.get(name)().parameters())), 0)

    def test_variation(self):
        rng = random.Random(0)
        for _ in range(200):
            a, b = gp.random_tree(rng, 4), gp.random_tree(rng, 4, "full")
            self.assertLessEqual(gp.depth(a), 4)
            m = gp.mutate(a, rng, 4, 15)
            c = gp.crossover(a, b, rng, 4, 15)
            for t in (m, c):
                self.assertLessEqual(gp.depth(t), 4)
                self.assertLessEqual(gp.size(t), max(15, gp.size(a)))
                self.assertEqual(gp.parse(gp.to_name(t)), t)

    def test_check_command(self):
        code, d = invoke(["gp", "check", "gp:mul(x;sin(mul(3;x)))"])
        self.assertEqual((code, d["schema"], d["result"]["ok"], d["result"]["infix"]),
                         (0, "rsi/gp-check@1", True, "x*sin(3*x)"))
        code, d = invoke(["gp", "check", "gp:mul(x)"])
        self.assertEqual((code, d["error"]["code"]), (3, "E_BAD_EXPR"))
        with self.assertRaises(RsiError):
            gp.check("nope(")

    def test_bad_expression_reports_why(self):
        deep = "gp:" + "neg(" * 300 + "x" + ")" * 300
        for name, why in (("gp:foo(x)", "unknown op 'foo'"), ("gp:neg(", "unexpected end"), (deep, "too deeply")):
            self.assertFalse(ACTIVATIONS.has(name))
            with self.assertRaises(configio.ConfigError) as cm:
                configio.resolve_name("activation", name)
            self.assertEqual(cm.exception.code, "E_BAD_EXPR")
            self.assertIn(why, str(cm.exception))
            self.assertIn("gp check", cm.exception.hint)
            with self.assertRaises(ValueError) as cm:
                ACTIVATIONS.get(name)
            self.assertIn(why, str(cm.exception))
        issues = configio.validate_config(Config(model="mlp", layers="8:gp:neg(,4"))
        self.assertEqual([(i["code"], i["field"]) for i in issues], [("E_BAD_EXPR", "layers")])
        code, d = invoke(["check", "--model", "mlp", "--activation", "gp:foo(x)"])
        self.assertEqual((code, d["error"]["code"]), (3, "E_BAD_EXPR"))
        self.assertIn("unknown op 'foo'", d["error"]["message"])
        code, d = invoke(["check", "--activation", "relux"])  # plain names keep E_UNKNOWN_PART
        self.assertEqual((code, d["error"]["code"]), (3, "E_UNKNOWN_PART"))


if __name__ == "__main__":
    unittest.main()
