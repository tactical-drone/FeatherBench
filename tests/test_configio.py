"""configio: JSON round trips, aliases, coercion, bounds, keys, settings files.
Part of nn-playground (AGPL-3.0; see COMMERCIAL.md)."""
import json
import math
import os
import tempfile
import unittest
from unittest import mock

from tests._util import quiet  # noqa: F401  (loads my_parts)
from nncore import Config, configio
from nncore.configio import (ConfigError, canonical_json, check_config, coerce_value, config_diff, config_from_dict,
                             config_key, config_to_command, config_to_dict, dumps, json_safe, load_settings,
                             normalise, resolve_name, save_settings, ui_issues, validate_config)


class RoundTrip(unittest.TestCase):
    def test_round_trip(self):
        d = json.loads(json.dumps(config_to_dict(Config())))
        cfg, warns = config_from_dict(d)
        self.assertEqual(cfg, Config())
        self.assertEqual(warns, [])
        c2 = Config(features=("x", "y", "r"), batch_size=None, extra={"a": [1, 2]})
        self.assertEqual(config_from_dict(json.loads(json.dumps(config_to_dict(c2))))[0], c2)

    def test_dict_order_and_lists(self):
        d = config_to_dict(Config())
        self.assertEqual(list(d), [f for f in Config().to_dict()])
        self.assertIsInstance(d["features"], list)

    def test_missing_fields_filled(self):
        cfg, warns = config_from_dict({"model": "mlp"})
        self.assertEqual(cfg, Config(model="mlp"))
        self.assertEqual(warns[0]["code"], "W_DEFAULT_FILLED")
        self.assertEqual(config_from_dict({"lr": 0.1}, base=Config(model="mlp"))[1], [])

    def test_normalise(self):
        self.assertEqual(normalise(Config()), Config())
        n = normalise(Config(features=["x", "y"], lr=1, width=8.0))
        self.assertEqual(n.features, ("x", "y"))
        self.assertIsInstance(n.lr, float)
        self.assertIsInstance(n.width, int)


class Names(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(resolve_name("activation", "relu"), ("relu", None))
        self.assertEqual(resolve_name("act_decide", "same"), ("same", None))

    def test_aliases(self):
        self.assertEqual(resolve_name("features", "x^2")[0], "x²")
        self.assertEqual(resolve_name("features", "x*y")[0], "x·y")
        self.assertEqual(resolve_name("schedule", "step (/10 every 2000)")[0], "step (÷10 every 2000)")
        self.assertEqual(resolve_name("schedule", "exp decay (x0.999/step)")[0], "exp decay (×0.999/step)")
        name, warn = resolve_name("schedule", "step")
        self.assertEqual(name, "step (÷10 every 2000)")
        self.assertEqual(warn["code"], "W_ALIAS_USED")
        self.assertEqual(resolve_name("optimizer", "adam")[0], "Adam")

    def test_features_in_dict(self):
        cfg, warns = config_from_dict({"features": "x,y,x^2"})
        self.assertEqual(cfg.features, ("x", "y", "x²"))
        self.assertIn("W_ALIAS_USED", [w["code"] for w in warns])
        self.assertEqual(config_from_dict({"features": ["y", "x"]})[0].features, ("y", "x"))  # order kept

    def test_ambiguous_and_unknown(self):
        with self.assertRaises(ConfigError) as cm:
            resolve_name("activation", "s")
        self.assertEqual(cm.exception.code, "E_AMBIGUOUS")
        self.assertTrue(cm.exception.did_you_mean)
        with self.assertRaises(ConfigError) as cm:
            resolve_name("activation", "relux")
        self.assertEqual(cm.exception.code, "E_UNKNOWN_PART")
        self.assertIn("relu", cm.exception.did_you_mean)
        d = cm.exception.to_dict()
        self.assertEqual(d["field"], "activation")
        self.assertIn("relu", d["allowed"])

    def test_unknown_field(self):
        with self.assertRaises(ConfigError) as cm:
            config_from_dict({"lrr": 0.1})
        self.assertEqual(cm.exception.code, "E_UNKNOWN_FIELD")
        self.assertIn("lr", cm.exception.did_you_mean)


class Values(unittest.TestCase):
    def test_coercion(self):
        self.assertEqual(coerce_value("lr", "0.01"), 0.01)
        self.assertIsInstance(coerce_value("lr", "0.01"), float)
        self.assertEqual(coerce_value("width", 8.0), 8)
        self.assertIsInstance(coerce_value("width", 8.0), int)
        self.assertEqual(coerce_value("width", "8"), 8)
        for bad in (8.5, "8.5", "eight", True):
            with self.assertRaises(ConfigError):
                coerce_value("width", bad)
        self.assertIsNone(coerce_value("batch_size", "full"))
        self.assertIsNone(coerce_value("batch_size", "null"))
        self.assertIsNone(coerce_value("batch_size", None))
        self.assertEqual(coerce_value("batch_size", "16"), 16)
        self.assertEqual(coerce_value("features", "x, y"), ("x", "y"))
        self.assertEqual(coerce_value("extra", '{"clip": 2}'), {"clip": 2})
        with self.assertRaises(ConfigError):
            coerce_value("lr", "nan")

    def test_bounds(self):
        with self.assertRaises(ConfigError) as cm:
            config_from_dict({"test_frac": 1.0})
        self.assertEqual(cm.exception.code, "E_OUT_OF_RANGE")
        codes = {i["code"] for i in validate_config(Config(width=-1, lr=0.0, seed=-1))}
        self.assertEqual(codes, {"E_OUT_OF_RANGE"})
        self.assertEqual(validate_config(Config()), [])
        check_config(Config(model="mlp", width=0))

    def test_layers_checked_for_mlp_only(self):
        issues = validate_config(Config(model="mlp", layers="8;8"))
        self.assertEqual(issues[0]["code"], "E_BAD_LAYERS")
        self.assertEqual(validate_config(Config(layers="8;8")), [])  # custom nn ignores layers

    def test_validate_names(self):
        issues = validate_config(Config(act_decide="Gauss", features=("x", "nope")))
        self.assertEqual({i["field"] for i in issues}, {"act_decide", "features"})
        self.assertIn("gauss", [i for i in issues if i["field"] == "act_decide"][0]["did_you_mean"])

    def test_ui_issues(self):
        self.assertEqual(ui_issues(Config()), [])
        codes = {i["code"] for i in ui_issues(Config(width=200, noise=0.123, features=("y", "x")))}
        self.assertEqual(codes, {"W_NOT_UI_EXACT", "W_FEATURE_ORDER"})


class Hashing(unittest.TestCase):
    def test_config_key(self):
        a = config_to_dict(Config(model="mlp", lr=1))
        b = dict(reversed(list(a.items())))
        b["lr"] = 1.0
        self.assertEqual(config_key(a), config_key(b))
        self.assertEqual(config_key(Config(model="mlp", lr=1)), config_key(a))
        self.assertNotEqual(config_key(Config()), config_key(Config(lr=0.02)))
        self.assertRegex(config_key(Config()), r"^[0-9a-f]{16}$")

    def test_canonical_json(self):
        self.assertEqual(canonical_json({"b": 1, "a": float("nan")}), '{"a":null,"b":1}')
        self.assertEqual(canonical_json({"x": "x²"}), '{"x":"x\\u00b2"}')

    def test_json_safe(self):
        nf = []
        out = json_safe({"a": (1, float("inf")), "b": {"c": float("nan")}}, nonfinite=nf)
        self.assertEqual(out, {"a": [1, None], "b": {"c": None}})
        self.assertEqual(nf, ["a[1]", "b.c"])
        text = dumps({"s": "÷", "v": math.nan})
        self.assertTrue(text.isascii())
        self.assertEqual(json.loads(text), {"s": "÷", "v": None})

    def test_diff_and_command(self):
        cfg = Config(model="mlp", layers="8:sin,8", batch_size=None, features=("x", "y", "x²"))
        self.assertEqual(set(config_diff(cfg)), {"model", "layers", "batch_size", "features"})
        cmd = config_to_command(cfg)
        self.assertTrue(cmd.startswith("python -m rsi run "))
        self.assertIn("--model mlp", cmd)
        self.assertIn("--batch-size full", cmd)
        self.assertIn('--features "x,y,x²"', cmd)


class SettingsFiles(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "s.json")

    def tearDown(self):
        self.dir.cleanup()

    def test_save_load(self):
        cfg = Config(model="mlp", features=("y", "x", "x²"), batch_size=None)
        save_settings(self.path, cfg, ui={"tab": "Boundary"}, train={"steps": 300}, meta={"note": "÷"})
        with open(self.path, encoding="utf-8") as f:
            text = f.read()
        self.assertEqual(json.loads(text)["format"], "nn-playground/settings")
        self.assertIn("x²", text)  # settings files keep UTF-8
        s = load_settings(self.path)
        self.assertEqual(s.config, cfg)
        self.assertEqual(s.train, {"steps": 300})
        self.assertEqual(s.ui, {"tab": "Boundary"})
        save_settings(self.path, Config())  # ui block kept on rewrite
        self.assertEqual(load_settings(self.path).ui, {"tab": "Boundary"})
        self.assertEqual([n for n in os.listdir(self.dir.name)], ["s.json"])  # no tmp files left

    def test_atomic_write_retries(self):
        real = os.replace
        calls = []

        def flaky(a, b):
            calls.append(1)
            if len(calls) < 3:
                raise PermissionError("busy")
            return real(a, b)
        with mock.patch.object(configio.os, "replace", flaky):
            save_settings(self.path, Config())
        self.assertEqual(len(calls), 3)
        self.assertEqual(load_settings(self.path).config, Config())

    @unittest.skipIf(os.name == "nt", "POSIX file modes")
    def test_atomic_write_file_mode(self):
        save_settings(self.path, Config())  # new file: 0666 & ~umask, not mkstemp's 0600
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o666 & ~configio._UMASK)
        os.chmod(self.path, 0o640)
        save_settings(self.path, Config(model="mlp"))  # overwrite keeps the mode
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o640)

    def test_accepted_shapes(self):
        trial = {"id": "t_1", "config": config_to_dict(Config(model="mlp"))}
        self.assertEqual(load_settings(trial).config, Config(model="mlp"))
        self.assertEqual(load_settings({"model": "mlp"}).config, Config(model="mlp"))

    def test_strict_vs_lenient(self):
        doc = {"format": "nn-playground/settings", "version": 1,
               "config": {"model": "mlp", "activation": "relux", "bogus": 1, "width": 9999}}
        with self.assertRaises(ConfigError):
            load_settings(doc)
        s = load_settings(doc, strict=False)
        self.assertEqual(s.config.activation, Config().activation)
        self.assertEqual(s.config.width, Config().width)
        codes = {w["code"] for w in s.warnings}
        self.assertTrue({"W_UNKNOWN_KEY", "W_UNKNOWN_PART"} <= codes)
        doc["config"] = {"activation": "relux"}
        doc["meta"] = {"parts": ["agent_parts.missing"]}
        with self.assertRaises(ConfigError) as cm:
            load_settings(doc)
        self.assertEqual(cm.exception.code, "E_PARTS_MISSING")

    def test_version(self):
        with self.assertRaises(ConfigError) as cm:
            load_settings({"format": "nn-playground/settings", "version": 2, "config": {}}, strict=False)
        self.assertEqual(cm.exception.code, "E_BAD_SETTINGS")


class RegistryApi(unittest.TestCase):
    def test_get_semantics(self):
        from nncore import ACTIVATIONS
        self.assertIsNone(ACTIVATIONS.get("nope", None))
        self.assertIs(ACTIVATIONS.get("relu"), ACTIVATIONS["relu"])
        with self.assertRaises(ValueError) as cm:
            ACTIVATIONS.get("Relu")
        self.assertIn("did you mean", str(cm.exception))
        self.assertIn("load_plugins", str(cm.exception))

    def test_register_doc_and_override(self):
        import warnings
        from nncore import Registry
        reg = Registry("thing")
        reg.register("a", lambda: 1, doc="first")
        self.assertEqual(reg.docs["a"], "first")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reg.register("a", print, override=False)  # print lives in another module: kept, warned
        self.assertEqual(reg["a"](), 1)
        self.assertTrue(any(issubclass(x.category, UserWarning) for x in w))
        reg.register("a", print)
        self.assertIs(reg["a"], print)

    def test_resolver(self):
        import numpy as np
        from nncore import DATASETS, Session
        from nncore.datasets import spiral

        def fam(name):
            if name.startswith("_test spiral[") and name.endswith("]"):
                return spiral(int(name[len("_test spiral["):-1]))
            return None
        DATASETS.add_resolver(fam)
        try:
            self.assertIsNone(DATASETS.get("_test other", None))
            fn = DATASETS.get("_test spiral[7]")
            self.assertIs(fn, DATASETS.get("_test spiral[7]"))  # cached
            self.assertNotIn("_test spiral[7]", list(DATASETS))  # not in the dropdown list
            self.assertEqual(resolve_name("dataset", "_test spiral[7]"), ("_test spiral[7]", None))
            cfg = config_from_dict({"model": "mlp", "dataset": "_test spiral[7]", "n_points": 700})[0]
            self.assertEqual(int(np.max(fn(70, 0.0, np.random.default_rng(0))[1])), 6)
            self.assertEqual(Session(cfg).n_classes, 7)
        finally:
            DATASETS._resolvers.remove(fam)
            DATASETS.resolved.pop("_test spiral[7]", None)

    def test_resolver_rejection(self):
        from nncore import Registry
        from nncore.registry import ResolveError
        reg = Registry("thing")

        def res(name):
            if name.startswith("t:"):
                if name == "t:bad":
                    raise ResolveError("t:bad is malformed", "E_BAD_EXPR", hint="fix it")
                return len
            return None
        reg.add_resolver(res)
        self.assertTrue(reg.has("t:ok"))
        self.assertFalse(reg.has("t:bad"))  # a rejection is a miss, never an exception
        self.assertIsNone(reg.get("t:bad", None))
        with self.assertRaises(ResolveError) as cm:
            reg.get("t:bad")
        self.assertEqual((str(cm.exception), cm.exception.code, cm.exception.hint),
                         ("t:bad is malformed", "E_BAD_EXPR", "fix it"))
        self.assertIsNone(reg.rejection("t:ok"))

    def test_call_accepting(self):
        from nncore import call_accepting
        self.assertEqual(call_accepting(lambda a: a, 1, cfg=2, step=3), 1)
        self.assertEqual(call_accepting(lambda a, cfg=None: (a, cfg), 1, cfg=2, step=3), (1, 2))
        self.assertEqual(call_accepting(lambda a, **kw: kw, 1, cfg=2), {"cfg": 2})

    def test_plugins(self):
        from nncore import load_plugins, loaded_plugins
        self.assertEqual(load_plugins("my_parts"), ["my_parts"])
        self.assertEqual(load_plugins(os.path.join(os.path.dirname(configio.__file__), "..", "my_parts.py")),
                         ["my_parts"])
        self.assertIn("my_parts", loaded_plugins())

    def test_lazy_plugins_in_fresh_process(self):
        import subprocess
        import sys
        root = os.path.dirname(os.path.dirname(os.path.abspath(configio.__file__)))
        code = "from nncore import Config, Session; s = Session(Config()); print(s.describe())"
        r = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip().splitlines()[-1], "wide 8")


if __name__ == "__main__":
    unittest.main()
