"""Session: transactional configure, private RNG, divergence, eval mode, fresh eval.
Part of nn-playground (AGPL-3.0; see COMMERCIAL.md)."""
import unittest

import numpy as np
import torch
import torch.nn as nn

from tests._util import quiet, same_params
from nncore import ACTIVATIONS, DATASETS, OPTIMIZERS, Config, Session
from nncore.configio import ConfigError

MLP = Config(model="mlp")


def _boom_act():
    raise RuntimeError("boom activation")


def _boom_opt(p, lr, wd):
    raise RuntimeError("boom optimizer")


def setUpModule():
    ACTIVATIONS.register("_test boom", _boom_act)
    OPTIMIZERS.register("_test boom", _boom_opt)
    DATASETS.register("_test one class", lambda n, noise, rng: (rng.uniform(-1, 1, (n, 2)).astype(np.float32),
                                                               np.zeros(n, dtype=np.int64)))


def tearDownModule():
    for reg in (ACTIVATIONS, OPTIMIZERS, DATASETS):
        reg.pop("_test boom", None)
    DATASETS.pop("_test one class", None)


def snapshot(s):
    return {"cfg": s.cfg.to_dict(), "X": s.X_all.copy(), "ytr": s.ytr.clone(), "n_classes": s.n_classes,
            "model": s.model, "opt": s.opt, "params": [p.detach().clone() for p in s.model.parameters()],
            "step": s.step_count, "Xtr": s.Xtr}


class Transactions(unittest.TestCase):
    def assertUnchanged(self, s, snap):
        self.assertEqual(s.cfg.to_dict(), snap["cfg"])
        np.testing.assert_array_equal(s.X_all, snap["X"])
        self.assertTrue(torch.equal(s.ytr, snap["ytr"]))
        self.assertEqual(s.n_classes, snap["n_classes"])
        self.assertIs(s.model, snap["model"])
        self.assertIs(s.opt, snap["opt"])
        self.assertIs(s.Xtr, snap["Xtr"])
        self.assertEqual(s.step_count, snap["step"])

    def _check(self, cfg, bad):
        """train 60, failed configure, train 60 == uninterrupted 120."""
        with quiet():
            ref = Session(cfg)
            ref.train(120)
            s = Session(cfg)
            s.train(60)
            snap = snapshot(s)
            with self.assertRaises((ValueError, RuntimeError)):
                s.configure(**bad)
            self.assertUnchanged(s, snap)
            s.train(60)
        self.assertTrue(same_params(s.model, ref.model))
        self.assertEqual(s.evaluate(), ref.evaluate())

    def test_data_change_failing_in_model_build(self):
        self._check(Config(), {"dataset": "Moons", "act_decide": "_test boom"})

    def test_model_change_failing_in_optimizer(self):
        self._check(MLP, {"layers": "4,4", "optimizer": "_test boom"})
        self._check(MLP, {"layers": "4,4", "lr": -1.0})

    def test_live_key_validated(self):
        self._check(MLP, {"loss": "nope"})
        self._check(MLP, {"train_step": "nope"})

    def test_failed_configure_does_not_reseed(self):
        self._check(Config(), {"act_decide": "nope"})
        self._check(Config(sampler="epoch shuffle"), {"width": 4, "init": "nope"})

    def test_new_data_failure(self):
        with quiet():
            s = Session(MLP)
        snap = snapshot(s)
        with self.assertRaises(ValueError):
            s.configure(dataset="_test one class")
        self.assertUnchanged(s, snap)


class Behaviour(unittest.TestCase):
    def test_no_op_configure(self):
        with quiet():
            s = Session(Config())
            s.train(5)
            self.assertIsNone(s.configure(features=["x", "y"]))
            self.assertIsNone(s.configure(lr="0.03", width=8.0))
        self.assertEqual(s.step_count, 5)

    def test_levels(self):
        s = Session(MLP)
        self.assertEqual(s.configure(lr=0.01), "optimizer")
        self.assertEqual(s.configure(batch_size="full"), "sampler")
        self.assertIsNone(s.configure(loss="mse on softmax"))
        self.assertEqual(s.configure(layers="4"), "model")
        self.assertEqual(s.configure(noise=0.1), "data")
        self.assertIsNone(s.cfg.batch_size)

    def test_caller_cfg_not_mutated(self):
        cfg = Config(model="mlp", extra={"k": [1]})
        s = Session(cfg)
        s.configure(lr=0.5)
        with self.assertRaises(ValueError):
            s.configure(layers="0")
        self.assertEqual(cfg.lr, 0.03)
        self.assertEqual(cfg.layers, "8,8")
        cfg.extra["k"].append(2)
        self.assertEqual(s.cfg.extra, {"k": [1]})
        d = {"z": 1}
        s.configure(extra=d)
        d["z"] = 2
        self.assertEqual(s.cfg.extra, {"z": 1})

    def test_dict_config(self):
        s = Session({"model": "mlp", "layers": "4"})
        self.assertEqual(s.hidden_sizes, [4])

    def test_stop_on_nonfinite(self):
        cfg = Config(model="mlp", layers="16:square,16:square,16:square", optimizer="SGD", lr=1.0, dataset="Moons")
        s = Session(cfg)
        loss = s.train(2000, stop_on_nonfinite=True)
        self.assertIsNotNone(s.diverged_at)
        self.assertLess(s.diverged_at, 2000)
        self.assertEqual(s.step_count, s.diverged_at)
        self.assertFalse(np.isfinite(loss))
        s2 = Session(cfg)
        s2.train(100)  # default: keeps stepping, but still records where it went bad
        self.assertEqual(s2.step_count, 100)
        self.assertEqual(s2.diverged_at, s.diverged_at)

    def test_train_arguments(self):
        s = Session(MLP)
        with self.assertRaises(ValueError):
            s.train(-1)
        s.train(2.0)
        self.assertEqual(s.step_count, 2)
        self.assertTrue(np.isnan(s.train(0)))

    def test_evaluate_keys_and_mode(self):
        with quiet():
            s = Session(Config())
        ev = s.evaluate()
        self.assertEqual(set(ev), {"train", "test"})
        self.assertEqual(set(ev["train"]), {"loss", "acc"})
        self.assertTrue(s.model.training)
        s2 = Session(Config(model="mlp", splitter="none (train on all)"))
        self.assertTrue(np.isnan(s2.evaluate()["test"]["acc"]))

    def test_eval_mode_used(self):
        ACTIVATIONS.register("_testdropout", lambda: nn.Dropout(0.5))
        try:
            s = Session(Config(model="mlp", layers="8:_testdropout"))
            a, b = s.evaluate(), s.evaluate()
            self.assertEqual(a, b)  # dropout off in eval mode
            self.assertTrue(s.model.training)
        finally:
            ACTIVATIONS.pop("_testdropout")

    def test_predict_inputs(self):
        s = Session(MLP)
        a = s.predict(np.array([[0.1, 0.2], [0.3, -0.4]], dtype=np.float64))
        b = s.predict([[0.1, 0.2], [0.3, -0.4]])
        c = s.predict(torch.tensor([[0.1, 0.2], [0.3, -0.4]]))
        self.assertTrue(torch.equal(a, b) and torch.equal(b, c))
        logits, hidden = s.predict([[0.0, 0.0]], collect=True)
        self.assertEqual(len(hidden), 2)
        with self.assertRaises(ValueError):
            s.predict([0.1, 0.2])

    def test_replace_config_equals_fresh(self):
        c = Config(model="mlp", layers="6:tanh", sampler="epoch shuffle", seed=3)
        a = Session(MLP)
        a.train(50)
        a.replace_config(c)
        b = Session(c)
        a.train(300)
        b.train(300)
        self.assertTrue(same_params(a.model, b.model))
        self.assertEqual(a.evaluate(), b.evaluate())

    def test_interleaved_sessions_equal_solo(self):
        solo = Session(Config(model="mlp", seed=0))
        solo.train(300)
        a, b = Session(Config(model="mlp", seed=0)), Session(Config(model="mlp", seed=1))
        for _ in range(3):
            a.train(100)
            b.train(100)
            torch.randn(5)
        self.assertTrue(same_params(a.model, solo.model))

    def test_evaluate_fresh(self):
        s = Session(MLP)
        s.train(100)
        f1, f2 = s.evaluate_fresh(500), s.evaluate_fresh(500)
        self.assertEqual(f1, f2)
        self.assertEqual(set(f1), {"loss", "acc"})
        ref = Session(MLP)
        ref.train(200)
        s.train(100)
        self.assertTrue(same_params(s.model, ref.model))

    def test_skip_irrelevant_and_keep_optimizer_state(self):
        s = Session(MLP)
        s.train(30)
        self.assertIsNone(s.configure(width=16, skip_irrelevant=True))
        self.assertEqual((s.step_count, s.cfg.width), (30, 16))
        self.assertEqual(s.configure(classes=3), "model")
        s.train(30)
        opt = s.opt
        self.assertEqual(s.configure(lr=0.01, keep_optimizer_state=True), "optimizer")
        self.assertIs(s.opt, opt)
        self.assertEqual(s.lr, 0.01)
        self.assertEqual(s.step_count, 30)

    def test_readable_data_errors(self):
        with self.assertRaisesRegex(ValueError, "fewer than 2 classes"):
            Session(Config(model="mlp", dataset="_test one class"))
        with self.assertRaises(ValueError):
            with quiet():
                Session(Config(n_points=3))
        with self.assertRaisesRegex(ConfigError, r"test_frac must be in \[0, 1\)"):
            Session(Config(model="mlp", test_frac=-0.2))

    def test_new_data_seed(self):
        s = Session(MLP)
        s.new_data(seed=4)
        self.assertEqual(s.cfg.seed, 4)
        self.assertTrue(same_params(s.model, Session(Config(model="mlp", seed=4)).model))


if __name__ == "__main__":
    unittest.main()
