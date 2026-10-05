"""The rewritten Session must train bit-identically to the legacy one (tests/_legacy_session.py),
even with evaluate() and foreign torch draws between chunks. Part of nn-playground (AGPL-3.0)."""
import platform
import unittest

import torch

from tests._util import SLOW, quiet, same_params
from tests._legacy_session import LegacySession
from nncore import Config, Session

CONFIGS = {
    "default": Config(),
    "mlp+xavier": Config(model="mlp", init="xavier uniform"),
    "epoch shuffle": Config(sampler="epoch shuffle"),
    "mlp batch=None": Config(model="mlp", batch_size=None),
    "wd 0 + exp decay": Config(weight_decay=0.0, schedule="exp decay (×0.999/step)"),
}


class Equivalence(unittest.TestCase):
    def test_against_legacy(self):
        """600 steps as 250+350 for the default config; the others run 100+150 unless RSI_SLOW=1
        (same code paths, a third of the time; RSI_SLOW=1 runs all of them at 250+350)."""
        for name, cfg in CONFIGS.items():
            a, b = (250, 350) if SLOW or name == "default" else (100, 150)
            with self.subTest(config=name), quiet():
                old = LegacySession(Config(**cfg.to_dict()))
                old.train(a + b)
                new = Session(cfg)
                new.train(a)
                new.evaluate()
                torch.rand(7)  # a foreign draw must not shift the new session
                new.predict(torch.zeros(3, 2))
                new.train(b)
                self.assertEqual(new.evaluate(), old.evaluate())
                self.assertTrue(same_params(new.model, old.model))
                self.assertEqual(new.step_count, old.step_count)

    def test_configure_matches_legacy(self):
        """Mid-run optimizer and sampler rebuilds continue the same RNG stream as before."""
        with quiet():
            old, new = LegacySession(Config(model="mlp")), Session(Config(model="mlp"))
            for s in (old, new):
                s.train(100)
                s.configure(lr=0.01)
                s.train(100)
                s.configure(batch_size=16)
                s.train(100)
                s.configure(layers="6:tanh,6")
                s.train(100)
        self.assertEqual(new.evaluate(), old.evaluate())
        self.assertTrue(same_params(new.model, old.model))

    def test_default_fingerprint(self):
        with quiet():
            s = Session(Config())
        self.assertAlmostEqual(sum(p.sum() for p in s.model.parameters()).item(), 17.098907, places=5)
        self.assertLess(abs(float(s.X_all.sum()) - 0.786433), 1e-6)
        if platform.system() != "Linux":
            self.skipTest("loss fingerprint recorded on Linux / torch 2.14")
        losses = [s.train(1), s.train(9), s.train(90)]
        for got, want in zip(losses, (1.873401, 1.620902, 1.452731)):
            self.assertAlmostEqual(got, want, places=5)

    def test_global_rng_untouched(self):
        torch.manual_seed(123)
        before = torch.get_rng_state()
        with quiet():
            s = Session(Config(sampler="epoch shuffle"))
            s.train(20)
            s.evaluate()
            s.configure(width=6)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
