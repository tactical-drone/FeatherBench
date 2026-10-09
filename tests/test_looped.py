"""Looped models and cross-loop self-distillation (after arXiv 2610.10623), and the parity
complexity family (linear equations mod 2). Part of nn-playground (AGPL-3.0; see COMMERCIAL.md).

Run:  python -m unittest tests.test_looped -v
"""
import math
import unittest

import numpy as np
import torch

from tests._util import quiet
from nncore import Config, Session
from nncore.datasets import DATASETS, FAMILIES, family_datasets
from nncore.training import TRAIN_STEPS


def session(**kw):
    with quiet():
        return Session(Config(**kw))


class Looped(unittest.TestCase):
    def test_params_do_not_grow_with_loops(self):
        sizes = {L: session(model="looped", extra={"loops": L}) for L in (1, 4, 12)}
        self.assertEqual(len({s.n_params for s in sizes.values()}), 1)
        self.assertEqual(sizes[12].hidden_sizes, [8] * 12)
        self.assertEqual(sizes[4].describe(), "looped 8x4")

    def test_loop_logits_and_forward_agree(self):
        s = session(model="looped", extra={"loops": 5})
        x = torch.rand(9, 2) * 2 - 1
        feats = s.Xtr[:9]
        logits = s.model.loop_logits(feats)
        self.assertEqual(len(logits), 5)
        self.assertTrue(torch.allclose(logits[-1], s.model(feats)))
        out, hidden = s.predict(x, collect=True)
        self.assertEqual([h.shape[-1] for h in hidden], s.hidden_sizes)

    def test_reads_the_documented_fields(self):
        from nncore import schema
        read = schema.fields_read("looped")
        self.assertTrue({"width", "expand", "fourier_freq", "activation"} <= read, read)


class CrossLoopDistill(unittest.TestCase):
    def test_trains_finitely_and_differs_from_standard(self):
        a = session(model="looped", train_step="cross-loop distill")
        b = session(model="looped")
        la, lb = a.train(50), b.train(50)
        self.assertTrue(math.isfinite(la) and math.isfinite(lb))
        x = a.Xtr[:16]
        self.assertFalse(torch.equal(a.model(x), b.model(x)))  # the distillation term changes training

    def test_knobs_and_fallback(self):
        s = session(model="looped", train_step="cross-loop distill",
                    extra={"loops": 6, "distill_loop": 2, "distill_weight": 0.5})
        self.assertTrue(math.isfinite(s.train(20)))
        m = session(model="mlp", train_step="cross-loop distill")  # no loop_logits: a standard step
        self.assertTrue(math.isfinite(m.train(20)))
        self.assertIn("cross-loop distill", TRAIN_STEPS)


class ParityFamily(unittest.TestCase):
    def test_family(self):
        self.assertIn("parity", FAMILIES)
        names = family_datasets("parity", [1, 3, 6])
        self.assertEqual(names, ["parity[1]", "parity[3]", "parity[6]"])
        for name in names:
            X, y = DATASETS.get(name)(600, 0.05, np.random.default_rng(0))
            self.assertEqual(X.shape, (600, 2))
            self.assertEqual(sorted(set(y.tolist())), [0, 1])
            self.assertLess(abs(int((y == 1).sum()) - 300), 30)  # stratified, roughly balanced


if __name__ == "__main__":
    unittest.main()


class LoopCap(unittest.TestCase):
    """FeatherBench rule: a looped submission may use at most MAX_LOOPS loops."""

    def test_cap(self):
        import json, tempfile
        from pathlib import Path
        from rsi.errors import RsiError
        from rsi.featherbench import MAX_LOOPS, check_submission
        with tempfile.TemporaryDirectory() as tmp:
            for loops, ok in ((4, True), (MAX_LOOPS, True), (MAX_LOOPS + 1, False), (0, False), (2.5, False),
                              ("lots", False), (True, False)):
                p = Path(tmp) / "alice.json"
                p.write_text(json.dumps({"model": "looped", "extra": {"loops": loops}}), encoding="utf-8")
                with self.subTest(loops=loops):
                    if ok:
                        self.assertEqual(check_submission(p)[0].extra["loops"], loops)
                    else:
                        with self.assertRaises(RsiError):
                            check_submission(p)
