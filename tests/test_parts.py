"""Tests for the parts layer (WP-parts): new parts, opt-in fixes, readable errors.
Part of nn-playground (AGPL-3.0; see COMMERCIAL.md).

Run:  PYTHONPATH=. python -m unittest tests.test_parts -v
"""
import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from tests._util import quiet
from nncore import (ACTIVATIONS, EXPANSIONS, FEATURES, INITIALIZERS, LOSSES, OPTIMIZERS, SAMPLERS,
                    SCHEDULES, SPLITTERS, TRAIN_STEPS, Config, Session, featurize, make_expansion)
import nncore.samplers as samplers_mod
from nncore.datasets import random_split, stratified_split
from nncore.layers import Fourier, Polar
from nncore.metrics import EXTRA_METRICS
from nncore.models import MLP, format_layers, parse_layers

# The registries as they were before WP-parts: new entries must come after these, in this order.
ORIGINAL = [
    (ACTIVATIONS, ["tanh", "relu", "leaky_relu", "sigmoid", "gelu", "silu", "mish", "elu", "softplus", "sin",
                   "gauss", "abs", "square", "linear"]),
    (FEATURES, ["x", "y", "x²", "y²", "x·y", "sin x", "sin y", "r"]),
    (INITIALIZERS, ["pytorch default", "xavier uniform", "kaiming normal"]),
    (LOSSES, ["cross entropy", "cross entropy (smoothed 0.1)", "mse on softmax"]),
    (OPTIMIZERS, ["Adam", "AdamW", "SGD", "SGD+momentum", "RMSprop"]),
    (SCHEDULES, ["constant", "exp decay (×0.999/step)", "step (÷10 every 2000)"]),
    (TRAIN_STEPS, ["standard", "clip grad norm 1.0"]),
    (SAMPLERS, ["random (with replacement)", "epoch shuffle"]),
    (SPLITTERS, ["random", "none (train on all)"]),
]
NEW_ACTS = ["snake", "cos", "selu", "softsign", "hardtanh", "prelu", "x·sin x"]
NEW_INITS = ["xavier uniform (keep expansion)", "kaiming normal (keep expansion)", "orthogonal", "xavier normal",
             "lecun normal"]


def session(**kw):
    with quiet():
        return Session(Config(**kw))


class RegistryOrder(unittest.TestCase):
    def test_new_parts_are_appended(self):
        for reg, names in ORIGINAL:
            with self.subTest(registry=reg.kind):
                self.assertEqual(list(reg)[:len(names)], names)

    def test_new_parts_registered(self):
        for reg, names in ((ACTIVATIONS, NEW_ACTS), (INITIALIZERS, NEW_INITS),
                           (OPTIMIZERS, ["NAdam", "RAdam", "Adamax", "SGD+nesterov"]), (LOSSES, ["focal (γ=2)"]),
                           (FEATURES, ["cos x", "cos y", "θ (atan2)"]),
                           (SCHEDULES, ["cosine (over extra.total_steps)", "cosine warm restarts (T0=1000)",
                                        "warmup 200 + constant"]),
                           (TRAIN_STEPS, ["skip non-finite", "clip grad norm (extra)"])):
            for n in names:
                self.assertIn(n, reg)

    def test_default_fingerprint(self):
        s = session()
        self.assertAlmostEqual(sum(float(p.detach().sum()) for p in s.model.parameters()), 17.098907, places=5)
        self.assertLess(abs(float(s.X_all.sum()) - 0.786433), 1e-6)


class Activations(unittest.TestCase):
    def test_finite_with_gradient(self):
        for name in NEW_ACTS:
            with self.subTest(activation=name):
                x = torch.linspace(-5, 5, 201, requires_grad=True)
                y = ACTIVATIONS.get(name)()(x)
                self.assertEqual(y.shape, x.shape)
                self.assertTrue(torch.isfinite(y).all())
                y.sum().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertGreater(float(x.grad.abs().sum()), 0)

    def test_values(self):
        x = torch.tensor([0.0, 1.0, -2.0])
        torch.testing.assert_close(ACTIVATIONS.get("snake")()(x), x + torch.sin(x) ** 2)
        torch.testing.assert_close(ACTIVATIONS.get("x·sin x")()(x), x * torch.sin(x))
        torch.testing.assert_close(ACTIVATIONS.get("cos")()(x), torch.cos(x))

    def test_usable_in_layers_and_training(self):
        for name in NEW_ACTS:
            with self.subTest(activation=name):
                s = session(model="mlp", dataset="Moons", layers=f"8:{name},4", activation="tanh", n_points=120)
                self.assertTrue(math.isfinite(s.train(20)))
                self.assertEqual(s.model.spec[0], (8, name))


class Initializers(unittest.TestCase):
    def test_run_on_mlp_and_custom_nn(self):
        for name in NEW_INITS:
            for model in ("mlp", "custom nn"):
                with self.subTest(init=name, model=model):
                    s = session(model=model, init=name, n_points=100)
                    self.assertTrue(all(torch.isfinite(p).all() for p in s.model.parameters()))
                    self.assertTrue(math.isfinite(s.train(10)))

    def test_keep_expansion_keeps_fourier_init(self):
        base = session()
        for name in NEW_INITS:
            with self.subTest(init=name):
                s = session(init=name)
                self.assertTrue(torch.equal(s.model.flank.lin.weight, base.model.flank.lin.weight))
                self.assertTrue(torch.equal(s.model.flank.lin.bias, base.model.flank.lin.bias))
                self.assertFalse(torch.equal(s.model.hid.weight, base.model.hid.weight))
                self.assertEqual(float(s.model.hid.bias.detach().abs().sum()), 0.0)

    def test_fourier_freq_matters_under_keep(self):
        a = session(init="xavier uniform (keep expansion)", fourier_freq=3.0)
        b = session(init="xavier uniform (keep expansion)", fourier_freq=10.0)
        self.assertFalse(torch.equal(a.model.flank.lin.weight, b.model.flank.lin.weight))

    def test_original_entries_unchanged(self):
        """The original xavier still overwrites the Fourier layer (old runs reproduce)."""
        a = session(init="xavier uniform", fourier_freq=3.0)
        b = session(init="xavier uniform", fourier_freq=10.0)
        self.assertTrue(torch.equal(a.model.flank.lin.weight, b.model.flank.lin.weight))
        self.assertTrue(Fourier.keep_init)

    def test_orthogonal_is_orthogonal(self):
        m = nn.Sequential(nn.Linear(8, 8))
        INITIALIZERS.get("orthogonal")(m)
        w = m[0].weight.detach()
        torch.testing.assert_close(w @ w.T, torch.eye(8), atol=1e-5, rtol=0)


class Layers(unittest.TestCase):
    def test_round_trip(self):
        for s in ("8:sin,4:tanh", "16:relu", "", "3:x·sin x,2:gauss,1:linear"):
            with self.subTest(layers=s):
                spec = parse_layers(s, "tanh")
                self.assertEqual(format_layers(spec), s)
                self.assertEqual(parse_layers(format_layers(spec), "relu"), spec)
        self.assertEqual(format_layers([(8, "sin"), (4, "tanh")], "tanh"), "8:sin,4")

    def test_valid_inputs_parse_as_before(self):
        self.assertEqual(parse_layers(" 8 : sin , 4 ", "tanh"), [(8, "sin"), (4, "tanh")])
        self.assertEqual(parse_layers("8:sin\n", "tanh"), [(8, "sin")])
        self.assertEqual(parse_layers("8:,,4", "tanh"), [(8, "tanh"), (4, "tanh")])
        self.assertEqual(parse_layers("8:leaky _relu", "tanh"), [(8, "leaky_relu")])
        self.assertEqual(parse_layers("1 6", "tanh"), [(16, "tanh")])

    def test_readable_errors(self):
        for text in (":tanh", "8.5", "abc", "8;8", "8x3"):
            with self.subTest(layers=text):
                with self.assertRaisesRegex(ValueError, r"expected WIDTH\[:ACTIVATION\] items separated by commas"):
                    parse_layers(text, "tanh")
        with self.assertRaisesRegex(ValueError, "layer width 600 in '600:sin' must be 1..512"):
            parse_layers("8,600:sin", "tanh")
        with self.assertRaisesRegex(ValueError, "unknown activation 'relux'"):
            parse_layers("8:relux", "tanh")

    def test_make_expansion_positional_and_keyword_scale(self):
        torch.manual_seed(0)
        a = make_expansion("fourier (sin)", 2, 5.0)
        torch.manual_seed(0)
        b = make_expansion("fourier (sin)", 2, scale=5.0)
        self.assertTrue(torch.equal(a.lin.weight, b.lin.weight))
        torch.manual_seed(0)
        big = make_expansion("fourier (sin)", 2, 5.0, k=4000)
        self.assertAlmostEqual(float(big.lin.weight.std()), 5.0, delta=0.15)
        torch.manual_seed(0)
        dflt = make_expansion("fourier (sin)", 2)  # scale=None: the factory default (3.0)
        torch.manual_seed(0)
        self.assertTrue(torch.equal(dflt.lin.weight, Fourier(2).lin.weight))

    def test_make_expansion_kwargs_factory_and_plain_factory(self):
        got = {}

        def kw_factory(n_in, **kw):
            got.update(kw)
            return nn.Identity()
        EXPANSIONS.register("_test kw", kw_factory)
        try:
            make_expansion("_test kw", 2, 7.0, other=1)
            self.assertEqual(got, {"scale": 7.0, "other": 1})
            self.assertIsInstance(make_expansion("none", 2, 3.0), nn.Identity)  # lambda n_in: ignores scale
        finally:
            del EXPANSIONS["_test kw"]

    def test_polar_needs_two_inputs(self):
        with self.assertRaisesRegex(ValueError, "polar spiral needs the first two inputs"):
            Polar(1)
        with self.assertRaisesRegex(ValueError, "polar spiral"):
            session(features=("x",), expand="polar spiral")


class MlpSkips(unittest.TestCase):
    def test_concat_is_sized_at_build(self):
        s = session(model="mlp", skip="concat", layers="8,8", dataset="Moons")
        self.assertEqual(s.hidden_sizes, [10, 18])
        self.assertTrue(math.isfinite(s.train(10)))
        logits, hidden = s.predict(np.zeros((3, 2)), collect=True)
        self.assertEqual([h.shape[-1] for h in hidden], s.hidden_sizes)

    def test_add_fails_readably_at_build(self):
        with self.assertRaisesRegex(ValueError, "skip 'add' cannot combine width 8 with 2"):
            session(model="mlp", skip="add")

    def test_failed_skip_leaves_session_unchanged(self):
        s = session(model="mlp", dataset="Moons")
        with self.assertRaises(ValueError):
            with quiet():
                s.configure(skip="add")
        self.assertEqual(s.cfg.skip, "none")
        self.assertTrue(math.isfinite(s.train(5)))

    def test_plain_widths_unchanged(self):
        m = MLP(2, [(8, "tanh"), (4, "relu")], 3, skip="residual (same width)")
        self.assertEqual(m.hidden_sizes, [8, 4])
        self.assertEqual(MLP(2, [], 3).hidden_sizes, [])


class Datasets(unittest.TestCase):
    def test_spiral_needs_n_at_least_k(self):
        with self.assertRaisesRegex(ValueError, r"Spiral \(5 arms\) needs n_points >= 5"):
            session(n_points=3)

    def test_random_split_validates_test_frac(self):
        rng = np.random.default_rng(0)
        for bad in (-0.2, 1.0, 1.5):
            with self.assertRaisesRegex(ValueError, r"test_frac must be in \[0, 1\)"):
                random_split(100, bad, rng)
        tr, te = random_split(10, 0.999, np.random.default_rng(0))
        self.assertGreaterEqual(len(tr), 1)
        self.assertEqual(len(tr) + len(te), 10)

    def test_random_split_default_unchanged(self):
        tr, te = random_split(600, 0.2, np.random.default_rng(5))
        idx = np.random.default_rng(5).permutation(600)
        np.testing.assert_array_equal(tr, idx[120:])
        np.testing.assert_array_equal(te, idx[:120])

    def test_stratified_split(self):
        y = np.repeat([0, 1, 2], [50, 30, 20])
        tr, te = stratified_split(100, 0.2, np.random.default_rng(0), y=y)
        self.assertEqual(sorted(np.concatenate([tr, te]).tolist()), list(range(100)))
        self.assertEqual(np.bincount(y[te]).tolist(), [10, 6, 4])
        with self.assertRaises(ValueError):
            stratified_split(100, 0.2, np.random.default_rng(0))


class Schedules(unittest.TestCase):
    def _lrs(self, name, steps, extra=None):
        opt = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
        from nncore.registry import call_accepting
        sch = call_accepting(SCHEDULES.get(name), opt, cfg=Config(extra=extra or {}))
        out = []
        for _ in range(steps):
            out.append(opt.param_groups[0]["lr"])
            opt.step()
            sch.step()
        return out

    def test_cosine_reads_total_steps(self):
        lrs = self._lrs("cosine (over extra.total_steps)", 120, {"total_steps": 100})
        self.assertAlmostEqual(lrs[0], 1.0)
        self.assertAlmostEqual(lrs[50], 0.5, places=6)
        self.assertAlmostEqual(lrs[100], 0.0, places=9)
        self.assertAlmostEqual(lrs[119], 0.0, places=9)
        self.assertAlmostEqual(self._lrs("cosine (over extra.total_steps)", 1001)[1000], 0.5, places=6)  # default 2000

    def test_warmup_and_restarts(self):
        w = self._lrs("warmup 200 + constant", 300)
        self.assertAlmostEqual(w[0], 1 / 200)
        self.assertAlmostEqual(w[199], 1.0)
        self.assertAlmostEqual(w[299], 1.0)
        r = self._lrs("cosine warm restarts (T0=1000)", 1001)
        self.assertAlmostEqual(r[500], 0.5, places=6)
        self.assertAlmostEqual(r[1000], 1.0, places=6)

    def test_in_session(self):
        s = session(model="mlp", dataset="Moons", schedule="cosine (over extra.total_steps)",
                    extra={"total_steps": 50})
        s.train(50)
        self.assertAlmostEqual(s.lr, 0.0, places=9)


class TrainSteps(unittest.TestCase):
    def _setup(self):
        torch.manual_seed(0)
        model = nn.Linear(2, 2)
        return model, torch.optim.SGD(model.parameters(), lr=1.0), torch.randn(8, 2), torch.randint(0, 2, (8,))

    def test_skip_nonfinite_keeps_weights(self):
        model, opt, xb, yb = self._setup()
        before = [p.detach().clone() for p in model.parameters()]
        nan_loss = lambda logits, y: logits.sum() * float("nan")
        loss = TRAIN_STEPS.get("skip non-finite")(model, opt, nan_loss, xb, yb)
        self.assertTrue(math.isnan(loss))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, model.parameters())))
        inf_grad = lambda logits, y: (logits * torch.tensor(float("inf"))).clamp(max=1.0).sum()  # finite loss, inf grad
        TRAIN_STEPS.get("skip non-finite")(model, opt, inf_grad, xb, yb)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_skip_nonfinite_equals_standard_when_finite(self):
        a = session(model="mlp", dataset="Moons", n_points=120)
        b = session(model="mlp", dataset="Moons", n_points=120, train_step="skip non-finite")
        a.train(30)
        b.train(30)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a.model.parameters(), b.model.parameters())))

    def test_clip_reads_extra(self):
        for clip in (0.01, 0.5):
            model, opt, xb, yb = self._setup()
            before = torch.cat([p.detach().flatten().clone() for p in model.parameters()])
            TRAIN_STEPS.get("clip grad norm (extra)")(model, opt, nn.functional.cross_entropy, xb, yb,
                                                     cfg=SimpleNamespace(extra={"clip": clip}))
            moved = torch.cat([p.detach().flatten() for p in model.parameters()]) - before
            self.assertLessEqual(float(moved.norm()), clip * 1.0001)  # SGD lr=1: step = clipped grad
        s = session(model="mlp", dataset="Moons", train_step="clip grad norm (extra)", extra={"clip": 0.5})
        self.assertTrue(math.isfinite(s.train(10)))


class LossesOptimizersFeatures(unittest.TestCase):
    def test_focal(self):
        logits = torch.tensor([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]], requires_grad=True)
        y = torch.tensor([0, 1, 0])
        fl = LOSSES.get("focal (γ=2)")(logits, y)
        ce = nn.functional.cross_entropy(logits, y, reduction="none")
        torch.testing.assert_close(fl, ((1 - torch.exp(-ce)) ** 2 * ce).mean())
        self.assertLess(float(fl), float(ce.mean()))
        fl.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_new_optimizers_train(self):
        for name in ("NAdam", "RAdam", "Adamax", "SGD+nesterov"):
            with self.subTest(optimizer=name):
                s = session(model="mlp", dataset="Moons", optimizer=name, n_points=120)
                self.assertTrue(math.isfinite(s.train(20)))

    def test_new_features(self):
        pts = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.3, -0.2]])
        out = featurize(pts, ["θ (atan2)", "cos x", "x", "cos y"])
        torch.testing.assert_close(out[:, 0], torch.atan2(pts[:, 1], pts[:, 0]) / math.pi)
        torch.testing.assert_close(out[:, 1], torch.cos(3 * pts[:, 0]))
        torch.testing.assert_close(out[:, 2], pts[:, 0])  # columns follow the order given
        self.assertLessEqual(float(out[:, 0].abs().max()), 1.0)


class ExtraMetrics(unittest.TestCase):
    def setUp(self):
        # true 0 0 0 0 1 1 2 2 ; predicted 0 0 0 1 1 0 2 2
        self.y = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])
        pred = torch.tensor([0, 0, 0, 1, 1, 0, 2, 2])
        self.logits = torch.full((8, 3), -2.0)
        self.logits[torch.arange(8), pred] = 2.0

    def m(self, name):
        return EXTRA_METRICS.get(name)(self.logits, self.y)

    def test_names(self):
        self.assertEqual(list(EXTRA_METRICS), ["bal_acc", "macro_f1", "worst_class_acc", "mean_conf", "ece"])

    def test_hand_example(self):
        # recalls: 3/4, 1/2, 1 ; precisions: 3/4, 1/2, 1
        self.assertAlmostEqual(self.m("bal_acc"), (0.75 + 0.5 + 1) / 3, places=6)
        self.assertAlmostEqual(self.m("worst_class_acc"), 0.5, places=6)
        self.assertAlmostEqual(self.m("macro_f1"), (0.75 + 0.5 + 1) / 3, places=6)
        conf = float(torch.softmax(torch.tensor([2.0, -2.0, -2.0]), 0)[0])
        self.assertAlmostEqual(self.m("mean_conf"), conf, places=6)
        self.assertAlmostEqual(self.m("ece"), abs(conf - 6 / 8), places=6)  # one bin holds every point

    def test_f1_differs_from_recall(self):
        y = torch.tensor([0, 0, 0, 1])
        logits = torch.tensor([[1.0, 0], [1, 0], [0, 1], [0, 1]])  # recall 2/3, 1 ; precision 1, 1/2
        f1 = EXTRA_METRICS.get("macro_f1")(logits, y)
        self.assertAlmostEqual(f1, (2 * (2 / 3) / (5 / 3) + 2 * 0.5 / 1.5) / 2, places=6)

    def test_absent_class_and_empty(self):
        y = torch.tensor([0, 0, 1, 1])  # class 2 never appears in y
        logits = torch.tensor([[1.0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0]])
        self.assertAlmostEqual(EXTRA_METRICS.get("bal_acc")(logits, y), 0.75, places=6)
        for name in EXTRA_METRICS:
            self.assertTrue(math.isnan(EXTRA_METRICS.get(name)(torch.zeros(0, 3), torch.zeros(0, dtype=torch.long))))

    def test_in_evaluate(self):
        s = session(model="mlp", dataset="Imbalanced (90/10)", n_points=200)
        s.train(20)
        ev = s.evaluate(extra_metrics=tuple(EXTRA_METRICS))
        self.assertEqual(list(ev["train"])[:2], ["loss", "acc"])
        for name in EXTRA_METRICS:
            self.assertTrue(0.0 <= ev["test"][name] <= 1.0)
        self.assertEqual(set(s.evaluate()["train"]), {"loss", "acc"})  # default output unchanged


class SamplersDoc(unittest.TestCase):
    def test_epoch_shuffle_drops_tail(self):
        torch.manual_seed(0)
        nb = SAMPLERS.get("epoch shuffle")(10, 4)
        first = [nb() for _ in range(2)]
        self.assertEqual(len(torch.cat(first).unique()), 8)  # 10 % 4 = 2 points skipped this epoch
        self.assertIn("drop_last", samplers_mod.__doc__)


if __name__ == "__main__":
    unittest.main()
