"""Skip connections in custom nn (Wide): one per stage, like the per-stage activations.
Part of nn-playground (AGPL-3.0; see COMMERCIAL.md).

Run:  python -m unittest tests.test_wide_skip -v
"""
import json
import math
import os
import tempfile
import unittest

import torch

from tests._util import quiet, same_params
from nncore import SKIPS, Config, Session, configio, schema

STAGES = ("skip", "skip_decide", "skip_relate", "skip_prepare")
ALL_NONE = dict(skip="none", skip_decide="none", skip_relate="none", skip_prepare="none")
V0 = dict(skip_decide="none")  # "fourier-gauss-v0": the default recipe before the decide residual


def session(**kw):
    with quiet():
        return Session(Config(**kw))


class WideSkip(unittest.TestCase):
    def test_none_and_same_are_the_old_network(self):
        """fourier-gauss-v0 ('none' + 'same' stages) == every stage 'none', bit for bit, and still
        the historical default fingerprint (init sum, first losses)."""
        a, b = session(**V0), session(**ALL_NONE)
        self.assertTrue(same_params(a.model, b.model))
        self.assertEqual(a.n_params, 341)
        self.assertEqual(a.hidden_sizes, [16, 8, 5, 5, 5, 2, 5, 5])
        self.assertAlmostEqual(sum(p.double().sum().item() for p in a.model.parameters()), 17.098907, places=5)
        self.assertAlmostEqual(a.train(1), 1.873401, places=5)
        self.assertAlmostEqual(b.train(1), 1.873401, places=5)

    def test_every_skip_at_every_stage(self):
        """Builds and trains finitely, or (explicit misfit only) a readable build-time error."""
        for classes in (5, 3, 2):
            for field in STAGES:
                for name in SKIPS:
                    with self.subTest(classes=classes, field=field, skip=name):
                        try:
                            s = session(classes=classes, **{field: name})
                        except ValueError as e:
                            # only an explicit 'add' on a stage that changes width (classes vs n_in=2)
                            self.assertEqual(name, "add")
                            self.assertIn(field, ("skip_relate", "skip_prepare"))
                            self.assertNotEqual(classes, 2)
                            self.assertIn("needs equal widths", str(e))
                            continue
                        loss = s.train(50)
                        self.assertTrue(math.isfinite(loss), loss)

    def test_default_is_the_decide_residual(self):
        """The default adds a residual around decide: same 341 params and weights as v0 (a skip
        has no parameters), different training from step 1."""
        d, v0 = session(), session(**V0)
        self.assertEqual(d.cfg.skip_decide, "residual (same width)")
        self.assertIs(d.model.skip_decide, SKIPS.get("residual (same width)"))
        self.assertTrue(same_params(d.model, v0.model))
        self.assertEqual((d.n_params, d.hidden_sizes), (341, v0.hidden_sizes))
        self.assertNotAlmostEqual(d.train(1), v0.train(1), places=6)

    def test_skip_changes_the_network(self):
        base = session(**ALL_NONE)
        for name in ("residual (same width)", "half residual", "add", "concat"):
            with self.subTest(skip=name):
                s = session(skip=name)
                x = torch.rand(7, 2) * 2 - 1
                self.assertFalse(torch.equal(s.predict(x), base.predict(x)))

    def test_same_falls_back_where_it_cannot_fit(self):
        """Main skip 'add' can't span relate (classes -> 2) or prepare (2 -> classes): those
        'same' stages are plain, decide and perc get the residual."""
        s = session(skip="add", skip_decide="same")
        m = s.model
        self.assertIs(m.skip_relate, SKIPS.get("none"))
        self.assertIs(m.skip_prepare, SKIPS.get("none"))
        self.assertIs(m.skip_decide, SKIPS.get("add"))
        self.assertIs(m.skip, SKIPS.get("add"))
        self.assertEqual(s.n_params, 341)

    def test_hidden_sizes_match_what_forward_collects(self):
        for kw in ({}, {"skip": "concat"}, {"skip_decide": "concat"}, {"skip_relate": "concat"},
                   {"skip_prepare": "concat", "classes": 3}, {"skip": "concat", "skip_relate": "none"}):
            with self.subTest(**kw):
                s = session(**kw)
                _, hidden = s.predict(torch.zeros(4, 2), collect=True)
                self.assertEqual([h.shape[-1] for h in hidden], s.hidden_sizes)

    def test_concat_widens_the_next_layer(self):
        s = session(skip="concat", skip_decide="same")  # 'same' stages concat too
        self.assertEqual(s.hidden_sizes, [16, 8, 5, 10, 15, 17, 22, 5])
        self.assertGreater(s.n_params, 341)

    def test_skip_fields_are_read_not_inert(self):
        read = schema.fields_read("custom nn")
        self.assertTrue(set(STAGES) <= read, read)
        self.assertFalse(set(STAGES) & schema.inert_fields(Config(skip="concat")))
        self.assertTrue({"skip_decide", "skip_relate", "skip_prepare"} <= schema.inert_fields(Config(model="mlp")))

    def test_settings_round_trip_and_old_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "s.json")
            cfg = Config(skip="half residual", skip_decide="concat", skip_relate="none", skip_prepare="same")
            configio.save_settings(p, cfg)
            back = configio.load_settings(p).config
            self.assertEqual([getattr(back, f) for f in STAGES], [getattr(cfg, f) for f in STAGES])
            old = {k: v for k, v in Config().to_dict().items() if not k.startswith("skip_")}
            with open(p, "w", encoding="utf-8") as f:
                json.dump(old, f)
            st = configio.load_settings(p, strict=False)  # missing fields take today's defaults
            self.assertEqual([getattr(st.config, f) for f in STAGES[1:]],
                             [getattr(Config(), f) for f in STAGES[1:]])
        with self.assertRaises(configio.ConfigError):
            configio.config_from_dict({"skip_decide": "nope"})


if __name__ == "__main__":
    unittest.main()
