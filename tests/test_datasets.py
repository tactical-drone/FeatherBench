"""Contract tests for every registered dataset and suite.

Run:  PYTHONPATH=. python -m unittest discover -s tests -v
"""
import re
import unittest

import numpy as np

import my_parts  # noqa: F401  (registers the user's parts, as the app does)
from nncore import DATASETS, SUITES, Config, Session, suite_datasets, suite_signature
from nncore import configio
from nncore.datasets import FAMILIES, INEXACT_N, family_datasets, parse_family_name

SIZES = (40, 601)
NOISES = (0.0, 0.3)
SEEDS = (0, 1)


class DatasetContract(unittest.TestCase):
    def _make(self, name, n, noise, seed):
        return DATASETS.get(name)(n, noise, np.random.default_rng(seed))

    def test_every_dataset(self):
        for name in DATASETS:
            for n in SIZES:
                for noise in NOISES:
                    for seed in SEEDS:
                        with self.subTest(dataset=name, n=n, noise=noise, seed=seed):
                            X, y = self._make(name, n, noise, seed)
                            self.assertIsInstance(X, np.ndarray)
                            self.assertEqual(X.dtype, np.float32)
                            self.assertEqual(y.dtype, np.int64)
                            self.assertEqual(X.ndim, 2)
                            self.assertEqual(X.shape[1], 2)
                            self.assertEqual(y.shape, (len(X),))
                            if name not in INEXACT_N:
                                self.assertEqual(len(X), n)
                            else:
                                self.assertGreater(len(X), 0)
                                self.assertLessEqual(len(X), n)
                            self.assertTrue(np.isfinite(X).all())
                            k = int(y.max()) + 1
                            self.assertGreaterEqual(int(y.min()), 0)
                            self.assertGreaterEqual(k, 2)
                            self.assertLessEqual(k, 8)
                            self.assertEqual(len(np.unique(y)), k, "every class 0..K-1 must appear")
                            if noise == 0.0:
                                self.assertLessEqual(float(np.abs(X).max()), 1.6)

    def test_deterministic(self):
        for name in DATASETS:
            with self.subTest(dataset=name):
                X1, y1 = self._make(name, 601, 0.2, 7)
                X2, y2 = self._make(name, 601, 0.2, 7)
                np.testing.assert_array_equal(X1, X2)
                np.testing.assert_array_equal(y1, y2)

    def test_noise_changes_points(self):
        for name in DATASETS:
            with self.subTest(dataset=name):
                X0, y0 = self._make(name, 601, 0.0, 3)
                X1, y1 = self._make(name, 601, 0.4, 3)
                self.assertTrue(not np.array_equal(X0, X1) or not np.array_equal(y0, y1),
                                "noise must perturb the data")

    def test_tiny_n(self):
        """n below the class count must not raise (except the classic spirals, which say
        'needs n_points >= k'); zoo datasets still return exactly n."""
        for name in DATASETS:
            for n in (1, 2, 5):
                with self.subTest(dataset=name, n=n):
                    if name.startswith("Spiral (") and name in INEXACT_N and n < int(name[8]):
                        with self.assertRaisesRegex(ValueError, rf"{re.escape(name)} needs n_points >= {name[8]}"):
                            self._make(name, n, 0.1, 0)
                        continue
                    X, y = self._make(name, n, 0.1, 0)
                    self.assertEqual(X.shape[1:], (2,))
                    self.assertEqual(y.shape, (len(X),))
                    if name not in INEXACT_N:
                        self.assertEqual(len(X), n)
                        self.assertGreaterEqual(int(y.min()), 0)
                        self.assertTrue(np.isfinite(X).all())

    def test_zoo_every_class_from_n_equals_k(self):
        """Zoo promise: every class appears as soon as n >= number of classes (labels come from
        fixed per-class quotas, so a few seeds suffice)."""
        for name in suite_datasets("zoo"):
            k = int(self._make(name, 600, 0.05, 0)[1].max()) + 1
            for n in range(k, k + 12):
                for seed in range(3):
                    for noise in (0.0, 0.5):
                        with self.subTest(dataset=name, n=n, seed=seed, noise=noise):
                            y = self._make(name, n, noise, seed)[1]
                            self.assertEqual(sorted(set(y.tolist())), list(range(k)))

    def test_moons4_moons_do_not_touch(self):
        X, y = self._make("Moons (4)", 4000, 0.0, 0)
        for a in range(4):
            for b in range(a + 1, 4):
                d = np.sqrt(((X[y == a][:, None] - X[y == b][None]) ** 2).sum(-1).min())
                self.assertGreater(d, 0.15, f"moons {a} and {b} touch")

    def test_inexact_n_documented_only_for_known_names(self):
        for name in INEXACT_N:
            self.assertIn(name, DATASETS)


    def test_fresh_sample_keeps_the_layout(self):
        """A fresh sample (seed + offset, as Session.evaluate_fresh draws it) is labelled
        by the same layout as the training data: 1-NN label agreement stays high."""
        from nncore.datasets import sample
        names = [d for d in DATASETS.names() if d not in ("XOR gate (4 points)", "Moons (label noise)")]
        names += [n for f in FAMILIES for n in family_datasets(f, FAMILIES.get(f).sizes[:3])]
        for name in names:
            X, y = DATASETS.get(name)(400, 0.0, np.random.default_rng(0))
            F, fy = sample(name, 400, 0.0, np.random.default_rng(10_000), layout_rng=np.random.default_rng(0))
            nn = ((F[:, None, :] - X[None]) ** 2).sum(-1).argmin(1)
            self.assertGreater(float((y[nn] == fy).mean()), 0.7, name)
        # layout_rng=None is the plain call (training data unchanged)
        a = DATASETS.get("Voronoi (6)")(50, 0.05, np.random.default_rng(3))
        b = sample("Voronoi (6)", 50, 0.05, np.random.default_rng(3))
        self.assertTrue(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))

    def test_evaluate_fresh_matches_test_on_random_layouts(self):
        s = Session(Config(model="mlp", dataset="Voronoi (6)", layers="16,16", lr=0.01))
        s.train(400)
        test, fresh = s.evaluate()["test"]["acc"], s.evaluate_fresh(2000)["acc"]
        self.assertLess(abs(test - fresh), 0.1)


class Suites(unittest.TestCase):
    def test_required_suites_exist(self):
        for s in ("classic", "zoo", "general", "stretch", "sanity", "all"):
            self.assertIn(s, SUITES)

    def test_suites_name_registered_datasets(self):
        for s in SUITES:
            with self.subTest(suite=s):
                names = suite_datasets(s)
                self.assertTrue(names)
                self.assertEqual(len(names), len(set(names)), "no duplicates")
                for d in names:
                    self.assertIn(d, DATASETS)

    def test_all_is_lazy_and_skips_gate(self):
        self.assertNotIn("XOR gate (4 points)", suite_datasets("all"))
        DATASETS.register("_tmp test dataset", DATASETS.get("Linear"))
        try:
            self.assertIn("_tmp test dataset", suite_datasets("all"))
        finally:
            del DATASETS["_tmp test dataset"]

    def test_general_has_no_stretch_or_sanity_entries(self):
        general = set(suite_datasets("general"))
        self.assertFalse(general & set(suite_datasets("stretch")))
        self.assertFalse(general & set(suite_datasets("sanity")))

    def test_zoo_covers_every_new_dataset(self):
        from nncore import datasets as mod
        builtin = [d for d, fn in DATASETS.items() if getattr(fn, "__module__", "") == mod.__name__
                   or getattr(fn, "__qualname__", "").startswith("spiral.")]
        new = set(builtin) - set(suite_datasets("classic")) - {"XOR gate (4 points)"}
        self.assertEqual(new, set(suite_datasets("zoo")))

    def test_signature(self):
        a = suite_signature("general")
        self.assertTrue(a.startswith("general#"))
        self.assertEqual(a, suite_signature("general"))
        self.assertNotEqual(a, suite_signature("zoo"))

    def test_unknown_suite(self):
        with self.assertRaises(ValueError):
            suite_datasets("no such suite")


class Families(unittest.TestCase):
    def test_required_families(self):
        for f in ("spiral", "checkerboard", "rings", "stripes", "blobs"):
            fam = FAMILIES.get(f)
            self.assertTrue(fam.sizes and fam.size_label and fam.symbol and fam.description)
            self.assertTrue(callable(fam.make(fam.sizes[0])))

    def test_every_member_keeps_the_contract(self):
        for f, fam in FAMILIES.items():
            for k in fam.sizes:
                for n, seed in ((600, 0), (37, 1)):
                    with self.subTest(family=f, size=k, n=n, seed=seed):
                        fn = DATASETS.get(fam.dataset_name(k))
                        X, y = fn(n, 0.05, np.random.default_rng(seed))
                        self.assertEqual((X.dtype, y.dtype), (np.float32, np.int64))
                        self.assertEqual(X.shape, (len(y), 2))
                        self.assertTrue(np.isfinite(X).all())
                        self.assertEqual(len(X), n) if fam.exact_n else self.assertLessEqual(len(X), n)
                        self.assertEqual(sorted(set(y.tolist())), list(range(fam.n_classes(k))))
                        X2, y2 = fn(n, 0.05, np.random.default_rng(seed))
                        np.testing.assert_array_equal(X, X2)
                        np.testing.assert_array_equal(y, y2)

    def test_size_changes_the_problem(self):
        for f, fam in FAMILIES.items():
            with self.subTest(family=f):
                a = DATASETS.get(fam.dataset_name(fam.sizes[0]))(300, 0.0, np.random.default_rng(0))
                b = DATASETS.get(fam.dataset_name(fam.sizes[-1]))(300, 0.0, np.random.default_rng(0))
                self.assertFalse(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))

    def test_spiral_family_reuses_the_classic(self):
        for k in (2, 3, 4, 5):
            X1, y1 = DATASETS.get(f"spiral[{k}]")(600, 0.05, np.random.default_rng(3))
            X2, y2 = DATASETS.get(f"Spiral ({k} arms)")(600, 0.05, np.random.default_rng(3))
            np.testing.assert_array_equal(X1, X2)
            np.testing.assert_array_equal(y1, y2)

    def test_resolver_caches_without_listing(self):
        before = list(DATASETS)
        fn = DATASETS.get("spiral[7]")
        self.assertIs(DATASETS.get("spiral[7]"), fn)
        self.assertEqual(list(DATASETS), before, "resolved names must not join the dropdown list")
        self.assertNotIn("spiral[7]", suite_datasets("all"))
        self.assertTrue(DATASETS.has("checkerboard[5]"))
        self.assertEqual((fn.family, fn.size), ("spiral", 7))

    def test_bad_names_do_not_resolve(self):
        for name in ("spiral[07]", "spiral[ 7]", "Spiral[7]", "nope[3]", "spiral", "spiral[]", "spiral[-2]",
                     "spiral[7"):
            with self.subTest(name=name):
                self.assertIsNone(parse_family_name(name))
                self.assertFalse(DATASETS.has(name))
        for name in ("spiral[1]", "spiral[33]"):  # well-formed but out of range
            self.assertIsNotNone(parse_family_name(name))
            self.assertFalse(DATASETS.has(name))
        self.assertEqual(parse_family_name("rings[12]"), ("rings", 12))
        with self.assertRaises(ValueError):
            FAMILIES.get("spiral").make(1)

    def test_config_accepts_family_names(self):
        self.assertEqual(configio.validate_config(Config(dataset="stripes[6]")), [])
        bad = configio.validate_config(Config(dataset="stripes[99]"))
        self.assertEqual([i["code"] for i in bad], ["E_OUT_OF_RANGE"])
        self.assertIn("2..48", bad[0]["message"])
        bad = configio.validate_config(Config(dataset="Stripes[6]"))  # not canonical: named, not resolved
        self.assertEqual([(i["code"], i["did_you_mean"]) for i in bad], [("E_UNKNOWN_PART", ["stripes[6]"])])
        self.assertEqual(configio.resolve_name("dataset", "Spiral [7]")[0], "spiral[7]")  # console spelling
        with self.assertRaises(configio.ConfigError) as cm:
            configio.resolve_name("dataset", "spirall[3]")
        self.assertEqual(cm.exception.did_you_mean, ["spiral[3]"])
        cfg, _ = configio.config_from_dict({"dataset": "blobs[6]"})
        s = Session(cfg)
        self.assertEqual(s.n_classes, 6)
        s.train(5)

    def test_family_datasets(self):
        self.assertEqual(family_datasets("spiral", [2, 3]), ["spiral[2]", "spiral[3]"])
        self.assertEqual(len(family_datasets("stripes")), len(FAMILIES.get("stripes").sizes))


class Fingerprint(unittest.TestCase):
    def test_default_data_unchanged(self):
        s = Session(Config())
        self.assertLess(abs(float(s.X_all.sum()) - 0.786433), 1e-6)


if __name__ == "__main__":
    unittest.main()
