"""rsi bench: frozen definitions, scoring and rank keys, a submission, ranking, the bench:<id>
evolve objective. Part of nn-playground (AGPL-3.0; see COMMERCIAL.md)."""
import json
import tempfile
import unittest
from pathlib import Path

from tests._util import quiet  # noqa: F401  (loads my_parts)
import rsi
from nncore.datasets import suite_datasets
from rsi.bench import BENCHMARKS, Benchmark, canonical_id, get_benchmark, rank_key, scalar, summarize_rows
from rsi.cli import invoke
from rsi.errors import RsiError

TEST = Benchmark("_test-v1", 1, ("Moons", "Circles"), seeds=(0,), steps=30, n_points=120, fresh_points=200,
                 threshold=0.6)


class BenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        BENCHMARKS["_test-v1"] = TEST
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        BENCHMARKS.pop("_test-v1", None)
        cls.tmp.cleanup()

    def test_definitions(self):
        self.assertEqual(list(get_benchmark("general-v1").datasets), suite_datasets("general"))
        self.assertEqual(list(get_benchmark("classic-v1").datasets), suite_datasets("classic"))
        q = get_benchmark("quick-v1")
        self.assertEqual((len(q.datasets), q.seeds, q.steps), (4, (0,), 1000))
        self.assertNotEqual(get_benchmark("general-v1").hash, get_benchmark("classic-v1").hash)
        code, d = invoke(["bench", "list"])
        self.assertEqual((code, d["schema"]), (0, "rsi/bench-list@1"))
        self.assertIn("featherbench-general-v1", [b["id"] for b in d["result"]["benchmarks"]])
        with self.assertRaises(RsiError) as cm:
            rsi.bench(benchmark="genral-v1")
        self.assertIn("general-v1", cm.exception.did_you_mean)

    def test_featherbench_names_and_aliases(self):
        # the hashes the v1 benchmarks were published with: renaming must not change them
        published = {"general-v1": "6e9b8a2dc8a4", "classic-v1": "206e382c2dd5", "quick-v1": "33d01d26d70f"}
        for old, h in published.items():
            new = "featherbench-" + old
            self.assertIs(get_benchmark(old), get_benchmark(new))
            self.assertEqual((get_benchmark(new).id, get_benchmark(new).hash), (new, h))
            self.assertEqual(canonical_id(old), new)
        code, d = invoke(["featherbench", "list"])
        self.assertEqual((code, d["schema"]), (0, "rsi/bench-list@1"))
        r = d["result"]
        self.assertEqual((r["name"], r["default"]), ("FeatherBench", "featherbench-general-v1"))
        self.assertIn("FeatherBench", r["objective"])
        self.assertEqual(r["aliases"]["quick-v1"], "featherbench-quick-v1")
        quick = next(b for b in r["benchmarks"] if b["id"] == "featherbench-quick-v1")
        self.assertEqual((quick["aliases"], quick["hash"]), (["quick-v1"], published["quick-v1"]))
        code, d = invoke(["describe", "--section", "objectives"])
        self.assertIn("bench:featherbench-general-v1", d["result"]["objectives"]["search"])
        self.assertIn("FeatherBench", rsi.describe(section="commands")["commands"]["bench"])

    def test_rank_keys(self):
        rows = lambda accs, p: [{"dataset": str(i), "fresh_acc_mean": a, "n_params": p, "solved": a >= 0.9}  # noqa
                                for i, a in enumerate(accs)]
        b = Benchmark("x", 1, ())
        small_solver = summarize_rows(rows([0.91, 0.95], 50), b)
        big_solver = summarize_rows(rows([0.99, 0.99], 500), b)
        near = summarize_rows(rows([0.99, 0.89], 10), b)
        weak = summarize_rows(rows([0.5, 0.5], 5), b)
        order = sorted([weak, near, big_solver, small_solver], key=lambda s: s["rank_key"])
        self.assertEqual(order, [small_solver, big_solver, near, weak])
        self.assertEqual(sorted([weak, near, big_solver, small_solver], key=scalar, reverse=True), order)
        self.assertEqual(rank_key(True, 2, 50, 0.93), [0, 50, -0.93, 0])
        self.assertEqual((small_solver["params_max"], small_solver["solved_all"], near["n_solved"]), (50, True, 1))

    def test_submission_rank_and_objective(self):
        a, b = self.root / "a.json", self.root / "b.json"
        with quiet():
            sa = rsi.bench(benchmark="_test-v1", overrides={"model": "mlp", "layers": "8,8", "dataset": "Moons"},
                           workers=0, submit=a, store=str(self.root))
            sb = rsi.bench(benchmark="_test-v1", overrides={"model": "mlp", "layers": "2"}, workers=0, submit=b,
                           store=str(self.root))
        for s in (sa, sb):
            self.assertEqual((s["format"], s["n_datasets"], len(s["cells"])), ("rsi/bench", 2, 2))
            self.assertEqual(s["benchmark"]["hash"], TEST.hash)
            self.assertIn("code_fp", s["environment"])
            self.assertEqual(s["params_max"], max(r["n_params"] for r in s["per_dataset"]))
        self.assertEqual(sa["config"]["n_points"], 120)  # the benchmark's task, not the recipe's
        self.assertEqual((sb["config"]["dataset"], sb["config_diff"]), ("Spiral (5 arms)", {"model": "mlp", "layers": "2"}))
        self.assertEqual((sa["config"]["dataset"], sa["config_diff"].get("dataset")), ("Moons", "Moons"))
        self.assertEqual(json.loads(a.read_text())["config_key"], sa["config_key"])
        code, d = invoke(["bench", "rank", str(a), str(b)])
        self.assertEqual((code, d["schema"], d["result"]["n_submissions"]), (0, "rsi/bench-leaderboard@1", 2))
        keys = [e["rank_key"] for e in d["result"]["entries"]]
        self.assertEqual(keys, sorted(keys))
        self.assertTrue(all(e["consistent"] for e in d["result"]["entries"]))
        other = json.loads(a.read_text())
        other["benchmark"]["hash"] = "000000000000"
        (self.root / "c.json").write_text(json.dumps(other))
        code, d = invoke(["bench", "rank", str(a), str(self.root / "c.json")])
        self.assertEqual((code, d["error"]["code"]), (3, "E_BAD_SETTINGS"))
        with quiet():
            lb = rsi.evolve(overrides={"model": "mlp"}, objective="bench:_test-v1", pop=3, generations=1,
                            holdout_top=0, workers=0, store=str(self.root), name="be")
        e = lb["entries"][0]
        self.assertEqual(lb["objective"], "bench:_test-v1")
        self.assertIn("solved_all", e["fitness"])
        self.assertEqual(e["config"]["n_points"], 120)
        self.assertEqual(e["config"]["dataset"], "Spiral (5 arms)")  # the recipe's own, not the first benchmark's
        self.assertTrue(all("dataset" not in x["config_diff"] for x in lb["entries"]))


if __name__ == "__main__":
    unittest.main()
