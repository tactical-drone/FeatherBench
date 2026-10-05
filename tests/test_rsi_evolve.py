"""rsi evolve: determinism across workers, resume equivalence, elitism, constraints, escalation,
holdout, failing genomes, stop + resume, export. Part of nn-playground (AGPL-3.0; see COMMERCIAL.md)."""
import json
import tempfile
import unittest
from pathlib import Path

from tests._util import quiet  # noqa: F401  (loads my_parts)
import rsi
from nncore import configio
from rsi.errors import RsiError
from rsi.evolve import EvolveSettings, escalation_seeds, escalation_targets, fitness, rank_key
from rsi.rundir import read_jsonl

BASE = {"model": "mlp", "dataset": "Moons", "n_points": 120}
FAST = dict(pop=4, steps=30, seeds=[0], max_seeds=2, holdout_seeds=[1000], holdout_top=2, workers=0)


def strip(lines):
    out = []
    for x in lines:
        x = {k: v for k, v in x.items() if k not in ("t", "seconds")}
        if isinstance(x.get("cell"), dict):
            x["cell"] = {k: v for k, v in x["cell"].items() if k not in ("seconds", "cached")}
        out.append(x)
    return out


def board(lb):
    return [(e["genome_id"], e["score"], (e["holdout"] or {}).get("score")) for e in lb["entries"]]


class Evolve(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def evolve(self, store="s", **kw):
        kw = {"overrides": BASE, "store": str(self.root / store), **FAST, **kw}
        with quiet():
            return rsi.evolve(**kw)

    def test_workers_resume_elitism_holdout(self):
        a = self.evolve("a", name="r", generations=3)
        b = self.evolve("b", name="r", generations=3, workers=2)
        da, db = self.root / "a" / "r", self.root / "b" / "r"
        self.assertEqual(strip(read_jsonl(da / "evals.jsonl")), strip(read_jsonl(db / "evals.jsonl")))
        self.assertEqual(board(a), board(b))
        # 3 generations straight == 1 generation + resume + 2
        self.evolve("c", name="r", generations=1)
        c = self.evolve("c", resume="r", generations=3)
        self.assertEqual(board(a), board(c))
        gen = lambda d: [{k: v for k, v in x.items() if k != "seconds"} for x in read_jsonl(d / "generations.jsonl")]
        self.assertEqual(gen(da), gen(self.root / "c" / "r"))
        sa, sc = (json.loads((d / "state.json").read_text()) for d in (da, self.root / "c" / "r"))
        self.assertEqual((sa["rng"], sa["hof"]), (sc["rng"], sc["hof"]))
        # a budget hit mid-generation is not committed: resuming takes the straight run's path
        self.assertEqual(self.evolve("d", name="r", generations=3, max_evals=8)["state"], "budget")
        self.assertEqual(json.loads((self.root / "d" / "r" / "state.json").read_text())["generation"], 0)
        self.evolve("d", resume="r", generations=3, max_evals=10 ** 6)
        sd = json.loads((self.root / "d" / "r" / "state.json").read_text())
        self.assertEqual((sa["rng"], sa["hof"]), (sd["rng"], sd["hof"]))
        key = lambda d: [(x["generation"], x["best"], x["best_genome_id"]) for x in read_jsonl(d / "generations.jsonl")]
        self.assertEqual(key(da), key(self.root / "d" / "r"))
        # status 'best' == leaderboard rank 1 == best.settings.json
        meta = json.loads((da / "best.settings.json").read_text())["meta"]
        self.assertEqual({rsi.status(da)["best"]["genome_id"], meta["genome_id"]}, {a["entries"][0]["genome_id"]})
        # elitism: the best never decreases; escalation reached max_seeds; holdout seeds are disjoint
        best = [x["best"] for x in gen(da)]
        self.assertEqual(best, sorted(best))
        ev = read_jsonl(da / "evals.jsonl")
        search = {x["seed"] for x in ev if "cell" in x and x["phase"] in ("search", "escalate")}
        hold = {x["seed"] for x in ev if x.get("phase") == "holdout"}
        self.assertEqual((hold, search & hold), ({1000}, set()))
        self.assertIn(1, {x["seed"] for x in ev if x.get("phase") == "escalate"})
        self.assertIsNotNone(a["entries"][0]["holdout"])
        self.assertEqual(a["state"], "done")
        # the run dir works with status / leaderboard / export, and export reproduces meta.expect
        self.assertEqual(rsi.status(da)["state"], "done")
        with quiet():
            out = rsi.export(da, rank=1, out=self.root / "w.json", store=str(self.root / "a"))
            rec = rsi.run(config=str(self.root / "w.json"), steps=30, store=str(self.root / "a"))
        exp = out["meta"]["expect"]
        self.assertEqual((rec["cells"][0]["train"]["acc"], rec["cells"][0]["test"]["acc"]),
                         (exp["train_acc"], exp["test_acc"]))
        with self.assertRaises(RsiError) as cm:
            self.evolve("c", resume="r", pop=6)
        self.assertEqual((cm.exception.code, list(cm.exception.details["diff"])), ("E_RESUME_MISMATCH", ["pop"]))

    def test_constraints_and_failures(self):
        lb = self.evolve(name="mp", generations=1, max_params=1, holdout_top=0)
        ev = read_jsonl(self.root / "s" / "mp" / "evals.jsonl")
        trained = {x["genome_id"] for x in ev if "cell" in x}
        self.assertTrue(any(x.get("status") == "rejected" for x in ev))
        self.assertEqual(trained, {lb["baseline"]["genome_id"]})  # rejected genomes never train
        # a base that fails to build (skip 'add' across widths) scores null, ranks last, the run goes on
        lb = self.evolve(name="bad", generations=1, overrides={**BASE, "skip": "add", "layers": "8,4"}, holdout_top=0)
        self.assertEqual(lb["state"], "done")
        self.assertIsNone(lb["baseline"]["score"])
        st = json.loads((self.root / "s" / "bad" / "state.json").read_text())
        self.assertEqual(st["hof"][-1], lb["baseline"]["genome_id"])
        with self.assertRaises(RsiError) as cm:
            self.evolve(name="x", holdout_seeds=[0])
        self.assertEqual(cm.exception.code, "E_USAGE")

    def test_stop_then_resume(self):
        d = self.root / "s" / "st"

        def stop_after_first(event, **f):
            if event == "generation":
                (d / "STOP").write_text("graceful\n")
        lb = self.evolve(name="st", generations=3, on_event=stop_after_first, holdout_top=0)
        self.assertEqual((lb["state"], lb["stats"]["generations_done"]), ("stopped", 1))
        lb = self.evolve(resume="st", generations=2, holdout_top=0)
        self.assertEqual((lb["state"], lb["stats"]["generations_done"]), ("done", 2))

    def test_scoring_helpers(self):
        s = EvolveSettings()
        ok = lambda acc, p=100: {"status": "ok", "test": {"acc": acc}, "train": {"acc": acc},
                                 "model": {"n_params": p}, "seconds": 1.0, "steps_done": 100}
        cells = {("D", 0): ok(0.9), ("D", 1): ok(0.7)}
        sc, fit = fitness(cells, ["D"], [0, 1], s, lambda d, sd: 2)
        self.assertAlmostEqual(fit["acc_agg"], 0.8)
        self.assertAlmostEqual(sc, 0.8 - 0.005 * __import__("math").log2(100 / 64))
        err = {("D", 0): {"status": "error"}, ("D", 1): {"status": "error"}}
        sc2, fit2 = fitness(err, ["D"], [0, 1], s, lambda d, sd: 4)
        self.assertIsNone(sc2)
        self.assertEqual(fit2["acc_agg"], 0.25)  # chance
        self.assertLess(rank_key("a", sc, fit), rank_key("b", sc2, fit2))
        self.assertEqual(escalation_seeds([0, 1, 2], 5), [0, 1, 2, 3, 4])
        scored = {"a": 0.9, "b": 0.85, "c": 0.7, "d": None}
        n = {"a": 1, "b": 1, "c": 1, "d": 1}
        self.assertEqual(escalation_targets(scored, list(scored), 0.25, 0.01, n, 3, 4), ["a"])
        self.assertEqual(escalation_targets(scored, list(scored), 0.5, 0.0, n, 3, 4), ["a", "b"])
        self.assertEqual(escalation_targets(scored, list(scored), 0.5, 0.0, {**n, "a": 3}, 3, 4), ["b"])
        self.assertEqual(EvolveSettings.from_dict(s.to_dict()), s)


if __name__ == "__main__":
    unittest.main()
