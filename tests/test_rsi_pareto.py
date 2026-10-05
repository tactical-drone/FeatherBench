"""rsi pareto: non-dominated sorting, crowding, NSGA-II selection, the leaderboard front and an
`evolve --objective pareto` run. Part of nn-playground (AGPL-3.0; see COMMERCIAL.md)."""
import math
import tempfile
import unittest

from tests._util import quiet  # noqa: F401  (loads my_parts)
import rsi
from rsi import pareto as par


class ParetoTests(unittest.TestCase):
    def test_sort_crowding_select(self):
        pts = [(0.9, -8), (0.8, -6), (0.95, -10), (0.7, -9), (0.8, -7), (0.6, -5)]
        fronts = par.non_dominated_sort(pts)
        self.assertEqual(fronts[0], [0, 1, 2, 5])
        self.assertEqual(fronts[1:], [[4], [3]])  # (0.7, -9) is dominated by (0.8, -7)
        cd = par.crowding_distance(pts, fronts[0])
        self.assertEqual((cd[2], cd[5]), (math.inf, math.inf))  # the ends of each objective
        self.assertTrue(0 < cd[0] < math.inf)
        ids = ["a", "b", "c", "d", "e", "f"]
        self.assertEqual(sorted(par.nsga2_select(pts, ids, 4)), [0, 1, 2, 5])
        self.assertEqual(len(par.nsga2_select(pts, ids, 5)), 5)
        self.assertTrue(par.dominates((1, 1), (1, 0)) and not par.dominates((1, 0), (0, 1)))
        self.assertEqual(par.objectives(None, 64), (-math.inf, -6.0))
        rows = [{"genome_id": "a", "acc_agg": 0.9, "n_params": 300}, {"genome_id": "b", "acc_agg": 0.8, "n_params": 50},
                {"genome_id": "c", "acc_agg": 0.7, "n_params": 400}, {"genome_id": "d", "acc_agg": None, "n_params": 1}]
        self.assertEqual([r["genome_id"] for r in par.front(rows)], ["b", "a"])

    def test_pareto_evolve(self):
        with tempfile.TemporaryDirectory() as tmp, quiet():
            lb = rsi.evolve(overrides={"model": "mlp", "dataset": "Moons", "n_points": 120}, objective="pareto", pop=4,
                            generations=2, steps=20, seeds=[0], max_seeds=1, holdout_top=0, workers=0, store=tmp,
                            name="p")
        self.assertEqual(lb["state"], "done")
        self.assertEqual(lb["objective"], "pareto(acc_agg, -log2 n_params)")
        front = lb["pareto"]
        self.assertTrue(front)
        for a in front:  # nothing on the front dominates another front member
            for b in front:
                self.assertFalse(a is not b and a["acc_agg"] >= b["acc_agg"] and a["n_params"] <= b["n_params"] and
                                 (a["acc_agg"], a["n_params"]) != (b["acc_agg"], b["n_params"]))


if __name__ == "__main__":
    unittest.main()
