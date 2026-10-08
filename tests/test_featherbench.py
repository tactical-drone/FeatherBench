"""FeatherBench submissions: data-only checks, CI verify, the leaderboard, and the
workflows' security properties. Part of nn-playground (AGPL-3.0; see COMMERCIAL.md).

Run:  python -m unittest tests.test_featherbench -v
"""
import json
import re
import tempfile
import unittest
from pathlib import Path

from tests._util import ROOT, quiet
from nncore import Config, configio
from rsi.errors import RsiError
from rsi.featherbench import (check_attempt, check_submission, extract_answer, feather_score, leaderboard,
                              rsi_index, score_answer, stamp, start, verify)

QUICK = "featherbench-quick-v1"


class Submissions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, obj):
        p = self.dir / name
        p.write_text(json.dumps(obj) if not isinstance(obj, str) else obj, encoding="utf-8")
        return p

    def refused(self, path, github=None, needle=""):
        with self.assertRaises(RsiError) as cm:
            check_submission(path, github)
        self.assertIn(needle, str(cm.exception))

    def test_accepts_settings_docs_and_bare_configs(self):
        p = self.dir / "alice.json"
        configio.save_settings(p, Config(model="mlp", layers="8"))
        self.assertEqual(check_submission(p, "Alice")[1], "alice")  # GitHub names are case-insensitive
        self.assertEqual(check_submission(self.write("bob.json", {"model": "mlp"}), "bob")[0].model, "mlp")
        self.assertEqual(check_submission(ROOT / "featherbench" / "submissions" / "tactical-drone.json",
                                          "tactical-drone")[1], "tactical-drone")

    def test_refusals(self):
        ok = {"model": "mlp"}
        self.refused(self.write("alice.json", ok), "bob", "belongs to 'alice'")
        self.refused(self.write("bad name!.json", ok), None, "GitHub username")
        self.refused(self.write("carol.txt", ok), None, ".json")
        self.refused(self.write("dan.json", "{not json"), None, "valid UTF-8 JSON")
        self.refused(self.write("erin.json", [1, 2]), None, "JSON object")
        self.refused(self.write("frank.json", {"model": "mlp", "nope": 1}), None, "unknown config field")
        self.refused(self.write("gina.json", {"activation": "my_secret_act"}), None, "")
        self.refused(self.write("hal.json", {"format": "nn-playground/settings", "version": 1, "config": ok,
                                             "parts": {"evil": {}}}), None, "data only")
        self.refused(self.write("ivy.json", {"format": "nn-playground/settings", "version": 1, "config": ok,
                                             "meta": {"parts": ["agent_parts.x"]}}), None, "data only")
        self.refused(self.write("jo.json", {"format": "nn-playground/settings", "config": ok, "code": "x"}),
                     None, "unexpected keys")
        self.refused(self.write("kim.json", " " * (70 * 1024) + "{}"), None, "limit")

    def test_params_cap(self):
        with self.assertRaises(RsiError) as cm, quiet():
            verify(self.write("big.json", {"model": "mlp", "layers": "256,256,256"}), benchmark=QUICK,
                   workers=0, store=str(self.dir / "store"))
        self.assertIn("capped", str(cm.exception))

    def test_verify_and_leaderboard(self):
        res = self.dir / "results"
        res.mkdir()
        for user, cfg in (("small", {"model": "mlp", "layers": "4"}), ("good", {"model": "mlp", "layers": "16,16"})):
            if user == "good":
                configio.save_settings(self.dir / "good.json", configio.config_from_dict(cfg)[0])
                stamp(self.dir / "good.json", model="Claude Opus 5.5", harness="Claude Code", tokens=1000,
                      cost_usd=1.5, human_assist="none")
            with quiet():
                path = self.dir / "good.json" if user == "good" else self.write(f"{user}.json", cfg)
                doc = verify(path, github=user, benchmark=QUICK, workers=0,
                             out=res / f"{user}.json", store=str(self.dir / "store"))
            self.assertEqual((doc["github"], doc["benchmark"]["id"]), (user, QUICK))
        (res / "junk.json").write_text("{}", encoding="utf-8")  # ignored: not a result doc
        board = leaderboard(res, self.dir / "lb.json", benchmark=QUICK)
        self.assertEqual(board["n_entries"], 2)
        self.assertEqual([e["rank"] for e in board["entries"]], [1, 2])
        first = board["entries"][0]
        self.assertTrue({"github", "params_max", "mean_acc", "solved_all", "config"} <= set(first))
        keys = [(not e["solved_all"], e["params_max"] if e["solved_all"] else -e["n_solved"]) for e in board["entries"]]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(json.loads((self.dir / "lb.json").read_text(encoding="utf-8"))["entries"], board["entries"])
        self.assertEqual(leaderboard(self.dir / "missing", None, benchmark=QUICK)["n_entries"], 0)
        by_user = {e["github"]: e for e in board["entries"]}
        self.assertEqual(by_user["good"]["attempt"]["model"], "Claude Opus 5.5")
        self.assertIsNone(by_user["small"]["attempt"]["model"])  # nothing reported stays unknown
        scores = [e["feather_score"] for e in board["entries"]]
        self.assertEqual(scores, sorted(scores, reverse=True))  # the score orders like the ranking
        models = {m["model"]: m for m in board["models"]}
        self.assertEqual((models["Claude Opus 5.5"]["attempts"], models["Claude Opus 5.5"]["median_tokens"]), (1, 1000))
        self.assertIn("unknown", models)
        self.assertTrue(all(isinstance(e["rsi_index"], float) for e in board["entries"]))
        self.assertEqual(doc["baseline"]["config_key"], configio.config_key(Config()))

    def test_attempt_fields(self):
        ok = check_attempt({"model": "GPT-6 Astra", "tokens": 5, "cost_usd": 0.25, "human_assist": "some",
                            "harness": "unknown"})
        self.assertEqual((ok["model"], ok["harness"], ok["tokens"]), ("GPT-6 Astra", None, 5))
        for bad in ({"model": "x" * 81}, {"tokens": -1}, {"tokens": 1.5}, {"tokens": True}, {"cost_usd": "free"},
                    {"human_assist": "maybe"}, {"secret": 1}, ["model"]):
            with self.subTest(bad=bad), self.assertRaises(RsiError):
                check_attempt(bad)
        p = self.dir / "alice.json"
        p.write_text(json.dumps({"model": "mlp"}), encoding="utf-8")  # a bare config gets wrapped
        stamp(p, model="human", human_assist="none")
        stamp(p, tokens=0)  # later stamps merge
        cfg, user, att = check_submission(p, "alice")
        self.assertEqual((cfg.model, att["model"], att["tokens"]), ("mlp", "human", 0))

    def test_feather_score_scale(self):
        base = {"n_datasets": 12, "solved_all": False, "n_solved": 0, "mean_acc": 0.0, "params_max": 1}
        unsolved = feather_score({**base, "n_solved": 11, "mean_acc": 0.99})
        solved_small = feather_score({**base, "solved_all": True, "n_solved": 12, "params_max": 100})
        solved_big = feather_score({**base, "solved_all": True, "n_solved": 12, "params_max": 20000})
        self.assertTrue(0 <= feather_score(base) < unsolved < 923 <= solved_big < solved_small <= 1000)


class RsiLoop(unittest.TestCase):
    """RSIGym-style loop: inherit a setup, record compute, measure the gap closed."""

    def test_rsi_index(self):
        base = [{"dataset": "a", "fresh_acc_mean": 0.8}, {"dataset": "b", "fresh_acc_mean": 1.0}]
        self.assertEqual(rsi_index([{"dataset": "a", "fresh_acc_mean": 0.9}, {"dataset": "b", "fresh_acc_mean": 1.0}], base),
                         0.25)  # (0.5 + 0) / 2: half of a's gap, nothing left on b
        self.assertEqual(rsi_index([{"dataset": "a", "fresh_acc_mean": 0.6}], base), -1.0)  # a regression
        self.assertIsNone(rsi_index([], base))

    def test_baseline_is_frozen(self):
        from rsi.bench import get_benchmark
        from rsi.featherbench import baseline_rows
        b = get_benchmark("featherbench-general-v1")
        rows, key = baseline_rows(b)  # no training: read from featherbench/baselines/
        self.assertEqual(len(rows), 12)
        self.assertEqual(key, json.loads(Path("featherbench/results/tactical-drone.json").read_text(encoding="utf-8"))["config_key"])
        with tempfile.TemporaryDirectory() as tmp:  # a stale file (other hash) is never used
            f = Path(tmp) / f"{b.id}.json"
            f.write_text(json.dumps({"hash": "000000000000", "config_key": "x", "per_dataset": []}), encoding="utf-8")
            from unittest import mock
            with mock.patch("rsi.api.bench", return_value={"per_dataset": ["fresh"]}) as run:
                self.assertEqual(baseline_rows(b, baselines=tmp)[0], ["fresh"])
                run.assert_called_once()

    def test_start_inherits_and_stamp_records(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            os.environ["RSI_STORE"] = str(tmp / "store")
            try:
                res = tmp / "results"
                res.mkdir()
                champ = configio.config_to_dict(Config(model="mlp", layers="6"))
                (tmp / "lb.json").write_text(json.dumps({"entries": [{"github": "Champ", "config": champ}]}), "utf-8")
                (res / "Champ.json").write_text(json.dumps({"per_dataset": [
                    {"dataset": "Moons", "fresh_acc_mean": 0.99, "solved": True},
                    {"dataset": "Smiley", "fresh_acc_mean": 0.7, "solved": False}]}), "utf-8")
                r = start("leader", tmp / "mine.json", results=res, board=tmp / "lb.json")
                self.assertEqual((r["parent"], r["weakest_patterns"][0]["dataset"]), ("Champ", "Smiley"))
                self.assertEqual(configio.load_settings(tmp / "mine.json").config.layers, "6")  # inherited
                with self.assertRaises(RsiError):
                    start("nobody", tmp / "x.json", results=res, board=tmp / "lb.json")
                self.assertEqual(start("default", tmp / "d.json", results=res, board=tmp / "lb.json")["parent"], "default")
                start("Champ", tmp / "mine.json", results=res, board=tmp / "lb.json")
                import rsi
                with quiet():
                    rsi.run(config=str(tmp / "mine.json"), steps=20, seeds="0-1", workers=0, datasets=["Moons"])
                att = stamp(tmp / "mine.json", model="Claude Opus 5.5")
                self.assertEqual(att["parent"], "Champ")
                self.assertEqual((att["compute"]["trials"], att["compute"]["cells"]), (1, 2))
                self.assertGreaterEqual(att["compute"]["cell_seconds"], 0)  # 20 steps round to ~0 s
                self.assertEqual(check_submission(tmp / "mine.json")[2]["parent"], "Champ")  # still a valid submission
            finally:
                os.environ.pop("RSI_STORE", None)

    def test_parent_and_compute_validation(self):
        self.assertEqual(check_attempt({"parent": "default"})["parent"], "default")
        for bad in ({"parent": "../etc"}, {"compute": {"cells": -1}}, {"compute": {"hack": 1}}, {"compute": [1]}):
            with self.subTest(bad=bad), self.assertRaises(RsiError):
                check_attempt(bad)


class HeldOut(unittest.TestCase):
    """Secret seeds only CI knows: scored, published as aggregates, never revealed."""

    def test_holdout_scored_without_revealing_seeds(self):
        import os
        from rsi.featherbench import HOLDOUT_ENV, holdout_seeds, overfit_flag
        secret = ("861733", "4242421", "9090907")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            res = tmp / "results"
            res.mkdir()
            (tmp / "neo.json").write_text(json.dumps({"model": "mlp", "layers": "8"}), "utf-8")
            os.environ[HOLDOUT_ENV] = ",".join(secret)
            try:
                with quiet():
                    doc = verify(tmp / "neo.json", github="neo", benchmark=QUICK, workers=0, out=res / "neo.json",
                                 store=str(tmp / "store"))
            finally:
                os.environ.pop(HOLDOUT_ENV, None)
            ho = doc["holdout"]
            self.assertEqual((ho["n_seeds"], ho["n_datasets"]), (3, doc["n_datasets"]))
            self.assertIsInstance(doc["overfit"], bool)
            board = leaderboard(res, tmp / "lb.json", benchmark=QUICK)
            published = (res / "neo.json").read_text(encoding="utf-8") + (tmp / "lb.json").read_text(encoding="utf-8")
            for seed in secret:
                self.assertNotIn(seed, published)  # the secret seeds never reach any published file
            self.assertEqual(board["entries"][0]["holdout"]["n_solved"], ho["n_solved"])
        self.assertIsNone(holdout_seeds())  # unset: no held-out run (local and pull-request CI)
        for bad in ("1,2", "1,1,2", "a,b,c", "-1,2,3"):
            os.environ[HOLDOUT_ENV] = bad
            try:
                with self.subTest(bad=bad), self.assertRaises(RsiError):
                    holdout_seeds()
            finally:
                os.environ.pop(HOLDOUT_ENV, None)
        pub = {"n_solved": 12, "mean_acc": 0.95}
        self.assertIsNone(overfit_flag(pub, None))
        self.assertFalse(overfit_flag(pub, {"n_solved": 12, "mean_acc": 0.94}))
        self.assertTrue(overfit_flag(pub, {"n_solved": 10, "mean_acc": 0.94}))
        self.assertTrue(overfit_flag(pub, {"n_solved": 12, "mean_acc": 0.91}))


class OneShot(unittest.TestCase):
    def test_extract_answer(self):
        fence = "`" * 3
        reply = f'think...\n{fence}json\n{{"a": 1}}\n{fence}\nthen\n{fence}json\n{{"b": 2}}\n{fence}'
        self.assertEqual(extract_answer(reply), {"b": 2})  # the last fenced block wins
        self.assertEqual(extract_answer('use {"model": "mlp", "x": {"y": 1}} ok'), {"model": "mlp", "x": {"y": 1}})
        self.assertIsNone(extract_answer("no json here {nope}"))

    def test_invalid_answers_score_zero(self):
        with tempfile.TemporaryDirectory() as tmp, quiet():
            import os
            os.environ["RSI_STORE"] = tmp
            try:
                fence = "`" * 3
                for text, why in (("I can't help with that.", "no JSON"),
                                  (f'{fence}json\n{{"activation": "snek"}}\n{fence}', "snek"),
                                  ('{"model": "mlp", "layers": "512,512,512"}', "cap"),
                                  ('{"model": "mlp", "evil": "import os"}', "unknown config field")):
                    with self.subTest(text=text):
                        r = score_answer(text, benchmark=QUICK, workers=0)
                        self.assertEqual((r["valid"], r["feather_score"]), (False, 0.0))
                        self.assertIn(why, r["reason"])
                r = score_answer(f'{fence}json\n{{"model": "mlp", "layers": "8"}}\n{fence}', benchmark=QUICK, workers=0)
                self.assertTrue(r["valid"] and 0 < r["feather_score"] <= 1000)
            finally:
                os.environ.pop("RSI_STORE", None)


class Workflows(unittest.TestCase):
    """The CI that runs on strangers' pull requests must stay read-only."""

    def read(self, name):
        return (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")

    def test_pr_workflow_is_read_only(self):
        wf = self.read("featherbench-pr.yml")
        code = "\n".join(l for l in wf.splitlines() if not l.lstrip().startswith("#"))
        self.assertIn("pull_request:", code)
        self.assertNotIn("pull_request_target", code)
        self.assertIn("contents: read", code)
        self.assertNotIn("write", re.sub(r"(?s)python - <<'PY'.*?PY", "", code).split("jobs:")[0])
        self.assertNotIn("secrets.", code)
        self.assertIn("persist-credentials: false", code)
        for line in code.splitlines():  # event data only through env:, never inlined into a script
            if "${{ github.event" in line and not line.strip().startswith("group:"):
                self.assertRegex(line.strip(), r"^[A-Z_]+: \$\{\{ github\.event\.[a-z_.]+ \}\}$")

    def test_holdout_secret_only_in_the_trusted_score_job(self):
        self.assertNotIn("HOLDOUT", self.read("featherbench-pr.yml"))
        wf = self.read("featherbench-leaderboard.yml")
        score, publish = wf.split("  publish:")
        self.assertIn("secrets.FEATHERBENCH_HOLDOUT_SEEDS", score)
        self.assertNotIn("secrets.", publish)

    def test_leaderboard_workflow_runs_on_main_only(self):
        code = "\n".join(l for l in self.read("featherbench-leaderboard.yml").splitlines()
                         if not l.lstrip().startswith("#"))
        self.assertNotIn("pull_request", code)
        self.assertIn("branches: [main]", code)


if __name__ == "__main__":
    unittest.main()
