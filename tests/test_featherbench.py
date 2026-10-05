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
from rsi.featherbench import check_submission, leaderboard, verify

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
            with quiet():
                doc = verify(self.write(f"{user}.json", cfg), github=user, benchmark=QUICK, workers=0,
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

    def test_leaderboard_workflow_runs_on_main_only(self):
        code = "\n".join(l for l in self.read("featherbench-leaderboard.yml").splitlines()
                         if not l.lstrip().startswith("#"))
        self.assertNotIn("pull_request", code)
        self.assertIn("branches: [main]", code)


if __name__ == "__main__":
    unittest.main()
