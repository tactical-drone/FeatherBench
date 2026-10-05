"""rsi console CLI: the stdout contract, envelopes, exit codes, config precedence, run /
check / repl / export / replay / runs / background. Part of nn-playground (AGPL-3.0; see COMMERCIAL.md)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests._util import ROOT, quiet  # noqa: F401  (loads my_parts)
from nncore import Config
from nncore.configio import load_settings, save_settings
from rsi.cli import invoke

MOONS = ["--model", "mlp", "--dataset", "Moons", "--n-points", "120"]


def proc(args, store, env=None, stdin=None, timeout=120):
    e = {**os.environ, "RSI_STORE": str(store), "PYTHONPATH": str(ROOT), **(env or {})}
    return subprocess.run([sys.executable, "-m", "rsi", *args], cwd=ROOT, env=e, input=stdin, capture_output=True,
                          timeout=timeout)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.store = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def rsi(self, *args, stdin=None):
        return invoke(["--store", str(self.store), "--log", "quiet", *args], stdin=stdin)


class Process(Base):
    def test_pipe_cp1252_one_ascii_document(self):
        """A real pipe in a cp1252 console: one ASCII JSON document on stdout, part prints kept off it."""
        p = proc(["run", "--set", "dataset=XOR quadrants", "--model", "mlp", "--steps", "20"], self.store,
                 env={"PYTHONIOENCODING": "cp1252"})
        self.assertEqual(p.returncode, 0, p.stderr[-500:])
        p.stdout.decode("ascii")  # ASCII only
        lines = [ln for ln in p.stdout.decode().splitlines() if ln.strip()]
        self.assertEqual(len(lines), 1)
        d = json.loads(lines[0])
        self.assertEqual(d["schema"], "rsi/trial@1")
        self.assertIn("my_parts", d["meta"]["parts"])

    def test_background_and_wait(self):
        code, d = self.rsi("sweep", "--vary", "lr=0.01,0.03", *MOONS, "--steps", "20", "--workers", "0", "--name",
                           "bg1", "--background")  # launched in-process; the run itself is a detached child
        self.assertEqual(code, 0, d)
        self.assertEqual((d["schema"], d["result"]["state"]), ("rsi/background@1", "running"))
        self.assertIsInstance(d["result"]["pid"], int)
        code, w = self.rsi("wait", "bg1", "--timeout", "120")
        self.assertEqual(code, 0, w)
        self.assertEqual(w["result"]["state"], "done")
        out = json.loads((self.store / "bg1" / "stdout.json").read_text())
        self.assertEqual(out["result"]["n_trials"], 2)


class Contract(Base):
    def test_errors_and_exit_codes(self):
        code, d = self.rsi("run", "--activation", "relux", "--steps", "5")
        self.assertEqual((code, d["error"]["code"], d["error"]["exit"]), (3, "E_UNKNOWN_PART", 3))
        self.assertIn("relu", d["error"]["did_you_mean"])
        code, d = self.rsi("run", "--bogus", "1")
        self.assertEqual((code, d["ok"], d["error"]["code"]), (2, False, "E_USAGE"))
        code, d = self.rsi("frobnicate")
        self.assertEqual(code, 2)
        code, d = self.rsi("run", "--set", "nope=1")
        self.assertEqual((code, d["error"]["code"]), (3, "E_UNKNOWN_FIELD"))
        code, d = self.rsi("run", "--test-frac", "1.0")
        self.assertEqual((code, d["error"]["code"]), (3, "E_OUT_OF_RANGE"))
        code, d = self.rsi("runs", "show", "t_ffffffffffff")
        self.assertEqual((code, d["error"]["code"]), (5, "E_NOT_FOUND"))
        code, d = self.rsi("runs", "query", "--where", "test_acc")
        self.assertEqual((code, d["error"]["code"]), (3, "E_BAD_QUERY"))
        code, d = self.rsi("run", "--help")
        self.assertEqual((code, d["schema"]), (0, "rsi/help@1"))
        self.assertIn("--steps", d["result"]["usage"])
        for cmd in (["evolve"], ["bench", "list"], ["complexity", "--family", "spiral", "--sizes", "2-3"],
                    ["gp", "check", "gp:x"]):
            mod = {"evolve": "evolve", "bench": "bench", "complexity": "complexity", "gp": "gp"}[cmd[0]]
            if not (ROOT / "rsi" / f"{mod}.py").exists():
                code, d = self.rsi(*cmd)
                self.assertEqual((code, d["error"]["code"]), (3, "E_UNSUPPORTED"), cmd)

    def test_diverged_is_exit_4(self):
        code, d = self.rsi("run", "--model", "mlp", "--layers", "16:square,16:square,16:square", "--optimizer", "SGD",
                           "--lr", "1", "--dataset", "Moons", "--steps", "500")
        self.assertEqual((code, d["error"]["code"]), (4, "E_RUN_FAILED"))
        cell = d["result"]["cells"][0]
        self.assertEqual(cell["status"], "diverged")
        self.assertIsNone(cell["test"]["acc"])
        self.assertIn("W_NONFINITE", [w["code"] for w in d["warnings"]])

    def test_run_cache_and_runs(self):
        args = ["run", *MOONS, "--steps", "50", "--seeds", "0-2", "--tag", "ab"]
        code, a = self.rsi(*args)
        self.assertEqual(code, 0, a)
        r = a["result"]
        self.assertEqual((r["status"], r["cached"], len(r["cells"]), r["seeds"]), ("ok", False, 3, [0, 1, 2]))
        self.assertEqual(r["config_diff"], {"dataset": "Moons", "n_points": 120, "model": "mlp"})
        code, b = self.rsi(*args)
        self.assertTrue(b["result"]["cached"])
        self.assertEqual(b["result"]["summary"]["test_acc"], r["summary"]["test_acc"])
        code, q = self.rsi("runs", "query", "--tag", "ab", "--sort", "-fitness", "--fields", "id,fitness,params")
        self.assertEqual((code, q["result"]["n"]), (0, 2))
        code, s = self.rsi("runs", "show", r["id"][:8])
        self.assertEqual(s["result"]["id"], r["id"])
        code, st = self.rsi("runs", "stats", "--group-by", "config.model", "--where", "tag==ab")
        self.assertEqual(st["result"]["groups"][0]["n_trials"], 2)
        code, rp = self.rsi("replay", r["id"])
        self.assertEqual(code, 0, rp)
        self.assertTrue(rp["result"]["identical"])
        self.assertEqual(rp["result"]["n_cells"], 3)

    def test_precedence_and_warnings(self):
        f = self.store / "base.json"
        save_settings(f, Config(model="mlp", lr=0.1, dataset="Moons", layers="4"))
        code, d = self.rsi("check", "--config", str(f), "--lr", "0.2", "--set-json", '{"lr": 0.3, "width": 16}',
                           "--set", "lr=0.4", "--set", "extra.clip=0.5", "--features", "x,y,x^2",
                           "--schedule", "step (/10 every 2000)")
        self.assertEqual(code, 0, d)
        c = d["result"]["config"]
        self.assertEqual((c["lr"], c["width"], c["layers"], c["extra"]), (0.4, 16, "4", {"clip": 0.5}))
        self.assertEqual(c["features"], ["x", "y", "x²"])
        codes = [w["code"] for w in d["warnings"]]
        self.assertIn("W_INERT_FIELD", codes)  # width with mlp
        self.assertIn("W_ALIAS_USED", codes)
        self.assertEqual(d["result"]["n_params"], 3 * 4 + 4 + 4 * 2 + 2)  # x,y,x² -> 4 -> 2 classes
        code, d = self.rsi("check", "--stdin", stdin='{"model": "mlp", "layers": "8:sin"}')
        self.assertEqual(d["result"]["describe"].split()[2], "8sin")
        code, d = self.rsi("check", "--model", "mlp", "--skip", "add")
        self.assertEqual(code, 4)

    def test_map_png_settings_and_text(self):
        png = self.store / "m.png"
        st = self.store / "s.json"
        code, d = self.rsi("run", *MOONS, "--steps", "40", "--map", "30x10", "--png", str(png), "--save-settings",
                           str(st), "--curve")
        self.assertEqual(code, 0, d)
        m = d["result"]["map"]
        self.assertEqual((len(m), len(m[0])), (10, 30))
        self.assertTrue(set("".join(m)) <= set("01#"))
        self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        s = load_settings(st)
        self.assertEqual(s.meta["expect"]["test_acc"], d["result"]["cells"][0]["test"]["acc"])
        self.assertEqual(d["result"]["cells"][0]["curve"]["step"][-1], 40)
        code, text = self.rsi("--format", "text", "run", *MOONS, "--steps", "40")
        self.assertIn("test_acc", text)

    def test_repl(self):
        reqs = '{"cmd":"train","steps":100}\n{"cmd":"set","set":{"lr":0.003}}\n{"cmd":"eval"}\n' \
               '{"cmd":"set","set":{"activation":"relux"}}\n{"cmd":"map","size":"20x6"}\n{"cmd":"quit"}\n'
        code, text = self.rsi("repl", "--model", "mlp", stdin=reqs)
        lines = [json.loads(ln) for ln in text.splitlines()]
        self.assertEqual(code, 0)
        self.assertEqual([ln["ok"] for ln in lines], [True, True, True, False, True])
        self.assertEqual(lines[1]["result"]["rebuilt"], "optimizer")
        self.assertEqual(lines[2]["result"]["step"], 100)
        self.assertEqual(lines[3]["error"]["code"], "E_UNKNOWN_PART")

    def test_sweep_export_open_leaderboard(self):
        code, d = self.rsi("sweep", "--vary", "activation=tanh,relu", *MOONS, "--steps", "30", "--name", "t_sw",
                           "--workers", "0")
        self.assertEqual((code, d["result"]["n_trials"]), (0, 2), d)
        load_settings(self.store / "t_sw" / "best.settings.json")
        w = self.store / "w.json"
        code, e = self.rsi("export", "t_sw", "--rank", "1", "-o", str(w))
        self.assertEqual(code, 0, e)
        code, r = self.rsi("run", "--config", str(w))  # steps: the file's train.steps
        exp = load_settings(w).meta["expect"]
        cell = r["result"]["cells"][0]
        self.assertEqual((r["result"]["budget"]["steps"], r["warnings"]), (30, []))
        self.assertEqual((cell["train"]["acc"], cell["test"]["acc"]), (exp["train_acc"], exp["test_acc"]))
        self.assertTrue(e["result"]["repro"].startswith("python -m rsi run ") and e["result"]["repro"].endswith(f" --config {w}"))
        code, r = self.rsi("run", "--config", str(w), "--steps", "20")
        self.assertIn("W_STEPS_DIFFER", [x["code"] for x in r["warnings"]])
        code, o = self.rsi("open", "t_sw", "--dry-run")
        self.assertEqual(o["result"]["command"][1:3], [str(ROOT / "nn_playground.py"), "--load"])
        self.assertFalse(o["result"]["launched"])
        code, lb = self.rsi("leaderboard", "t_sw", "--top", "1")
        self.assertEqual(len(lb["result"]["entries"]), 1)
        code, st = self.rsi("status", "t_sw")
        self.assertEqual(st["result"]["state"], "done")
        code, rl = self.rsi("runs", "list")
        self.assertIn("t_sw", [x["name"] for x in rl["result"]["runs"]])

    def test_describe_and_schema(self):
        code, d = self.rsi("describe", "--lever", "act_decide")
        self.assertEqual(d["result"]["lever"]["special"], ["same"])
        code, d = self.rsi("describe", "--lever", "widht")
        self.assertEqual((code, d["error"]["did_you_mean"][0]), (3, "width"))
        code, d = self.rsi("describe", "--section", "errors")
        self.assertIn("E_RUN_FAILED", d["result"]["errors"])
        for name in ("envelope", "error", "settings", "cell", "trial", "sweep", "space", "evolve-status",
                     "leaderboard", "repl"):
            code, d = self.rsi("schema", name)
            self.assertEqual((code, d["result"]["$schema"]), (0, "https://json-schema.org/draft/2020-12/schema"))


class ReviewFixes(Base):
    def test_text_output_on_cp1252(self):
        import io as _io
        from rsi.cli import main
        raw = _io.BytesIO()
        out = _io.TextIOWrapper(raw, encoding="cp1252")
        code = main(["--log", "quiet", "--format", "text", "describe", "--section", "parts"], out=out)
        out.flush()
        text = raw.getvalue().decode("cp1252")
        self.assertEqual(code, 0)
        self.assertIn("theta (atan2)", text)  # 'θ' folded to ASCII instead of a UnicodeEncodeError

    def test_usage_suggestions_and_headless_routing(self):
        import headless
        code, d = self.rsi("check", "--model", "mlp", "--activaton", "relu")
        self.assertEqual((code, d["error"]["code"]), (2, "E_USAGE"))
        self.assertIn("--activation", d["error"]["did_you_mean"])
        self.assertEqual(d["error"]["hint"], "python -m rsi check --help")
        code, d = self.rsi("runn")
        self.assertEqual((code, d["error"]["did_you_mean"][0]), (2, "run"))
        code, d = self.rsi("runs", "querry")
        self.assertEqual(d["error"]["did_you_mean"], ["query"])
        self.assertTrue(headless._rsi_command(["runn"]) and headless._rsi_command(["--format", "text", "describe"]))
        self.assertFalse(headless._rsi_command(["--steps", "20"]))
        for bad in (["--fresh-points", "-1"], ["--max-seconds", "-1"]):
            code, d = self.rsi("run", "--model", "mlp", "--steps", "5", *bad)
            self.assertEqual((code, d["error"]["code"]), (2, "E_USAGE"), bad)

    def test_parts_provenance(self):
        from rsi.store import Store
        code, d = self.rsi("--parts", "my_parts", "--parts", "tests._crash_parts", "check", "--model", "_test mlp")
        self.assertEqual(code, 0, d)
        self.assertIn("--parts tests._crash_parts", d["result"]["command"])
        code, r = self.rsi("run", *MOONS, "--steps", "5")
        self.assertIn("my_parts", r["result"]["parts"])
        rec = dict(r["result"], id="t_00000000beef", parts=["my_parts", "zz_not_loaded"],
                   config=dict(r["result"]["config"], activation="zz_act"))
        with Store(self.store) as st:
            st.put_trial(rec)
        for cmd in (["replay", rec["id"]], ["run", "--from", rec["id"], "--steps", "5"], ["export", rec["id"]]):
            code, d = self.rsi(*cmd)
            self.assertEqual((code, d["error"]["code"]), (5, "E_PARTS_MISSING"), cmd)
            self.assertIn("--parts zz_not_loaded", d["error"]["hint"])

    def test_gp_names_resolve_lazily(self):
        from rsi.pool import Evaluator, _gp_lazy
        self.assertIsNotNone(_gp_lazy("gp:mul(x;sin(x))"))
        self.assertIsNone(_gp_lazy("relu"))
        with Evaluator(0, ["my_parts"]) as ev:
            plain = Config(model="mlp").__dict__
            self.assertEqual(ev.fp_for(Config(model="mlp")), ev.code_fp)
            self.assertNotEqual(ev.fp_for(Config(model="mlp", activation="gp:sin(x)")), ev.code_fp)
            self.assertTrue(plain)
        code, d = self.rsi("run", *MOONS, "--activation", "gp:mul(x;sin(x))", "--steps", "5")
        self.assertEqual((code, d["result"]["status"]), (0, "ok"), d)


if __name__ == "__main__":
    unittest.main()
