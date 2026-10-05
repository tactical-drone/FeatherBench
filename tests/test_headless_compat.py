"""Legacy headless.py: text formats unchanged, plus the fixes (report-every 0, validation
before runs, kebab flags, exit codes) and `headless.py <command>` == `python -m rsi <command>`.
Part of nn-playground (AGPL-3.0; see COMMERCIAL.md)."""
import contextlib
import io
import json
import re
import subprocess
import sys
import unittest

from tests._util import ROOT, quiet  # noqa: F401  (loads my_parts)
import headless

STEP = re.compile(r"^  step +\d+  train loss \d+\.\d{4}  acc \d+\.\d{4}   test loss \d+\.\d{4}  acc \d+\.\d{4}$")
HEAD = re.compile(r"^== (run|summary|\S+=.+)$")
NET = re.compile(r"^  net .+  params \d+  \d+\.\d\ds$")
SUMMARY = re.compile(r"^  .{32} train loss \d+\.\d{4}  acc \d+\.\d{4}   test loss \d+\.\d{4}  acc \d+\.\d{4}$")
WIDTH = re.compile(r"^Width = \d+;?$")  # my_parts' debug print (until the other team removes it)


def legacy(*args):
    """In-process headless.main -> (exit code, stdout lines, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = headless.main(list(args))
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
    return code, [ln for ln in out.getvalue().splitlines() if ln.strip()], err.getvalue()


def check_lines(tc, lines):
    for ln in lines:
        tc.assertTrue(any(r.match(ln) for r in (STEP, HEAD, NET, SUMMARY, WIDTH)), ln)


class Legacy(unittest.TestCase):
    def test_list(self):
        code, lines, _ = legacy("--list")
        self.assertEqual(code, 0)
        self.assertTrue(lines[0].startswith("dataset      fn(n, noise, rng)"))
        self.assertIn("               - Moons", lines)

    def test_run_subprocess(self):
        p = subprocess.run([sys.executable, "headless.py", "--steps", "20", "--report-every", "10"], cwd=ROOT,
                           capture_output=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr[-400:])
        lines = [ln for ln in p.stdout.decode("utf-8").splitlines() if ln.strip()]
        check_lines(self, lines)
        self.assertEqual(sum(bool(STEP.match(ln)) for ln in lines), 2)

    def test_compare(self):
        code, lines, err = legacy("--compare", "activation=tanh, relu", "--model", "mlp", "--steps", "20",
                                  "--report-every", "20")
        self.assertEqual(code, 0, err)
        check_lines(self, lines)
        self.assertIn("== activation=tanh", lines)
        self.assertIn("== activation=relu", lines)
        self.assertIn("== summary", lines)

    def test_report_every_zero_terminates(self):
        code, lines, _ = legacy("--steps", "10", "--report-every", "0", "--model", "mlp")
        self.assertEqual(code, 0)
        self.assertEqual(sum(bool(STEP.match(ln)) for ln in lines), 1)

    def test_validation_and_exit_codes(self):
        self.assertEqual(legacy("--compare", "foo=1,2")[0], 2)
        self.assertEqual(legacy("--compare", "lr")[0], 2)
        self.assertEqual(legacy("--steps", "-1")[0], 2)
        code, lines, err = legacy("--compare", "activation=tanh,relux", "--steps", "5")
        self.assertEqual(code, 3)
        self.assertEqual(lines, [])  # nothing ran before the bad value was found
        self.assertIn("relu", err)
        self.assertEqual(legacy("--width", "abc")[0], 3)
        self.assertEqual(legacy("--dataset", "Nope")[0], 3)

    def test_kebab_aliases_and_inert_warning(self):
        code, lines, err = legacy("--n-points", "100", "--batch-size", "full", "--features", "x,y,x^2",
                                  "--report_every", "5", "--steps", "5", "--layers", "4,4")
        self.assertEqual(code, 0, err)
        self.assertIn("layers is not read by model 'custom nn'", err)
        self.assertTrue(any(NET.match(ln) for ln in lines))

    def test_subcommand_dispatch(self):
        buf = io.StringIO()
        from rsi.cli import main
        code = main(["describe", "--section", "suites", "--log", "quiet"], out=buf)
        self.assertEqual(code, 0)
        self.assertTrue(headless._rsi_command(["describe"]))
        self.assertFalse(headless._rsi_command(["--steps", "5"]))
        self.assertEqual(json.loads(buf.getvalue())["schema"], "rsi/describe@1")


if __name__ == "__main__":
    unittest.main()
