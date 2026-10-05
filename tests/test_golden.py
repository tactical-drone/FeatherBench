"""Golden numbers for the default config (7000 steps), keyed by platform and torch version,
plus the platform-independent guard: run_unit equals the legacy Session bit for bit.
Slow: runs only with RSI_SLOW=1. Part of nn-playground (AGPL-3.0; see COMMERCIAL.md)."""
import json
import platform
import unittest
from pathlib import Path

import torch

from tests._util import SLOW, quiet
from tests._legacy_session import LegacySession
from nncore import Config
from nncore.configio import config_to_dict
from nncore.run import RunSpec, run_unit

GOLDEN = json.loads((Path(__file__).parent / "golden.json").read_text(encoding="utf-8"))


def golden_entry(entries=None):
    """The entry for this platform/torch, or None. torch null = any version."""
    for e in GOLDEN["entries"] if entries is None else entries:
        if e["system"] != platform.system() or e["machine"] not in (None, platform.machine()):
            continue
        if e["torch"] is None or torch.__version__.startswith(e["torch"]):
            return e
    return None


V0 = GOLDEN["named"]["fourier-gauss-v0"]


@unittest.skipUnless(SLOW, "slow: set RSI_SLOW=1")
class Golden(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell = run_unit(RunSpec(config=config_to_dict(Config()), steps=GOLDEN["steps"]))
        cls.v0_cfg = Config(**V0["overrides"])
        cls.v0 = run_unit(RunSpec(config=config_to_dict(cls.v0_cfg), steps=GOLDEN["steps"]))

    def test_equals_legacy(self):
        for name, cfg, cell in (("default", Config(), self.cell), ("fourier-gauss-v0", self.v0_cfg, self.v0)):
            with self.subTest(recipe=name), quiet():
                s = LegacySession(cfg)
                s.train(GOLDEN["steps"])
                ev = s.evaluate()
                self.assertEqual(cell["train"], ev["train"])
                self.assertEqual(cell["test"], ev["test"])

    def check(self, cell, entries):
        e = golden_entry(entries)
        if e is None:
            self.skipTest(f"no golden for {platform.system()}-{platform.machine()}/{torch.__version__}")
        self.assertAlmostEqual(cell["train"]["acc"], e["train_acc"], places=GOLDEN["places"])
        self.assertAlmostEqual(cell["test"]["acc"], e["test_acc"], places=GOLDEN["places"])

    def test_golden_numbers(self):
        self.check(self.cell, GOLDEN["entries"])

    def test_golden_fourier_gauss_v0(self):
        """The pre-skip default: 99.6 / 97.5 on V's Windows box, kept as a named recipe."""
        self.check(self.v0, V0["entries"])


if __name__ == "__main__":
    unittest.main()
