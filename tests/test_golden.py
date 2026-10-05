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


def golden_entry():
    """The entry for this platform/torch, or None. torch null = any version."""
    for e in GOLDEN["entries"]:
        if e["system"] != platform.system() or e["machine"] not in (None, platform.machine()):
            continue
        if e["torch"] is None or torch.__version__.startswith(e["torch"]):
            return e
    return None


@unittest.skipUnless(SLOW, "slow: set RSI_SLOW=1")
class Golden(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell = run_unit(RunSpec(config=config_to_dict(Config()), steps=GOLDEN["steps"]))

    def test_equals_legacy(self):
        with quiet():
            s = LegacySession(Config())
            s.train(GOLDEN["steps"])
        ev = s.evaluate()
        self.assertEqual(self.cell["train"], ev["train"])
        self.assertEqual(self.cell["test"], ev["test"])

    def test_golden_numbers(self):
        e = golden_entry()
        if e is None:
            self.skipTest(f"no golden for {platform.system()}-{platform.machine()}/{torch.__version__}")
        self.assertAlmostEqual(self.cell["train"]["acc"], e["train_acc"], places=GOLDEN["places"])
        self.assertAlmostEqual(self.cell["test"]["acc"], e["test_acc"], places=GOLDEN["places"])


if __name__ == "__main__":
    unittest.main()
