"""Shared test helpers. Part of nn-playground (AGPL-3.0; see COMMERCIAL.md)."""
import contextlib
import io
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

with contextlib.redirect_stdout(io.StringIO()):
    import my_parts  # noqa: F401,E402  (registers 'custom nn'; its build print is swallowed here)

SLOW = os.environ.get("RSI_SLOW") == "1"


def params(model):
    return [p.detach().clone() for p in model.parameters()]


def same_params(a, b):
    pa, pb = list(a.parameters()), list(b.parameters())
    return len(pa) == len(pb) and all(torch.equal(x, y) for x, y in zip(pa, pb))


@contextlib.contextmanager
def quiet():
    """Swallow stdout (Wide prints 'Width = 8' on every build)."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield
