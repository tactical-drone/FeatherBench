"""Test-only parts: a model that kills the worker process that builds it (never the
coordinator). Part of nn-playground (AGPL-3.0; see COMMERCIAL.md)."""
import multiprocessing
import os

from nncore import MODELS
from nncore.models import build_mlp


def _crash(n_in, n_out, cfg):
    if multiprocessing.parent_process() is not None:
        os._exit(3)  # a hard crash: BrokenProcessPool in the coordinator
    raise RuntimeError("_test crash model only crashes inside workers")


MODELS.register("_test crash", _crash)
MODELS.register("_test mlp", build_mlp)
