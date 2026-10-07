"""
nncore // the neural-net side of the playground, with no UI, Qt or plotting.

Every compartment is a Registry of interchangeable parts:

    DATASETS      fn(n, noise, rng) -> (X, y)                      datasets.py
    SPLITTERS     fn(n, test_frac, rng) -> (train_idx, test_idx)   datasets.py
    FEATURES      fn(x, y) -> tensor                               features.py
    MODELS        build(n_in, n_out, cfg) -> nn.Module             models.py
    LAYERS        factory(d_in, d_out) -> nn.Module                layers.py
    SKIPS         fn(h, x) -> tensor                               layers.py
    EXPANSIONS    factory(n_in, **knobs) -> nn.Module              layers.py  (make_expansion passes
                                                                              only the knobs it accepts,
                                                                              e.g. scale=fourier_freq)
    ACTIVATIONS   factory() -> nn.Module                           activations.py
    INITIALIZERS  fn(model) -> None                                initializers.py
    LOSSES        fn(logits, y) -> scalar tensor                   losses.py
    OPTIMIZERS    fn(params, lr, weight_decay) -> Optimizer        optimizers.py
    SCHEDULES     fn(optimizer[, cfg=]) -> scheduler | None        optimizers.py
    TRAIN_STEPS   fn(model, opt, loss_fn, xb, yb[, cfg=, step=])   training.py
    SAMPLERS      factory(n, batch_size) -> next_batch()           samplers.py
    METRICS       fn(logits, y) -> float                           metrics.py

Optional keyword arguments in [] are passed only to parts whose signature
accepts them (registry.call_accepting), so plain parts keep working.

Config (config.py) picks one entry per compartment by name; Session
(session.py) wires them together and trains. Your own parts go in
my_parts.py (or any module): load_plugins("my_parts", "agent_parts.x")
imports them, and a name lookup that misses tries DEFAULT_PLUGINS
(env NNCORE_PLUGINS, default my_parts) once by itself.

Machine-facing helpers: configio (Config <-> JSON, validation, settings
files), schema (lever table, param counts), run (run_unit: one JSON cell).

Concurrency: one Session per process; run parallel work in spawned
processes, not threads. Importing nncore sets torch to NNCORE_THREADS
threads (default 1: for these tiny nets threading costs more than it saves).
"""
import os

import torch

torch.set_num_threads(int(os.environ.get("NNCORE_THREADS", "1")))

__version__ = "0.3.0"

from . import configio, schema  # noqa: E402
from .activations import ACTIVATIONS  # noqa: E402
from .config import Config  # noqa: E402
from .datasets import DATASETS, SPLITTERS, SUITES, suite_datasets, suite_signature  # noqa: E402
from .features import FEATURES, featurize  # noqa: E402
from .initializers import INITIALIZERS  # noqa: E402
from .layers import EXPANSIONS, LAYERS, SKIPS, make_expansion  # noqa: E402
from .losses import LOSSES  # noqa: E402
from .metrics import METRICS  # noqa: E402
from .models import MLP, MODELS, parse_layers  # noqa: E402
from .optimizers import OPTIMIZERS, SCHEDULES  # noqa: E402
from .registry import DEFAULT_PLUGINS, Registry, call_accepting, load_plugins, loaded_plugins  # noqa: E402
from .run import RunSpec, run_unit  # noqa: E402
from .samplers import SAMPLERS  # noqa: E402
from .session import Session  # noqa: E402
from .training import TRAIN_STEPS  # noqa: E402

try:
    from .datasets import FAMILIES  # noqa: E402  (dataset families, e.g. "spiral[7]")
except ImportError:
    FAMILIES = None

REGISTRIES = {
    "dataset": DATASETS, "splitter": SPLITTERS, "features": FEATURES, "model": MODELS,
    "layer": LAYERS, "skip": SKIPS, "expand": EXPANSIONS, "activation": ACTIVATIONS, "init": INITIALIZERS,
    "loss": LOSSES, "optimizer": OPTIMIZERS, "schedule": SCHEDULES,
    "train_step": TRAIN_STEPS, "sampler": SAMPLERS, "metrics": METRICS,
}  # name -> registry, as listed by headless --list; configio.FIELD_REGISTRY maps Config fields

__all__ = ["Config", "Session", "Registry", "REGISTRIES", "MLP", "parse_layers", "featurize", "make_expansion",
           "ACTIVATIONS", "DATASETS", "SPLITTERS", "FEATURES", "INITIALIZERS", "LAYERS", "SKIPS", "EXPANSIONS",
           "LOSSES", "METRICS", "MODELS", "OPTIMIZERS", "SCHEDULES", "SAMPLERS", "TRAIN_STEPS",
           "SUITES", "suite_datasets", "suite_signature", "FAMILIES",
           "load_plugins", "loaded_plugins", "DEFAULT_PLUGINS", "call_accepting", "configio", "schema",
           "run_unit", "RunSpec", "__version__"]
