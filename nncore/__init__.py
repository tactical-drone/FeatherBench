"""
nncore // the neural-net side of the playground, with no UI, Qt or plotting.

Every compartment is a Registry of interchangeable parts:

    DATASETS      fn(n, noise, rng) -> (X, y)                      datasets.py
    SPLITTERS     fn(n, test_frac, rng) -> (train_idx, test_idx)   datasets.py
    FEATURES      fn(x, y) -> tensor                               features.py
    MODELS        build(n_in, n_out, cfg) -> nn.Module             models.py
    LAYERS        factory(d_in, d_out) -> nn.Module                layers.py
    SKIPS         fn(h, x) -> tensor                               layers.py
    EXPANSIONS    factory(n_in) -> nn.Module                       layers.py
    ACTIVATIONS   factory() -> nn.Module                           activations.py
    INITIALIZERS  fn(model) -> None                                initializers.py
    LOSSES        fn(logits, y) -> scalar tensor                   losses.py
    OPTIMIZERS    fn(params, lr, weight_decay) -> Optimizer        optimizers.py
    SCHEDULES     fn(optimizer) -> scheduler | None                optimizers.py
    TRAIN_STEPS   fn(model, opt, loss_fn, xb, yb) -> float         training.py
    SAMPLERS      factory(n, batch_size) -> next_batch()           samplers.py
    METRICS       fn(logits, y) -> float                           metrics.py

Config (config.py) picks one entry per compartment by name; Session
(session.py) wires them together and trains. Your own parts go in
my_parts.py (or any module imported before the Session is built).
"""
import torch

from .activations import ACTIVATIONS
from .config import Config
from .datasets import DATASETS, SPLITTERS
from .features import FEATURES, featurize
from .initializers import INITIALIZERS
from .layers import EXPANSIONS, LAYERS, SKIPS, make_expansion
from .losses import LOSSES
from .metrics import METRICS
from .models import MLP, MODELS, parse_layers
from .optimizers import OPTIMIZERS, SCHEDULES
from .registry import Registry
from .samplers import SAMPLERS
from .session import Session
from .training import TRAIN_STEPS

torch.set_num_threads(1)  # tiny nets: threading overhead > work

REGISTRIES = {
    "dataset": DATASETS, "splitter": SPLITTERS, "features": FEATURES, "model": MODELS,
    "layer": LAYERS, "skip": SKIPS, "expand": EXPANSIONS, "activation": ACTIVATIONS, "init": INITIALIZERS,
    "loss": LOSSES, "optimizer": OPTIMIZERS, "schedule": SCHEDULES,
    "train_step": TRAIN_STEPS, "sampler": SAMPLERS, "metrics": METRICS,
}  # Config field -> registry it names

__all__ = ["Config", "Session", "Registry", "REGISTRIES", "MLP", "parse_layers", "featurize",
           "ACTIVATIONS", "DATASETS", "SPLITTERS", "FEATURES", "INITIALIZERS", "LAYERS", "SKIPS", "EXPANSIONS", "make_expansion",
           "LOSSES", "METRICS", "MODELS", "OPTIMIZERS", "SCHEDULES", "SAMPLERS", "TRAIN_STEPS"]
