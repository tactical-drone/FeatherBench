"""
Compartment: INITIALIZERS  (how weights start)

Contract:  fn(model) -> None, mutates parameters in place.
Runs right after the model is built, under the session's torch seed.
"""
import torch.nn as nn

from .registry import Registry

INITIALIZERS = Registry("initializer", "fn(model) -> None")


def _each_linear(model, weight_fn):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            weight_fn(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


@INITIALIZERS.register("pytorch default")
def default_init(model):
    pass  # nn.Linear already initialised itself (Kaiming-uniform, a=√5)


@INITIALIZERS.register("xavier uniform")
def xavier(model):
    _each_linear(model, nn.init.xavier_uniform_)


@INITIALIZERS.register("kaiming normal")
def kaiming(model):
    _each_linear(model, lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu"))
