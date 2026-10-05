"""
Compartment: INITIALIZERS  (how weights start)

Contract:  fn(model) -> None, mutates parameters in place.
Runs right after the model is built, under the session's torch seed.

Modules can protect their own init: a module class with `keep_init = True`
(e.g. the Fourier expansion, whose weight spread is fourier_freq) is skipped,
with everything inside it, by the "(keep expansion)" entries and by every
entry added after them. The original "xavier uniform" and "kaiming normal"
still overwrite every nn.Linear (unchanged, so old runs reproduce).
"""
import torch.nn as nn

from .registry import Registry

INITIALIZERS = Registry("initializer", "fn(model) -> None")


def _each_linear(model, weight_fn, keep=False):
    """weight_fn(weight) and zero bias for every nn.Linear; keep=True skips modules
    inside any module whose class sets keep_init = True."""
    skip = set()
    if keep:
        for m in model.modules():
            if getattr(m, "keep_init", False):
                skip.update(id(c) for c in m.modules())
    for m in model.modules():
        if isinstance(m, nn.Linear) and id(m) not in skip:
            weight_fn(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


@INITIALIZERS.register("pytorch default")
def default_init(model):
    pass  # nn.Linear already initialised itself (Kaiming-uniform, a=√5)


@INITIALIZERS.register("xavier uniform")
def xavier(model):
    """Xavier-uniform weights, zero biases, on every nn.Linear (also the Fourier expansion)."""
    _each_linear(model, nn.init.xavier_uniform_)


@INITIALIZERS.register("kaiming normal")
def kaiming(model):
    """Kaiming-normal with the relu gain (whatever the activation), zero biases, every nn.Linear."""
    _each_linear(model, lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu"))


# appended (opt-in); these leave keep_init modules (the Fourier expansion) alone
@INITIALIZERS.register("xavier uniform (keep expansion)")
def xavier_keep(model):
    """Xavier-uniform, zero biases; keep_init modules (Fourier) keep their own init."""
    _each_linear(model, nn.init.xavier_uniform_, keep=True)


@INITIALIZERS.register("kaiming normal (keep expansion)")
def kaiming_keep(model):
    """Kaiming-normal (relu gain), zero biases; keep_init modules keep their own init."""
    _each_linear(model, lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu"), keep=True)


@INITIALIZERS.register("orthogonal")
def orthogonal(model):
    """Orthogonal weights (gain 1), zero biases; keep_init modules keep their own init."""
    _each_linear(model, nn.init.orthogonal_, keep=True)


@INITIALIZERS.register("xavier normal")
def xavier_normal(model):
    """Xavier-normal weights, zero biases; keep_init modules keep their own init."""
    _each_linear(model, nn.init.xavier_normal_, keep=True)


@INITIALIZERS.register("lecun normal")
def lecun_normal(model):
    """Normal(0, 1/fan_in) weights (pairs with selu), zero biases; keep_init modules keep theirs."""
    _each_linear(model, lambda w: nn.init.kaiming_normal_(w, nonlinearity="linear"), keep=True)
