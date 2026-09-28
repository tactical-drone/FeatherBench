"""
Compartments: LAYERS and SKIPS  (the building blocks an architecture uses)

LAYERS  factory(d_in, d_out) -> nn.Module      the weighted transform per layer
SKIPS   fn(h, x) -> tensor                      how a layer's output h combines
                                                with its input x (residuals etc.)
"""
import torch.nn as nn

from .registry import Registry

LAYERS = Registry("layer", "factory(d_in, d_out) -> nn.Module")
SKIPS = Registry("skip", "fn(h, x) -> tensor")

LAYERS.register("linear", nn.Linear)
LAYERS.register("linear (no bias)", lambda d_in, d_out: nn.Linear(d_in, d_out, bias=False))


@SKIPS.register("none")
def no_skip(h, x):
    return h


@SKIPS.register("residual (same width)")
def residual(h, x):
    return h + x if h.shape[-1] == x.shape[-1] else h
