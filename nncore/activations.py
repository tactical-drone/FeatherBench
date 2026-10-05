"""
Compartment: ACTIVATIONS

Contract:  factory() -> nn.Module   (called once per layer, so it may hold params)
"""
import torch
import torch.nn as nn

from .registry import Registry

ACTIVATIONS = Registry("activation", "factory() -> nn.Module")


class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class Gauss(nn.Module):
    def forward(self, x):
        return torch.exp(-x * x)


class Abs(nn.Module):
    def forward(self, x):
        return x.abs()


class Square(nn.Module):
    def forward(self, x):
        return x * x


for _name, _factory in {
    "tanh": nn.Tanh, "relu": nn.ReLU, "leaky_relu": lambda: nn.LeakyReLU(0.1),
    "sigmoid": nn.Sigmoid, "gelu": nn.GELU, "silu": nn.SiLU, "mish": nn.Mish,
    "elu": nn.ELU, "softplus": nn.Softplus, "sin": Sine, "gauss": Gauss,
    "abs": Abs, "square": Square, "linear": nn.Identity,
}.items():
    ACTIVATIONS.register(_name, _factory)


# --- more parts for search (appended, so the original order is unchanged) ----
class Snake(nn.Module):
    """x + sin²(x): periodic wiggles on a rising trend."""
    def forward(self, x):
        return x + torch.sin(x) ** 2


class Cos(nn.Module):
    def forward(self, x):
        return torch.cos(x)


class XSin(nn.Module):
    """x·sin(x): a wave whose amplitude grows with |x|."""
    def forward(self, x):
        return x * torch.sin(x)


ACTIVATIONS.register("snake", Snake, doc="x + sin²(x): periodic wiggles on a rising trend")
ACTIVATIONS.register("cos", Cos, doc="cos(x)")
ACTIVATIONS.register("selu", nn.SELU, doc="self-normalising ELU")
ACTIVATIONS.register("softsign", nn.Softsign, doc="x / (1 + |x|)")
ACTIVATIONS.register("hardtanh", nn.Hardtanh, doc="clamp(x, -1, 1)")
ACTIVATIONS.register("prelu", nn.PReLU, doc="leaky relu with a learnable slope (one per layer, starts at 0.25)")
ACTIVATIONS.register("x·sin x", XSin, doc="x·sin(x): a wave whose amplitude grows with |x|")
