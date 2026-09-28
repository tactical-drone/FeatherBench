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
