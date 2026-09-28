"""
Compartment: FEATURES  (input feature engineering, TF-Playground style)

Contract:  fn(x, y) -> tensor, same shape as x
  x, y are 1-D float tensors of raw point coordinates.
The selected features are stacked into the network's input columns in the
order they are registered.
"""
import torch

from .registry import Registry

FEATURES = Registry("feature", "fn(x, y) -> tensor like x")

FEATURES.register("x", lambda x, y: x)
FEATURES.register("y", lambda x, y: y)
FEATURES.register("x²", lambda x, y: x * x)
FEATURES.register("y²", lambda x, y: y * y)
FEATURES.register("x·y", lambda x, y: x * y)
FEATURES.register("sin x", lambda x, y: torch.sin(x * 3))
FEATURES.register("sin y", lambda x, y: torch.sin(y * 3))
FEATURES.register("r", lambda x, y: torch.sqrt(x * x + y * y + 1e-9))


def featurize(points, names):
    """points (N, 2) raw coordinates -> (N, len(names)) network input."""
    x, y = points[:, 0], points[:, 1]
    return torch.stack([FEATURES.get(f)(x, y) for f in names], 1)
