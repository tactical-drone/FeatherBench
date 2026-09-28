"""
Compartment: METRICS  (extra numbers reported next to the loss)

Contract:  fn(logits (N, K), y (N,)) -> float
Every registered metric is computed on train and test each evaluation.
"""
from .registry import Registry

METRICS = Registry("metric", "fn(logits, y) -> float")


@METRICS.register("acc")
def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()
