"""
Compartment: METRICS  (extra numbers reported next to the loss)

Contract:  fn(logits (N, K), y (N,)) -> float
Every registered metric is computed on train and test each evaluation.

EXTRA_METRICS has the same contract but is computed only when asked for
(Session.evaluate(extra_metrics=("bal_acc", ...)), rsi --metrics), so the
default evaluate() / headless output is unchanged. Per-class metrics use only
the classes present in y (a class with no true points has no recall); an
empty y gives NaN.
"""
import math

import torch

from .registry import Registry

METRICS = Registry("metric", "fn(logits, y) -> float")
EXTRA_METRICS = Registry("extra metric", "fn(logits, y) -> float")


@METRICS.register("acc")
def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


def _per_class(logits, y):
    """(recall, precision) per class present in y, as float tensors."""
    pred = logits.argmax(1)
    classes = torch.unique(y)
    hit = torch.stack([((pred == c) & (y == c)).sum() for c in classes]).float()
    true = torch.stack([(y == c).sum() for c in classes]).float()
    said = torch.stack([(pred == c).sum() for c in classes]).float()
    return hit / true, torch.where(said > 0, hit / said.clamp(min=1), torch.zeros_like(hit))


@EXTRA_METRICS.register("bal_acc", doc="balanced accuracy: mean recall over the classes present")
def balanced_accuracy(logits, y):
    if len(y) == 0:
        return math.nan
    return _per_class(logits, y)[0].mean().item()


@EXTRA_METRICS.register("macro_f1", doc="mean F1 over the classes present")
def macro_f1(logits, y):
    if len(y) == 0:
        return math.nan
    rec, prec = _per_class(logits, y)
    f1 = torch.where(rec + prec > 0, 2 * rec * prec / (rec + prec).clamp(min=1e-12), torch.zeros_like(rec))
    return f1.mean().item()


@EXTRA_METRICS.register("worst_class_acc", doc="lowest recall over the classes present")
def worst_class_accuracy(logits, y):
    if len(y) == 0:
        return math.nan
    return _per_class(logits, y)[0].min().item()


@EXTRA_METRICS.register("mean_conf", doc="mean softmax probability of the predicted class")
def mean_confidence(logits, y):
    if len(y) == 0:
        return math.nan
    return logits.softmax(1).max(1).values.mean().item()


@EXTRA_METRICS.register("ece", doc="expected calibration error, 10 equal-width confidence bins")
def expected_calibration_error(logits, y, bins=10):
    if len(y) == 0:
        return math.nan
    conf, pred = logits.softmax(1).max(1)
    right = (pred == y).float()
    b = (conf * bins).long().clamp(max=bins - 1)  # bin i holds conf in [i/bins, (i+1)/bins)
    err = 0.0
    for i in range(bins):
        m = b == i
        if m.any():
            err += m.float().mean().item() * abs(right[m].mean().item() - conf[m].mean().item())
    return err
