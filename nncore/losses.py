"""
Compartment: LOSSES

Contract:  fn(logits (N, K), y (N,) int64) -> scalar tensor
The same function drives training and the loss numbers / curve you see.
"""
import torch.nn.functional as F

from .registry import Registry

LOSSES = Registry("loss", "fn(logits, y) -> scalar tensor")

LOSSES.register("cross entropy", F.cross_entropy)


@LOSSES.register("cross entropy (smoothed 0.1)")
def smoothed_ce(logits, y):
    return F.cross_entropy(logits, y, label_smoothing=0.1)


@LOSSES.register("mse on softmax")
def mse_softmax(logits, y):
    return F.mse_loss(logits.softmax(1), F.one_hot(y, logits.shape[1]).float())
