"""
Compartment: TRAIN_STEPS  (what one optimisation step does)

Contract:  fn(model, optimizer, loss_fn, xb, yb) -> float loss
Swap this to try gradient clipping, noise injection, custom update rules, ...
"""
import torch

from .registry import Registry

TRAIN_STEPS = Registry("train step", "fn(model, opt, loss_fn, xb, yb) -> float")


@TRAIN_STEPS.register("standard")
def standard(model, opt, loss_fn, xb, yb):
    loss = loss_fn(model(xb), yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss.item()


@TRAIN_STEPS.register("clip grad norm 1.0")
def clipped(model, opt, loss_fn, xb, yb):
    loss = loss_fn(model(xb), yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()
