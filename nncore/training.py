"""
Compartment: TRAIN_STEPS  (what one optimisation step does)

Contract:  fn(model, optimizer, loss_fn, xb, yb[, cfg=, step=]) -> float loss
  cfg (the whole Config) and step (steps done so far) are passed only to
  functions that accept them, so plain 5-argument steps keep working.
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


# appended (opt-in), so the default entries and their order are unchanged
def _finite_grads(model):
    return all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@TRAIN_STEPS.register("skip non-finite")
def skip_nonfinite(model, opt, loss_fn, xb, yb):
    """Standard step, but a non-finite loss or gradient skips the update, so one bad
    batch cannot write NaN into the weights. The loss is still returned as is (a
    non-finite one still sets Session.diverged_at)."""
    loss = loss_fn(model(xb), yb)
    opt.zero_grad(set_to_none=True)
    if not torch.isfinite(loss):
        return loss.item()
    loss.backward()
    if _finite_grads(model):
        opt.step()
    else:
        opt.zero_grad(set_to_none=True)
    return loss.item()


@TRAIN_STEPS.register("clip grad norm (extra)")
def clipped_extra(model, opt, loss_fn, xb, yb, cfg=None):
    """Clip the gradient norm to cfg.extra['clip'] (default 1.0) before stepping."""
    clip = float(((cfg.extra if cfg is not None else None) or {}).get("clip", 1.0))
    loss = loss_fn(model(xb), yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()
    return loss.item()
