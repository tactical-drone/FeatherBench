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


@TRAIN_STEPS.register("cross-loop distill")
def cross_loop_distill(model, opt, loss_fn, xb, yb, cfg=None):
    """D-LoopOPD (arXiv 2610.10623) for looped models: the task loss on the last loop, plus
    a reverse KL that pulls an intermediate loop (extra.distill_loop, default the middle one)
    toward the current last loop, stop-gradient. The teacher is the model's own deeper
    computation, refreshed every step, so student updates also improve the teacher through
    the shared weights. extra.distill_weight (default 1.0). Models without loop_logits get a
    standard step."""
    loop_logits = getattr(model, "loop_logits", None)
    if loop_logits is None:
        return standard(model, opt, loss_fn, xb, yb)
    extra = (cfg.extra if cfg is not None else None) or {}
    logits = loop_logits(xb)
    teacher_logits = logits[-1]
    loss = loss_fn(teacher_logits, yb)
    if len(logits) > 1:
        k = int(extra.get("distill_loop", max(1, len(logits) // 2)))
        student = logits[min(max(k, 1), len(logits) - 1) - 1]
        log_p = torch.log_softmax(student, -1)
        log_q = torch.log_softmax(teacher_logits.detach(), -1)  # sg[teacher]: the dynamic, self-refreshing teacher
        kl = (log_p.exp() * (log_p - log_q)).sum(-1).mean()     # reverse KL(student || teacher), as in OPD
        loss = loss + float(extra.get("distill_weight", 1.0)) * kl
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss.item()
