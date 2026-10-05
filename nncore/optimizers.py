"""
Compartments: OPTIMIZERS and SCHEDULES

OPTIMIZERS  fn(params, lr, weight_decay) -> torch.optim.Optimizer
SCHEDULES   fn(optimizer[, cfg=]) -> object with .step() (called once per train step), or None
            cfg (the whole Config) is passed only to factories that accept it.

Changing an optimizer field (optimizer, lr, weight_decay, schedule) through
Session.configure builds a fresh optimizer, so Adam moments and the schedule
position restart (unless configure(keep_optimizer_state=True) for lr / wd).
"""
import math

import torch

from .registry import Registry

OPTIMIZERS = Registry("optimizer", "fn(params, lr, weight_decay) -> Optimizer")
SCHEDULES = Registry("schedule", "fn(optimizer) -> scheduler with .step(), or None")

OPTIMIZERS.register("Adam", lambda p, lr, wd: torch.optim.Adam(p, lr, weight_decay=wd))
OPTIMIZERS.register("AdamW", lambda p, lr, wd: torch.optim.AdamW(p, lr, weight_decay=wd))
OPTIMIZERS.register("SGD", lambda p, lr, wd: torch.optim.SGD(p, lr, weight_decay=wd))
OPTIMIZERS.register("SGD+momentum",
                    lambda p, lr, wd: torch.optim.SGD(p, lr, momentum=0.9, weight_decay=wd))
OPTIMIZERS.register("RMSprop", lambda p, lr, wd: torch.optim.RMSprop(p, lr, weight_decay=wd))

SCHEDULES.register("constant", lambda opt: None)
SCHEDULES.register("exp decay (×0.999/step)",
                   lambda opt: torch.optim.lr_scheduler.ExponentialLR(opt, 0.999))
SCHEDULES.register("step (÷10 every 2000)",
                   lambda opt: torch.optim.lr_scheduler.StepLR(opt, 2000, 0.1))

# appended (opt-in), so the default entries and their order are unchanged
OPTIMIZERS.register("NAdam", lambda p, lr, wd: torch.optim.NAdam(p, lr, weight_decay=wd))
OPTIMIZERS.register("RAdam", lambda p, lr, wd: torch.optim.RAdam(p, lr, weight_decay=wd))
OPTIMIZERS.register("Adamax", lambda p, lr, wd: torch.optim.Adamax(p, lr, weight_decay=wd))
OPTIMIZERS.register("SGD+nesterov",
                    lambda p, lr, wd: torch.optim.SGD(p, lr, momentum=0.9, nesterov=True, weight_decay=wd))


@SCHEDULES.register("cosine (over extra.total_steps)")
def cosine(opt, cfg=None):
    """Cosine from lr down to 0 over cfg.extra['total_steps'] steps (default 2000), then 0."""
    total = max(1, int(((cfg.extra if cfg is not None else None) or {}).get("total_steps", 2000)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))


SCHEDULES.register("cosine warm restarts (T0=1000)",
                   lambda opt: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 1000),
                   doc="cosine from lr to 0 every 1000 steps, then restart at lr")
SCHEDULES.register("warmup 200 + constant",
                   lambda opt: torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200)),
                   doc="lr ramps up linearly over the first 200 steps, then stays")
