"""
Compartments: OPTIMIZERS and SCHEDULES

OPTIMIZERS  fn(params, lr, weight_decay) -> torch.optim.Optimizer
SCHEDULES   fn(optimizer) -> object with .step() (called once per train step), or None
"""
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
