"""
my_parts.py // your workbench. Anything registered here shows up in the UI
dropdowns and is usable from headless.py, no other file needs editing.

Both launchers import this module before building anything. Uncomment an
example to see it appear, then write your own.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from nncore import (ACTIVATIONS, DATASETS, FEATURES, INITIALIZERS, LAYERS, LOSSES,  # noqa: F401
                    METRICS, MODELS, OPTIMIZERS, SAMPLERS, SCHEDULES, SKIPS, TRAIN_STEPS)

# --- activation --------------------------------------------------------------
# class Snake(nn.Module):
#     """x + sin²(x): periodic but monotone-ish."""
#     def forward(self, x):
#         return x + torch.sin(x) ** 2
#
# ACTIVATIONS.register("snake", Snake)

# --- dataset -----------------------------------------------------------------
# import numpy as np
#
# @DATASETS.register("Two stripes")
# def stripes(n, noise, rng):
#     X = rng.uniform(-1.2, 1.2, (n, 2))
#     y = (np.abs(X[:, 0]) < 0.4).astype(np.int64)
#     return (X + rng.normal(0, noise, X.shape)).astype(np.float32), y

# --- input feature -----------------------------------------------------------
# FEATURES.register("cos x·y", lambda x, y: torch.cos(3 * x * y))

# --- optimizer ---------------------------------------------------------------
# OPTIMIZERS.register("Adagrad", lambda p, lr, wd: torch.optim.Adagrad(p, lr, weight_decay=wd))

# --- skip connection ---------------------------------------------------------
# SKIPS.register("half residual", lambda h, x: h + 0.5 * x if h.shape == x.shape else h)

# --- whole architecture ------------------------------------------------------
# Must return a module with forward(x, collect=False), hidden_sizes, describe().
class Wide(nn.Module):
    def __init__(self, n_in, n_out, width, activation):
        super().__init__()
        self.hid, self.head = nn.Linear(n_in*2, width), nn.Linear(width, n_out)
        self.flank = nn.Linear(n_in, n_in*2, False)

        self.act = ACTIVATIONS.get(activation)()   # factory() -> nn.Module
        self.hidden_sizes = [width]
        print("Width =", width);
    def describe(self):
        return f"wide {self.hidden_sizes[0]}"
    def forward(self, x, collect=False):
        #h = torch.relu(self.hid(x))
        #h = F.gelu(self.hid(x))
        h = self.act(self.hid(self.flank(x))) 
        out = self.head(h)
        return (out, [h]) if collect else out

MODELS.register("custom nn", lambda n_in, n_out, cfg: Wide(n_in, n_out, cfg.width, cfg.activation))
