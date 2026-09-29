"""
my_parts.py // your workbench. Anything registered here shows up in the UI
dropdowns and is usable from headless.py, no other file needs editing.

Both launchers import this module before building anything. Uncomment an
example to see it appear, then write your own.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from nncore import (ACTIVATIONS, DATASETS, EXPANSIONS, FEATURES, INITIALIZERS, LAYERS, LOSSES,  # noqa: F401
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
SKIPS.register("half residual", lambda h, x: h + 0.5 * x if h.shape == x.shape else h)
SKIPS.register("add",    lambda h, x: h + x)                   # ResNet style, your l + head(l)
SKIPS.register("concat", lambda h, x: torch.cat([h, x], -1))   # DenseNet style
# --- whole architecture ------------------------------------------------------
# Must return a module with forward(x, collect=False), hidden_sizes, describe().
class Wide(nn.Module):
    def __init__(self, n_in, n_out, width, activation, skip="none", expand="linear"):
        super().__init__()
        n_perc = n_out*2
        self.flank = EXPANSIONS.get(expand)(n_in)   # factory(n_in) -> nn.Module
        n_x = self.flank(torch.zeros(1, n_in)).shape[-1]  # what hid sees: expanded width
        self.hid, self.head = nn.Linear(n_x, width), nn.Linear(width, n_perc)
        self.perc = nn.Linear(n_perc, n_perc)
        self.decide = nn.Linear(n_perc, n_perc)
        self._tmp = nn.Linear(width, n_out)
        self.act = ACTIVATIONS.get(activation)()   # factory() -> nn.Module
        self.skip = SKIPS.get(skip)                # fn(h, x) -> tensor
        #d = n_out*2
        #pw = self.skip(torch.zeros(1, d), torch.zeros(1, d)).shape[-1]  # concat widens perc
        #self.focus = nn.Linear(pw, n_out, False)
        self.focus = nn.Linear(n_perc, n_out)
        self.width = width
        self.hidden_sizes = [n_x, width, n_perc, n_perc, n_perc, n_out]  # one per tensor forward() collects
        print("Width =", width);
    def describe(self):
        return f"wide {self.width}"
    def forward(self, x, collect=False):
        #h = torch.relu(self.hid(x))
        #h = F.gelu(self.hid(x))
        flank = self.flank(x) # expanded input
        hidden = self.act(self.hid(flank)) # w
        head = self.head(hidden)
        ##gut = self.act(flank)
        decide = self.act(self.decide(head))
        perc = self.perc(decide)                
        #out = self.focus(perc) 
        out = self._tmp(hidden)
        return (out, [flank, hidden, head, decide, perc, out]) if collect else out

MODELS.register("custom nn", lambda n_in, n_out, cfg: Wide(n_in, n_out, cfg.width, cfg.activation, cfg.skip, cfg.expand))
