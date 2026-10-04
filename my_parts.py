"""
my_parts.py // your workbench. Anything registered here shows up in the UI
dropdowns and is usable from headless.py, no other file needs editing.

Both launchers import this module before building anything. Uncomment an
example to see it appear, then write your own.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from nncore import (ACTIVATIONS, DATASETS, EXPANSIONS, FEATURES, make_expansion, INITIALIZERS, LAYERS, LOSSES,  # noqa: F401
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
    def __init__(self, n_in, n_out, width, activation, skip="none", expand="linear", classes=5,
                 act_decide="same", act_relate="same", act_prepare="same", fourier_freq=3.0):
        super().__init__()
        n_perc = classes                            # was n_out*2
        self.flank = make_expansion(expand, n_in, scale=fourier_freq)  # factory(n_in) -> nn.Module
        n_x = self.flank(torch.zeros(1, n_in)).shape[-1]  # what hid sees: expanded width
        self.hid, self.head = nn.Linear(n_x, width), nn.Linear(width, n_perc)
        self.perc = nn.Linear(n_perc, n_perc)
        self.relate = nn.Linear(n_perc, n_in)
        self.prepare = nn.Linear(n_in, n_perc)
        self.decide = nn.Linear(n_perc, n_perc)        
        self.act = ACTIVATIONS.get(activation)()   # factory() -> nn.Module
        pick = lambda name: ACTIVATIONS.get(activation if name == "same" else name)()
        self.act_decide, self.act_relate, self.act_prepare = pick(act_decide), pick(act_relate), pick(act_prepare)
        self.skip = SKIPS.get(skip)                # fn(h, x) -> tensor
        #d = n_out*2
        #pw = self.skip(torch.zeros(1, d), torch.zeros(1, d)).shape[-1]  # concat widens perc
        #self.focus = nn.Linear(pw, n_out, False)
        self.focus = nn.Linear(n_perc, n_out,False)
        self.width = width
        self.hidden_sizes = [n_x, width, n_perc, n_perc, n_perc, n_in, n_perc, n_out]  # one per tensor forward() collects
        print("Width =", width);
    def describe(self):
        return f"wide {self.width}"
    def forward(self, x, collect=False):
        
        flank = self.flank(x)
        hidden = self.act(self.hid(flank))
        head = self.head(hidden)

        decide = self.act_decide(self.decide(head))
        perc = self.perc(decide)        
        relate = self.act_relate(self.relate(perc))

        prepare = self.act_prepare(self.prepare(relate))           
        out = self.focus(prepare)         
        return (out, [flank, hidden, head, decide, perc, relate, prepare, out]) if collect else out

MODELS.register("custom nn", lambda n_in, n_out, cfg: Wide(n_in, n_out, cfg.width, cfg.activation, cfg.skip, cfg.expand, cfg.classes,
                                                                   cfg.act_decide, cfg.act_relate, cfg.act_prepare, cfg.fourier_freq))
