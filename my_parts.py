"""
my_parts.py // your workbench. Anything registered here shows up in the UI
dropdowns and is usable from headless.py, no other file needs editing.

Both launchers import this module before building anything. Uncomment an
example to see it appear, then write your own. The live entries at the bottom
(the extra skips and the "custom nn" model) are the playground's defaults.
"""
import torch
import torch.nn as nn

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
# Both models check at build time that a skip fits: "add" needs equal widths (readable
# error otherwise), "concat" widens the next layer. Wide has one per stage (skip_*).
SKIPS.register("add",    lambda h, x: h + x)                   # ResNet style, your l + head(l)
SKIPS.register("concat", lambda h, x: torch.cat([h, x], -1))   # DenseNet style

# --- whole architecture ------------------------------------------------------
def _skip_width(fn, name, stage, h_w, x_w):
    """Output width of skip fn(h, x) for a (1, h_w) layer output and its (1, x_w) input.
    Zero probe, no RNG. A skip that can't combine them raises a readable ValueError here,
    at build time, instead of a shape crash mid-training."""
    try:
        out = fn(torch.zeros(1, h_w), torch.zeros(1, x_w))
    except Exception as e:
        raise ValueError(f"skip '{name}' at {stage} needs equal widths (got {h_w} vs {x_w}); "
                         f"try concat, a residual, or none there") from None
    if not isinstance(out, torch.Tensor) or out.dim() != 2 or out.shape[0] != 1:
        raise ValueError(f"skip '{name}' at {stage} must return a (batch, width) tensor, got "
                         f"{getattr(out, 'shape', type(out).__name__)}")
    return out.shape[-1]


# Must return a module with forward(x, collect=False), hidden_sizes, describe().
class Wide(nn.Module):
    """The default model ("custom nn"). Dataflow, one Linear per step:

        x -> flank   = expand(x)        EXPANSIONS[expand], scale = fourier_freq
          -> hid     (width)    + activation
          -> head    (classes)
          -> decide  (classes)  + act_decide    skip_decide(decide, head)
          -> perc    (classes)                  skip(perc, decide)
          -> relate  (n_in)     + act_relate    skip_relate(relate, perc)
          -> prepare (classes)  + act_prepare   skip_prepare(prepare, relate)
          -> focus   (n_out, no bias)

    Each skip combines a stage's output with that stage's input, like the per-stage
    activations: skip_* = "same" uses skip where it fits (else none). "none" everywhere is
    the plain chain above.
    A skip that changes the width (concat) widens the next Linear; widths come from a
    zero probe, so building draws no extra randomness. Reads width, activation, expand,
    fourier_freq, classes, act_decide, act_relate, act_prepare ("same" = activation), skip
    and skip_decide, skip_relate, skip_prepare. layers and layer are mlp fields; this model
    ignores them.
    """
    def __init__(self, n_in, n_out, width, activation, skip="none", expand="linear", classes=5,
                 act_decide="same", act_relate="same", act_prepare="same", fourier_freq=3.0,
                 skip_decide="same", skip_relate="same", skip_prepare="same"):
        super().__init__()
        n_perc = classes                            # was n_out*2
        self.flank = make_expansion(expand, n_in, scale=fourier_freq)  # factory(n_in) -> nn.Module
        n_x = self.flank(torch.zeros(1, n_in)).shape[-1]  # what hid sees: expanded width
        # skips by stage, fn(h, x) -> tensor with h = the stage output, x = its input. Widths
        # are probed before any layer exists (same build order as always). A stage on "same"
        # inherits skip only where it fits (e.g. "add" can't span relate's width change)
        # and is plain there; a skip picked explicitly for a stage must fit or raises.
        def stage(name, st, h_w, x_w):
            fn = SKIPS.get(skip if name == "same" else name)
            try:
                return fn, _skip_width(fn, skip if name == "same" else name, st, h_w, x_w)
            except ValueError:
                if name != "same":
                    raise
                return SKIPS.get("none"), h_w
        self.skip_decide, dw = stage(skip_decide, "decide", n_perc, n_perc)
        self.skip = SKIPS.get(skip)
        pw = _skip_width(self.skip, skip, "perc", n_perc, dw)
        self.skip_relate, rw = stage(skip_relate, "relate", n_in, pw)
        self.skip_prepare, prw = stage(skip_prepare, "prepare", n_perc, rw)
        self.hid, self.head = nn.Linear(n_x, width), nn.Linear(width, n_perc)
        self.perc = nn.Linear(dw, n_perc)
        self.relate = nn.Linear(pw, n_in)
        self.prepare = nn.Linear(rw, n_perc)
        self.decide = nn.Linear(n_perc, n_perc)
        self.act = ACTIVATIONS.get(activation)()   # factory() -> nn.Module
        pick = lambda name: ACTIVATIONS.get(activation if name == "same" else name)()
        self.act_decide, self.act_relate, self.act_prepare = pick(act_decide), pick(act_relate), pick(act_prepare)
        #d = n_out*2
        #pw = self.skip(torch.zeros(1, d), torch.zeros(1, d)).shape[-1]  # concat widens perc
        #self.focus = nn.Linear(pw, n_out, False)
        self.focus = nn.Linear(prw, n_out,False)
        self.width = width
        self.hidden_sizes = [n_x, width, n_perc, dw, pw, rw, prw, n_out]  # one per tensor forward() collects
    def describe(self):
        return f"wide {self.width}"
    def forward(self, x, collect=False):
        flank = self.flank(x)
        hidden = self.act(self.hid(flank))
        head = self.head(hidden)

        decide = self.skip_decide(self.act_decide(self.decide(head)), head)
        perc = self.skip(self.perc(decide), decide)
        relate = self.skip_relate(self.act_relate(self.relate(perc)), perc)

        prepare = self.skip_prepare(self.act_prepare(self.prepare(relate)), relate)
        out = self.focus(prepare)
        return (out, [flank, hidden, head, decide, perc, relate, prepare, out]) if collect else out

MODELS.register("custom nn", lambda n_in, n_out, cfg: Wide(n_in, n_out, cfg.width, cfg.activation, cfg.skip, cfg.expand, cfg.classes,
                                                                   cfg.act_decide, cfg.act_relate, cfg.act_prepare, cfg.fourier_freq,
                                                                   cfg.skip_decide, cfg.skip_relate, cfg.skip_prepare))
