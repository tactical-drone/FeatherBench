"""
Compartments: LAYERS, SKIPS and EXPANSIONS  (the building blocks an architecture uses)

LAYERS      factory(d_in, d_out) -> nn.Module   the weighted transform per layer
SKIPS       fn(h, x) -> tensor                  how a layer's output h combines
                                                with its input x (residuals etc.)
EXPANSIONS  factory(n_in, **knobs) -> nn.Module widens the raw inputs before the
                                                first hidden layer; output width
                                                is whatever the module returns.
                                                Knobs are passed only when the
                                                factory accepts them (by name, or
                                                all of them for **kwargs); custom nn
                                                passes scale=cfg.fourier_freq.
"""
import math

import torch
import torch.nn as nn

from .registry import Registry, call_accepting

LAYERS = Registry("layer", "factory(d_in, d_out) -> nn.Module")
SKIPS = Registry("skip", "fn(h, x) -> tensor")
EXPANSIONS = Registry("expansion", "factory(n_in, **knobs) -> nn.Module")

LAYERS.register("linear", nn.Linear)
LAYERS.register("linear (no bias)", lambda d_in, d_out: nn.Linear(d_in, d_out, bias=False))


@SKIPS.register("none")
def no_skip(h, x):
    return h


@SKIPS.register("residual (same width)")
def residual(h, x):
    return h + x if h.shape[-1] == x.shape[-1] else h


# --- input expansions --------------------------------------------------------
# Only a nonlinear expansion adds expressiveness: a plain linear one is just
# another mix of x,y that the next layer could already make itself.
EXPANSIONS.register("none", lambda n_in: nn.Identity())
EXPANSIONS.register("linear", lambda n_in: nn.Linear(n_in, n_in * 2, bias=False))
EXPANSIONS.register("linear + tanh", lambda n_in: nn.Sequential(nn.Linear(n_in, n_in * 2), nn.Tanh()))


class Fourier(nn.Module):
    """sin(Wx + b): each unit is a wave with a learned direction and frequency."""
    keep_init = True  # its weight spread is the scale knob; "(keep expansion)" inits leave it alone

    def __init__(self, n_in, k=16, scale=3.0):
        super().__init__()
        self.lin = nn.Linear(n_in, k)
        nn.init.normal_(self.lin.weight, std=scale)          # spread of starting frequencies
        nn.init.uniform_(self.lin.bias, -math.pi, math.pi)   # random phases

    def forward(self, x):
        return torch.sin(self.lin(x))


class Polar(nn.Module):
    """Fixed features: x, y, r, and spiral waves sin/cos(k·r - θ) for a few k.
    The first two inputs are taken as the plane (x, y), so keep x, y first in features."""
    def __init__(self, n_in, ks=(1, 2, 4)):
        super().__init__()
        if n_in < 2:
            raise ValueError(f"polar spiral needs the first two inputs to be the plane (x, y); got {n_in} input")
        self.register_buffer("ks", torch.tensor(ks, dtype=torch.float32))

    def forward(self, x):
        a, b = x[:, :1], x[:, 1:2]  # first two inputs are the plane
        r, th = torch.sqrt(a * a + b * b), torch.atan2(b, a)
        phase = r * self.ks * math.pi - th
        return torch.cat([x, r, torch.sin(phase), torch.cos(phase)], -1)


class RBF(nn.Module):
    """exp(-|x - c_k|² / σ_k²): a learnable bump around each of k centers."""
    def __init__(self, n_in, k=16):
        super().__init__()
        self.centers = nn.Parameter(torch.empty(k, n_in).uniform_(-1, 1))
        self.log_sigma = nn.Parameter(torch.full((k,), math.log(0.4)))

    def forward(self, x):
        d2 = torch.cdist(x, self.centers) ** 2
        return torch.exp(-d2 / torch.exp(self.log_sigma) ** 2)


EXPANSIONS.register("fourier (sin)", Fourier)
EXPANSIONS.register("polar spiral", Polar)
EXPANSIONS.register("rbf bumps", RBF)


def make_expansion(name, n_in, scale=None, **knobs):
    """Build EXPANSIONS[name](n_in, ...) passing only the knobs its factory accepts
    (all of them to a **kwargs factory), so plain factory(n_in) lambdas keep working.
    scale (custom nn passes fourier_freq, by name or positionally) is one more knob,
    left out when None."""
    if scale is not None:
        knobs["scale"] = scale
    return call_accepting(EXPANSIONS.get(name), n_in, **knobs)
