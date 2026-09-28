"""
Compartment: MODELS  (architectures)

Contract:  build(n_in, n_out, cfg) -> nn.Module with
  forward(x, collect=False) -> logits (N, n_out)
                               or (logits, [hidden (N, w_i), ...]) when collect=True
  hidden_sizes              -> list[int], one entry per hidden tensor returned
  describe()                -> short string such as "2 → 8tanh → 8tanh → 2"

`cfg` is the whole Config, so an architecture can read whatever knobs it needs
(cfg.layers, cfg.activation, cfg.layer, cfg.skip, or new ones you add).
"""
import torch.nn as nn

from .activations import ACTIVATIONS
from .layers import LAYERS, SKIPS
from .registry import Registry

MODELS = Registry("model", "build(n_in, n_out, cfg) -> nn.Module")


def parse_layers(text, default_act):
    """'8:sin,4' -> [(8, 'sin'), (4, default_act)]. Empty text -> no hidden layers."""
    out = []
    for tok in text.replace(" ", "").split(","):
        if not tok:
            continue
        w, _, a = tok.partition(":")
        a = a or default_act
        ACTIVATIONS.get(a)  # raises a readable error for unknown names
        w = int(w)
        if not 1 <= w <= 512:
            raise ValueError("layer width must be 1..512")
        out.append((w, a))
    return out


class MLP(nn.Module):
    def __init__(self, n_in, layers, n_out, layer="linear", skip="none"):
        super().__init__()
        make_layer, self.skip = LAYERS.get(layer), SKIPS.get(skip)
        self.spec = layers
        self.n_in, self.n_out = n_in, n_out
        self.lins = nn.ModuleList()
        self.acts = nn.ModuleList()
        d = n_in
        for w, a in layers:
            self.lins.append(make_layer(d, w))
            self.acts.append(ACTIVATIONS.get(a)())
            d = w
        self.head = make_layer(d, n_out)

    @property
    def hidden_sizes(self):
        return [w for w, _ in self.spec]

    def describe(self):
        return " → ".join([str(self.n_in)] + [f"{w}{a[:4]}" for w, a in self.spec]
                          + [str(self.n_out)])

    def forward(self, x, collect=False):
        hidden = []
        for lin, act in zip(self.lins, self.acts):
            x = self.skip(act(lin(x)), x)
            if collect:
                hidden.append(x)
        out = self.head(x)
        return (out, hidden) if collect else out


@MODELS.register("mlp")
def build_mlp(n_in, n_out, cfg):
    return MLP(n_in, parse_layers(cfg.layers, cfg.activation), n_out, cfg.layer, cfg.skip)
