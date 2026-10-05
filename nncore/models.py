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
import torch
import torch.nn as nn

from .activations import ACTIVATIONS
from .layers import LAYERS, SKIPS
from .registry import Registry

MODELS = Registry("model", "build(n_in, n_out, cfg) -> nn.Module")


_GRAMMAR = "expected WIDTH[:ACTIVATION] items separated by commas, e.g. '8:sin,4'"


def parse_layers(text, default_act):
    """'8:sin,4' -> [(8, 'sin'), (4, default_act)]. Empty text -> no hidden layers.
    Whitespace is ignored (an activation name may contain inner spaces, e.g. 'x·sin x')."""
    out = []
    for raw in str(text).split(","):
        tok = raw.strip()
        if not tok:
            continue
        w, _, a = tok.partition(":")
        w, a = "".join(w.split()), a.strip()
        if a and a not in ACTIVATIONS and "".join(a.split()) in ACTIVATIONS:
            a = "".join(a.split())  # '8:leaky _relu' parsed as before
        a = a or default_act
        try:
            w = int(w)
        except ValueError:
            raise ValueError(f"bad layer '{tok}' in '{text}': {_GRAMMAR}") from None
        if not 1 <= w <= 512:
            raise ValueError(f"layer width {w} in '{tok}' must be 1..512")
        ACTIVATIONS.get(a)  # raises a readable error for unknown names
        out.append((w, a))
    return out


def format_layers(spec, default_act=None):
    """[(8, 'sin'), (4, 'tanh')] -> '8:sin,4:tanh' (inverse of parse_layers); an activation
    equal to default_act is left out ('8:sin,4' with default_act='tanh')."""
    return ",".join(str(int(w)) if a == default_act else f"{int(w)}:{a}" for w, a in spec)


class MLP(nn.Module):
    def __init__(self, n_in, layers, n_out, layer="linear", skip="none"):
        super().__init__()
        make_layer, self.skip = LAYERS.get(layer), SKIPS.get(skip)
        self.spec = layers
        self.n_in, self.n_out = n_in, n_out
        self.lins = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.widths = []  # real width after each skip (concat widens)
        d = n_in
        for w, a in layers:
            self.lins.append(make_layer(d, w))
            self.acts.append(ACTIVATIONS.get(a)())
            d = self._probe(skip, w, d)
            self.widths.append(d)
        self.head = make_layer(d, n_out)

    def _probe(self, name, w, d):
        """Width skip(h, x) returns for h of width w and x of width d (zeros: no RNG use)."""
        try:
            with torch.no_grad():
                out = self.skip(torch.zeros(1, w), torch.zeros(1, d))
            return int(out.shape[-1])
        except Exception as e:
            raise ValueError(f"skip '{name}' cannot combine width {w} with {d} "
                             f"(the layer output with its input): {e}") from None

    @property
    def hidden_sizes(self):
        return list(self.widths)

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
