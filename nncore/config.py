"""
Config: every knob of an experiment in one plain dataclass.

String fields name an entry in a registry (see the comment beside each).
Add a field here when a new part needs a knob; models receive the whole
Config, so they can read it directly.

The *_KEYS sets tell Session.configure() how much to rebuild when a field
changes: new data > new model > new optimizer > new sampler > nothing.
"""
from dataclasses import asdict, dataclass, field, fields


@dataclass
class Config:
    # data
    dataset: str = "Spiral (5 arms)"      # DATASETS
    n_points: int = 600
    noise: float = 0.05
    splitter: str = "random"              # SPLITTERS
    test_frac: float = 0.2
    seed: int = 0                         # data, split, init and batch order

    # model
    features: tuple = ("x", "y")          # FEATURES (one or more)
    model: str = "custom nn"              # MODELS (custom nn lives in my_parts.py)
    width: int = 8                        # width
    classes: int = 5                      # custom nn: width of the head/decide/perc stage
    expand: str = "fourier (sin)"         # EXPANSIONS (custom nn: widens x,y before hid)
    fourier_freq: float = 3.0             # custom nn: spread of starting frequencies for "fourier (sin)"
    layers: str = "8,8"                   # mlp: "4,4" or "8:sin,4:tanh", empty = linear
    activation: str = "softplus"              # ACTIVATIONS (default for layers without :act)
    act_decide: str = "gauss"             # custom nn: ACTIVATIONS for decide, "same" = activation
    act_relate: str = "same"              # custom nn: ACTIVATIONS for relate
    act_prepare: str = "linear"            # custom nn: ACTIVATIONS for prepare
    layer: str = "linear"                 # LAYERS
    skip: str = "none"                    # SKIPS
    init: str = "pytorch default"         # INITIALIZERS

    # optimisation
    loss: str = "cross entropy"           # LOSSES
    optimizer: str = "Adam"               # OPTIMIZERS
    lr: float = 0.03
    weight_decay: float = 1e-5
    schedule: str = "step (÷10 every 2000)"  # SCHEDULES
    train_step: str = "standard"          # TRAIN_STEPS

    # batching
    sampler: str = "random (with replacement)"  # SAMPLERS
    batch_size: int | None = 32           # None = full batch

    extra: dict = field(default_factory=dict)  # free-form knobs for your own parts

    def to_dict(self):
        return asdict(self)


DATA_KEYS = {"dataset", "n_points", "noise", "splitter", "test_frac", "seed"}
MODEL_KEYS = {"features", "model", "layers", "activation", "act_decide", "act_relate", "act_prepare", "width", "classes", "expand", "fourier_freq", "layer", "skip", "init", "extra"}
OPT_KEYS = {"optimizer", "lr", "weight_decay", "schedule"}
SAMPLER_KEYS = {"sampler", "batch_size"}
LIVE_KEYS = {"loss", "train_step"}  # looked up every step, take effect immediately

_all = DATA_KEYS | MODEL_KEYS | OPT_KEYS | SAMPLER_KEYS | LIVE_KEYS
assert _all == {f.name for f in fields(Config)}, "every Config field needs a rebuild level"
