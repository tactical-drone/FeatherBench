# NN Playground

Tiny neural nets trained live: decision boundary, per-neuron maps and loss curve,
redrawn every frame. Every piece of the net is a swappable part.

## Install and run (Windows 11)

    pip install torch numpy pyqtgraph PyQt6
    python nn_playground.py              # the live window (--opengl if frames feel heavy)
    python headless.py --list            # every registered part
    python headless.py --compare activation=tanh,relu,gelu --steps 3000

## Layout

    nncore/          the neural net. No Qt, no plotting, no timers.
      registry.py      Registry: a named slot of interchangeable parts
      config.py        Config dataclass: one field per knob / part choice
      session.py       Session: wires the parts together, trains, evaluates
      datasets.py  features.py  models.py  layers.py  activations.py
      initializers.py  losses.py  optimizers.py  training.py  samplers.py  metrics.py
    ui/              the window: widgets -> Session.configure(), paint results
      app.py  painters.py  theme.py
    my_parts.py      your workbench (examples inside, commented out)
    headless.py      train from the command line, A/B any part
    nn_playground.py launcher

## Compartments

| Config field | Registry | Contract | File |
|---|---|---|---|
| `dataset` | `DATASETS` | `fn(n, noise, rng) -> (X (n,2), y (n,))` | datasets.py |
| `splitter` | `SPLITTERS` | `fn(n, test_frac, rng) -> (train_idx, test_idx)` | datasets.py |
| `features` | `FEATURES` | `fn(x, y) -> tensor` | features.py |
| `model` | `MODELS` | `build(n_in, n_out, cfg) -> nn.Module` | models.py |
| `layer` | `LAYERS` | `factory(d_in, d_out) -> nn.Module` | layers.py |
| `skip` | `SKIPS` | `fn(h, x) -> tensor` | layers.py |
| `activation` | `ACTIVATIONS` | `factory() -> nn.Module` | activations.py |
| `init` | `INITIALIZERS` | `fn(model) -> None` | initializers.py |
| `loss` | `LOSSES` | `fn(logits, y) -> scalar` | losses.py |
| `optimizer` | `OPTIMIZERS` | `fn(params, lr, weight_decay) -> Optimizer` | optimizers.py |
| `schedule` | `SCHEDULES` | `fn(optimizer) -> scheduler or None` | optimizers.py |
| `train_step` | `TRAIN_STEPS` | `fn(model, opt, loss_fn, xb, yb) -> float` | training.py |
| `sampler` | `SAMPLERS` | `factory(n, batch_size) -> next_batch()` | samplers.py |
| (all) | `METRICS` | `fn(logits, y) -> float` | metrics.py |

The UI adds two of its own: `BOUNDARY_PAINTERS` and `NEURON_PAINTERS` in ui/painters.py.

A model must provide `forward(x, collect=False)` (returning `(logits, [hidden...])`
when `collect` is set), a `hidden_sizes` list and `describe()`. See `MLP` in models.py.

### Adding a part

Register it in `my_parts.py`; it appears in the UI dropdown and in `headless.py`:

    from nncore import ACTIVATIONS
    ACTIVATIONS.register("snake", Snake)

A part that needs its own knob can read `cfg.extra["..."]`, or you can add a field
to `Config` and put its name in one of the `*_KEYS` sets, which say what to rebuild
when it changes.

### Using the core from code

    from nncore import Config, Session
    s = Session(Config(dataset="Moons", layers="4:sin", lr=0.01))
    s.train(1000)
    print(s.evaluate())            # {'train': {'loss':.., 'acc':..}, 'test': {...}}
    s.configure(optimizer="SGD")   # rebuilds only what the change needs

## Controls

- **Hidden layers**: `4,4` = two layers of 4. `8:sin,4:tanh` sets per-layer activations. Empty = linear model.
- **Inputs**: extra features (x², y², x·y, sin, r) like TF Playground. Checking x² and y² solves Circles with zero hidden neurons.
- **Skip**: `residual (same width)` adds skip connections between same-width layers.
- **Steps / frame**: training speed. Frames render as fast as the display allows.
- **Grid res**: boundary resolution (128 default; 256 is sharper, costs ~4x render).
- **Neuron view**: which hidden layer's neurons to show (blue = positive, orange = negative, up to 32).
- Keys: Space play/pause, S step, R reset weights, N new data.

## Challenges

- XOR gate (4 points): 2 tanh neurons solve it (most seeds). 1 neuron works too: `1:square` every seed I tried, `1:abs` only on some. Why?
- Spiral (2 arms) with x, y only: find the smallest net that hits 100%. Try `sin`.
- Same spiral, residual on, `16,16,16,16` relu vs gelu.
