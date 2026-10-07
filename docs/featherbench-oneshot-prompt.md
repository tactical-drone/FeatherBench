<!-- Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md. -->

# FeatherBench One-Shot (for bridgebench.ai/test-prompts and model benchmarks)

A test any model can take in one reply, with no tools and no repo: it designs a tiny neural
net as one JSON object, and a deterministic scorer turns that reply into a number from 0
to 1000. Same reply, same score, on any machine with the same code, so it can be re-run
on every model and every model update (NerfBench-style tracking over time).

**Grading a reply** (anyone, about 1.5 minutes on 5 CPU cores):

    git clone https://github.com/tactical-drone/FeatherBench && cd FeatherBench
    pip install -r requirements.txt
    python -m rsi featherbench score-answer reply.txt --workers 5 --answer-png maps.png

It prints `feather_score` (0-1000), patterns solved, params_max and per-pattern accuracy,
and draws the 12 decision maps. A reply without a valid setup scores 0 and says why.
Scores: below 923 = not every pattern solved (more solved and higher accuracy score
higher); 923-1000 = all 12 solved, fewer parameters score higher.

**Prompt name:** Recursive Super Intelligence Featherweight Championship, One-Shot: design the smallest neural net that solves 12 patterns (auto-scored 0-1000)

**Linked repository:** https://github.com/tactical-drone/FeatherBench

**Prompt content** (everything between the lines):

---

RSI (Recursive Super Intelligence) Featherweight Championship: an AI designing a smarter AI.
Your answer is scored the same way every time, so it shows how well your model improves
intelligence, and it's re-run over time. Nerds vs AI: humans will try to beat it by hand.
Design a tiny neural network, in one reply, that learns 12 different 2-D patterns. You
can't run anything: reason about what each pattern needs, then commit to one setup.

The patterns (each a classification of points in the square [-1.2, 1.2]²): Bullseye (5
rings), Sectors (8 wedges), Egg crate (3), Checkerboard (4x4), Hex tiling (3 classes,
coarse), Random lines (parity), Voronoi (6 cells), Spiral (3 arms), Blobs + tiny cluster,
Mandelbrot set, Yin-yang, Smiley. Each is trained from scratch with your setup: 600 points
(80 % train), noise 0.05, 3000 training steps, 3 random seeds. A pattern counts as solved
when the mean accuracy over the 3 seeds, on 2000 fresh points, is at least 90 %.

Score: solve as many patterns as you can; once all 12 are solved, fewer parameters is
better (params_max = the parameter count on the dataset with the most classes). At most
20,000 parameters. The default setup below solves 9-10 of the 12.

Your setup is a JSON object. Any field you leave out keeps its default. Two models:

"custom nn" (the default). Dataflow, one linear layer per arrow:
  x -> expand(x) -> hid (width, + activation) -> head (classes) -> decide (classes,
  + act_decide) -> perc (classes) -> relate (2, + act_relate) -> prepare (classes,
  + act_prepare) -> output
  Fields: width (int), classes (int), expand, fourier_freq (float, the frequency spread
  for "fourier (sin)"), activation, act_decide / act_relate / act_prepare (an activation,
  or "same" = activation), skip (wraps perc), skip_decide / skip_relate / skip_prepare
  (wrap those stages; "same" = skip where the widths allow).

"mlp": a plain stack. Fields: layers ("16,16" or per layer "16:sin,8:tanh"; "" = linear),
  activation (for layers without one), layer, skip (between layers).

Shared fields: features (input columns, in order), init, optimizer, lr, weight_decay,
schedule, sampler, batch_size (int or null for full batch), loss, train_step.

Allowed values:
- features: "x", "y", "x²", "y²", "x·y", "sin x", "sin y", "r", "cos x", "cos y", "θ (atan2)"
  (ASCII spellings work too: "x^2", "x*y", "theta")
- expand: "none", "linear", "linear + tanh", "fourier (sin)", "polar spiral", "rbf bumps"
- activation (and act_*): "tanh", "relu", "leaky_relu", "sigmoid", "gelu", "silu", "mish",
  "elu", "softplus", "sin", "gauss", "abs", "square", "linear", "snake", "cos", "selu",
  "softsign", "hardtanh", "prelu", "x·sin x"
- skip (and skip_*): "none", "residual (same width)", "half residual", "add", "concat"
- layer: "linear", "linear (no bias)"
- init: "pytorch default", "xavier uniform", "kaiming normal", "orthogonal", "xavier normal",
  "lecun normal", "xavier uniform (keep expansion)", "kaiming normal (keep expansion)"
- optimizer: "Adam", "AdamW", "SGD", "SGD+momentum", "SGD+nesterov", "RMSprop", "NAdam",
  "RAdam", "Adamax"
- schedule: "constant", "exp decay (×0.999/step)", "step (÷10 every 2000)",
  "cosine warm restarts (T0=1000)", "warmup 200 + constant"
- sampler: "random (with replacement)", "epoch shuffle"
- loss: "cross entropy", "cross entropy (smoothed 0.1)", "mse on softmax", "focal (γ=2)"
- train_step: "standard", "clip grad norm 1.0", "skip non-finite"
- ranges: width 1-512, classes 1-512, lr 1e-6 to 100, weight_decay 0-1, batch_size 1-100000

The default setup:
{"model": "custom nn", "width": 8, "classes": 5, "expand": "fourier (sin)",
 "fourier_freq": 3.0, "activation": "softplus", "act_decide": "gauss", "act_relate": "same",
 "act_prepare": "linear", "skip": "none", "skip_decide": "residual (same width)",
 "skip_relate": "same", "skip_prepare": "same", "features": ["x", "y"], "init": "pytorch default",
 "optimizer": "Adam", "lr": 0.03, "weight_decay": 0.00001, "schedule": "step (÷10 every 2000)",
 "sampler": "random (with replacement)", "batch_size": 32, "loss": "cross entropy",
 "train_step": "standard"}

Think about what makes each pattern hard (high frequency, angles, parity, a tiny minority
class, fractal edges) and what in the network can represent it, then decide. Explain your
reasoning briefly, then end your reply with exactly one ```json code block holding your
setup. Only the last JSON block is scored.

---
