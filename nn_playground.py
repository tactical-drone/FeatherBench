"""
NN Playground // tiny neural nets, trained live.

Pick a dataset, type a network shape, pick an activation, hit Play and watch
the decision boundary, per-neuron maps and loss curve evolve at display rate.

  Train:  PyTorch on CPU (tiny MLPs are faster on CPU than on a GPU: kernel
          launch overhead dwarfs the math).
  Draw:   pyqtgraph (Qt scene graph, textures blitted by the GPU compositor),
          no matplotlib redraws. Boundary is evaluated on a grid every frame.

Layer syntax (the "Hidden layers" box):
  4,4           two hidden layers of 4 neurons, global activation
  8:sin,4:tanh  per-layer activation override
  (empty)       no hidden layer: pure linear / logistic regression

Keys: Space play/pause, S single step, R reset weights, N new data.

Run:  python nn_playground.py        (add --opengl to try the GL viewport)
"""
import argparse
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

torch.set_num_threads(1)  # tiny nets: threading overhead > work

VIEW = 1.6  # half-width of the plotted plane
PLANE = (-VIEW, -VIEW, 2 * VIEW, 2 * VIEW)  # x, y, w, h handed to ImageItem.setImage
BG = np.array([11, 12, 16], dtype=np.float32)
CLASS_COLORS = np.array([
    [245, 147, 34],   # orange
    [8, 150, 230],    # blue
    [255, 42, 109],   # magenta
    [0, 240, 180],    # teal
    [245, 230, 99],   # yellow
], dtype=np.float32)
NEG = np.array([245, 147, 34], dtype=np.float32)
POS = np.array([8, 150, 230], dtype=np.float32)
BG_T, CLASS_COLORS_T = torch.from_numpy(BG), torch.from_numpy(CLASS_COLORS)
# diverging LUT for neuron maps: -1 orange .. 0 background .. +1 blue
_t = np.linspace(-1, 1, 256, dtype=np.float32)[:, None]
NEURON_LUT = np.where(_t > 0, BG + (POS - BG) * _t, BG + (NEG - BG) * -_t).astype(np.uint8)

# --------------------------------------------------------------------------- data


def make_dataset(name, n, noise, rng):
    """Returns X (n,2) float32 in roughly [-1.2, 1.2], y (n,) int64."""
    if name == "XOR gate (4 points)":
        X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float32)
        y = np.array([0, 1, 1, 0])
        X = X + rng.normal(0, noise * 0.3, X.shape)
        return X.astype(np.float32), y.astype(np.int64)
    if name == "XOR quadrants":
        X = rng.uniform(-1.2, 1.2, (n, 2))
        X += np.sign(X) * 0.05
        y = (X[:, 0] * X[:, 1] < 0).astype(int)
        X += rng.normal(0, noise, X.shape)
    elif name.startswith("Spiral"):
        k = 2 if "2" in name else 3
        per = n // k
        Xs, ys = [], []
        for c in range(k):
            r = np.linspace(0.05, 1.0, per)
            t = r * 3.2 * math.pi + c * 2 * math.pi / k
            t = t + rng.normal(0, noise * 1.5, per)
            Xs.append(np.stack([r * np.sin(t), r * np.cos(t)], 1) * 1.15)
            ys.append(np.full(per, c))
        X, y = np.concatenate(Xs), np.concatenate(ys)
    elif name == "Circles":
        r = np.concatenate([rng.uniform(0, 0.45, n // 2), rng.uniform(0.75, 1.15, n - n // 2)])
        t = rng.uniform(0, 2 * math.pi, n)
        X = np.stack([r * np.cos(t), r * np.sin(t)], 1) + rng.normal(0, noise, (n, 2))
        y = np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)])
    elif name == "Moons":
        m = n // 2
        t1, t2 = rng.uniform(0, math.pi, m), rng.uniform(0, math.pi, n - m)
        a = np.stack([np.cos(t1), np.sin(t1)], 1)
        b = np.stack([1 - np.cos(t2), 0.5 - np.sin(t2)], 1)
        X = (np.concatenate([a, b]) - [0.5, 0.25]) * 0.8 + rng.normal(0, noise, (n, 2))
        y = np.concatenate([np.zeros(m), np.ones(n - m)])
    elif name == "Checkerboard":
        X = rng.uniform(-1.2, 1.2, (n, 2))
        y = ((np.floor((X[:, 0] + 1.2) / 0.6) + np.floor((X[:, 1] + 1.2) / 0.6)) % 2).astype(int)
        X += rng.normal(0, noise * 0.5, X.shape)
    elif name == "Gaussian blobs (5)":
        k = 5
        ang = np.arange(k) * 2 * math.pi / k
        centers = np.stack([np.cos(ang), np.sin(ang)], 1) * 0.8
        y = rng.integers(0, k, n)
        X = centers[y] + rng.normal(0, 0.12 + noise, (n, 2))
    else:
        raise ValueError(name)
    return X.astype(np.float32), y.astype(np.int64)


DATASETS = ["XOR gate (4 points)", "XOR quadrants", "Spiral (2 arms)", "Spiral (3 arms)",
            "Circles", "Moons", "Checkerboard", "Gaussian blobs (5)"]

FEATURES = {
    "x": lambda x, y: x, "y": lambda x, y: y,
    "x²": lambda x, y: x * x, "y²": lambda x, y: y * y, "x·y": lambda x, y: x * y,
    "sin x": lambda x, y: torch.sin(x * 3), "sin y": lambda x, y: torch.sin(y * 3),
    "r": lambda x, y: torch.sqrt(x * x + y * y + 1e-9),
}

# --------------------------------------------------------------------------- model


class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class Gauss(nn.Module):
    def forward(self, x):
        return torch.exp(-x * x)


class Abs(nn.Module):
    def forward(self, x):
        return x.abs()


class Square(nn.Module):
    def forward(self, x):
        return x * x


ACTIVATIONS = {
    "tanh": nn.Tanh, "relu": nn.ReLU, "leaky_relu": lambda: nn.LeakyReLU(0.1),
    "sigmoid": nn.Sigmoid, "gelu": nn.GELU, "silu": nn.SiLU, "mish": nn.Mish,
    "elu": nn.ELU, "softplus": nn.Softplus, "sin": Sine, "gauss": Gauss,
    "abs": Abs, "square": Square, "linear": nn.Identity,
}


def parse_layers(text, default_act):
    """'8:sin,4' -> [(8,'sin'), (4, default_act)]"""
    out = []
    for tok in text.replace(" ", "").split(","):
        if not tok:
            continue
        w, _, a = tok.partition(":")
        a = a or default_act
        if a not in ACTIVATIONS:
            raise ValueError(f"unknown activation '{a}' (have: {', '.join(ACTIVATIONS)})")
        w = int(w)
        if not 1 <= w <= 512:
            raise ValueError("layer width must be 1..512")
        out.append((w, a))
    return out


class MLP(nn.Module):
    def __init__(self, n_in, layers, n_out, residual=False):
        super().__init__()
        self.lins = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.residual = residual
        d = n_in
        for w, a in layers:
            self.lins.append(nn.Linear(d, w))
            self.acts.append(ACTIVATIONS[a]())
            d = w
        self.head = nn.Linear(d, n_out)

    def forward(self, x, collect=False):
        hidden = []
        for lin, act in zip(self.lins, self.acts):
            h = act(lin(x))
            if self.residual and h.shape[-1] == x.shape[-1]:
                h = h + x
            x = h
            if collect:
                hidden.append(h)
        out = self.head(x)
        return (out, hidden) if collect else out

# --------------------------------------------------------------------------- UI


class GrowBuf:
    def __init__(self):
        self.a = np.zeros(4096, np.float32)
        self.n = 0

    def add(self, v):
        if self.n == len(self.a):
            self.a = np.concatenate([self.a, np.zeros_like(self.a)])
        self.a[self.n] = v
        self.n += 1

    def view(self):
        return self.a[: self.n]

    def clear(self):
        self.n = 0


class Playground(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NN Playground // Jarvis build")
        self.resize(1500, 900)
        self.rng = np.random.default_rng(0)
        self.running = False
        self.step_count = 0
        self.frame_times = []
        self.last_frame = time.perf_counter()
        self.train_hist, self.test_hist, self.hist_x = GrowBuf(), GrowBuf(), GrowBuf()
        self.thumbs = []

        self._build_ui()
        self.new_data()

        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.tick)
        self.timer.start(0)  # as fast as the event loop and vsync allow

    # ---------------------------------------------------------------- layout
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        panel = QtWidgets.QWidget()
        panel.setFixedWidth(300)
        form = QtWidgets.QFormLayout(panel)
        root.addWidget(panel)

        def combo(items, cur=None, cb=None, editable=False):
            c = QtWidgets.QComboBox()
            c.addItems([str(i) for i in items])
            c.setEditable(editable)
            if cur is not None:
                c.setCurrentText(str(cur))
            if cb:
                c.currentTextChanged.connect(cb)
            return c

        self.cb_data = combo(DATASETS, "Spiral (2 arms)", lambda _: self.new_data())
        form.addRow("Dataset", self.cb_data)
        self.sp_n = QtWidgets.QSpinBox()
        self.sp_n.setRange(20, 5000)
        self.sp_n.setValue(600)
        self.sp_n.setSingleStep(100)
        self.sp_n.editingFinished.connect(self.new_data)
        form.addRow("Points", self.sp_n)
        self.sl_noise = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_noise.setRange(0, 50)
        self.sl_noise.setValue(5)
        self.sl_noise.sliderReleased.connect(self.new_data)
        form.addRow("Noise", self.sl_noise)

        feat_box = QtWidgets.QWidget()
        fl = QtWidgets.QGridLayout(feat_box)
        fl.setContentsMargins(0, 0, 0, 0)
        self.feat_checks = {}
        for i, f in enumerate(FEATURES):
            ch = QtWidgets.QCheckBox(f)
            ch.setChecked(f in ("x", "y"))
            ch.toggled.connect(self.reset_model)
            fl.addWidget(ch, i // 4, i % 4)
            self.feat_checks[f] = ch
        form.addRow("Inputs", feat_box)

        self.le_layers = QtWidgets.QLineEdit("8,8")
        self.le_layers.setToolTip("e.g. 4,4   or   8:sin,4:tanh   or empty for linear")
        self.le_layers.editingFinished.connect(self.reset_model)
        form.addRow("Hidden layers", self.le_layers)
        self.cb_act = combo(ACTIVATIONS, "tanh", lambda _: self.reset_model())
        form.addRow("Activation", self.cb_act)
        self.ch_res = QtWidgets.QCheckBox("skip connections (same width)")
        self.ch_res.toggled.connect(self.reset_model)
        form.addRow("Residual", self.ch_res)

        self.cb_opt = combo(["Adam", "AdamW", "SGD", "SGD+momentum", "RMSprop"], "Adam",
                            lambda _: self.reset_opt())
        form.addRow("Optimizer", self.cb_opt)
        self.cb_lr = combo([0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3], 0.03,
                           lambda _: self.reset_opt(), editable=True)
        form.addRow("Learning rate", self.cb_lr)
        self.cb_wd = combo([0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1], 0, lambda _: self.reset_opt())
        form.addRow("L2 / wd", self.cb_wd)
        self.cb_batch = combo(["full", 8, 16, 32, 64, 128], 32)
        form.addRow("Batch", self.cb_batch)
        self.sp_steps = QtWidgets.QSpinBox()
        self.sp_steps.setRange(1, 1000)
        self.sp_steps.setValue(5)
        form.addRow("Steps / frame", self.sp_steps)
        self.cb_res = combo([64, 96, 128, 192, 256], 128,
                            lambda _: self._make_grid(), editable=True)
        form.addRow("Grid res", self.cb_res)
        self.cb_layer_view = combo(["layer 1"], "layer 1", lambda _: self._rebuild_thumbs())
        form.addRow("Neuron view", self.cb_layer_view)
        self.sp_seed = QtWidgets.QSpinBox()
        self.sp_seed.setRange(0, 99999)
        form.addRow("Seed", self.sp_seed)

        btns = QtWidgets.QGridLayout()
        self.bt_play = QtWidgets.QPushButton("▶ Play  [Space]")
        self.bt_play.clicked.connect(self.toggle)
        bt_step = QtWidgets.QPushButton("Step [S]")
        bt_step.clicked.connect(self.single_step)
        bt_reset = QtWidgets.QPushButton("Reset weights [R]")
        bt_reset.clicked.connect(self.reset_model)
        bt_data = QtWidgets.QPushButton("New data [N]")
        bt_data.clicked.connect(lambda: self.new_data(reseed=True))
        btns.addWidget(self.bt_play, 0, 0, 1, 2)
        btns.addWidget(bt_step, 1, 0)
        btns.addWidget(bt_reset, 1, 1)
        btns.addWidget(bt_data, 2, 0, 1, 2)
        form.addRow(btns)

        self.lb_stats = QtWidgets.QLabel()
        self.lb_stats.setStyleSheet("font-family: Consolas, monospace; color: #7fffd4;")
        self.lb_stats.setWordWrap(True)
        form.addRow(self.lb_stats)

        for key, fn in (("Space", self.toggle), ("S", self.single_step),
                        ("R", self.reset_model), ("N", lambda: self.new_data(reseed=True))):
            QtGui.QShortcut(QtGui.QKeySequence(key), self, activated=fn)

        # right side: boundary | (neurons over loss)
        self.gl = pg.GraphicsLayoutWidget()
        root.addWidget(self.gl, 1)
        self.p_main = self.gl.addPlot(row=0, col=0, rowspan=2, title="decision boundary")
        self.p_main.setAspectLocked(True)
        self.p_main.setRange(xRange=(-VIEW, VIEW), yRange=(-VIEW, VIEW), padding=0)
        self.p_main.hideButtons()
        self.p_main.hideAxis("left")  # axis text is the priciest thing to repaint
        self.p_main.hideAxis("bottom")
        self.img = pg.ImageItem()
        self.p_main.addItem(self.img)
        self.sc_train = pg.ScatterPlotItem(size=7, pen=pg.mkPen((255, 255, 255, 200), width=1))
        self.sc_test = pg.ScatterPlotItem(size=7, pen=pg.mkPen((0, 0, 0, 220), width=1.5))
        self.p_main.addItem(self.sc_train)
        self.p_main.addItem(self.sc_test)

        self.gl_neurons = self.gl.addLayout(row=0, col=1)
        self.p_loss = self.gl.addPlot(row=1, col=1, title="loss (train cyan, test magenta)")
        self.p_loss.setLogMode(y=True)
        self.p_loss.showGrid(x=True, y=True, alpha=0.2)
        self.p_loss.setClipToView(True)
        self.p_loss.setDownsampling(auto=True, mode="peak")
        self.c_train = self.p_loss.plot(pen=pg.mkPen("#00f0ff", width=2))
        self.c_test = self.p_loss.plot(pen=pg.mkPen("#ff2a6d", width=2))
        self.gl.ci.layout.setColumnStretchFactor(0, 3)
        self.gl.ci.layout.setColumnStretchFactor(1, 2)

    # ---------------------------------------------------------------- state
    def _make_grid(self):
        try:
            r = int(self.cb_res.currentText())
        except ValueError:
            return
        r = max(16, min(r, 1024))
        xs = torch.linspace(-VIEW, VIEW, r)
        yy, xx = torch.meshgrid(xs, xs, indexing="ij")  # row = y, col = x
        self.grid_res = r
        self.grid_raw = torch.stack([xx.reshape(-1), yy.reshape(-1)], 1)
        self.grid_feat = self.featurize(self.grid_raw) if hasattr(self, "feats") else None

    def featurize(self, P):
        x, y = P[:, 0], P[:, 1]
        return torch.stack([FEATURES[f](x, y) for f in self.feats], 1)

    def new_data(self, reseed=False):
        if reseed:
            self.sp_seed.setValue(int(self.rng.integers(0, 99999)))
        rng = np.random.default_rng(self.sp_seed.value())
        X, y = make_dataset(self.cb_data.currentText(), self.sp_n.value(),
                            self.sl_noise.value() / 100, rng)
        idx = rng.permutation(len(X))
        n_test = 0 if len(X) <= 8 else len(X) // 5
        te, tr = idx[:n_test], idx[n_test:]
        self.Xtr_raw, self.ytr = torch.from_numpy(X[tr]), torch.from_numpy(y[tr])
        self.Xte_raw, self.yte = torch.from_numpy(X[te]), torch.from_numpy(y[te])
        self.n_classes = int(y.max()) + 1
        brushes = [pg.mkBrush(*CLASS_COLORS[c]) for c in range(self.n_classes)]
        self.sc_train.setData(X[tr, 0], X[tr, 1], brush=[brushes[c] for c in y[tr]],
                              size=14 if len(X) <= 8 else 7)
        self.sc_test.setData(X[te, 0], X[te, 1], brush=[brushes[c] for c in y[te]])
        self.reset_model()

    def reset_model(self):
        self.feats = [f for f, ch in self.feat_checks.items() if ch.isChecked()] or ["x", "y"]
        try:
            layers = parse_layers(self.le_layers.text(), self.cb_act.currentText())
            self.le_layers.setStyleSheet("")
        except ValueError as e:
            self.le_layers.setStyleSheet("background: #501020;")
            self.le_layers.setToolTip(str(e))
            return
        self.layers = layers
        torch.manual_seed(self.sp_seed.value())
        self.model = MLP(len(self.feats), layers, self.n_classes, self.ch_res.isChecked())
        self.Xtr = self.featurize(self.Xtr_raw)
        self.Xte = self.featurize(self.Xte_raw) if len(self.Xte_raw) else None
        self._make_grid()
        self.reset_opt()
        self.step_count = 0
        for b in (self.train_hist, self.test_hist, self.hist_x):
            b.clear()
        cur = self.cb_layer_view.currentText()
        self.cb_layer_view.blockSignals(True)
        self.cb_layer_view.clear()
        self.cb_layer_view.addItems([f"layer {i + 1}" for i in range(len(layers))] or ["(none)"])
        self.cb_layer_view.setCurrentText(cur)
        self.cb_layer_view.blockSignals(False)
        self._rebuild_thumbs()
        self.render()

    def reset_opt(self):
        if not hasattr(self, "model"):
            return
        try:
            lr = float(self.cb_lr.currentText())
        except ValueError:
            return
        wd = float(self.cb_wd.currentText())
        p = self.model.parameters()
        o = self.cb_opt.currentText()
        self.opt = {
            "Adam": lambda: torch.optim.Adam(p, lr, weight_decay=wd),
            "AdamW": lambda: torch.optim.AdamW(p, lr, weight_decay=wd),
            "SGD": lambda: torch.optim.SGD(p, lr, weight_decay=wd),
            "SGD+momentum": lambda: torch.optim.SGD(p, lr, momentum=0.9, weight_decay=wd),
            "RMSprop": lambda: torch.optim.RMSprop(p, lr, weight_decay=wd),
        }[o]()

    def _rebuild_thumbs(self):
        self.gl_neurons.clear()
        self.thumbs = []
        txt = self.cb_layer_view.currentText()
        if not self.layers or not txt.startswith("layer"):
            self.view_layer = None
            return
        self.view_layer = min(int(txt.split()[1]) - 1, len(self.layers) - 1)
        n = min(self.layers[self.view_layer][0], 32)
        cols = max(1, math.ceil(math.sqrt(n * 1.5)))
        for i in range(n):
            vb = self.gl_neurons.addViewBox(row=i // cols, col=i % cols, lockAspect=True,
                                            enableMouse=False)
            im = pg.ImageItem(lut=NEURON_LUT)
            vb.addItem(im)
            vb.setRange(xRange=(-VIEW, VIEW), yRange=(-VIEW, VIEW), padding=0)
            self.thumbs.append(im)

    # ---------------------------------------------------------------- loop
    def toggle(self):
        self.running = not self.running
        self.bt_play.setText("⏸ Pause  [Space]" if self.running else "▶ Play  [Space]")

    def single_step(self):
        self.train(1)
        self.render()

    def train(self, steps):
        m, opt, X, y = self.model, self.opt, self.Xtr, self.ytr
        bs = self.cb_batch.currentText()
        bs = len(X) if bs == "full" else min(int(bs), len(X))
        for _ in range(steps):
            if bs < len(X):
                idx = torch.randint(0, len(X), (bs,))
                xb, yb = X[idx], y[idx]
            else:
                xb, yb = X, y
            loss = F.cross_entropy(m(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            self.step_count += 1

    def tick(self):
        now = time.perf_counter()
        self.frame_times.append(now - self.last_frame)
        self.last_frame = now
        if self.running:
            t0 = time.perf_counter()
            self.train(self.sp_steps.value())
            self.t_train = time.perf_counter() - t0
            self.render()

    @torch.no_grad()
    def render(self):
        t0 = time.perf_counter()
        m = self.model
        tr_logits = m(self.Xtr)
        tr_loss = F.cross_entropy(tr_logits, self.ytr).item()
        tr_acc = (tr_logits.argmax(1) == self.ytr).float().mean().item()
        te_loss = te_acc = float("nan")
        if self.Xte is not None:
            te_logits = m(self.Xte)
            te_loss = F.cross_entropy(te_logits, self.yte).item()
            te_acc = (te_logits.argmax(1) == self.yte).float().mean().item()
        self.hist_x.add(self.step_count)
        self.train_hist.add(max(tr_loss, 1e-8))
        self.test_hist.add(max(te_loss, 1e-8) if te_loss == te_loss else np.nan)

        logits, hidden = m(self.grid_feat, collect=True)
        r = self.grid_res
        probs = torch.softmax(logits, 1)
        K = probs.shape[1]
        conf = ((probs.amax(1, keepdim=True) - 1 / K) / (1 - 1 / K)).sqrt_()
        rgb = BG_T + (probs @ CLASS_COLORS_T[:K] - BG_T) * (0.25 + 0.6 * conf)
        rgb = rgb.view(r, r, 3)
        cls = probs.argmax(1).view(r, r)
        edge = torch.zeros((r, r), dtype=torch.bool)
        edge[:, 1:] |= cls[:, 1:] != cls[:, :-1]
        edge[1:, :] |= cls[1:, :] != cls[:-1, :]
        rgb[edge] = 235.0
        self.img.setImage(rgb.to(torch.uint8).numpy(), autoLevels=False, levels=(0, 255),
                              rect=PLANE)

        if self.view_layer is not None and self.thumbs:
            H = hidden[self.view_layer][:, : len(self.thumbs)]
            H = H / H.abs().amax(0).clamp_min(1e-9)  # each neuron scaled to [-1, 1]
            idx = ((H + 1) * 127.5).to(torch.uint8).T.contiguous().numpy()
            for i, im in enumerate(self.thumbs):
                im.setImage(idx[i].reshape(r, r), autoLevels=False, levels=(0, 255), rect=PLANE)

        xs = self.hist_x.view()
        self.c_train.setData(xs, self.train_hist.view())
        if self.Xte is not None:
            self.c_test.setData(xs, self.test_hist.view())
        else:
            self.c_test.setData([], [])

        t_render = time.perf_counter() - t0
        ft = self.frame_times[-60:]
        fps = len(ft) / max(sum(ft), 1e-9) if ft else 0
        self.frame_times = ft
        n_params = sum(p.numel() for p in m.parameters())
        n_neurons = sum(w for w, _ in self.layers)
        shape = " → ".join([str(len(self.feats))] + [f"{w}{a[:4]}" for w, a in self.layers]
                           + [str(K)])
        self.lb_stats.setText(
            f"net     {shape}\n"
            f"neurons {n_neurons}   params {n_params}\n"
            f"step    {self.step_count}\n"
            f"train   loss {tr_loss:.4f}  acc {tr_acc * 100:5.1f}%\n"
            f"test    loss {te_loss:.4f}  acc {te_acc * 100:5.1f}%\n"
            f"fps     {fps:5.1f}\n"
            f"train   {getattr(self, 't_train', 0) * 1e3:5.2f} ms/frame\n"
            f"render  {t_render * 1e3:5.2f} ms/frame")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opengl", action="store_true", help="use the OpenGL viewport")
    args = ap.parse_args()
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True, useOpenGL=args.opengl,
                        background="#0b0c10", foreground="#c8d0e0")
    app = pg.mkQApp("NN Playground")
    w = Playground()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
