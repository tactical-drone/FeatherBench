"""
Qt front end. All it does is: read widgets -> Session.configure(), call
Session.train() on a timer, and paint Session.evaluate() / Session.predict().
Dropdowns are filled from the nncore registries, so new parts appear here
without touching this file.
"""
import math
import time

import numpy as np
import pyqtgraph as pg
import torch
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from nncore import (ACTIVATIONS, DATASETS, FEATURES, INITIALIZERS, LAYERS, LOSSES, MODELS,
                    EXPANSIONS, OPTIMIZERS, SAMPLERS, SCHEDULES, SKIPS, SPLITTERS, TRAIN_STEPS, Config,
                    Session)

from . import theme
from .painters import BOUNDARY_PAINTERS, NEURON_PAINTERS, make_grid

MAX_THUMBS = 32


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
    def __init__(self, cfg=None):
        super().__init__()
        self.setWindowTitle("NN Playground // Jarvis build")
        self.resize(1500, 950)
        self.rng = np.random.default_rng()
        self.session = Session(cfg or Config())
        self.running = False
        self.frame_times = []
        self.last_frame = time.perf_counter()
        self.t_train = 0.0
        self.train_hist, self.test_hist, self.hist_x = GrowBuf(), GrowBuf(), GrowBuf()
        self.thumbs = []

        self._build_ui()
        self._set_grid()
        self._on_rebuilt("data")

        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.tick)
        self.timer.start(0)  # as fast as the event loop and vsync allow

    # ---------------------------------------------------------------- layout
    def _build_ui(self):
        c = self.session.cfg
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(330)
        panel = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(panel)
        scroll.setWidget(panel)
        root.addWidget(scroll)

        def combo(items, cur, cb, editable=False):
            w = QtWidgets.QComboBox()
            w.addItems([str(i) for i in items])
            w.setEditable(editable)
            w.setCurrentText(str(cur))
            w.currentTextChanged.connect(lambda _: cb())
            return w

        def section(title):
            lb = QtWidgets.QLabel(title)
            lb.setStyleSheet("color: #ff2a6d; font-weight: bold; margin-top: 6px;")
            form.addRow(lb)

        apply = self.apply

        section("DATA")
        self.cb_data = combo(DATASETS, c.dataset, apply)
        form.addRow("Dataset", self.cb_data)
        self.sp_n = QtWidgets.QSpinBox()
        self.sp_n.setRange(20, 5000)
        self.sp_n.setSingleStep(100)
        self.sp_n.setValue(c.n_points)
        self.sp_n.editingFinished.connect(apply)
        form.addRow("Points", self.sp_n)
        self.sl_noise = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_noise.setRange(0, 50)
        self.sl_noise.setValue(round(c.noise * 100))
        self.sl_noise.sliderReleased.connect(apply)
        form.addRow("Noise", self.sl_noise)
        self.cb_split = combo(SPLITTERS, c.splitter, apply)
        form.addRow("Split", self.cb_split)
        self.sp_seed = QtWidgets.QSpinBox()
        self.sp_seed.setRange(0, 99999)
        self.sp_seed.setValue(c.seed)
        self.sp_seed.editingFinished.connect(apply)
        form.addRow("Seed", self.sp_seed)

        section("MODEL")
        feat_box = QtWidgets.QWidget()
        fl = QtWidgets.QGridLayout(feat_box)
        fl.setContentsMargins(0, 0, 0, 0)
        self.feat_checks = {}
        for i, f in enumerate(FEATURES):
            ch = QtWidgets.QCheckBox(f)
            ch.setChecked(f in c.features)
            ch.toggled.connect(apply)
            fl.addWidget(ch, i // 4, i % 4)
            self.feat_checks[f] = ch
        form.addRow("Inputs", feat_box)
        self.cb_model = combo(MODELS, c.model, apply)
        form.addRow("Model", self.cb_model)
        self.le_layers = QtWidgets.QLineEdit(c.layers)
        self.le_layers.setToolTip("e.g. 4,4   or   8:sin,4:tanh   or empty for linear")
        self.le_layers.editingFinished.connect(apply)
        form.addRow("Hidden layers", self.le_layers)

        self.sp_width = QtWidgets.QSpinBox()
        self.sp_width.setRange(0, 128)
        self.sp_width.setValue(c.width)
        self.sp_width.editingFinished.connect(apply)
        form.addRow("Width", self.sp_width)
        self.sp_classes = QtWidgets.QSpinBox()
        self.sp_classes.setRange(1, 128)
        self.sp_classes.setValue(c.classes)
        self.sp_classes.editingFinished.connect(apply)
        form.addRow("Classes", self.sp_classes)
        self.cb_expand = combo(EXPANSIONS, c.expand, apply)
        form.addRow("Expand input", self.cb_expand)
        self.sp_ffreq = QtWidgets.QDoubleSpinBox()
        self.sp_ffreq.setRange(0.1, 20)
        self.sp_ffreq.setSingleStep(0.5)
        self.sp_ffreq.setValue(c.fourier_freq)
        self.sp_ffreq.setToolTip("fourier (sin) only: spread of the starting wave frequencies")
        self.sp_ffreq.editingFinished.connect(apply)
        form.addRow("  Fourier freq", self.sp_ffreq)
        
        self.cb_act = combo(ACTIVATIONS, c.activation, apply)
        form.addRow("Activation", self.cb_act)
        same_or_act = ["same", *ACTIVATIONS]  # "same" follows Activation
        self.cb_act_decide = combo(same_or_act, c.act_decide, apply)
        form.addRow("  act: decide", self.cb_act_decide)
        self.cb_act_relate = combo(same_or_act, c.act_relate, apply)
        form.addRow("  act: relate", self.cb_act_relate)
        self.cb_act_prepare = combo(same_or_act, c.act_prepare, apply)
        form.addRow("  act: prepare", self.cb_act_prepare)
        self.cb_layer = combo(LAYERS, c.layer, apply)
        form.addRow("Layer type", self.cb_layer)
        self.cb_skip = combo(SKIPS, c.skip, apply)
        form.addRow("Skip", self.cb_skip)
        self.cb_init = combo(INITIALIZERS, c.init, apply)
        form.addRow("Init", self.cb_init)

        section("TRAINING")
        self.cb_loss = combo(LOSSES, c.loss, apply)
        form.addRow("Loss", self.cb_loss)
        self.cb_opt = combo(OPTIMIZERS, c.optimizer, apply)
        form.addRow("Optimizer", self.cb_opt)
        self.cb_lr = combo([0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3], c.lr, apply,
                           editable=True)
        form.addRow("Learning rate", self.cb_lr)
        self.cb_wd = combo([0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1], c.weight_decay, apply)
        form.addRow("L2 / wd", self.cb_wd)
        self.cb_sched = combo(SCHEDULES, c.schedule, apply)
        form.addRow("LR schedule", self.cb_sched)
        self.cb_step = combo(TRAIN_STEPS, c.train_step, apply)
        form.addRow("Train step", self.cb_step)
        self.cb_sampler = combo(SAMPLERS, c.sampler, apply)
        form.addRow("Sampler", self.cb_sampler)
        self.cb_batch = combo(["full", 8, 16, 32, 64, 128],
                              "full" if c.batch_size is None else c.batch_size, apply)
        form.addRow("Batch", self.cb_batch)

        section("VIEW")
        self.sp_steps = QtWidgets.QSpinBox()
        self.sp_steps.setRange(1, 1000)
        self.sp_steps.setValue(10)
        form.addRow("Steps / frame", self.sp_steps)
        self.cb_res = combo([64, 96, 128, 192, 256], 192, self._set_grid, editable=True)
        form.addRow("Grid res", self.cb_res)
        self.cb_paint = combo(BOUNDARY_PAINTERS, next(iter(BOUNDARY_PAINTERS)), self.render)
        form.addRow("Boundary", self.cb_paint)
        self.cb_npaint = combo(NEURON_PAINTERS, "per-neuron max", self.render)
        form.addRow("Neuron scale", self.cb_npaint)
        self.cb_layer_view = combo(["layer 6"], "layer 6", self._rebuild_thumbs)  # the 2-neuron relate bottleneck
        form.addRow("Neuron view", self.cb_layer_view)

        btns = QtWidgets.QGridLayout()
        self.bt_play = QtWidgets.QPushButton("▶ Play  [Space]")
        self.bt_play.clicked.connect(self.toggle)
        bt_step = QtWidgets.QPushButton("Step [S]")
        bt_step.clicked.connect(self.single_step)
        bt_reset = QtWidgets.QPushButton("Reset weights [R]")
        bt_reset.clicked.connect(self.reset_weights)
        bt_data = QtWidgets.QPushButton("New data [N]")
        bt_data.clicked.connect(self.reseed)
        btns.addWidget(self.bt_play, 0, 0, 1, 2)
        btns.addWidget(bt_step, 1, 0)
        btns.addWidget(bt_reset, 1, 1)
        btns.addWidget(bt_data, 2, 0, 1, 2)
        form.addRow(btns)

        self.lb_stats = QtWidgets.QLabel()
        self.lb_stats.setStyleSheet(theme.STATS_STYLE)
        self.lb_stats.setWordWrap(True)
        form.addRow(self.lb_stats)

        for key, fn in (("Space", self.toggle), ("S", self.single_step),
                        ("R", self.reset_weights), ("N", self.reseed)):
            QtGui.QShortcut(QtGui.QKeySequence(key), self, activated=fn)

        # right side: boundary | (neurons over [loss | activation] tabs)
        # Everything stays in ONE GraphicsLayoutWidget: each extra pyqtgraph view is
        # another OpenGL viewport with --opengl, and several of them froze the app.
        self.gl = pg.GraphicsLayoutWidget()
        root.addWidget(self.gl, 1)
        self.p_main = self.gl.addPlot(row=0, col=0, rowspan=2, title="decision boundary")
        self.p_main.setAspectLocked(True)
        self.p_main.setRange(xRange=(-theme.VIEW, theme.VIEW), yRange=(-theme.VIEW, theme.VIEW),
                             padding=0)
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
        self.gl_bottom = self.gl.addLayout(row=1, col=1)  # tab strip over the shown plot
        self.tabbar = QtWidgets.QTabBar()
        self.tabbar.addTab("Loss")
        self.tabbar.addTab("Activation")
        tab_proxy = QtWidgets.QGraphicsProxyWidget()
        tab_proxy.setWidget(self.tabbar)
        self.gl_bottom.addItem(tab_proxy, row=0, col=0)

        self.p_loss = pg.PlotItem(title="loss (train cyan, test magenta)")
        self.p_loss.setLogMode(y=True)
        self.p_loss.showGrid(x=True, y=True, alpha=0.2)
        self.p_loss.setClipToView(True)
        self.p_loss.setDownsampling(auto=True, mode="peak")
        self.c_train = self.p_loss.plot(pen=pg.mkPen(theme.TRAIN_PEN, width=2))
        self.c_test = self.p_loss.plot(pen=pg.mkPen(theme.TEST_PEN, width=2))
        for c in (self.c_train, self.c_test):  # wide antialiased path strokes get slow
            c.curve.setSegmentedLineMode("on")  # on long noisy curves; plain segments stay fast

        # activation preview: shows whichever activation dropdown was changed last
        self.p_act = pg.PlotItem()
        self.p_act.showGrid(x=True, y=True, alpha=0.2)
        self.p_act.addLegend(offset=(10, 10))
        self.p_act.addLine(x=0, pen=pg.mkPen(theme.WINDOW_FG, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.p_act.addLine(y=0, pen=pg.mkPen(theme.WINDOW_FG, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.c_act = self.p_act.plot(pen=pg.mkPen(theme.TRAIN_PEN, width=2.5), name="f(x)")
        self.c_dact = self.p_act.plot(pen=pg.mkPen(theme.TEST_PEN, width=1.5,
                                                   style=QtCore.Qt.PenStyle.DashLine), name="f'(x)")
        for label, cb in (("Activation", self.cb_act), ("act: decide", self.cb_act_decide),
                          ("act: relate", self.cb_act_relate), ("act: prepare", self.cb_act_prepare)):
            cb.currentTextChanged.connect(lambda name, label=label: self._show_activation(name, label))
        self._show_activation(self.cb_act.currentText(), "Activation")
        self.gl_bottom.addItem(self.p_loss, row=1, col=0)
        self.tabbar.currentChanged.connect(self._switch_tab)
        self.gl.ci.layout.setColumnStretchFactor(0, 3)
        self.gl.ci.layout.setColumnStretchFactor(1, 2)

    # ---------------------------------------------------------------- widgets -> config
    def _read_config(self):
        """Current widget values as Config fields. Unparseable numbers keep the old value."""
        c = self.session.cfg
        try:
            lr = float(self.cb_lr.currentText())
        except ValueError:
            lr = c.lr
        bs = self.cb_batch.currentText()
        return dict(
            dataset=self.cb_data.currentText(), n_points=self.sp_n.value(),
            noise=self.sl_noise.value() / 100, splitter=self.cb_split.currentText(),
            seed=self.sp_seed.value(),
            features=tuple(f for f, ch in self.feat_checks.items() if ch.isChecked()) or ("x", "y"),
            model=self.cb_model.currentText(), layers=self.le_layers.text(), width=self.sp_width.value(),
            classes=self.sp_classes.value(),
            expand=self.cb_expand.currentText(), fourier_freq=self.sp_ffreq.value(),
            activation=self.cb_act.currentText(),
            act_decide=self.cb_act_decide.currentText(), act_relate=self.cb_act_relate.currentText(),
            act_prepare=self.cb_act_prepare.currentText(), layer=self.cb_layer.currentText(),
            skip=self.cb_skip.currentText(), init=self.cb_init.currentText(),
            loss=self.cb_loss.currentText(), optimizer=self.cb_opt.currentText(), lr=lr,
            weight_decay=float(self.cb_wd.currentText()), schedule=self.cb_sched.currentText(),
            train_step=self.cb_step.currentText(), sampler=self.cb_sampler.currentText(),
            batch_size=None if bs == "full" else int(bs),
        )

    def apply(self):
        try:
            level = self.session.configure(**self._read_config())
            self.le_layers.setStyleSheet("")
            self.le_layers.setToolTip("e.g. 4,4   or   8:sin,4:tanh   or empty for linear")
        except ValueError as e:
            self.le_layers.setStyleSheet(theme.ERROR_STYLE)
            self.le_layers.setToolTip(str(e))
            return
        if level:
            self._on_rebuilt(level)

    def reset_weights(self):
        self.session.reset_model()
        self._on_rebuilt("model")

    def reseed(self):
        self.sp_seed.setValue(int(self.rng.integers(0, 99999)))
        self.apply()

    def _on_rebuilt(self, level):
        s = self.session
        if level == "data":
            X, y = s.X_all, s.y_all
            tr, te = s.train_idx, s.test_idx
            brushes = [pg.mkBrush(*theme.CLASS_COLORS[k % len(theme.CLASS_COLORS)])
                       for k in range(s.n_classes)]
            self.sc_train.setData(X[tr, 0], X[tr, 1], brush=[brushes[k] for k in y[tr]],
                                  size=14 if len(X) <= 8 else 7)
            self.sc_test.setData(X[te, 0], X[te, 1], brush=[brushes[k] for k in y[te]])
        if level in ("data", "model"):
            for b in (self.train_hist, self.test_hist, self.hist_x):
                b.clear()
            cur = self.cb_layer_view.currentText()
            sizes = s.hidden_sizes
            self.cb_layer_view.blockSignals(True)
            self.cb_layer_view.clear()
            self.cb_layer_view.addItems([f"layer {i + 1}" for i in range(len(sizes))] or ["(none)"])
            self.cb_layer_view.setCurrentText(cur)
            self.cb_layer_view.blockSignals(False)
            self._rebuild_thumbs()
        self.render()

    def _set_grid(self):
        try:
            self.grid_res, self.grid = make_grid(int(self.cb_res.currentText()))
        except ValueError:
            return
        if hasattr(self, "thumbs"):
            self.render()

    def _rebuild_thumbs(self):
        self.gl_neurons.clear()
        self.thumbs = []
        sizes = self.session.hidden_sizes
        txt = self.cb_layer_view.currentText()
        if not sizes or not txt.startswith("layer"):
            self.view_layer = None
            return
        self.view_layer = min(int(txt.split()[1]) - 1, len(sizes) - 1)
        size = sizes[self.view_layer]
        n = min(size, MAX_THUMBS)
        cols = max(1, math.ceil(math.sqrt(n * 1.5)))
        shown = f"  (first {n})" if n < size else ""
        self.gl_neurons.addLabel(f"layer {self.view_layer + 1}  ·  {size} neurons{shown}",
                                 row=0, col=0, colspan=cols, color=theme.TRAIN_PEN, size="11pt")
        for i in range(n):
            vb = self.gl_neurons.addViewBox(row=1 + i // cols, col=i % cols, lockAspect=True,
                                            enableMouse=False)
            im = pg.ImageItem(lut=theme.NEURON_LUT)
            vb.addItem(im)
            vb.setRange(xRange=(-theme.VIEW, theme.VIEW), yRange=(-theme.VIEW, theme.VIEW),
                        padding=0)
            self.thumbs.append(im)
        self.render()

    def _switch_tab(self, i):
        """Swap the plot under the tab strip (both plots live in the one shared view)."""
        old, new = (self.p_act, self.p_loss) if i == 0 else (self.p_loss, self.p_act)
        self.gl_bottom.removeItem(old)
        self.gl_bottom.addItem(new, row=1, col=0)

    def _show_activation(self, name, label):
        """Plot f(x) and its slope f'(x) for the activation just picked."""
        if name == "same":
            name = self.cb_act.currentText()
        x = torch.linspace(-4, 4, 401, requires_grad=True)
        with torch.enable_grad():  # render() runs under no_grad
            y = ACTIVATIONS.get(name)()(x)
            dy, = torch.autograd.grad(y.sum(), x)
        xs = x.detach().numpy()
        self.c_act.setData(xs, y.detach().numpy())
        self.c_dact.setData(xs, dy.numpy())
        self.p_act.setTitle(f"{label}: {name}")

    # ---------------------------------------------------------------- loop
    def toggle(self):
        self.running = not self.running
        self.bt_play.setText("⏸ Pause  [Space]" if self.running else "▶ Play  [Space]")

    def single_step(self):
        self.session.train(1)
        self.render()

    def tick(self):
        now = time.perf_counter()
        self.frame_times.append(now - self.last_frame)
        self.last_frame = now
        if self.running:
            t0 = time.perf_counter()
            self.session.train(self.sp_steps.value())
            self.t_train = time.perf_counter() - t0
            self.render()

    @torch.no_grad()
    def render(self):
        if not hasattr(self, "grid"):
            return
        t0 = time.perf_counter()
        s = self.session
        ev = s.evaluate()
        tr, te = ev["train"], ev["test"]
        self.hist_x.add(s.step_count)
        self.train_hist.add(max(tr["loss"], 1e-8))
        self.test_hist.add(max(te["loss"], 1e-8) if te["loss"] == te["loss"] else np.nan)

        logits, hidden = s.predict(self.grid, collect=True)
        r = self.grid_res
        probs = torch.softmax(logits, 1)
        rgb = BOUNDARY_PAINTERS.get(self.cb_paint.currentText())(probs, r)
        self.img.setImage(rgb, autoLevels=False, levels=(0, 255), rect=theme.PLANE)

        if getattr(self, "view_layer", None) is not None and self.thumbs:
            H = hidden[self.view_layer][:, : len(self.thumbs)]
            idx = NEURON_PAINTERS.get(self.cb_npaint.currentText())(H)
            for i, im in enumerate(self.thumbs):
                im.setImage(idx[i].reshape(r, r), autoLevels=False, levels=(0, 255),
                            rect=theme.PLANE)

        xs = self.hist_x.view()
        self.c_train.setData(xs, self.train_hist.view())
        if s.Xte is not None:
            self.c_test.setData(xs, self.test_hist.view())
        else:
            self.c_test.setData([], [])

        t_render = time.perf_counter() - t0
        ft = self.frame_times[-60:]
        fps = len(ft) / max(sum(ft), 1e-9) if ft else 0
        self.frame_times = ft
        extra = "  ".join(f"{k} {v * 100:5.1f}%" for k, v in tr.items() if k != "loss")
        extra_te = "  ".join(f"{k} {v * 100:5.1f}%" for k, v in te.items() if k != "loss")
        self.lb_stats.setText(
            f"net     {s.describe()}\n"
            f"neurons {sum(s.hidden_sizes)}   params {s.n_params}\n"
            f"step    {s.step_count}   lr {s.lr:.3g}\n"
            f"train   loss {tr['loss']:.4f}  {extra}\n"
            f"test    loss {te['loss']:.4f}  {extra_te}\n"
            f"fps     {fps:5.1f}\n"
            f"train   {self.t_train * 1e3:5.2f} ms/frame\n"
            f"render  {t_render * 1e3:5.2f} ms/frame")


def run(opengl=False, cfg=None):
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True, useOpenGL=opengl,
                        background=theme.WINDOW_BG, foreground=theme.WINDOW_FG)
    app = pg.mkQApp("NN Playground")
    w = Playground(cfg)
    w.show()
    return app.exec()
