"""
Qt front end. All it does is: read widgets -> Session.configure(), call
Session.train() on a timer, and paint Session.evaluate() / Session.predict().
Dropdowns are filled from the nncore registries, so new parts appear here
without touching this file.
"""
import json
import math
import sys
import time
import traceback
from dataclasses import fields

import numpy as np
import pyqtgraph as pg
import torch
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

import nncore
from nncore import (ACTIVATIONS, DATASETS, FEATURES, INITIALIZERS, LAYERS, LOSSES, MODELS,
                    EXPANSIONS, OPTIMIZERS, SAMPLERS, SCHEDULES, SKIPS, SPLITTERS, TRAIN_STEPS, Config,
                    Session, parse_layers)

from . import theme
from .painters import BOUNDARY_PAINTERS, NEURON_PAINTERS, make_grid

VERSION = getattr(nncore, "__version__", "1.0.0")
MAX_THUMBS = 32
TITLE = "NN Playground // Jarvis build"
LAYERS_TIP = "e.g. 4,4   or   8:sin,4:tanh   or empty for a linear model"
SETTINGS_ORG, SETTINGS_APP = "tactical-drone", "NN Playground"

# hover help for every control (shown as tooltips and in the status bar)
TIPS = {
    "Dataset": "The 2-D problem to learn. Each colour is one class.",
    "Points": "How many points to generate (train + test).",
    "Noise": "How much the points are scattered. More noise = harder, blurrier classes.",
    "Split": "How points are split into training data and held-out test data.",
    "Seed": "Random seed for the data and the starting weights. Same seed = same run.",
    "Inputs": "Features fed into the network. x, y are the raw coordinates; the rest are "
              "hand-made extras (TF Playground style).",
    "Model": "The architecture. 'mlp' is a plain stack of layers set by Hidden layers; "
             "'custom nn' is the Wide model from my_parts.py.",
    "Hidden layers": "mlp only. " + LAYERS_TIP,
    "Width": "custom nn only. Neurons in the first hidden layer.",
    "Classes": "custom nn only. Width of the head / decide / perc stage.",
    "Expand input": "custom nn only. Widens x, y before the first layer. "
                    "Nonlinear ones (fourier, polar, rbf) add real expressive power.",
    "Fourier freq": "fourier (sin) only. Spread of the starting wave frequencies. "
                    "Spirals like 3 to 6; smooth problems are fine lower.",
    "Activation": "The nonlinearity. In custom nn it drives the first hidden layer, "
                  "and any act: dropdown set to 'same'.",
    "act: decide": "custom nn only. Activation after the decide layer ('same' = Activation).",
    "act: relate": "custom nn only. Activation after relate, the narrow bottleneck.",
    "act: prepare": "custom nn only. Activation after prepare ('linear' = none).",
    "Layer type": "mlp only. The weighted transform used for every layer.",
    "Skip": "mlp only. How a layer's output combines with its input (residuals etc.).",
    "Init": "How the starting weights are drawn.",
    "Loss": "What training minimises.",
    "Optimizer": "How weights are updated from the gradients.",
    "Learning rate": "Step size. Too high = chaos, too low = crawl. You can type any value > 0.",
    "L2 / wd": "Weight decay: pulls weights toward 0 to fight overfitting. You can type any value >= 0.",
    "LR schedule": "How the learning rate changes during training.",
    "Train step": "What one training step does with a batch.",
    "Sampler": "How each batch is drawn from the training set.",
    "Batch": "Points per training step ('full' = the whole training set). You can type any number.",
    "Steps / frame": "Training steps between redraws. Higher = faster training, choppier view.",
    "Stop at step": "Pause automatically when training reaches this step (0 = never).",
    "Grid res": "Resolution of the boundary and neuron images (higher = sharper, slower).",
    "Boundary": "How the decision boundary is painted.",
    "Neuron scale": "How neuron values map to colours. 'full range' stretches each neuron "
                    "over the whole palette; 'per-neuron max' keeps 0 in the middle.",
    "Neuron view": "Which layer's neurons to show (up to 32 thumbnails).",
}

# Ready-made experiments: Config defaults plus these overrides. They only name
# parts that ship with the playground (custom nn lives in my_parts.py).
RECIPES = [
    ("Fourier + gauss, 5-arm spiral (default)", {}),
    ("Circles with zero hidden neurons (x², y²)",
     dict(dataset="Circles", model="mlp", layers="", features=("x", "y", "x²", "y²"),
          schedule="constant")),
    ("XOR gate with one square neuron",
     dict(dataset="XOR gate (4 points)", model="mlp", layers="1:square", batch_size=None,
          schedule="constant")),
    ("XOR gate with two tanh neurons",
     dict(dataset="XOR gate (4 points)", model="mlp", layers="2:tanh", batch_size=None,
          schedule="constant")),
    ("Moons, tiny tanh net",
     dict(dataset="Moons", model="mlp", layers="4,4", activation="tanh", schedule="constant")),
    ("2-arm spiral, deep residual relu",
     dict(dataset="Spiral (2 arms)", model="mlp", layers="16,16,16,16", activation="relu",
          skip="residual (same width)", lr=0.01)),
    ("Checkerboard, sine MLP",
     dict(dataset="Checkerboard", model="mlp", layers="16:sin,16:sin", activation="sin",
          lr=0.01, noise=0.0)),
    ("Gaussian blobs, rbf bumps",
     dict(dataset="Gaussian blobs (5)", expand="rbf bumps")),
]

HELP_HTML = """
<h3>NN Playground</h3>
<p>Tiny neural networks, trained live. Pick a dataset and a model on the left (or a
ready-made experiment from the <b>Recipes</b> menu), press <b>Play</b>, and watch the
decision boundary, the neurons and the loss evolve.</p>
<h4>Keys</h4>
<table cellpadding="2">
<tr><td><b>Space</b></td><td>play / pause</td></tr>
<tr><td><b>S</b></td><td>one training step</td></tr>
<tr><td><b>R</b></td><td>reset weights (same data)</td></tr>
<tr><td><b>N</b></td><td>new data (new random seed)</td></tr>
<tr><td><b>Ctrl+S / Ctrl+O</b></td><td>save / load settings</td></tr>
<tr><td><b>Ctrl+Shift+C</b></td><td>copy the current setup as a headless.py command</td></tr>
<tr><td><b>F12</b></td><td>save a screenshot</td></tr>
<tr><td><b>Ctrl+Shift+S</b></td><td>save the trained model (.pt)</td></tr>
<tr><td><b>F11</b></td><td>full screen</td></tr>
<tr><td><b>Ctrl+D</b></td><td>restore the default settings</td></tr>
<tr><td><b>F1</b></td><td>this help</td></tr>
</table>
<h4>Reading the view</h4>
<ul>
<li><b>Decision boundary</b>: the colour is the predicted class, brighter = more confident.
Dots with a white rim are training points, a black rim marks test points. Hover the
plane to read the prediction at that spot in the status bar.</li>
<li><b>Neurons</b>: one thumbnail per neuron of the chosen layer, showing what it
responds to across the plane.</li>
<li><b>Loss</b> tab: train (cyan) and test (magenta) loss. When test rises while train
falls, the net is overfitting. <b>Accuracy</b> tab: the same for accuracy.
<b>Activation</b> tab: the last activation you picked, with its slope f'(x) dashed.</li>
</ul>
<p>Hover any control for a short explanation. Greyed-out controls don't apply to the
current model. Drop a settings .json on the window to load it. Your settings are remembered between runs (start with <code>--fresh</code>
to skip that). Add your own parts in <code>my_parts.py</code>.</p>
"""


def clean_config(data):
    """Loaded dict -> (values for Session.configure, dropped unknown keys)."""
    clean = getattr(Config, "clean", None)
    if clean is not None:
        return clean(data)
    known = {f.name for f in fields(Config)}
    values = {k: v for k, v in data.items() if k in known}
    if "features" in values:
        values["features"] = tuple(values["features"])
    return values, sorted(set(data) - known)


def headless_command(cfg):
    """The headless.py command line that reproduces cfg (only non-default fields)."""
    base, parts = Config().to_dict(), ["python headless.py"]
    for k, v in cfg.to_dict().items():
        if k == "extra" or v == base[k]:
            continue
        if k == "features":
            v = ",".join(v)
        elif k == "batch_size" and v is None:
            v = "full"
        v = str(v)
        if not v or any(ch in v for ch in ' "&|<>^()÷×,;'):
            v = '"' + v.replace('"', '\\"') + '"'
        parts.append(f"--{k} {v}")
    return " ".join(parts)


class GrowBuf:
    def __init__(self):
        self.a = np.zeros(4096, np.float32)
        self.n = 0

    def add(self, v):
        if self.n == len(self.a):
            self.a = np.concatenate([self.a, np.zeros_like(self.a)])
        self.a[self.n] = v
        self.n += 1

    def last(self):
        return self.a[self.n - 1] if self.n else None

    def view(self):
        return self.a[: self.n]

    def clear(self):
        self.n = 0


class Playground(QtWidgets.QMainWindow):
    def __init__(self, cfg=None, restore=False, settings=None):
        super().__init__()
        self.setWindowTitle(TITLE)
        self.resize(1500, 950)
        self.setAcceptDrops(True)  # drop a settings .json to load it
        self.settings = settings or QtCore.QSettings(SETTINGS_ORG, SETTINGS_APP)
        self.rng = np.random.default_rng()
        self.running = False
        self.diverged = False
        self.frame_times = []
        self.last_frame = time.perf_counter()
        self.t_train = 0.0
        self.best_test = math.nan
        self.train_hist, self.test_hist, self.hist_x = GrowBuf(), GrowBuf(), GrowBuf()
        self.train_acc, self.test_acc = GrowBuf(), GrowBuf()
        self.thumbs = []
        self.invalid = []

        note = None
        if cfg is None and restore:
            cfg, note = self._restored_config()
        self.session = Session(cfg or Config())

        self._building = True  # widgets fire signals while they are set up; ignore them
        self._build_ui()
        self._build_menus()
        if restore:
            self._restore_view()
        self._building = False
        self._update_enabled()
        self._set_grid()
        self._on_rebuilt("data")
        self.statusBar().showMessage(note or "Ready. Press Space to train, F1 for help.", 10000)

        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.tick)  # started by Play: no busy loop while paused

    # ---------------------------------------------------------------- layout
    def _build_ui(self):
        c = self.session.cfg
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(340)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panel = QtWidgets.QWidget()
        form = self.form = QtWidgets.QFormLayout(panel)
        scroll.setWidget(panel)
        root.addWidget(scroll)

        def combo(items, cur, cb, editable=False):
            w = QtWidgets.QComboBox()
            w.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            w.setMinimumContentsLength(12)  # long part names elide instead of widening the panel
            w.addItems([str(i) for i in items])
            w.setEditable(editable)
            w.setCurrentText(str(cur))
            if editable:  # apply typed values on Enter / focus out, not every keystroke
                w.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
                w.lineEdit().editingFinished.connect(lambda: cb())
                w.activated.connect(lambda _: cb())
            else:
                w.currentTextChanged.connect(lambda _: cb())
            return w

        def spin(lo, hi, value, step=1, double=False, cb=None):
            w = QtWidgets.QDoubleSpinBox() if double else QtWidgets.QSpinBox()
            w.setRange(lo, hi)
            w.setSingleStep(step)
            w.setValue(value)
            w.setKeyboardTracking(False)  # typed values apply on Enter, arrows apply at once
            w.valueChanged.connect(lambda _: (cb or self.apply)())
            return w

        def row(label, widget, tip_key=None):
            tip = TIPS.get(tip_key or label.strip(), "")
            widget.setToolTip(tip)
            widget.setStatusTip(tip)
            form.addRow(label, widget)
            return widget

        def section(title):
            lb = QtWidgets.QLabel(title)
            lb.setStyleSheet(theme.SECTION_STYLE)
            form.addRow(lb)

        apply = self.apply

        section("DATA")
        self.cb_data = row("Dataset", combo(DATASETS, c.dataset, apply))
        self.cb_data.setMaxVisibleItems(40)  # the dataset zoo is long; show it without scrolling
        self.sp_n = row("Points", spin(20, 5000, c.n_points, 100))
        noise_box = QtWidgets.QWidget()
        nl = QtWidgets.QHBoxLayout(noise_box)
        nl.setContentsMargins(0, 0, 0, 0)
        self.sl_noise = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_noise.setRange(0, 50)
        self.sl_noise.setValue(round(c.noise * 100))
        self.lb_noise = QtWidgets.QLabel()
        self.lb_noise.setMinimumWidth(32)
        self.sl_noise.valueChanged.connect(self._noise_changed)
        self.sl_noise.sliderReleased.connect(apply)
        nl.addWidget(self.sl_noise, 1)
        nl.addWidget(self.lb_noise)
        self._noise_changed(self.sl_noise.value())
        row("Noise", noise_box)
        self.cb_split = row("Split", combo(SPLITTERS, c.splitter, apply))
        self.sp_seed = row("Seed", spin(0, 99999, c.seed))

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
        row("Inputs", feat_box)
        self.cb_model = row("Model", combo(MODELS, c.model, apply))
        self.le_layers = row("Hidden layers", QtWidgets.QLineEdit(c.layers))
        self.le_layers.setPlaceholderText("empty = linear model")
        self.le_layers.editingFinished.connect(apply)
        self.sp_width = row("Width", spin(1, 128, c.width))
        self.sp_classes = row("Classes", spin(1, 128, c.classes))
        self.cb_expand = row("Expand input", combo(EXPANSIONS, c.expand, apply))
        self.sp_ffreq = row("  Fourier freq", spin(0.1, 20, c.fourier_freq, 0.5, double=True))

        self.cb_act = row("Activation", combo(ACTIVATIONS, c.activation, apply))
        same_or_act = ["same", *ACTIVATIONS]  # "same" follows Activation
        self.cb_act_decide = row("  act: decide", combo(same_or_act, c.act_decide, apply))
        self.cb_act_relate = row("  act: relate", combo(same_or_act, c.act_relate, apply))
        self.cb_act_prepare = row("  act: prepare", combo(same_or_act, c.act_prepare, apply))
        self.cb_layer = row("Layer type", combo(LAYERS, c.layer, apply))
        self.cb_skip = row("Skip", combo(SKIPS, c.skip, apply))
        self.cb_init = row("Init", combo(INITIALIZERS, c.init, apply))

        section("TRAINING")
        self.cb_loss = row("Loss", combo(LOSSES, c.loss, apply))
        self.cb_opt = row("Optimizer", combo(OPTIMIZERS, c.optimizer, apply))
        self.cb_lr = row("Learning rate", combo([0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
                                                c.lr, apply, editable=True))
        self.cb_wd = row("L2 / wd", combo([0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1], c.weight_decay, apply,
                                          editable=True))
        self.cb_sched = row("LR schedule", combo(SCHEDULES, c.schedule, apply))
        self.cb_step = row("Train step", combo(TRAIN_STEPS, c.train_step, apply))
        self.cb_sampler = row("Sampler", combo(SAMPLERS, c.sampler, apply))
        self.cb_batch = row("Batch", combo(["full", 8, 16, 32, 64, 128],
                                           "full" if c.batch_size is None else c.batch_size, apply,
                                           editable=True))
        self.editable = {self.cb_lr: "Learning rate", self.cb_wd: "L2 / wd", self.cb_batch: "Batch"}

        section("VIEW")
        self.sp_steps = row("Steps / frame", spin(1, 1000, 10, cb=lambda: None))
        self.sp_stop = row("Stop at step", spin(0, 10_000_000, 0, 1000, cb=lambda: None))
        self.sp_stop.setSpecialValueText("never")
        self.cb_res = row("Grid res", combo([64, 96, 128, 192, 256], 192, self._set_grid,
                                            editable=True))
        self.cb_paint = row("Boundary", combo(BOUNDARY_PAINTERS, next(iter(BOUNDARY_PAINTERS)),
                                              self.render))
        self.cb_npaint = row("Neuron scale", combo(NEURON_PAINTERS, "per-neuron max", self.render))
        # layer 6 of custom nn is the 2-neuron relate bottleneck
        self.cb_layer_view = row("Neuron view", combo(["layer 6"], "layer 6", self._rebuild_thumbs))

        btns = QtWidgets.QGridLayout()
        self.bt_play = QtWidgets.QPushButton("▶ Play  [Space]")
        self.bt_play.clicked.connect(self.toggle)
        bt_step = QtWidgets.QPushButton("Step [S]")
        bt_step.clicked.connect(self.single_step)
        bt_reset = QtWidgets.QPushButton("Reset weights [R]")
        bt_reset.clicked.connect(self.reset_weights)
        bt_data = QtWidgets.QPushButton("New data [N]")
        bt_data.clicked.connect(self.reseed)
        for b, tip in ((self.bt_play, "Start or pause training"),
                       (bt_step, "Run exactly one training step"),
                       (bt_reset, "Start the weights over (same seed, same data)"),
                       (bt_data, "Pick a new random seed: new points and new weights")):
            b.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)  # keep Space for play/pause
            b.setToolTip(tip)
            b.setStatusTip(tip)
        btns.addWidget(self.bt_play, 0, 0, 1, 2)
        btns.addWidget(bt_step, 1, 0)
        btns.addWidget(bt_reset, 1, 1)
        btns.addWidget(bt_data, 2, 0, 1, 2)
        form.addRow(btns)

        self.lb_stats = QtWidgets.QLabel()
        self.lb_stats.setStyleSheet(theme.STATS_STYLE)
        self.lb_stats.setWordWrap(True)
        self.lb_stats.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self.lb_stats)

        self.lb_state = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self.lb_state)
        self._show_state()

        # right side: boundary | (neurons over [loss | accuracy | activation] tabs)
        # Everything stays in ONE GraphicsLayoutWidget: each extra pyqtgraph view is
        # another OpenGL viewport with --opengl, and several of them froze the app.
        self.gl = pg.GraphicsLayoutWidget()
        root.addWidget(self.gl, 1)
        self.p_main = self.gl.addPlot(row=0, col=0, rowspan=2, title="decision boundary")
        self.p_main.setAspectLocked(True)
        self.p_main.setRange(xRange=(-theme.VIEW, theme.VIEW), yRange=(-theme.VIEW, theme.VIEW),
                             padding=0)
        self.p_main.setMouseEnabled(x=False, y=False)  # the plane is fixed; no accidental pans
        self.p_main.hideButtons()
        self.p_main.hideAxis("left")  # axis text is the priciest thing to repaint
        self.p_main.hideAxis("bottom")
        self.img = pg.ImageItem()
        self.p_main.addItem(self.img)
        self.sc_train = pg.ScatterPlotItem(size=7, pen=pg.mkPen((255, 255, 255, 200), width=1))
        self.sc_test = pg.ScatterPlotItem(size=7, pen=pg.mkPen((0, 0, 0, 220), width=1.5))
        self.p_main.addItem(self.sc_train)
        self.p_main.addItem(self.sc_test)
        self.gl.scene().sigMouseMoved.connect(self._hover)

        self.gl_neurons = self.gl.addLayout(row=0, col=1)
        self.gl_bottom = self.gl.addLayout(row=1, col=1)  # tab strip over the shown plot
        self.tabbar = QtWidgets.QTabBar()
        for name in ("Loss", "Accuracy", "Activation"):
            self.tabbar.addTab(name)
        self.tabbar.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        tab_proxy = QtWidgets.QGraphicsProxyWidget()
        tab_proxy.setWidget(self.tabbar)
        self.gl_bottom.addItem(tab_proxy, row=0, col=0)

        def curve_plot(title, log):
            p = pg.PlotItem(title=title)
            p.setLogMode(y=log)
            p.showGrid(x=True, y=True, alpha=0.2)
            p.setClipToView(True)
            p.setDownsampling(auto=True, mode="peak")
            p.setLabel("bottom", "step")
            curves = [p.plot(pen=pg.mkPen(pen, width=2)) for pen in (theme.TRAIN_PEN, theme.TEST_PEN)]
            for cv in curves:  # wide antialiased path strokes get slow
                cv.curve.setSegmentedLineMode("on")  # on long noisy curves; plain segments stay fast
            return p, curves

        self.p_loss, (self.c_train, self.c_test) = curve_plot("loss (train cyan, test magenta)", True)
        self.p_acc, (self.c_acc_train, self.c_acc_test) = curve_plot(
            "accuracy % (train cyan, test magenta)", False)
        self.p_acc.setYRange(0, 101, padding=0)
        self.p_acc.enableAutoRange(axis="y", enable=False)

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

        self.tab_plots = [self.p_loss, self.p_acc, self.p_act]
        self.shown_tab = self.p_loss
        self.gl_bottom.addItem(self.p_loss, row=1, col=0)
        self.tabbar.currentChanged.connect(self._switch_tab)
        self.gl.ci.layout.setColumnStretchFactor(0, 3)
        self.gl.ci.layout.setColumnStretchFactor(1, 2)

        # which controls belong to which model; the others are greyed out
        self.mlp_only = [self.le_layers, self.cb_layer, self.cb_skip]
        self.custom_only = [self.sp_width, self.sp_classes, self.cb_expand, self.sp_ffreq,
                            self.cb_act_decide, self.cb_act_relate, self.cb_act_prepare]

    def _build_menus(self):
        def action(menu, text, fn, key=None, tip=""):
            a = QtGui.QAction(text, self)
            if key:
                a.setShortcut(QtGui.QKeySequence(key))
            a.setStatusTip(tip)
            a.triggered.connect(lambda _=False: fn())
            menu.addAction(a)
            self.addAction(a)  # shortcuts work even with the menu bar hidden
            return a

        mb = self.menuBar()
        m = mb.addMenu("&File")
        action(m, "&Save settings…", self.save_settings, "Ctrl+S", "Save every setting to a .json file")
        action(m, "&Load settings…", self.load_settings, "Ctrl+O", "Load settings from a .json file")
        action(m, "&Copy as headless command", self.copy_command, "Ctrl+Shift+C",
               "Copy a headless.py command line that reproduces the current setup")
        action(m, "Save &screenshot…", self.save_screenshot, "F12", "Save the window as a .png")
        action(m, "Save &model…", self.save_model, "Ctrl+Shift+S",
               "Save the trained weights and the settings that built them (.pt)")
        action(m, "Export &history (CSV)…", self.export_history, None,
               "Save the loss and accuracy curves as a .csv table")
        m.addSeparator()
        action(m, "&Quit", self.close, "Ctrl+Q")

        m = mb.addMenu("&Run")
        action(m, "&Play / pause", self.toggle, "Space")
        action(m, "&Step", self.single_step, "S", "One training step")
        action(m, "&Reset weights", self.reset_weights, "R", "Fresh weights, same data")
        action(m, "&New data", self.reseed, "N", "New random seed for data and weights")
        m.addSeparator()
        action(m, "Restore &defaults", self.restore_defaults, "Ctrl+D",
               "Back to the default settings")

        m = mb.addMenu("&View")
        action(m, "&Full screen", self.toggle_fullscreen, "F11")

        m = mb.addMenu("Re&cipes")
        for name, overrides in RECIPES:
            action(m, name, lambda name=name, o=overrides: self.load_recipe(name, o),
                   tip="Load this ready-made experiment")

        m = mb.addMenu("&Help")
        action(m, "&Help", self.show_help, "F1")
        action(m, "&About", self.show_about)

    # ---------------------------------------------------------------- widgets <-> config
    def _noise_changed(self, v):
        self.lb_noise.setText(f"{v / 100:.2f}")
        if not self.sl_noise.isSliderDown():  # keyboard / wheel; drags apply on release
            self.apply()

    def _read_config(self):
        """Current widget values as Config fields. Values that don't parse keep the old
        setting and their widget is listed in self.invalid."""
        c = self.session.cfg
        self.invalid = []

        def num(widget, old, cast=float, ok=lambda v: True):
            try:
                v = cast(widget.currentText().strip())
                if ok(v):
                    return v
            except ValueError:
                pass
            self.invalid.append(widget)
            return old

        if self.cb_batch.currentText().strip().lower() in ("full", "none", ""):
            batch = None
        else:
            batch = num(self.cb_batch, c.batch_size, int, lambda v: v > 0)
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
            loss=self.cb_loss.currentText(), optimizer=self.cb_opt.currentText(),
            lr=num(self.cb_lr, c.lr, float, lambda v: 0 < v < math.inf),
            weight_decay=num(self.cb_wd, c.weight_decay, float, lambda v: 0 <= v < math.inf),
            schedule=self.cb_sched.currentText(),
            train_step=self.cb_step.currentText(), sampler=self.cb_sampler.currentText(),
            batch_size=batch,
        )

    def _set_widgets(self, c):
        """Show Config c in the widgets without triggering apply()."""
        widgets = [w for w in vars(self).values() if isinstance(w, QtWidgets.QWidget)]
        widgets += list(self.feat_checks.values())
        blocked = [(w, w.blockSignals(True)) for w in widgets]
        try:
            self.cb_data.setCurrentText(c.dataset)
            self.sp_n.setValue(c.n_points)
            self.sl_noise.setValue(round(c.noise * 100))
            self.lb_noise.setText(f"{c.noise:.2f}")
            self.cb_split.setCurrentText(c.splitter)
            self.sp_seed.setValue(c.seed)
            for f, ch in self.feat_checks.items():
                ch.setChecked(f in c.features)
            self.cb_model.setCurrentText(c.model)
            self.le_layers.setText(c.layers)
            self.sp_width.setValue(c.width)
            self.sp_classes.setValue(c.classes)
            self.cb_expand.setCurrentText(c.expand)
            self.sp_ffreq.setValue(c.fourier_freq)
            self.cb_act.setCurrentText(c.activation)
            self.cb_act_decide.setCurrentText(c.act_decide)
            self.cb_act_relate.setCurrentText(c.act_relate)
            self.cb_act_prepare.setCurrentText(c.act_prepare)
            self.cb_layer.setCurrentText(c.layer)
            self.cb_skip.setCurrentText(c.skip)
            self.cb_init.setCurrentText(c.init)
            self.cb_loss.setCurrentText(c.loss)
            self.cb_opt.setCurrentText(c.optimizer)
            self.cb_lr.setCurrentText(str(c.lr))
            self.cb_wd.setCurrentText(str(c.weight_decay))
            self.cb_sched.setCurrentText(c.schedule)
            self.cb_step.setCurrentText(c.train_step)
            self.cb_sampler.setCurrentText(c.sampler)
            self.cb_batch.setCurrentText("full" if c.batch_size is None else str(c.batch_size))
        finally:
            for w, was in blocked:
                w.blockSignals(was)
        self._clear_errors()
        self._update_enabled()
        self._show_activation(c.activation, "Activation")

    def _update_enabled(self):
        """Grey out controls that the selected model ignores."""
        model = self.cb_model.currentText()
        for w in self.mlp_only + self.custom_only:
            on = not ((model == "mlp" and w in self.custom_only)
                      or (model == "custom nn" and w in self.mlp_only))
            if w is self.sp_ffreq:
                on = on and self.cb_expand.currentText() == "fourier (sin)"
            w.setEnabled(on)
            lb = self.form.labelForField(w)
            if lb is not None:
                lb.setEnabled(on)

    def _clear_errors(self):
        for w in (self.le_layers, *self.editable):
            w.setStyleSheet("")
        self.le_layers.setToolTip(TIPS["Hidden layers"])

    def apply(self):
        if getattr(self, "_building", True):
            return
        self._update_enabled()
        cfg = self._read_config()
        self._clear_errors()
        for w in self.invalid:
            w.setStyleSheet(theme.ERROR_STYLE)
        if self.invalid:
            names = ", ".join(self.editable[w] for w in self.invalid)
            self.statusBar().showMessage(f"Not a valid value: {names} (kept the previous setting)", 8000)
        try:
            if cfg["model"] == "mlp":
                parse_layers(cfg["layers"], cfg["activation"])  # flag typos on the field itself
        except ValueError as e:
            self.le_layers.setStyleSheet(theme.ERROR_STYLE)
            self.le_layers.setToolTip(str(e))
            self.statusBar().showMessage(f"Hidden layers: {e}", 8000)
            return
        try:
            level = self.session.configure(**cfg)
        except Exception as e:  # a bad part or combination: keep the old model running
            self.statusBar().showMessage(f"Could not apply that setting: {e}", 10000)
            return
        if level:
            self._on_rebuilt(level)

    def _apply_config(self, values, what):
        """Replace every setting (load / defaults / recipe), then show it in the widgets."""
        try:
            Session(Config(**{**self.session.cfg.to_dict(), **values}))  # dry run: fail before touching anything
            self.session.configure(**values)
        except Exception as e:
            self.statusBar().showMessage(f"Could not {what}: {e}", 10000)
            return False
        self._set_widgets(self.session.cfg)
        self.session.new_data()
        self._on_rebuilt("data")
        return True

    def restore_defaults(self):
        if self._apply_config(Config().to_dict(), "restore the defaults"):
            self.statusBar().showMessage("Default settings restored.", 5000)

    def load_recipe(self, name, overrides):
        values = {**Config().to_dict(), **overrides}
        if self._apply_config(values, f"load the recipe '{name}'"):
            self.statusBar().showMessage(f"Recipe: {name}.  Press Space to train.", 8000)

    def save_settings(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save settings", "playground_settings.json", "Settings (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.session.cfg.to_dict(), f, indent=2, ensure_ascii=False)
        except OSError as e:
            self.statusBar().showMessage(f"Could not save: {e}", 10000)
            return
        self.statusBar().showMessage(f"Settings saved to {path}", 6000)

    def load_settings(self, path=None):
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Load settings", "", "Settings (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("expected a JSON object of settings")
        except (OSError, ValueError) as e:
            self.statusBar().showMessage(f"Could not read {path}: {e}", 10000)
            return
        values, skipped = clean_config(data)
        if self._apply_config(values, "load those settings"):
            note = f" (ignored unknown keys: {', '.join(skipped)})" if skipped else ""
            self.statusBar().showMessage(f"Settings loaded from {path}{note}", 8000)

    def copy_command(self):
        cmd = headless_command(self.session.cfg)
        QtWidgets.QApplication.clipboard().setText(cmd)
        self.statusBar().showMessage(f"Copied: {cmd}", 10000)

    def save_screenshot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save screenshot", f"playground_step{self.session.step_count}.png",
            "PNG image (*.png)")
        if path:
            ok = self.grab().save(path)
            self.statusBar().showMessage(f"Screenshot saved to {path}" if ok
                                         else f"Could not save {path}", 6000)

    def save_model(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save model", f"playground_step{self.session.step_count}.pt", "PyTorch model (*.pt)")
        if not path:
            return
        s = self.session
        try:
            torch.save({"config": s.cfg.to_dict(), "step": s.step_count, "params": s.n_params,
                        "net": s.describe(), "state_dict": s.model.state_dict()}, path)
        except Exception as e:
            self.statusBar().showMessage(f"Could not save the model: {e}", 10000)
            return
        self.statusBar().showMessage(f"Model saved to {path} ({s.n_params} params, step {s.step_count})", 8000)

    def export_history(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export history", "playground_history.csv", "CSV table (*.csv)")
        if not path:
            return
        cols = (self.hist_x, self.train_hist, self.test_hist, self.train_acc, self.test_acc)
        try:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write("step,train_loss,test_loss,train_acc_pct,test_acc_pct\n")
                for row in zip(*(b.view() for b in cols)):
                    cells = ("" if v != v else f"{v:.6g}" for v in row[1:])  # NaN -> empty cell
                    f.write(f"{int(row[0])}," + ",".join(cells) + "\n")
        except OSError as e:
            self.statusBar().showMessage(f"Could not export: {e}", 10000)
            return
        self.statusBar().showMessage(f"History ({self.hist_x.n} points) saved to {path}", 8000)

    def toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if len(urls) == 1 and urls[0].toLocalFile().lower().endswith(".json"):
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.load_settings(event.mimeData().urls()[0].toLocalFile())

    def show_help(self):
        QtWidgets.QMessageBox.information(self, "Help", HELP_HTML)

    def show_about(self):
        QtWidgets.QMessageBox.about(
            self, "About", f"<h3>{TITLE}</h3><p>Version {VERSION}. Tiny neural nets, trained live.</p>"
            "<p>Copyright (C) 2026 tactical-drone. Licensed under the GNU AGPL v3.0; "
            "commercial licenses available (see COMMERCIAL.md).</p>")

    # ---------------------------------------------------------------- persistence
    def _restored_config(self):
        """Last session's settings, or (None, note) if there are none or they no longer build."""
        raw = self.settings.value("config")
        if not raw:
            return None, None
        try:
            values, _ = clean_config(json.loads(raw))
            cfg = Config(**values)
            Session(Config(**cfg.to_dict()))  # must still build (parts may have been renamed)
            return cfg, "Welcome back: restored your last settings (Ctrl+D for defaults)."
        except Exception as e:
            return None, f"Could not restore your last settings ({e}); using the defaults."

    def _restore_view(self):
        geo = self.settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        try:
            view = json.loads(self.settings.value("view") or "{}")
        except ValueError:
            view = {}
        for key, w in (("steps", self.sp_steps), ("stop", self.sp_stop)):
            if isinstance(view.get(key), int):
                w.setValue(view[key])
        for key, w in (("res", self.cb_res), ("boundary", self.cb_paint), ("neurons", self.cb_npaint)):
            if isinstance(view.get(key), str) and (w.isEditable() or w.findText(view[key]) >= 0):
                w.setCurrentText(view[key])
        if isinstance(view.get("layer"), str):
            self.cb_layer_view.addItem(view["layer"])  # replaced with the real list on rebuild
            self.cb_layer_view.setCurrentText(view["layer"])
        if isinstance(view.get("tab"), int) and 0 <= view["tab"] < self.tabbar.count():
            self.tabbar.setCurrentIndex(view["tab"])

    def save_state(self):
        self.settings.setValue("config", json.dumps(self.session.cfg.to_dict(), ensure_ascii=False))
        self.settings.setValue("view", json.dumps(dict(
            steps=self.sp_steps.value(), stop=self.sp_stop.value(), res=self.cb_res.currentText(),
            boundary=self.cb_paint.currentText(), neurons=self.cb_npaint.currentText(),
            layer=self.cb_layer_view.currentText(), tab=self.tabbar.currentIndex())))
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.sync()

    def closeEvent(self, event):
        self.pause()
        try:
            self.save_state()
        except Exception:
            traceback.print_exc()  # never block closing the window
        super().closeEvent(event)

    # ---------------------------------------------------------------- rebuilding
    def reset_weights(self):
        self.session.reset_model()
        self._on_rebuilt("model")

    def reseed(self):
        self.sp_seed.setValue(int(self.rng.integers(0, 99999)))  # valueChanged -> apply()

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
            for b in (self.train_hist, self.test_hist, self.hist_x, self.train_acc, self.test_acc):
                b.clear()
            self.best_test = math.nan
            self._set_diverged(False)
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
            self.statusBar().showMessage("Grid res must be a whole number (16 to 1024).", 6000)
            return
        if hasattr(self, "session") and not self._building:
            self._rebuild_thumbs()

    def _rebuild_thumbs(self):
        if self._building:
            return
        self.gl_neurons.clear()
        self.thumbs = []
        sizes = self.session.hidden_sizes
        txt = self.cb_layer_view.currentText()
        if not sizes or not txt.startswith("layer"):
            self.view_layer = None
            self.gl_neurons.addLabel("this model has no hidden layers to show",
                                     color=theme.WINDOW_FG, size="10pt")
            self.render(record=False)
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
        self.render(record=False)

    def _switch_tab(self, i):
        """Swap the plot under the tab strip (all plots live in the one shared view)."""
        new = self.tab_plots[i]
        if new is self.shown_tab:
            return
        self.gl_bottom.removeItem(self.shown_tab)
        self.gl_bottom.addItem(new, row=1, col=0)
        self.shown_tab = new

    def _show_activation(self, name, label):
        """Plot f(x) and its slope f'(x) for the activation just picked."""
        if name == "same":
            name = self.cb_act.currentText()
        x = torch.linspace(-4, 4, 401, requires_grad=True)
        try:
            with torch.enable_grad():  # render() runs under no_grad
                y = ACTIVATIONS.get(name)()(x)
                dy, = torch.autograd.grad(y.sum(), x, allow_unused=True)
        except Exception as e:  # e.g. a custom activation that isn't differentiable
            self.statusBar().showMessage(f"Can't plot activation '{name}': {e}", 8000)
            return
        xs = x.detach().numpy()
        self.c_act.setData(xs, y.detach().numpy())
        self.c_dact.setData(xs, np.zeros_like(xs) if dy is None else dy.numpy())
        self.p_act.setTitle(f"{label}: {name}")

    def _hover(self, pos):
        """Status-bar readout of the prediction under the mouse on the boundary plot."""
        if not self.p_main.sceneBoundingRect().contains(pos):
            return
        pt = self.p_main.vb.mapSceneToView(pos)
        x, y = pt.x(), pt.y()
        if abs(x) > theme.VIEW or abs(y) > theme.VIEW:
            return
        with torch.no_grad():
            p = torch.softmax(self.session.predict(torch.tensor([[x, y]], dtype=torch.float32)), 1)[0]
        k = int(p.argmax())
        colour = theme.CLASS_NAMES[k % len(theme.CLASS_NAMES)]
        self.statusBar().showMessage(f"x {x:+.2f}   y {y:+.2f}   →   class {k} ({colour}), "
                                     f"{p[k].item() * 100:.0f}% sure", 4000)

    # ---------------------------------------------------------------- loop
    def toggle(self):
        self.running = not self.running
        self.bt_play.setText("⏸ Pause  [Space]" if self.running else "▶ Play  [Space]")
        if self.running:
            self.frame_times = []
            self.last_frame = time.perf_counter()
            self.timer.start(0)  # as fast as the event loop and vsync allow
        else:
            self.timer.stop()
        self._show_state()

    def pause(self):
        if self.running:
            self.toggle()

    def _show_state(self):
        if self.running:
            self.lb_state.setText("● training")
            self.lb_state.setStyleSheet(theme.RUNNING_STYLE)
        else:
            self.lb_state.setText("❚❚ paused")
            self.lb_state.setStyleSheet(theme.PAUSED_STYLE)

    def _set_diverged(self, on):
        self.diverged = on
        self.lb_stats.setStyleSheet(theme.STATS_DIVERGED_STYLE if on else theme.STATS_STYLE)

    def _train(self, steps):
        stop = self.sp_stop.value()
        limited = bool(stop) and self.session.step_count < stop  # already past it: keep going
        if limited:
            steps = min(steps, stop - self.session.step_count)
        loss = self.session.train(steps)
        if not math.isfinite(loss):
            self.pause()
            self._set_diverged(True)
            self.statusBar().showMessage("Training diverged (loss is NaN or infinite). Lower the "
                                         "learning rate, then press R to reset the weights.", 20000)
        elif limited and self.session.step_count >= stop and self.running:
            self.pause()
            self.statusBar().showMessage(f"Reached step {stop}: paused.", 8000)

    def single_step(self):
        self._train(1)
        self.render()

    def tick(self):
        now = time.perf_counter()
        self.frame_times.append(now - self.last_frame)
        del self.frame_times[:-60]
        self.last_frame = now
        if self.running:
            t0 = time.perf_counter()
            self._train(self.sp_steps.value())
            self.t_train = time.perf_counter() - t0
            self.render()

    @torch.no_grad()
    def render(self, record=True):
        if not hasattr(self, "grid") or self._building:
            return
        t0 = time.perf_counter()
        s = self.session
        ev = s.evaluate()
        tr, te = ev["train"], ev["test"]
        if record and self.hist_x.last() != s.step_count:  # view-only redraws add no points
            self.hist_x.add(s.step_count)
            for buf, v in ((self.train_hist, tr["loss"]), (self.test_hist, te["loss"])):
                buf.add(max(v, 1e-8) if math.isfinite(v) else np.nan)
            self.train_acc.add(tr.get("acc", math.nan) * 100)
            self.test_acc.add(te.get("acc", math.nan) * 100)
            acc_te = te.get("acc", math.nan)
            if acc_te == acc_te and not acc_te <= self.best_test:  # also replaces NaN
                self.best_test = acc_te

        logits, hidden = s.predict(self.grid, collect=True)
        r = self.grid_res
        probs = torch.softmax(logits, 1).nan_to_num_(1 / max(logits.shape[1], 1))
        rgb = BOUNDARY_PAINTERS.get(self.cb_paint.currentText())(probs, r)
        self.img.setImage(rgb, autoLevels=False, levels=(0, 255), rect=theme.PLANE)

        if getattr(self, "view_layer", None) is not None and self.thumbs:
            H = hidden[self.view_layer][:, : len(self.thumbs)].nan_to_num()
            idx = NEURON_PAINTERS.get(self.cb_npaint.currentText())(H)
            for i, im in enumerate(self.thumbs):
                im.setImage(idx[i].reshape(r, r), autoLevels=False, levels=(0, 255),
                            rect=theme.PLANE)

        xs = self.hist_x.view()
        has_test = s.Xte is not None
        self.c_train.setData(xs, self.train_hist.view())
        self.c_test.setData(xs if has_test else [], self.test_hist.view() if has_test else [])
        self.c_acc_train.setData(xs, self.train_acc.view())
        self.c_acc_test.setData(xs if has_test else [], self.test_acc.view() if has_test else [])

        t_render = time.perf_counter() - t0
        ft = self.frame_times
        fps = len(ft) / max(sum(ft), 1e-9) if (ft and self.running) else 0
        extra = "  ".join(f"{k} {v * 100:5.1f}%" for k, v in tr.items() if k != "loss")
        extra_te = "  ".join(f"{k} {v * 100:5.1f}%" for k, v in te.items() if k != "loss")
        best = f"   best {self.best_test * 100:.1f}%" if self.best_test == self.best_test else ""
        test_line = (f"test    loss {te['loss']:.4f}  {extra_te}{best}" if has_test
                     else "test    (no test set)")
        self.lb_stats.setText(
            f"net     {s.describe()}\n"
            f"neurons {sum(s.hidden_sizes)}   params {s.n_params}\n"
            f"data    {len(s.train_idx)} train / {len(s.test_idx)} test, {s.n_classes} classes\n"
            f"step    {s.step_count}   lr {s.lr:.3g}\n"
            f"train   loss {tr['loss']:.4f}  {extra}\n"
            f"{test_line}\n"
            + ("DIVERGED: lower the learning rate, press R\n" if self.diverged else "")
            + f"fps     {fps:5.1f}\n"
            f"train   {self.t_train * 1e3:5.2f} ms/frame\n"
            f"render  {t_render * 1e3:5.2f} ms/frame")


def _install_error_handler(window):
    """PyQt6 aborts the whole app on an exception inside a slot. Show it instead,
    pause training so it doesn't repeat every frame, and keep the window alive."""
    def hook(exc_type, exc, tb):
        traceback.print_exception(exc_type, exc, tb)
        try:
            window.pause()
            window.statusBar().showMessage(f"Error: {exc_type.__name__}: {exc}  (training paused, "
                                           "details in the console)", 15000)
        except Exception:
            pass
    sys.excepthook = hook


def run(opengl=False, cfg=None, fresh=False, settings_file=None):
    """Open the window. cfg overrides everything; otherwise the last session's
    settings are restored unless fresh is set. settings_file loads a saved .json."""
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True, useOpenGL=opengl,
                        background=theme.WINDOW_BG, foreground=theme.WINDOW_FG)
    app = pg.mkQApp("NN Playground")
    w = Playground(cfg, restore=cfg is None and not fresh)
    _install_error_handler(w)
    if settings_file:
        w.load_settings(settings_file)
    w.show()
    return app.exec()
