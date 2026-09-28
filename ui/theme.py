"""Colours and plane geometry shared by the UI painters."""
import numpy as np
import torch

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

WINDOW_BG, WINDOW_FG = "#0b0c10", "#c8d0e0"
TRAIN_PEN, TEST_PEN = "#00f0ff", "#ff2a6d"
STATS_STYLE = "font-family: Consolas, monospace; color: #7fffd4;"
ERROR_STYLE = "background: #501020;"
