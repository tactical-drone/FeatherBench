"""Colours and plane geometry shared by the UI painters."""
import numpy as np
import torch

VIEW = 1.6  # half-width of the plotted plane
PLANE = (-VIEW, -VIEW, 2 * VIEW, 2 * VIEW)  # x, y, w, h handed to ImageItem.setImage

BG = np.array([11, 12, 16], dtype=np.float32)
# the juicy palette below, ordered so neighbouring classes swap warm/cool
CLASS_COLORS = np.array([
    [0, 245, 212],    # aqua
    [241, 91, 181],   # pink
    [0, 187, 249],    # sky
    [254, 228, 64],   # lemon
    [155, 93, 229],   # violet
    [255, 150, 40],   # orange
    [140, 240, 90],   # lime
    [220, 225, 255],  # ice
], dtype=np.float32)
CLASS_NAMES = ["aqua", "pink", "sky", "lemon", "violet", "orange", "lime", "ice"]
BG_T, CLASS_COLORS_T = torch.from_numpy(BG), torch.from_numpy(CLASS_COLORS)


def class_colors(k):
    """(k, 3) float colours for k classes; past the palette they repeat."""
    return CLASS_COLORS_T[torch.arange(k) % len(CLASS_COLORS_T)]


# juicy LUT for neuron maps, no dark middle so every neuron stays colourful:
# aqua .. sky .. violet (0) .. pink .. lemon
_STOPS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
_JUICY = np.array([
    [0, 245, 212],    # aqua
    [0, 187, 249],    # sky
    [155, 93, 229],   # violet
    [241, 91, 181],   # pink
    [254, 228, 64],   # lemon
], dtype=np.float32)
_t = np.linspace(0, 1, 256)
NEURON_LUT = np.stack([np.interp(_t, _STOPS, _JUICY[:, c]) for c in range(3)], 1).astype(np.uint8)

WINDOW_BG, WINDOW_FG = "#0b0c10", "#c8d0e0"
TRAIN_PEN, TEST_PEN = "#00f0ff", "#ff2a6d"
STATS_STYLE = "font-family: Consolas, monospace; color: #7fffd4;"
STATS_DIVERGED_STYLE = "font-family: Consolas, monospace; color: #ff2a6d;"
SECTION_STYLE = "color: #ff2a6d; font-weight: bold; margin-top: 6px;"
ERROR_STYLE = "background: #501020;"
RUNNING_STYLE = "color: #00f0ff; font-weight: bold; padding: 0 8px;"
PAUSED_STYLE = "color: #8890a0; padding: 0 8px;"
JOB_STYLE = "color: #fee440; padding: 0 8px;"
GHOST_PEN = "#8890a0"
