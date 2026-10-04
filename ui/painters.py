"""
Painters: turn model outputs on the grid into images. Pure tensor maths, no Qt,
so they are easy to swap or test.

  BOUNDARY_PAINTERS  fn(probs (r*r, K), r) -> uint8 (r, r, 3)
  NEURON_PAINTERS    fn(hidden (r*r, n)) -> uint8 (n, r*r) indices into NEURON_LUT
"""
import torch

from nncore import Registry

from .theme import BG_T, CLASS_COLORS_T, VIEW

BOUNDARY_PAINTERS = Registry("boundary painter", "fn(probs, r) -> uint8 (r, r, 3)")
NEURON_PAINTERS = Registry("neuron painter", "fn(hidden) -> uint8 (n, r*r)")


def make_grid(res):
    """(res*res, 2) raw points covering the view plane, row = y, col = x."""
    r = max(16, min(int(res), 1024))
    xs = torch.linspace(-VIEW, VIEW, r)
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    return r, torch.stack([xx.reshape(-1), yy.reshape(-1)], 1)


@BOUNDARY_PAINTERS.register("confidence + edges")
def confidence_edges(probs, r):
    K = probs.shape[1]
    conf = ((probs.amax(1, keepdim=True) - 1 / K) / (1 - 1 / K)).sqrt_()
    rgb = BG_T + (probs @ CLASS_COLORS_T[:K] - BG_T) * (0.25 + 0.6 * conf)
    rgb = rgb.view(r, r, 3)
    cls = probs.argmax(1).view(r, r)
    edge = torch.zeros((r, r), dtype=torch.bool)
    edge[:, 1:] |= cls[:, 1:] != cls[:, :-1]
    edge[1:, :] |= cls[1:, :] != cls[:-1, :]
    rgb[edge] = 235.0
    return rgb.to(torch.uint8).numpy()


@BOUNDARY_PAINTERS.register("hard classes")
def hard_classes(probs, r):
    K = probs.shape[1]
    rgb = BG_T + (CLASS_COLORS_T[:K][probs.argmax(1)] - BG_T) * 0.6
    return rgb.view(r, r, 3).to(torch.uint8).numpy()


@NEURON_PAINTERS.register("full range")
def full_range(H):
    """Each neuron stretched min..max over the whole palette, however flat it is."""
    lo, hi = H.amin(0), H.amax(0)
    H = (H - lo) / (hi - lo).clamp_min(1e-9)
    return (H * 255).to(torch.uint8).T.contiguous().numpy()


@NEURON_PAINTERS.register("per-neuron max")
def per_neuron_max(H):
    H = H / H.abs().amax(0).clamp_min(1e-9)  # each neuron scaled to [-1, 1]
    return ((H + 1) * 127.5).to(torch.uint8).T.contiguous().numpy()


@NEURON_PAINTERS.register("tanh squash")
def tanh_squash(H):
    return ((torch.tanh(H) + 1) * 127.5).to(torch.uint8).T.contiguous().numpy()
