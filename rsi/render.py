"""
render: decision maps without a window. ascii_map gives text rows (class
digits 0-9a-z, train points '#'); png_map writes a PNG with the UI's palette
(PyQt6 QImage when available, else a stdlib zlib/struct encoder).

Both predict through Session.predict (eval mode, forked RNG), so drawing a
map never changes training.

Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md.
"""
import struct
import zlib

import numpy as np
import torch

VIEW = 1.6  # same plane as the UI
DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"
BG = np.array([11, 12, 16], dtype=np.float32)
CLASS_COLORS = np.array([[0, 245, 212], [241, 91, 181], [0, 187, 249], [254, 228, 64], [155, 93, 229]],
                        dtype=np.float32)  # the UI's juicy palette, cycled for more classes
TRAIN_RGB, TEST_RGB = (0, 240, 255), (255, 42, 109)


def parse_size(text, default=(48, 24)):
    """'48x24' -> (48, 24)."""
    if text in (None, True, ""):
        return default
    if isinstance(text, (tuple, list)):
        return int(text[0]), int(text[1])
    w, _, h = str(text).lower().partition("x")
    w, h = int(w), int(h or max(1, int(w) // 2))
    if not (4 <= w <= 400 and 2 <= h <= 200):
        raise ValueError(f"map size {text!r} out of range (4..400 x 2..200)")
    return w, h


def _probs(session, xs, ys):
    yy, xx = torch.meshgrid(torch.as_tensor(ys, dtype=torch.float32), torch.as_tensor(xs, dtype=torch.float32),
                            indexing="ij")
    pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], 1)
    logits = session.predict(pts)
    return torch.softmax(logits.float(), 1)


def _cell(v, n):
    """Raw coordinate in [-VIEW, VIEW] -> index 0..n-1."""
    return int(np.clip((v + VIEW) / (2 * VIEW) * n, 0, n - 1))


def ascii_map(session, w=48, h=24, points=True):
    """h strings of w chars: the predicted class per cell (top row = +y), '#' where train points fall."""
    xs = (np.arange(w) + 0.5) / w * 2 * VIEW - VIEW
    ys = VIEW - (np.arange(h) + 0.5) / h * 2 * VIEW
    cls = _probs(session, xs, ys).argmax(1).view(h, w).numpy()
    rows = [[DIGITS[c % len(DIGITS)] for c in row] for row in cls]
    if points:
        for x, y in session.Xtr_raw.numpy():
            rows[h - 1 - _cell(y, h)][_cell(x, w)] = "#"
    return ["".join(r) for r in rows]


def map_rgb(session, size=256, points=True):
    """uint8 (size, size, 3) decision image like the UI's 'confidence + edges' painter."""
    xs = (np.arange(size) + 0.5) / size * 2 * VIEW - VIEW
    ys = VIEW - (np.arange(size) + 0.5) / size * 2 * VIEW
    probs = _probs(session, xs, ys).numpy()
    K = probs.shape[1]
    colors = CLASS_COLORS[np.arange(K) % len(CLASS_COLORS)]
    conf = np.sqrt(np.clip((probs.max(1, keepdims=True) - 1 / K) / max(1e-9, 1 - 1 / K), 0, 1))
    rgb = (BG + (probs @ colors - BG) * (0.25 + 0.6 * conf)).reshape(size, size, 3)
    cls = probs.argmax(1).reshape(size, size)
    edge = np.zeros((size, size), bool)
    edge[:, 1:] |= cls[:, 1:] != cls[:, :-1]
    edge[1:, :] |= cls[1:, :] != cls[:-1, :]
    rgb[edge] = 235.0
    img = np.clip(rgb, 0, 255).astype(np.uint8)
    if points:
        r = max(1, size // 128)
        for X, col in ((session.Xtr_raw, TRAIN_RGB), (session.Xte_raw, TEST_RGB)):
            for x, y in X.numpy():
                cx, cy = _cell(x, size), size - 1 - _cell(y, size)
                img[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1] = col
    return img


def write_png(path, img):
    """Encode uint8 (h, w, 3) with zlib/struct (no dependencies)."""
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[i].tobytes() for i in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


def _qt_png(path, img):
    from PyQt6.QtGui import QImage
    h, w, _ = img.shape
    buf = np.ascontiguousarray(img)
    q = QImage(buf.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return q.copy().save(str(path), "PNG")


def png_map(session, path, size=256):
    """Write the decision map as a PNG; returns the path."""
    img = map_rgb(session, int(size))
    try:
        ok = _qt_png(path, img)
    except Exception:
        ok = False
    if not ok:
        write_png(path, img)
    return str(path)
