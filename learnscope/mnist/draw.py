"""Drawing pad support: stamp brush strokes onto a canvas and convert it to an MNIST-style 28x28 image."""

from __future__ import annotations

import numpy as np

CANVAS = 224
BRUSH = 8  # radius in canvas pixels


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y) <= r * r


_MASK = _disk(BRUSH)


def stamp(canvas: np.ndarray, x: float, y: float) -> None:
    """Draw a white disk centred at (x, y), clipped to the canvas."""
    cx, cy = int(round(x)), int(round(y))
    h, w = canvas.shape
    x0, x1 = max(0, cx - BRUSH), min(w, cx + BRUSH + 1)
    y0, y1 = max(0, cy - BRUSH), min(h, cy + BRUSH + 1)
    if x0 >= x1 or y0 >= y1:
        return
    m = _MASK[y0 - (cy - BRUSH) : y1 - (cy - BRUSH), x0 - (cx - BRUSH) : x1 - (cx - BRUSH)]
    canvas[y0:y1, x0:x1][m] = 255


def stroke(canvas: np.ndarray, p0: tuple[float, float] | None, p1: tuple[float, float]) -> None:
    """Draw from p0 to p1 (or just a dot when there is no previous point)."""
    if p0 is None:
        stamp(canvas, *p1)
        return
    dist = max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))
    for t in np.linspace(0.0, 1.0, int(dist / 2) + 2):
        stamp(canvas, p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)


def _area_resize(img: np.ndarray, out: int) -> np.ndarray:
    edges = np.linspace(0, img.shape[0], out + 1).astype(int)
    rows = np.add.reduceat(img, edges[:-1], axis=0) / np.diff(edges)[:, None]
    return np.add.reduceat(rows, edges[:-1], axis=1) / np.diff(edges)[None, :]


def to_mnist(canvas: np.ndarray) -> np.ndarray:
    """Canvas (uint8) -> (28, 28) float32 in [0, 1], prepared like MNIST digits: crop to the drawn
    area, fit it (aspect ratio kept) into a 20x20 box inside the 28x28 frame, then shift so the
    centre of mass is at the centre."""
    a = canvas.astype(np.float64) / 255.0
    rows, cols = np.nonzero(a.max(axis=1) > 0)[0], np.nonzero(a.max(axis=0) > 0)[0]
    if rows.size == 0:
        return np.zeros((28, 28), dtype=np.float32)
    a = a[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    side = max(a.shape)
    sq = np.zeros((side, side))  # square, drawing centred
    r0, c0 = (side - a.shape[0]) // 2, (side - a.shape[1]) // 2
    sq[r0 : r0 + a.shape[0], c0 : c0 + a.shape[1]] = a
    k = max(1, -(-40 // side))  # make sure every output pixel averages at least a 2x2 block
    small = _area_resize(np.repeat(np.repeat(sq, k, axis=0), k, axis=1), 20)
    img = np.zeros((28, 28))
    img[4:24, 4:24] = small
    total = img.sum()
    if total < 1e-3:
        return img.astype(np.float32)
    ys, xs = np.mgrid[0:28, 0:28]
    dx = int(round(13.5 - (img * xs).sum() / total))
    dy = int(round(13.5 - (img * ys).sum() / total))
    out = np.zeros_like(img)
    sy0, sy1 = max(0, -dy), min(28, 28 - dy)
    sx0, sx1 = max(0, -dx), min(28, 28 - dx)
    out[sy0 + dy : sy1 + dy, sx0 + dx : sx1 + dx] = img[sy0:sy1, sx0:sx1]
    return out.astype(np.float32)
