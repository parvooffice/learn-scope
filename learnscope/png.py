"""A tiny PNG encoder (standard library + NumPy only) so images can be shown without PIL."""

from __future__ import annotations

import base64
import struct
import zlib

import numpy as np


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def encode_png(img: np.ndarray) -> bytes:
    """Encode a uint8 array of shape (h, w) [grayscale] or (h, w, 3) [RGB] as PNG bytes."""
    a = np.ascontiguousarray(img, dtype=np.uint8)
    if a.ndim == 2:
        color_type = 0
    elif a.ndim == 3 and a.shape[2] == 3:
        color_type = 2
    else:
        raise ValueError("expected (h, w) or (h, w, 3)")
    h, w = a.shape[:2]
    rows = a.reshape(h, -1)
    raw = np.concatenate([np.zeros((h, 1), dtype=np.uint8), rows], axis=1)  # filter byte 0 for every row
    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw.tobytes(), 3)) + _chunk(b"IEND", b"")


def data_uri(img: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(encode_png(img)).decode("ascii")


def upscale(img: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbour upscale (keeps tiny images crisp)."""
    return np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)
