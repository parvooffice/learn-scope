"""Convert the MNIST subset PNG (10,000 train + 2,000 test, one 28x28 image per row of a 784-wide
grayscale PNG) into learnscope/data/mnist-12k.npz.

    python scripts/make_mnist_npz.py <mnist-12k.png> <mnist-12k-labels.bin> [out.npz]

Uses only the standard library (zlib) to decode the PNG, so no imaging package is needed.
The subset was cut from the public "model-builder" MNIST sprite
(https://storage.googleapis.com/learnjs-data/model-builder/): rows 0..9999 train, 55000..56999 test.
"""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

import numpy as np


def decode_gray_png(data: bytes) -> np.ndarray:
    """Decode an 8-bit grayscale, non-interlaced PNG into a (height, width) uint8 array."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    pos, idat, width = 8, [], 0
    height = depth = ctype = interlace = None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height, depth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", chunk)
        elif kind == b"IDAT":
            idat.append(chunk)
        elif kind == b"IEND":
            break
    if (depth, ctype, interlace) != (8, 0, 0):
        raise ValueError(f"only 8-bit grayscale non-interlaced PNGs are supported (got {depth}, {ctype}, {interlace})")
    raw = np.frombuffer(zlib.decompress(b"".join(idat)), dtype=np.uint8)
    stride = width + 1
    raw = raw.reshape(height, stride)
    out = np.zeros((height, width), dtype=np.uint8)
    prev = np.zeros(width, dtype=np.int32)
    for y in range(height):
        f = raw[y, 0]
        line = raw[y, 1:].astype(np.int32)
        if f == 0:
            cur = line
        elif f == 1:  # Sub
            cur = np.cumsum(line) & 255
        elif f == 2:  # Up
            cur = (line + prev) & 255
        elif f == 3:  # Average (sequential)
            cur = np.zeros(width, dtype=np.int32)
            for x in range(width):
                left = cur[x - 1] if x else 0
                cur[x] = (line[x] + ((left + prev[x]) >> 1)) & 255
        elif f == 4:  # Paeth (sequential)
            cur = np.zeros(width, dtype=np.int32)
            for x in range(width):
                a = cur[x - 1] if x else 0
                b = prev[x]
                c = prev[x - 1] if x else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                cur[x] = (line[x] + pred) & 255
        else:
            raise ValueError(f"bad PNG filter type {f}")
        out[y] = cur
        prev = cur
    return out


def main() -> None:
    png, labels_path = Path(sys.argv[1]), Path(sys.argv[2])
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(__file__).resolve().parents[1] / "learnscope" / "data" / "mnist-12k.npz"
    pixels = decode_gray_png(png.read_bytes())
    labels = np.frombuffer(labels_path.read_bytes(), dtype=np.uint8)
    if pixels.shape != (12000, 784) or labels.shape != (12000,):
        raise SystemExit(f"unexpected shapes {pixels.shape}, {labels.shape}")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, images=pixels.reshape(12000, 28, 28), labels=labels, n_train=np.array(10000))
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB); classes: {np.bincount(labels).tolist()}")


if __name__ == "__main__":
    main()
