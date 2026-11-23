import numpy as np

from learnscope.mnist.draw import BRUSH, CANVAS, stamp, stroke, to_mnist


def test_stamp_draws_a_clipped_disk():
    c = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    stamp(c, 100, 100)
    assert c[100, 100] == 255 and c[100, 100 + BRUSH] == 255 and c[100, 100 + BRUSH + 2] == 0
    corner = np.zeros_like(c)
    stamp(corner, 0, 0)  # partly outside: must clip, not crash
    assert corner[0, 0] == 255 and corner.sum() > 0
    stamp(corner, -500, -500)  # completely outside: nothing happens
    stamp(corner, 10_000, 3)


def test_stroke_connects_two_points():
    c = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    stroke(c, (20, 20), (200, 20))
    assert c[20, 20] == 255 and c[20, 110] == 255 and c[20, 200] == 255  # continuous, no gaps
    dot = np.zeros_like(c)
    stroke(dot, None, (50, 50))
    assert dot[50, 50] == 255


def test_empty_canvas_gives_an_empty_digit():
    out = to_mnist(np.zeros((CANVAS, CANVAS), dtype=np.uint8))
    assert out.shape == (28, 28) and out.dtype == np.float32 and out.sum() == 0


def test_digit_is_shrunk_into_the_frame_and_centred_whatever_its_position():
    for (x, y) in [(40, 60), (180, 170), (112, 112)]:
        c = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
        stroke(c, (x, y - 50), (x, y + 50))  # a vertical bar somewhere on the pad
        img = to_mnist(c)
        assert 0 <= img.min() and img.max() <= 1 and img.sum() > 5
        ys, xs = np.mgrid[0:28, 0:28]
        cx, cy = (img * xs).sum() / img.sum(), (img * ys).sum() / img.sum()
        assert abs(cx - 13.5) < 1.0 and abs(cy - 13.5) < 1.0, (cx, cy)


def test_drawing_fills_the_box_whatever_its_size():
    small, big = np.zeros((CANVAS, CANVAS), dtype=np.uint8), np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    stroke(small, (100, 100), (100, 120))  # a short mark
    stroke(big, (100, 20), (100, 200))  # a tall one
    for c in (small, big):
        img = to_mnist(c)
        rows = np.nonzero(img.max(axis=1) > 0.2)[0]
        assert rows[-1] - rows[0] + 1 >= 18  # both span (almost) the 20-pixel box
