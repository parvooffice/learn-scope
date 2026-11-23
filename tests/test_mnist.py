import time

import numpy as np
import pytest

from learnscope.mnist.data import load_mnist
from learnscope.mnist.layers import Conv2D, Dense, Flatten, MaxPool2, ReLU, softmax, softmax_cross_entropy
from learnscope.mnist.trainer import MnistConfig, MnistRun, MnistTrainer, Net, build_model


def numeric_check(net: Net, x: np.ndarray, y: np.ndarray, h=1e-6, tol=2e-5, samples=6):
    """Compare analytic gradients of every parameter with central differences (float64)."""
    loss, d = softmax_cross_entropy(net.forward(x), y)
    net.backward(d)
    rng = np.random.default_rng(0)
    for p, g in zip(net.params, [g.copy() for g in net.grads]):
        flat_idx = rng.choice(p.size, size=min(samples, p.size), replace=False)
        for fi in flat_idx:
            idx = np.unravel_index(fi, p.shape)
            old = p[idx]
            p[idx] = old + h
            lp = softmax_cross_entropy(net.forward(x), y)[0]
            p[idx] = old - h
            lm = softmax_cross_entropy(net.forward(x), y)[0]
            p[idx] = old
            assert abs(g[idx] - (lp - lm) / (2 * h)) < tol, (p.shape, idx)


def small_cnn(dtype=np.float64):
    rng = np.random.default_rng(5)
    return Net([Conv2D(1, 2, 3, rng, dtype), ReLU(), MaxPool2(), Conv2D(2, 3, 2, rng, dtype), ReLU(), Flatten(), Dense(3 * 2 * 2, 4, rng, dtype)])


def test_conv_pool_dense_backprop_matches_numerical_gradients():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 1, 8, 8))  # conv3 -> 6x6, pool -> 3x3, conv2 -> 2x2
    numeric_check(small_cnn(), x, np.array([0, 3, 1]))


def test_mlp_backprop_matches_numerical_gradients():
    rng = np.random.default_rng(2)
    net = Net([Flatten(), Dense(12, 7, rng, np.float64), ReLU(), Dense(7, 5, rng, np.float64)])
    numeric_check(net, rng.normal(size=(4, 3, 2, 2)), np.array([0, 1, 4, 2]))


def test_conv_forward_matches_a_hand_computed_example():
    c = Conv2D(1, 1, 2, np.random.default_rng(0), np.float64)
    c.W[...] = [[1, 1, 1, 1]]
    c.b[...] = 0
    x = np.arange(1, 10, dtype=np.float64).reshape(1, 1, 3, 3)
    assert np.allclose(c.forward(x)[0, 0], [[12, 16], [24, 28]])


def test_maxpool_forward_backward_and_ties():
    p = MaxPool2()
    x = np.array([[[[1, 2], [3, 4]]]], dtype=np.float64)
    assert p.forward(x)[0, 0, 0, 0] == 4
    assert np.array_equal(p.backward(np.ones((1, 1, 1, 1)))[0, 0], [[0, 0], [0, 1]])
    tie = np.ones((1, 1, 2, 2))
    p.forward(tie)
    assert np.allclose(p.backward(np.ones((1, 1, 1, 1))), 0.25)  # ties share the gradient
    with pytest.raises(ValueError):
        p.forward(np.ones((1, 1, 3, 3)))


def test_softmax_is_stable_and_cross_entropy_gradient_is_p_minus_onehot():
    assert np.isfinite(softmax(np.array([[1000.0, 0.0, -1000.0]]))).all()
    logits = np.array([[2.0, 1.0, 0.1]])
    loss, d = softmax_cross_entropy(logits, np.array([0]))
    p = softmax(logits)
    assert abs(loss + np.log(p[0, 0])) < 1e-12 and np.allclose(d, p - np.array([[1, 0, 0]]))


def test_parameter_counts_and_shapes():
    assert build_model("cnn").count_params() == 5386  # 208 + 1168 + 4010
    assert build_model("mlp").count_params() == 109386
    out = build_model("cnn").forward(np.zeros((2, 1, 28, 28), dtype=np.float32))
    assert out.shape == (2, 10)


def test_shipped_data_has_the_right_shape_and_classes():
    d = load_mnist()
    assert d.train_x.shape == (10000, 1, 28, 28) and d.test_x.shape == (2000, 1, 28, 28)
    assert 0 <= d.train_x.min() and d.train_x.max() <= 1
    assert set(np.unique(d.train_y)) == set(range(10)) and d.n_test == 2000


def test_cnn_learns_real_digits_quickly():
    d = load_mnist()
    t = MnistTrainer(MnistConfig("cnn", lr=0.003, batch_size=64, seed=1), d)
    acc0 = t.evaluate().test_acc
    t0 = time.time()
    for _ in range(120):
        t.train_step()
    ev = t.evaluate()
    assert acc0 < 0.3 and ev.test_acc > 0.8, (acc0, ev.test_acc)
    assert ev.confusion.sum() == 2000 and np.trace(ev.confusion) == round(ev.test_acc * 2000)
    assert len(ev.worst) == 12 and all(w["pred"] != w["truth"] for w in ev.worst)
    assert time.time() - t0 < 120


def test_mlp_learns_too_and_exposes_visualisation_data():
    d = load_mnist()
    t = MnistTrainer(MnistConfig("mlp", lr=0.003, batch_size=64, seed=2), d)
    for _ in range(80):
        t.train_step()
    assert t.evaluate().test_acc > 0.8
    f = t.focus(d.test_x[3, 0])
    assert f.hidden.shape == (128,) and abs(f.probs.sum() - 1) < 1e-5 and t.filters().shape == (16, 28, 28)


def test_cnn_focus_returns_feature_maps():
    t = MnistTrainer(MnistConfig("cnn"), load_mnist())
    f = t.focus(np.zeros((28, 28)))
    assert f.conv1.shape == (8, 24, 24) and f.conv2.shape == (16, 10, 10) and t.filters().shape == (8, 5, 5)


def test_background_run_records_curves_and_supports_focus_and_drawing():
    r = MnistRun(MnistConfig("mlp", lr=0.003), load_mnist(), speed=8)
    try:
        r.start()
        t0 = time.time()
        while time.time() - t0 < 30 and not (len(r.points) > 15 and r.points[-1].test_acc is not None or any(p.test_acc for p in r.points)):
            time.sleep(0.05)
        r.pause()
        assert any(p.test_acc is not None for p in r.points) and r.error is None
        r.focus_test(5)
        assert r.focus_label == int(r.data.test_y[5])
        r.focus_pixels(np.zeros((28, 28)))
        assert r.focus_label is None and abs(r.focus.probs.sum() - 1) < 1e-5
        assert r.images_per_second > 0
    finally:
        r.close()
