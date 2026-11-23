import numpy as np
import pytest

from learnscope.config import (
    DEFAULT_CONFIG,
    HiddenLayer,
    decode_config,
    encode_config,
    layer_specs,
    sanitize_config,
)
from learnscope.datasets import generate_dataset, split_dataset
from learnscope.forward import evaluate, forward_all, forward_grid
from learnscope.mlp import MLP, Optimizer
from learnscope.snapshots import HistoryBuffer, snapshot_bytes
from learnscope.stats import build_layer_stats, histogram
from learnscope.trainer import Trainer


# ------------------------------------------------------------------ datasets
@pytest.mark.parametrize("kind", ["moons", "circles", "spiral", "xor", "blobs"])
def test_dataset_size_classes_bounds_and_determinism(kind):
    a = generate_dataset(kind, 300, 0.1, 5)
    b = generate_dataset(kind, 300, 0.1, 5)
    assert a.n == 300 and a.x.shape == (300, 2)
    assert np.array_equal(a.x, b.x)
    assert 90 < a.y.sum() < 210
    assert np.abs(a.x).max() < 2.5


def test_different_seeds_differ_and_split_keeps_every_point():
    assert not np.array_equal(generate_dataset("moons", 50, 0.1, 1).x, generate_dataset("moons", 50, 0.1, 2).x)
    ds = generate_dataset("circles", 200, 0.05, 3)
    train, test = split_dataset(ds, 0.25, 3)
    assert test.n == 50 and train.n == 150
    both = np.vstack([train.x, test.x])
    assert sorted(map(tuple, both)) == sorted(map(tuple, ds.x))


# ------------------------------------------------------------------ config + share links
def test_config_round_trips_through_the_url_encoding():
    cfg = DEFAULT_CONFIG.with_(lr=0.5, dataset="xor", layers=(HiddenLayer(5, "gelu"),))
    assert decode_config(encode_config(cfg)) == cfg


def test_hostile_links_are_repaired_or_rejected():
    assert decode_config("%%%not-base64") is None
    assert decode_config("A" * 10000) is None
    fixed = sanitize_config({"lr": 1e9, "layers": [{"units": 999, "activation": "evil"}], "dataset": "nope", "seed": "x"})
    assert fixed.lr <= 3 and fixed.layers[0].units == 16 and fixed.layers[0].activation == "tanh"
    assert fixed.dataset == DEFAULT_CONFIG.dataset and fixed.seed == DEFAULT_CONFIG.seed
    assert sanitize_config("not a dict") == DEFAULT_CONFIG
    assert sanitize_config({"lr": float("nan")}).lr == DEFAULT_CONFIG.lr
    assert len(sanitize_config({"layers": [{"units": 4, "activation": "relu"}] * 50}).layers) == 6
    assert sanitize_config({"layers": []}).layers == DEFAULT_CONFIG.layers


def test_layer_specs_end_in_a_single_sigmoid_unit():
    specs = layer_specs(DEFAULT_CONFIG)
    assert [(s.nin, s.nout) for s in specs] == [(2, 8), (8, 8), (8, 1)] and specs[-1].activation == "sigmoid"


# ------------------------------------------------------------------ backprop vs numerical gradients
@pytest.mark.parametrize("acts", [["relu", "tanh"], ["gelu", "sigmoid"], ["tanh", "gelu", "relu"]])
@pytest.mark.parametrize("l2", [0.0, 0.05])
def test_numpy_backprop_matches_numerical_gradients(acts, l2):
    cfg = DEFAULT_CONFIG.with_(layers=tuple(HiddenLayer(4, a) for a in acts), seed=3)
    net = MLP(layer_specs(cfg), cfg.seed)
    ds = generate_dataset("moons", 24, 0.1, 2)
    _, dW, db = net.loss_and_grads(ds.x, ds.y, l2)
    h = 1e-6
    for params, grads in ((net.W, dW), (net.b, db)):
        for p, g in zip(params, grads):
            it = np.nditer(p, flags=["multi_index"])
            for _ in it:
                idx = it.multi_index
                old = p[idx]
                p[idx] = old + h
                lp = net.loss_and_grads(ds.x, ds.y, l2)[0]
                p[idx] = old - h
                lm = net.loss_and_grads(ds.x, ds.y, l2)[0]
                p[idx] = old
                assert abs(g[idx] - (lp - lm) / (2 * h)) < 1e-6


def test_stable_loss_for_extreme_logits():
    assert np.isfinite(MLP.bce(np.array([[1000.0], [-1000.0]]), np.array([0.0, 1.0])))


# ------------------------------------------------------------------ trainer
def make_trainer(**over):
    cfg = DEFAULT_CONFIG.with_(**over)
    ds = generate_dataset(cfg.dataset, cfg.samples, cfg.noise, cfg.seed)
    train, test = split_dataset(ds, cfg.test_split, cfg.seed)
    return Trainer(cfg, train, test), train, test


def test_learns_two_blobs_quickly():
    t, _, _ = make_trainer(dataset="blobs", layers=(HiddenLayer(4, "relu"),), lr=0.05, samples=200)
    first = t.snapshot()
    for _ in range(80):
        t.train_step()
    last = t.snapshot()
    assert last.step == 80 and last.train_loss < first.train_loss and last.test_acc > 0.9


@pytest.mark.parametrize("opt,lr", [("sgd", 0.3), ("momentum", 0.05), ("adam", 0.03)])
def test_every_optimizer_reduces_the_loss(opt, lr):
    t, _, _ = make_trainer(dataset="moons", optimizer=opt, lr=lr, layers=(HiddenLayer(6, "tanh"),))
    first = t.snapshot()
    for _ in range(150):
        t.train_step()
    assert t.snapshot().train_loss < first.train_loss


def test_training_is_deterministic_for_a_fixed_seed():
    a, _, _ = make_trainer(dataset="xor")
    b, _, _ = make_trainer(dataset="xor")
    for _ in range(15):
        a.train_step()
        b.train_step()
    assert a.snapshot().train_loss == b.snapshot().train_loss


def test_snapshot_contents_and_agreement_with_the_independent_forward_pass():
    t, train, test = make_trainer(
        layers=(HiddenLayer(5, "relu"), HiddenLayer(4, "gelu"), HiddenLayer(3, "tanh"), HiddenLayer(3, "sigmoid"))
    )
    for _ in range(10):
        t.train_step()
    snap = t.snapshot()
    loss, acc = evaluate(t.specs, snap.weights, train.x, train.y)
    assert abs(snap.train_loss - loss) < 1e-6 and abs(snap.train_acc - acc) < 1e-9
    assert len(snap.layers) == 5 and len(snap.weights) == 5
    assert all(np.isfinite(l.g_norm) and l.g_norm > 0 for l in snap.layers)
    assert snapshot_bytes(snap) > 0


def test_snapshot_weights_are_copies_not_views():
    t, _, _ = make_trainer()
    s0 = t.snapshot()
    w0 = s0.weights[0][0].copy()
    for _ in range(20):
        t.train_step()
    assert np.array_equal(s0.weights[0][0], w0)  # later training must not rewrite recorded history


def test_live_settings_apply_without_a_restart():
    t, train, _ = make_trainer()
    t.set_live(lr=0.001, l2=0.02, batch_size=10_000)
    assert t.lr == 0.001 and t.l2 == 0.02 and t.batch_size == train.n
    t.train_step()


def test_l2_penalty_shrinks_weights():
    plain, _, _ = make_trainer(layers=(HiddenLayer(8, "tanh"),), optimizer="sgd", lr=0.05)
    reg, _, _ = make_trainer(layers=(HiddenLayer(8, "tanh"),), optimizer="sgd", lr=0.05, l2=0.05)
    for _ in range(300):
        plain.train_step()
        reg.train_step()
    norm = lambda t: sum(float(np.sum(w * w)) for w in t.net.W)  # noqa: E731
    assert norm(reg) < norm(plain)


def test_optimizer_rejects_unknown_kind():
    with pytest.raises(ValueError):
        Optimizer("lion", [np.zeros(2)])


# ------------------------------------------------------------------ forward passes
def test_forward_grid_agrees_with_pointwise_forward():
    t, _, _ = make_trainer()
    for _ in range(5):
        t.train_step()
    w = t.net.weights()
    grid = forward_grid(t.specs, w, g=6, extent=1.0)
    assert grid[-1].shape == (6, 6, 1) and grid[0].shape == (6, 6, 8)
    x0, x1 = -1 + 0.5 / 3, 1 - 0.5 / 3  # cell (row 0, col 0): left, TOP of the picture
    expect = forward_all(t.specs, w, np.array([[x0, x1]]))[-1][0, 0]
    assert abs(grid[-1][0, 0, 0] - expect) < 1e-12


# ------------------------------------------------------------------ stats
def test_layer_stats_detect_dead_relus_and_saturation():
    acts = np.array([[0, 1], [0, 2], [0, 0.5], [0, 3]], dtype=float)  # unit 0 never fires
    s = build_layer_stats("relu", 2, np.array([[0.1, -0.2], [0.3, 0.4]]), np.array([[0.01, 0.02], [-0.01, 0]]), acts)
    assert abs(s.dead_pct - 0.5) < 1e-12 and abs(s.zero_pct - 0.5) < 1e-12 and abs(s.unit_mean[1] - 1.625) < 1e-12
    t = build_layer_stats("tanh", 1, np.array([[1.0]]), np.array([[0.0]]), np.array([[0.99], [-0.99], [0.2], [0.97]]))
    assert abs(t.sat_pct - 0.75) < 1e-12 and t.dead_pct == 0


def test_histograms_count_every_value_and_survive_edge_cases():
    h = histogram(np.array([0, 0.1, 0.2, 0.9, 1.0]), 4)
    assert sum(h.counts) == 5 and h.lo == 0 and h.hi == 1
    assert sum(histogram(np.array([3.0, 3.0, 3.0]), 5).counts) == 3
    assert sum(histogram(np.array([1.0, np.nan, np.inf, 2.0]), 4).counts) == 2  # non-finite values ignored
    assert sum(histogram(np.array([]), 4).counts) == 0


# ------------------------------------------------------------------ history buffer
def test_history_thins_old_items_but_keeps_first_and_last():
    buf = HistoryBuffer(lambda x: 1, 10, 10**9)
    for i in range(100):
        buf.push(i)
    items = buf.to_list()
    assert len(buf) <= 10 and items[0] == 0 and items[-1] == 99
    assert all(b > a for a, b in zip(items, items[1:]))


def test_history_respects_the_byte_limit_and_finds_nearest():
    buf = HistoryBuffer(lambda x: 100, 10**9, 1000)
    for i in range(200):
        buf.push(i)
    assert buf.total_bytes <= 1000 and buf.last() == 199
    small = HistoryBuffer(lambda x: 1)
    for n in (0, 10, 20, 30):
        small.push(n)
    ident = lambda x: x  # noqa: E731
    assert small.nearest(14, ident) == 1 and small.nearest(16, ident) == 2 and small.nearest(1000, ident) == 3
    assert HistoryBuffer(lambda x: 1).nearest(5, ident) == -1
