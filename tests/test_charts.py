import base64
import json
import struct
import zlib

import numpy as np

from learnscope.colors import diverging, rgb_hex, viridis
from learnscope.config import DEFAULT_CONFIG, HiddenLayer, layer_specs
from learnscope.forward import forward_grid
from learnscope.landscape import compute_landscape
from learnscope.png import data_uri, encode_png, upscale
from learnscope.runner import Run
from learnscope.ui import charts


def decode_png(data: bytes) -> np.ndarray:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, w, h, ctype = 8, b"", 0, 0, 0
    while pos < len(data):
        (n,) = struct.unpack(">I", data[pos : pos + 4])
        kind, body = data[pos + 4 : pos + 8], data[pos + 8 : pos + 8 + n]
        (crc,) = struct.unpack(">I", data[pos + 8 + n : pos + 12 + n])
        assert crc == zlib.crc32(kind + body) & 0xFFFFFFFF  # chunk checksums are valid
        if kind == b"IHDR":
            w, h, _, ctype = struct.unpack(">IIBB", body[:10])
        if kind == b"IDAT":
            idat += body
        pos += 12 + n
    channels = 1 if ctype == 0 else 3
    raw = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(h, 1 + w * channels)
    assert (raw[:, 0] == 0).all()
    return raw[:, 1:].reshape(h, w) if channels == 1 else raw[:, 1:].reshape(h, w, 3)


def test_png_round_trips_gray_and_rgb():
    g = np.random.default_rng(0).integers(0, 256, (7, 11), dtype=np.uint8)
    assert np.array_equal(decode_png(encode_png(g)), g)
    c = np.random.default_rng(1).integers(0, 256, (5, 9, 3), dtype=np.uint8)
    assert np.array_equal(decode_png(encode_png(c)), c)
    assert data_uri(g).startswith("data:image/png;base64,")
    assert base64.b64decode(data_uri(g).split(",")[1]) == encode_png(g)
    assert upscale(np.arange(4).reshape(2, 2), 3).shape == (6, 6)


def test_png_rejects_bad_shapes():
    try:
        encode_png(np.zeros((2, 2, 4), dtype=np.uint8))
    except ValueError:
        return
    raise AssertionError("RGBA must be rejected")


def test_colormaps_have_the_right_endpoints_and_shapes():
    v = viridis(np.array([0.0, 1.0]))
    assert tuple(v[0]) == (68, 1, 84) and tuple(v[1]) == (253, 231, 37)
    assert viridis(np.zeros((3, 4))).shape == (3, 4, 3)
    d = diverging(np.array([-1.0, 0.0, 1.0]))
    assert d[0][2] > d[0][0] and d[2][0] > d[2][2]  # negative -> blue, positive -> red
    assert rgb_hex((255, 0, 16)) == "#ff0010"


def make_run(n=30):
    run = Run(DEFAULT_CONFIG, speed=8)
    run.step(n)
    run.step(n)
    return run


def strict_json(fig) -> None:
    """Figures must be valid JSON for the browser: no NaN / Infinity, and the right top-level shape."""
    assert set(fig) == {"data", "layout"}
    json.dumps(fig, allow_nan=False)


def test_every_playground_chart_builds_and_is_valid_json():
    run = make_run()
    s, snap = run.series(), run.view_snapshot()
    grid = forward_grid(run.specs, snap.weights)
    for fig in (
        charts.loss_figure(s, snap.step), charts.accuracy_figure(s, None), charts.gradient_figure(s, snap.step),
        charts.layer_hist_figure([l.w_hist for l in snap.layers], ["a", "b", "c"], charts.LAYER_COLORS),
        charts.layer_hist_figure([l.a_hist for l in snap.layers], ["a", "b", "c"], charts.LAYER_COLORS, zero_line=False),
        charts.boundary_figure(grid[-1][:, :, 0], run.train, run.test), charts.boundary_figure(None, run.train, run.test, show_test=False),
        charts.network_figure(run.specs, snap, (0, 1)), charts.network_figure(run.specs, None, None),
        charts.neuron_figure(grid[0], 2),
    ):
        strict_json(fig)
    assert len(charts.loss_figure(s, 5)["data"]) == 2 and len(charts.gradient_figure(s, None)["data"]) == 3
    assert charts.loss_figure(s, 5)["layout"]["shapes"][0]["x0"] == 5 and charts.loss_figure(s, None)["layout"]["shapes"] == []
    run.close()


def test_histogram_subplots_have_one_axis_pair_and_one_title_per_layer():
    run = make_run(10)
    snap = run.view_snapshot()
    fig = charts.layer_hist_figure([l.w_hist for l in snap.layers], ["t1", "t2", "t3"], charts.LAYER_COLORS)
    assert [t["xaxis"] for t in fig["data"]] == ["x", "x2", "x3"]
    assert all(k in fig["layout"] for k in ("xaxis", "xaxis2", "xaxis3", "yaxis3"))
    assert [a["text"] for a in fig["layout"]["annotations"]] == ["t1", "t2", "t3"]
    doms = [fig["layout"][k]["domain"] for k in ("yaxis", "yaxis2", "yaxis3")]
    assert doms[0][0] > doms[1][1] and doms[1][0] > doms[2][1]  # stacked top to bottom without overlap
    run.close()


def test_non_finite_numbers_become_null():
    assert charts.L([1.0, float("nan"), float("inf")]) == [1.0, None, None]
    assert charts.L(np.array([[1.0, np.nan]])) == [[1.0, None]]
    s = make_run(5).series()
    s.test_loss[0] = np.nan
    strict_json(charts.loss_figure(s, None))


def test_network_figure_has_nodes_for_every_unit_and_edges_in_both_signs():
    run = make_run(10)
    snap = run.view_snapshot()
    fig = charts.network_figure(run.specs, snap, None)
    nodes = fig["data"][-1]
    assert len(nodes["x"]) == 2 + 8 + 8 + 1
    colors = {t["line"]["color"] for t in fig["data"][:-1]}
    assert charts.RED in colors and charts.BLUE in colors
    assert nodes["customdata"][0] == [-1, 0]
    run.close()


def test_landscape_and_compare_figures():
    run = make_run(10)
    snap = run.view_snapshot()
    land = compute_landscape(run.specs, snap.weights, run.train.x, run.train.y, grid=7)
    fig = charts.landscape_figure(land, [(0.1, 0.1), (0.0, 0.0)])
    strict_json(fig)
    assert len(fig["data"]) == 4  # heatmap, training path, path start, current weights
    assert len(charts.landscape_figure(land, [])["data"]) == 2  # no path: heatmap + centre marker
    other = make_run(10)
    for kind in ("loss", "acc"):
        f = charts.compare_figure(run.series(), other.series(), kind)
        strict_json(f)
        assert len(f["data"]) == 4
    run.close()
    other.close()


def test_mnist_figures():
    conf = np.diag(np.arange(1, 11)) + 1
    strict_json(charts.confusion_figure(conf))
    maps = np.random.default_rng(0).normal(size=(8, 5, 5))
    z = charts.mosaic(maps, 4)
    assert z.shape == (2 * 6 - 1, 4 * 6 - 1) and np.isnan(z[5, 0])
    strict_json(charts.mosaic_figure(maps, 4, signed=True))
    strict_json(charts.mosaic_figure(np.abs(maps), 4, signed=False))
    strict_json(charts.mosaic_figure(np.zeros((2, 3, 3)), 2, signed=False))  # constant images must not divide by zero


def test_layer_specs_used_by_layout():
    specs = layer_specs(DEFAULT_CONFIG.with_(layers=(HiddenLayer(3, "relu"),)))
    pos = charts.network_layout(specs)
    assert [len(c) for c in pos] == [2, 3, 1] and pos[0][0][0] < pos[1][0][0] < pos[2][0][0]


def test_building_figures_is_fast_enough_not_to_starve_training():
    """Regression: validated plotly Figure objects cost ~40-60 ms each and starved the training thread."""
    import time

    run = make_run(200)
    s, snap = run.series(), run.view_snapshot()
    grid = forward_grid(run.specs, snap.weights)
    t0 = time.perf_counter()
    for _ in range(10):
        charts.boundary_figure(grid[-1][:, :, 0], run.train, run.test)
        charts.layer_hist_figure([l.w_hist for l in snap.layers], ["a", "b", "c"], charts.LAYER_COLORS)
        charts.network_figure(run.specs, snap, None)
        charts.loss_figure(s, None)
    per_refresh_ms = (time.perf_counter() - t0) / 10 * 1000
    assert per_refresh_ms < 25, per_refresh_ms
    run.close()
