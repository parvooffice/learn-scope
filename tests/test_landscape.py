import numpy as np

from learnscope.config import DEFAULT_CONFIG, HiddenLayer, layer_specs
from learnscope.datasets import generate_dataset
from learnscope.forward import evaluate
from learnscope.landscape import compute_landscape, flatten, project, random_directions, unflatten
from learnscope.mlp import MLP

CFG = DEFAULT_CONFIG.with_(layers=(HiddenLayer(4, "tanh"),), seed=3)
SPECS = layer_specs(CFG)
CENTER = MLP(SPECS, 3).weights()
DS = generate_dataset("moons", 60, 0.1, 1)


def test_flatten_unflatten_round_trip():
    back = unflatten(CENTER, flatten(CENTER))
    for (w, b), (w2, b2) in zip(CENTER, back):
        assert np.array_equal(w, w2) and np.array_equal(b, b2)


def test_directions_are_orthogonal_and_scaled_per_layer():
    d1, d2 = random_directions(CENTER, 5)
    assert abs(d1 @ d2) / (np.linalg.norm(d1) * np.linalg.norm(d2)) < 1e-6
    n0 = CENTER[0][0].size + CENTER[0][1].size
    c = flatten(CENTER)
    assert abs(np.linalg.norm(d1[:n0]) / np.linalg.norm(c[:n0]) - 1) < 0.1


def test_projection_recovers_the_coordinates_used_to_build_a_point():
    d1, d2 = random_directions(CENTER, 9)
    c = flatten(CENTER)
    a, b = project(c + 0.3 * d1 - 0.2 * d2, c, d1, d2)
    assert abs(a - 0.3) < 1e-6 and abs(b + 0.2) < 1e-6


def test_centre_cell_equals_the_loss_at_the_centre_weights_and_progress_is_reported():
    seen = []
    land = compute_landscape(SPECS, CENTER, DS.x, DS.y, grid=9, on_progress=seen.append)
    direct = evaluate(SPECS, CENTER, DS.x, DS.y)[0]
    assert abs(land.values[4, 4] - direct) < 1e-9 and abs(land.center_loss - direct) < 1e-12
    assert land.values.shape == (9, 9) and land.clip >= land.min and seen[-1] == 1.0


def test_can_be_cancelled():
    assert compute_landscape(SPECS, CENTER, DS.x, DS.y, grid=9, cancelled=lambda: True) is None
