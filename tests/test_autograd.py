import numpy as np

from learnscope.autograd import Value
from learnscope.explainer import apply_step, initial_params, run_explainer
from learnscope.tinynet import MLP, train_mse


def numerical_grad(f, xs, h=1e-6):
    out = []
    for i in range(len(xs)):
        plus, minus = list(xs), list(xs)
        plus[i] += h
        minus[i] -= h
        out.append((f([Value(x) for x in plus]).data - f([Value(x) for x in minus]).data) / (2 * h))
    return out


def check(f, xs):
    inputs = [Value(x) for x in xs]
    f(inputs).backward()
    for g, n in zip([i.grad for i in inputs], numerical_grad(f, xs)):
        assert abs(g - n) < 1e-6


def test_add_mul():
    check(lambda v: (v[0] * v[1] + v[2]) * v[0], [0.3, -1.2, 2.5])


def test_sub_div():
    check(lambda v: (v[0] - v[1]) / (v[1] * v[1] + 1), [1.7, -0.4])


def test_pow():
    check(lambda v: v[0] ** 3 + v[1] ** 2, [0.8, -1.1])


def test_tanh():
    check(lambda v: (v[0] * v[1]).tanh(), [0.5, -0.9])


def test_sigmoid():
    check(lambda v: (v[0] + v[1]).sigmoid(), [0.5, -0.9])


def test_exp_log():
    check(lambda v: v[0].exp() + (v[1] * v[1] + 1).log(), [0.2, 1.3])


def test_relu_away_from_kink():
    check(lambda v: (v[0] * v[1]).relu() + v[1].relu(), [0.7, 1.4])


def test_value_used_twice_accumulates():
    a = Value(3.0)
    out = a * a  # a^2 -> d/da = 2a = 6
    out.backward()
    assert abs(a.grad - 6.0) < 1e-12


def test_python_operators_and_reverse_ops():
    a = Value(2.0)
    out = 3 * a + 1 - a / 2  # 2.5a + 1
    out.backward()
    assert abs(out.data - 6.0) < 1e-12 and abs(a.grad - 2.5) < 1e-12
    b = Value(4.0)
    (10 - b).backward()
    assert b.grad == -1.0


def test_deep_graph_does_not_hit_the_recursion_limit():
    x = Value(1.0)
    y = x
    for _ in range(5000):
        y = y * 1.0001
    y.backward()
    assert x.grad > 1.0


def test_whole_mlp_loss_matches_numerical_gradients():
    net = MLP([2, 3, 1], ["tanh", "tanh"], seed=7)
    xs, ys = [[0.5, -0.3], [-1.0, 0.8]], [1.0, -1.0]

    def loss():
        total = Value(0.0)
        for x, y in zip(xs, ys):
            total = total + (net(x)[0] - y) ** 2
        return total

    net.zero_grad()
    loss().backward()
    analytic = [p.grad for p in net.params()]
    for p, a in zip(net.params(), analytic):
        orig = p.data
        p.data = orig + 1e-6
        fp = loss().data
        p.data = orig - 1e-6
        fm = loss().data
        p.data = orig
        assert abs(a - (fp - fm) / 2e-6) < 1e-6


def test_tiny_mlp_learns_xor():
    xs = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
    ys = [-1, 1, 1, -1]
    for seed in range(1, 8):  # a few seeds: a 4-unit net can land in a bad basin from some starts
        net = MLP([2, 4, 1], ["tanh", "tanh"], seed=seed)
        hist = train_mse(net, xs, ys, lr=0.1, steps=500)
        if hist[-1] < 0.02 and all(np.sign(net(x)[0].data) == y for x, y in zip(xs, ys)):
            return
    raise AssertionError("XOR was not learned from any of 7 seeds")


# ------------------------------------------------------------------ the Backprop explainer
def test_hand_written_chain_rule_equals_autograd():
    for seed in (1, 2, 3, 4, 5):
        assert run_explainer(initial_params(seed)).max_diff < 1e-12


def test_explainer_forward_values_follow_the_formulas():
    p = initial_params(4)
    r = run_explainer(p)
    z0 = p.w[0][0] * p.x[0] + p.w[0][1] * p.x[1] + p.b[0]
    assert abs(r.z[0] - z0) < 1e-12 and abs(r.h[0] - np.tanh(z0)) < 1e-12
    yhat = sum(vj * hj for vj, hj in zip(p.v, r.h)) + p.c
    assert abs(r.yhat - yhat) < 1e-12 and abs(r.loss - (yhat - p.t) ** 2) < 1e-12


def test_explainer_gradients_match_numerical_differentiation():
    from dataclasses import replace

    p = initial_params(4)
    r = run_explainer(p)
    h = 1e-6

    def bump(make):
        return (run_explainer(make(h)).loss - run_explainer(make(-h)).loss) / (2 * h)

    w10 = lambda d: replace(p, w=(p.w[0], (p.w[1][0] + d, p.w[1][1]), p.w[2]))  # noqa: E731
    v2 = lambda d: replace(p, v=(p.v[0], p.v[1], p.v[2] + d))  # noqa: E731
    assert abs(r.auto.w[1][0] - bump(w10)) < 1e-6
    assert abs(r.auto.v[2] - bump(v2)) < 1e-6
    assert abs(r.auto.c - bump(lambda d: replace(p, c=p.c + d))) < 1e-6


def test_explainer_updates_lower_the_loss():
    p = initial_params(4)
    before = run_explainer(p)
    assert run_explainer(apply_step(p, before.auto, 0.05)).loss < before.loss
    for _ in range(300):
        p = apply_step(p, run_explainer(p).auto, 0.1)
    assert run_explainer(p).loss < 1e-4
