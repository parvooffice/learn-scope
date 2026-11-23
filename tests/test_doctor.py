import math

from helpers import cfg_with, layers, mk_history

from learnscope.doctor import diagnose
from learnscope.doctor.rules import (
    dead_relus,
    exploding_gradients,
    learning_rate_too_high,
    learning_rate_too_low,
    overfitting,
    saturated_units,
    vanishing_gradients,
)
from learnscope.doctor.types import DoctorContext

RELU = cfg_with(layers=layers(8, "relu", 3))
SIGMOID = cfg_with(layers=layers(8, "sigmoid", 3))
TANH = cfg_with(layers=layers(8, "tanh", 2))


def ctx(history, cfg):
    return DoctorContext(history, cfg)


def healthy(cfg=RELU, n=40):
    """A boring, healthy run: loss falls smoothly, test tracks train."""
    return mk_history(
        cfg, n,
        lambda i: dict(
            train_loss=0.7 * math.exp(-i / 12) + 0.08,
            test_loss=0.7 * math.exp(-i / 12) + 0.1,
            train_acc=0.98, test_acc=0.96,
        ),
    )


# ------------------------------------------------------------------ healthy runs are not flagged
def test_healthy_run_is_not_flagged():
    d = diagnose(ctx(healthy(), RELU))
    assert [x.id for x in d] == ["healthy"] and d[0].severity == "info"


def test_short_history_stays_quiet():
    assert diagnose(ctx(healthy(RELU, 2), RELU)) == []


# ------------------------------------------------------------------ exploding
def test_exploding_fires_on_huge_gradient_with_a_restart_fix():
    h = mk_history(RELU, 10, lambda i: dict(layer_over=lambda l: {"g_norm": 500} if i >= 6 and l == 0 else {}))
    d = exploding_gradients(ctx(h, RELU))
    assert d.severity == "critical" and d.evidence.step == h[6].step and d.evidence.chart == "grad"
    assert abs(d.fix.patch["lr"] - RELU.lr / 10) < 1e-9 and d.fix.reset is True


def test_exploding_fires_on_nan_loss():
    h = mk_history(RELU, 5, lambda i: dict(train_loss=float("nan")) if i == 3 else {})
    assert "NaN" in exploding_gradients(ctx(h, RELU)).title


def test_exploding_does_not_fire_on_normal_gradients():
    assert exploding_gradients(ctx(healthy(), RELU)) is None


def test_explosion_hides_downstream_symptoms():
    h = mk_history(SIGMOID, 30, lambda i: dict(train_loss=50 if i > 20 else 0.7, layer_over=lambda l: {"g_norm": 1e4 if i > 20 else 1e-6}))
    assert [d.id for d in diagnose(ctx(h, SIGMOID))] == ["exploding"]


# ------------------------------------------------------------------ vanishing
def _vanishing(loss=0.69, cfg=SIGMOID, norms=(1e-6, 1e-5, 1e-4, 0.3)):
    return mk_history(cfg, 30, lambda i: dict(train_loss=loss, layer_over=lambda l: {"g_norm": norms[l]}))


def test_vanishing_fires_when_early_layers_get_a_tiny_fraction():
    d = vanishing_gradients(ctx(_vanishing(), SIGMOID))
    assert d.severity == "critical" and d.evidence.chart == "grad" and "ReLU" in d.fix.label
    assert all(l.activation == "relu" for l in d.fix.patch["layers"])


def test_vanishing_is_quiet_for_balanced_gradients_converged_nets_and_single_layers():
    assert vanishing_gradients(ctx(healthy(), RELU)) is None
    assert vanishing_gradients(ctx(_vanishing(loss=0.02), SIGMOID)) is None
    one = cfg_with(layers=layers(4, "sigmoid", 1))
    assert vanishing_gradients(ctx(_vanishing(cfg=one, norms=(1e-7, 0.3)), one)) is None


# ------------------------------------------------------------------ dead ReLUs
def test_dead_relu_fires_above_half_and_points_at_the_first_crossing():
    h = mk_history(RELU, 20, lambda i: dict(layer_over=lambda l: {"dead_pct": 0.1 if i < 8 else 0.75} if l == 1 else {}))
    d = dead_relus(ctx(h, RELU))
    assert d.severity == "warn" and d.evidence.step == h[8].step and d.evidence.chart == "activations" and "layer 2" in d.explanation


def test_dead_relu_is_critical_above_80_percent():
    h = mk_history(RELU, 10, lambda i: dict(layer_over=lambda l: {"dead_pct": 0.9}))
    assert dead_relus(ctx(h, RELU)).severity == "critical"


def test_dead_relu_ignores_tanh_and_healthy_layers():
    h = mk_history(TANH, 10, lambda i: dict(layer_over=lambda l: {"dead_pct": 0.9}))
    assert dead_relus(ctx(h, TANH)) is None and dead_relus(ctx(healthy(), RELU)) is None


# ------------------------------------------------------------------ learning rate too high
def test_lr_high_fires_on_zigzag_with_a_safe_rate_and_restart():
    h = mk_history(RELU, 30, lambda i: dict(train_loss=1.4 if i % 2 else 0.6))
    d = learning_rate_too_high(ctx(h, RELU))
    assert d.evidence.chart == "loss" and abs(d.fix.patch["lr"] - 0.01) < 1e-9 and d.fix.reset is True


def test_lr_high_suggestion_is_capped_for_the_optimizer():
    adam = cfg_with(layers=layers(8, "relu", 3), lr=1.5, optimizer="adam")
    sgd = adam.with_(optimizer="sgd")
    h = mk_history(adam, 30, lambda i: dict(train_loss=1.4 if i % 2 else 0.6))
    assert abs(learning_rate_too_high(ctx(h, adam)).fix.patch["lr"] - 0.01) < 1e-9
    assert abs(learning_rate_too_high(ctx(h, sgd)).fix.patch["lr"] - 0.1) < 1e-9


def test_lr_high_fires_when_loss_climbs_and_not_on_smooth_or_short_runs():
    h = mk_history(RELU, 30, lambda i: dict(train_loss=0.5 - i * 0.02 if i < 15 else 0.2 + (i - 15) * 0.1))
    assert learning_rate_too_high(ctx(h, RELU)) is not None
    assert learning_rate_too_high(ctx(healthy(), RELU)) is None
    short = mk_history(RELU, 5, lambda i: dict(train_loss=1.4 if i % 2 else 0.6))
    assert learning_rate_too_high(ctx(short, RELU)) is None


# ------------------------------------------------------------------ learning rate too low
def test_lr_low_fires_when_a_high_loss_barely_moves():
    h = mk_history(RELU, 60, lambda i: dict(train_loss=0.69 - i * 0.00002, train_acc=0.5))
    d = learning_rate_too_low(ctx(h, RELU))
    assert abs(d.fix.patch["lr"] - 0.09) < 1e-9 and d.fix.reset is False


def test_lr_low_is_quiet_when_falling_converged_or_early():
    assert learning_rate_too_low(ctx(healthy(RELU, 60), RELU)) is None
    assert learning_rate_too_low(ctx(mk_history(RELU, 60, lambda i: dict(train_loss=0.05, train_acc=0.99)), RELU)) is None
    assert learning_rate_too_low(ctx(mk_history(RELU, 20, lambda i: dict(train_loss=0.69, train_acc=0.5)), RELU)) is None


# ------------------------------------------------------------------ overfitting
OVERFIT = mk_history(
    RELU, 60,
    lambda i: dict(
        train_loss=max(0.02, 0.6 - i * 0.012),
        test_loss=0.6 - i * 0.01 if i < 25 else 0.35 + (i - 25) * 0.02,
        train_acc=0.99, test_acc=0.8,
    ),
)


def test_overfitting_points_at_the_best_test_step_and_suggests_l2():
    d = overfitting(ctx(OVERFIT, RELU))
    assert d.evidence.step == OVERFIT[25].step and d.fix.patch["l2"] == 0.01 and d.fix.reset is False


def test_overfitting_suggests_a_smaller_net_when_l2_is_already_on():
    d = overfitting(ctx(OVERFIT, RELU.with_(l2=0.01)))
    assert d.fix.patch["layers"][0].units == 4 and d.fix.reset is True


def test_overfitting_is_quiet_for_a_healthy_run():
    assert overfitting(ctx(healthy(), RELU)) is None


# ------------------------------------------------------------------ saturation
def test_saturation_fires_when_tanh_outputs_are_pinned_and_loss_is_high():
    h = mk_history(TANH, 12, lambda i: dict(layer_over=lambda l: {"sat_pct": 0.1 if i < 4 else 0.9} if l == 0 else {}))
    d = saturated_units(ctx(h, TANH))
    assert d.severity == "critical" and d.evidence.step == h[4].step
    assert all(l.activation == "relu" for l in d.fix.patch["layers"])


def test_saturation_is_allowed_once_the_task_is_learned():
    h = mk_history(TANH, 12, lambda i: dict(train_loss=0.05, layer_over=lambda l: {"sat_pct": 0.9}))
    assert saturated_units(ctx(h, TANH)) is None


def test_saturation_ignores_relu_nets_and_low_saturation():
    assert saturated_units(ctx(healthy(), RELU)) is None
    assert saturated_units(ctx(healthy(TANH), TANH)) is None


# ------------------------------------------------------------------ determinism
def test_same_history_gives_the_same_diagnoses():
    h = mk_history(SIGMOID, 30, lambda i: dict(train_loss=0.69, layer_over=lambda l: {"g_norm": (1e-6, 1e-5, 1e-4, 0.3)[l], "sat_pct": 0.9}))
    a, b = diagnose(ctx(h, SIGMOID)), diagnose(ctx(h, SIGMOID))
    assert a == b and len(a) > 0
