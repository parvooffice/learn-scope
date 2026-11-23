"""The seven diagnostic rules. Each is a pure function of the training history and the config,
with its thresholds exposed as constants so tests (and you) can see exactly what triggers it."""

from __future__ import annotations

from ..config import HiddenLayer
from .types import Diagnosis, DoctorContext, Evidence, Fix
from .util import avg, fmt, last, nice_lr, snapshot_has_nan

# ---------------------------------------------------------------------------- thresholds
EXPLODING_GRAD_NORM = 100.0
EXPLODING_LOSS = 20.0
VANISH_RATIO = 0.05
VANISH_ABS = 1e-5
VANISH_MIN_LOSS = 0.2
DEAD_WARN = 0.5
DEAD_CRITICAL = 0.8
SAT_WARN = 0.6
SAT_CRITICAL = 0.85
SAT_MIN_LOSS = 0.25
LR_HIGH_WINDOW = 20
LR_HIGH_UP_FRACTION = 0.3
LR_HIGH_RISE = 1.3
LR_CEILING = {"sgd": 0.1, "momentum": 0.05, "adam": 0.01}  # comfortably stable rates per optimizer
LR_LOW_MIN_STEPS = 150
LR_LOW_WINDOW = 30
LR_LOW_IMPROVEMENT = 0.01
LR_LOW_LOSS = 0.45
OVERFIT_GAP = 0.15
OVERFIT_RISE = 1.1
OVERFIT_ACC_GAP = 0.05


def _all_layers(cfg, activation: str) -> tuple[HiddenLayer, ...]:
    return tuple(HiddenLayer(l.units, activation) for l in cfg.layers)


# ---------------------------------------------------------------------------- 1. exploding
def exploding_gradients(ctx: DoctorContext) -> Diagnosis | None:
    """Gradients or loss have blown up (or became NaN / Infinity)."""
    h, cfg = ctx.history, ctx.config
    if len(h) < 2:
        return None
    hit = next(
        (
            s
            for s in h
            if snapshot_has_nan(s) or s.train_loss > EXPLODING_LOSS or any(l.g_norm > EXPLODING_GRAD_NORM for l in s.layers)
        ),
        None,
    )
    if hit is None:
        return None
    broken = snapshot_has_nan(hit)
    worst = max((l.g_norm if l.g_norm == l.g_norm and l.g_norm != float("inf") else float("inf")) for l in hit.layers)
    target = nice_lr(cfg.lr / 10)
    return Diagnosis(
        id="exploding",
        severity="critical",
        title="Training produced NaN / Infinity" if broken else "Exploding gradients",
        explanation=(
            "The numbers overflowed, so the weights are now NaN and the network cannot recover. "
            "This usually means the learning rate was too large for this network."
            if broken
            else f'The gradient (the "push" on the weights) became huge (norm {fmt(worst)}). Giant steps overshoot the '
            "valley of the loss surface and make training unstable. Lowering the learning rate shrinks every step."
        ),
        evidence=Evidence(
            hit.step, "grad", "first step with non-finite values" if broken else f"gradient norm {fmt(worst)} at this step"
        ),
        fix=Fix(f"Lower the learning rate to {target} and restart", {"lr": target}, True),
    )


# ---------------------------------------------------------------------------- 2. vanishing
def vanishing_gradients(ctx: DoctorContext) -> Diagnosis | None:
    """Gradients shrink layer by layer on the way back: the first layers barely learn."""
    h, cfg = ctx.history, ctx.config
    if len(cfg.layers) < 2 or len(h) < 5:
        return None
    recent = last(h, 5)
    tail = recent[-1]
    if tail.step < 20 or tail.train_loss < VANISH_MIN_LOSS:  # converged nets have small gradients too
        return None
    n_layers = len(tail.layers)
    first = avg([s.layers[0].g_norm for s in recent])
    last_l = avg([s.layers[n_layers - 1].g_norm for s in recent])
    if not last_l > 0:
        return None
    ratio = first / last_l
    if not (ratio < VANISH_RATIO or first < VANISH_ABS):
        return None
    squashing = any(l.activation in ("sigmoid", "tanh") for l in cfg.layers)
    return Diagnosis(
        id="vanishing",
        severity="critical" if ratio < 0.01 or first < VANISH_ABS else "warn",
        title="Vanishing gradients",
        explanation=(
            f"The first layer's gradient is {fmt(ratio * 100, 2)}% of the last layer's. Each layer multiplies the "
            "gradient by a number below 1 (sigmoid/tanh slopes are at most 1 and 0.25), so by the time it reaches the "
            "early layers almost nothing is left and they stop learning."
        ),
        evidence=Evidence(tail.step, "grad", f"layer 1 gradient {fmt(first)} vs output layer {fmt(last_l)}"),
        fix=Fix("Switch all hidden layers to ReLU", {"layers": _all_layers(cfg, "relu")}, True) if squashing else None,
    )


# ---------------------------------------------------------------------------- 3. dead ReLUs
def dead_relus(ctx: DoctorContext) -> Diagnosis | None:
    """ReLU units that output exactly 0 for every training sample: no gradient, they never recover."""
    h, cfg = ctx.history, ctx.config
    if len(h) < 3:
        return None
    tail = h[-1]
    worst, worst_layer = 0.0, -1
    for i, l in enumerate(cfg.layers):
        if l.activation not in ("relu", "gelu") or l.units < 2:
            continue
        dead = tail.layers[i].dead_pct if i < len(tail.layers) else 0.0
        if dead > worst:
            worst, worst_layer = dead, i
    if worst_layer < 0 or worst <= DEAD_WARN:
        return None
    first = next((s for s in h if (s.layers[worst_layer].dead_pct if worst_layer < len(s.layers) else 0.0) > DEAD_WARN), tail)
    return Diagnosis(
        id="dead-relu",
        severity="critical" if worst > DEAD_CRITICAL else "warn",
        title="Dead ReLU units",
        explanation=(
            f"{fmt(worst * 100)}% of the units in hidden layer {worst_layer + 1} output 0 for every sample. A ReLU outputs "
            "0 for negative inputs and passes no gradient there, so a \"dead\" unit can never wake up again. It is often "
            "caused by a learning rate that kicked the weights too far."
        ),
        evidence=Evidence(first.step, "activations", f"layer {worst_layer + 1}: {fmt(worst * 100)}% dead"),
        fix=Fix(
            "Switch ReLU layers to GELU (smooth, no hard zero) and restart",
            {"layers": tuple(HiddenLayer(l.units, "gelu" if l.activation == "relu" else l.activation) for l in cfg.layers)},
            True,
        ),
    )


# ---------------------------------------------------------------------------- 4. saturation
def saturated_units(ctx: DoctorContext) -> Diagnosis | None:
    """tanh/sigmoid units pinned at their flat ends have almost zero slope, so they barely learn."""
    h, cfg = ctx.history, ctx.config
    if len(h) < 3:
        return None
    tail = h[-1]
    # A network that has already learned the task is allowed saturated units (a spiral boundary needs
    # sharp turns). Only flag saturation while the loss is still high.
    if tail.train_loss < SAT_MIN_LOSS:
        return None
    worst, worst_layer = 0.0, -1
    for i, l in enumerate(cfg.layers):
        if l.activation not in ("tanh", "sigmoid"):
            continue
        sat = tail.layers[i].sat_pct if i < len(tail.layers) else 0.0
        if sat > worst:
            worst, worst_layer = sat, i
    if worst_layer < 0 or worst <= SAT_WARN:
        return None
    first = next((s for s in h if (s.layers[worst_layer].sat_pct if worst_layer < len(s.layers) else 0.0) > SAT_WARN), tail)
    act = cfg.layers[worst_layer].activation
    return Diagnosis(
        id="saturated",
        severity="critical" if worst > SAT_CRITICAL else "warn",
        title=f"Saturated {act} units",
        explanation=(
            f"{fmt(worst * 100)}% of the {act} outputs in hidden layer {worst_layer + 1} sit at the flat ends of the "
            "curve. There the slope is almost 0, so gradients through them nearly vanish. Weights that grew too large "
            "(or a learning rate that was too high) push units into this regime."
        ),
        evidence=Evidence(first.step, "activations", f"layer {worst_layer + 1}: {fmt(worst * 100)}% saturated"),
        fix=Fix(
            "Switch saturated layers to ReLU and restart",
            {"layers": tuple(HiddenLayer(l.units, "relu" if l.activation in ("tanh", "sigmoid") else l.activation) for l in cfg.layers)},
            True,
        ),
    )


# ---------------------------------------------------------------------------- 5. lr too high
def learning_rate_too_high(ctx: DoctorContext) -> Diagnosis | None:
    """Training loss going UP, or zig-zagging: the optimizer keeps overshooting the minimum."""
    h, cfg = ctx.history, ctx.config
    if len(h) < 12:
        return None
    win = last(h, LR_HIGH_WINDOW)
    losses = [s.train_loss for s in win]
    if any(x != x or x in (float("inf"), float("-inf")) for x in losses):
        return None
    ups, worst_jump, worst_step = 0, 0.0, win[-1].step
    for i in range(1, len(losses)):
        if losses[i] > losses[i - 1] * 1.05:
            ups += 1
            jump = losses[i] - losses[i - 1]
            if jump > worst_jump:
                worst_jump, worst_step = jump, win[i].step
    up_fraction = ups / (len(losses) - 1)
    best = min(s.train_loss for s in h)
    now_avg = avg(last(losses, 5))
    rising = now_avg > best * LR_HIGH_RISE and now_avg - best > 0.05
    if not (up_fraction >= LR_HIGH_UP_FRACTION or rising):
        return None
    target = nice_lr(min(cfg.lr / 3, LR_CEILING[cfg.optimizer]))
    severe = now_avg > best * 2 and now_avg - best > 0.2
    return Diagnosis(
        id="lr-high",
        severity="critical" if severe else "warn",
        title="Learning rate looks too high",
        explanation=(
            f"The training loss keeps jumping back up ({ups} rises in the last {len(losses) - 1} snapshots; best "
            f"{fmt(best)}, now {fmt(now_avg)}). With steps that are too big the optimizer leaps over the bottom of the "
            "valley and lands on the other side, again and again."
        ),
        evidence=Evidence(worst_step, "loss", f"loss jumped by {fmt(worst_jump)} here"),
        # restart: a net that already overshot may be stuck in saturated/dead regions
        fix=Fix(f"Lower the learning rate to {target} and restart", {"lr": target}, True),
    )


# ---------------------------------------------------------------------------- 6. lr too low
def learning_rate_too_low(ctx: DoctorContext) -> Diagnosis | None:
    """Loss is high but has hardly moved for a long time: the steps are too small to make progress."""
    h, cfg = ctx.history, ctx.config
    if not h:
        return None
    tail = h[-1]
    if len(h) < LR_LOW_WINDOW or tail.step < LR_LOW_MIN_STEPS:
        return None
    win = last(h, LR_LOW_WINDOW)
    start, end = win[0].train_loss, win[-1].train_loss
    if not (start == start and end == end) or abs(start) == float("inf") or abs(end) == float("inf"):
        return None
    if win[-1].step - win[0].step < 60:
        return None
    if end < LR_LOW_LOSS or tail.train_acc > 0.95:
        return None
    improvement = (start - end) / max(start, 1e-9)
    if improvement >= LR_LOW_IMPROVEMENT:
        return None
    target = nice_lr(cfg.lr * 3)
    return Diagnosis(
        id="lr-low",
        severity="warn",
        title="Learning rate looks too low (or training is stuck)",
        explanation=(
            f"Over the last {win[-1].step - win[0].step} steps the loss only changed by {fmt(improvement * 100, 2)}%, and "
            f"it is still high ({fmt(end)}). Tiny steps crawl across a flat region. Try a bigger learning rate; if that "
            "doesn't help, the model may need more capacity."
        ),
        evidence=Evidence(win[0].step, "loss", f"loss {fmt(start)} → {fmt(end)} over this stretch"),
        fix=Fix(f"Raise the learning rate to {target}", {"lr": target}, False),
    )


# ---------------------------------------------------------------------------- 7. overfitting
def overfitting(ctx: DoctorContext) -> Diagnosis | None:
    """Training loss keeps improving but test loss has turned around: the model is memorising."""
    h, cfg = ctx.history, ctx.config
    if len(h) < 15:
        return None
    win = last(h, 10)
    gap = avg([s.test_loss - s.train_loss for s in win])
    acc_gap = avg([s.train_acc - s.test_acc for s in win])
    now_test = avg([s.test_loss for s in win])
    best = min(h, key=lambda s: s.test_loss)
    if gap != gap:
        return None
    if gap < OVERFIT_GAP or acc_gap < OVERFIT_ACC_GAP:
        return None
    if now_test < best.test_loss * OVERFIT_RISE:
        return None
    if cfg.l2 < 0.005:
        fix = Fix("Add L2 regularization (0.01)", {"l2": 0.01}, False)
    else:
        fix = Fix(
            "Use a smaller network (halve the units per layer) and restart",
            {"layers": tuple(HiddenLayer(max(2, round(l.units / 2)), l.activation) for l in cfg.layers)},
            True,
        )
    return Diagnosis(
        id="overfitting",
        severity="critical" if gap > 0.4 else "warn",
        title="Overfitting",
        explanation=(
            f"Test loss ({fmt(now_test)}) is {fmt(gap)} above training loss and has risen since its best value of "
            f"{fmt(best.test_loss)}. The network is memorising training points instead of learning the pattern, so it "
            "does worse on data it hasn't seen."
        ),
        evidence=Evidence(best.step, "loss", f"test loss was lowest here ({fmt(best.test_loss)}); it gets worse afterwards"),
        fix=fix,
    )
