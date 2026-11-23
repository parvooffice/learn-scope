"""Model Doctor: deterministic, rule-based diagnosis of training problems."""

from __future__ import annotations

from .rules import (
    dead_relus,
    exploding_gradients,
    learning_rate_too_high,
    learning_rate_too_low,
    overfitting,
    saturated_units,
    vanishing_gradients,
)
from .types import Diagnosis, DoctorContext, Evidence, Fix, Rule

RULES: list[Rule] = [
    exploding_gradients,
    vanishing_gradients,
    dead_relus,
    saturated_units,
    learning_rate_too_high,
    learning_rate_too_low,
    overfitting,
]

_ORDER = {"critical": 0, "warn": 1, "info": 2}


def diagnose(ctx: DoctorContext) -> list[Diagnosis]:
    """Run every rule over the training history. Deterministic: same input, same output."""
    found = [d for rule in RULES if (d := rule(ctx)) is not None]
    found.sort(key=lambda d: _ORDER[d.severity])
    # if the run just exploded, the other symptoms are consequences: show the root cause alone
    if any(d.id == "exploding" for d in found):
        return [d for d in found if d.id == "exploding"]
    tail = ctx.history[-1] if ctx.history else None
    if not found and tail is not None and tail.step >= 30:
        found.append(
            Diagnosis(
                id="healthy",
                severity="info",
                title="No problems detected",
                explanation=(
                    "Gradients flow through all layers, units are active, and training and test loss move together. "
                    "Keep watching: some problems (like overfitting) only show up later."
                ),
                evidence=Evidence(tail.step, "loss", "all checks passed at this step"),
            )
        )
    return found


__all__ = ["Diagnosis", "DoctorContext", "Evidence", "Fix", "RULES", "Rule", "diagnose"]
