from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..config import PlaygroundConfig
from ..snapshots import Snapshot

# severity: "info" | "warn" | "critical"      chart: "loss" | "grad" | "weights" | "activations" | "network"


@dataclass(frozen=True)
class Fix:
    label: str
    patch: dict[str, Any]  # config fields to change, e.g. {"lr": 0.01}
    reset: bool  # True when the run must restart (architecture change, NaNs in the weights, ...)

    def apply(self, cfg: PlaygroundConfig) -> PlaygroundConfig:
        return cfg.with_(**self.patch)


@dataclass(frozen=True)
class Evidence:
    step: int
    chart: str
    detail: str


@dataclass(frozen=True)
class Diagnosis:
    id: str  # stable, so the UI can keep the same card between runs
    severity: str
    title: str
    explanation: str  # plain English: what is happening and why it matters
    evidence: Evidence  # where to look: jump to this step and highlight this chart
    fix: Fix | None = None


@dataclass
class DoctorContext:
    history: list[Snapshot]
    config: PlaygroundConfig
    extra: dict = field(default_factory=dict)


# A rule is a pure function: same history in, same diagnosis out.
Rule = Callable[[DoctorContext], "Diagnosis | None"]
