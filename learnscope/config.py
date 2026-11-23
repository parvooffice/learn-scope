"""Experiment configuration + shareable-link encoding.

A config is plain data, so it can be sanitised, compared, copied and put in a URL. Anything that
comes from a URL is untrusted: `sanitize_config` repairs or clamps every field.
"""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass, replace
from typing import Any

ACTIVATIONS = ("relu", "tanh", "sigmoid", "gelu")
OPTIMIZERS = ("sgd", "momentum", "adam")
DATASETS = ("moons", "circles", "spiral", "xor", "blobs")
DATASET_LABELS = {
    "moons": "Two moons",
    "circles": "Circles",
    "spiral": "Spiral",
    "xor": "XOR",
    "blobs": "Gaussian blobs",
}
ACTIVATION_LABELS = {"relu": "ReLU", "tanh": "tanh", "sigmoid": "Sigmoid", "gelu": "GELU"}
OPTIMIZER_LABELS = {"sgd": "SGD", "momentum": "SGD + momentum", "adam": "Adam"}

LIVE_KEYS = ("lr", "l2", "batch_size")  # can change while training without a restart


@dataclass(frozen=True)
class HiddenLayer:
    units: int
    activation: str


@dataclass(frozen=True)
class PlaygroundConfig:
    dataset: str = "spiral"
    noise: float = 0.08
    samples: int = 400
    test_split: float = 0.3
    layers: tuple[HiddenLayer, ...] = (HiddenLayer(8, "tanh"), HiddenLayer(8, "tanh"))
    lr: float = 0.03
    batch_size: int = 32
    optimizer: str = "adam"
    l2: float = 0.0
    seed: int = 3  # a spiral + 2x8 tanh net is init-sensitive: seed 3 learns reliably (seed 7, for example, plateaus)

    def with_(self, **changes: Any) -> "PlaygroundConfig":
        return replace(self, **changes)


DEFAULT_CONFIG = PlaygroundConfig()


@dataclass(frozen=True)
class LayerSpec:
    """One dense layer. The output layer is always 1 unit with a sigmoid."""

    nin: int
    nout: int
    activation: str


def layer_specs(cfg: PlaygroundConfig) -> list[LayerSpec]:
    specs: list[LayerSpec] = []
    nin = 2
    for layer in cfg.layers:
        specs.append(LayerSpec(nin, layer.units, layer.activation))
        nin = layer.units
    specs.append(LayerSpec(nin, 1, "sigmoid"))
    return specs


# ---------------------------------------------------------------------------- sanitising
def _clamp(v: Any, lo: float, hi: float, fallback: float) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
        return fallback
    return min(hi, max(lo, float(v)))


def _pick(v: Any, allowed: tuple[str, ...], fallback: str) -> str:
    return v if isinstance(v, str) and v in allowed else fallback


def sanitize_config(raw: Any) -> PlaygroundConfig:
    """Make an untrusted object into a valid config: unknown values fall back to defaults and
    numbers are clamped into range, so a hostile link can never crash the app or request
    absurd sizes."""
    r = raw if isinstance(raw, dict) else {}
    d = DEFAULT_CONFIG
    layers_raw = r.get("layers", None)
    if not isinstance(layers_raw, list):
        layers = d.layers
    else:
        layers = tuple(
            HiddenLayer(
                units=int(round(_clamp((l or {}).get("units") if isinstance(l, dict) else None, 1, 16, 4))),
                activation=_pick((l or {}).get("activation") if isinstance(l, dict) else None, ACTIVATIONS, "tanh"),
            )
            for l in layers_raw[:6]
        )
        if not layers:
            layers = d.layers
    return PlaygroundConfig(
        dataset=_pick(r.get("dataset"), DATASETS, d.dataset),
        noise=_clamp(r.get("noise"), 0.0, 0.5, d.noise),
        samples=int(round(_clamp(r.get("samples"), 100, 1000, d.samples))),
        test_split=_clamp(r.get("test_split"), 0.1, 0.5, d.test_split),
        layers=layers,
        lr=_clamp(r.get("lr"), 0.0001, 3.0, d.lr),
        batch_size=int(round(_clamp(r.get("batch_size"), 1, 256, d.batch_size))),
        optimizer=_pick(r.get("optimizer"), OPTIMIZERS, d.optimizer),
        l2=_clamp(r.get("l2"), 0.0, 0.1, d.l2),
        seed=int(round(_clamp(r.get("seed"), 0, 999999, d.seed))),
    )


def config_to_dict(cfg: PlaygroundConfig) -> dict:
    d = asdict(cfg)
    d["layers"] = [{"units": l.units, "activation": l.activation} for l in cfg.layers]
    return d


def encode_config(cfg: PlaygroundConfig) -> str:
    """Config -> short URL-safe string (JSON, base64url, no padding)."""
    raw = json.dumps(config_to_dict(cfg), separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_config(s: str) -> PlaygroundConfig | None:
    """URL-safe string -> config, or None if it cannot be parsed."""
    try:
        if len(s) > 4096:
            return None
        raw = base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))
        return sanitize_config(json.loads(raw))
    except (ValueError, TypeError):
        return None
