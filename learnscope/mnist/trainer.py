"""MNIST models, trainer and background run."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from .data import MnistData, load_mnist
from .layers import Adam, Conv2D, Dense, Flatten, Layer, MaxPool2, ReLU, softmax, softmax_cross_entropy

MODEL_LABELS = {
    "cnn": "Small CNN (conv → pool → conv → pool → dense)",
    "mlp": "MLP (784 → 128 → 64 → 10)",
}


@dataclass(frozen=True)
class MnistConfig:
    model: str = "cnn"
    lr: float = 0.003
    batch_size: int = 64
    seed: int = 1


class Net:
    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers

    @property
    def params(self) -> list[np.ndarray]:
        return [p for l in self.layers for p in l.params]

    @property
    def grads(self) -> list[np.ndarray]:
        return [g for l in self.layers for g in l.grads]

    def count_params(self) -> int:
        return int(sum(p.size for p in self.params))

    def forward(self, x: np.ndarray, upto: int | None = None) -> np.ndarray:
        for layer in self.layers[:upto]:
            x = layer.forward(x)
        return x

    def backward(self, d: np.ndarray) -> None:
        for layer in reversed(self.layers):
            d = layer.backward(d)


def build_model(kind: str, seed: int = 1, dtype=np.float32) -> Net:
    rng = np.random.default_rng(seed)
    if kind == "cnn":
        return Net([
            Conv2D(1, 8, 5, rng, dtype), ReLU(), MaxPool2(),   # 28 -> 24 -> 12
            Conv2D(8, 16, 3, rng, dtype), ReLU(), MaxPool2(),  # 12 -> 10 -> 5
            Flatten(), Dense(16 * 5 * 5, 10, rng, dtype),
        ])
    if kind == "mlp":
        return Net([Flatten(), Dense(784, 128, rng, dtype), ReLU(), Dense(128, 64, rng, dtype), ReLU(), Dense(64, 10, rng, dtype)])
    raise ValueError(f"unknown model {kind!r}")


@dataclass
class MnistEval:
    test_acc: float
    test_loss: float
    confusion: np.ndarray  # (10, 10): [true, predicted]
    worst: list[dict]  # most confidently wrong test images


@dataclass
class MnistFocus:
    probs: np.ndarray
    conv1: np.ndarray | None = None  # (8, 24, 24) feature maps (after ReLU)
    conv2: np.ndarray | None = None  # (16, 10, 10)
    hidden: np.ndarray | None = None  # (128,) MLP first hidden layer


@dataclass
class MnistPoint:
    step: int
    epoch: float
    train_loss: float
    train_acc: float
    test_loss: float | None = None
    test_acc: float | None = None


class MnistTrainer:
    def __init__(self, cfg: MnistConfig, data: MnistData) -> None:
        self.cfg, self.data = cfg, data
        self.net = build_model(cfg.model, cfg.seed)
        self.opt = Adam(self.net.params, cfg.lr)
        self.step = 0
        self.train_ms = 0.0
        self._rng = np.random.default_rng(cfg.seed * 7 + 3)
        self._perm = self._rng.permutation(data.n_train)
        self._pos = 0
        self._acc = [0.0, 0.0, 0]

    def train_step(self) -> None:
        k = self.cfg.batch_size
        if self._pos + k > len(self._perm):
            self._perm = self._rng.permutation(self.data.n_train)
            self._pos = 0
        idx = self._perm[self._pos : self._pos + k]
        self._pos += k
        x, y = self.data.train_x[idx], self.data.train_y[idx]
        t0 = time.perf_counter()
        logits = self.net.forward(x)
        loss, dlogits = softmax_cross_entropy(logits, y)
        self.net.backward(dlogits.astype(logits.dtype))
        self.opt.step(self.net.params, self.net.grads)
        self.train_ms += (time.perf_counter() - t0) * 1000
        self.step += 1
        self._acc[0] += loss
        self._acc[1] += float(np.mean(logits.argmax(axis=1) == y))
        self._acc[2] += 1

    def take_train_metrics(self, last: tuple[float, float]) -> tuple[float, float]:
        if self._acc[2] == 0:
            return last
        out = (self._acc[0] / self._acc[2], self._acc[1] / self._acc[2])
        self._acc = [0.0, 0.0, 0]
        return out

    def evaluate(self, chunk: int = 500) -> MnistEval:
        d = self.data
        confusion = np.zeros((10, 10), dtype=np.int64)
        loss, wrong = 0.0, []
        for s in range(0, d.n_test, chunk):
            logits = self.net.forward(d.test_x[s : s + chunk])
            p = softmax(logits)
            y = d.test_y[s : s + chunk]
            pred = p.argmax(axis=1)
            loss += float(-np.sum(np.log(np.maximum(p[np.arange(len(y)), y], 1e-12))))
            np.add.at(confusion, (y, pred), 1)
            for i in np.nonzero(pred != y)[0]:
                wrong.append({"index": s + int(i), "pred": int(pred[i]), "truth": int(y[i]), "conf": float(p[i, pred[i]])})
        wrong.sort(key=lambda w: -w["conf"])
        return MnistEval(float(np.trace(confusion) / d.n_test), loss / d.n_test, confusion, wrong[:12])

    def focus(self, image: np.ndarray) -> MnistFocus:
        """Class probabilities and feature maps for ONE image of shape (28, 28) in [0, 1]."""
        x = image.reshape(1, 1, 28, 28).astype(np.float32)
        if self.cfg.model == "cnn":
            a1 = self.net.forward(x, upto=2)  # conv1 + relu
            a2 = self.net.forward(x, upto=5)  # ... pool, conv2 + relu
            probs = softmax(self.net.forward(x))[0]
            return MnistFocus(probs, conv1=a1[0].copy(), conv2=a2[0].copy())
        hidden = self.net.forward(x, upto=3)[0]
        return MnistFocus(softmax(self.net.forward(x))[0], hidden=hidden.copy())

    def filters(self) -> np.ndarray:
        """Layer-1 filters as images: CNN -> (8, 5, 5); MLP -> first 16 neurons as (16, 28, 28)."""
        if self.cfg.model == "cnn":
            return self.net.layers[0].W.reshape(8, 5, 5).copy()
        return self.net.layers[1].W[:, :16].T.reshape(16, 28, 28).copy()


class MnistRun:
    """Background MNIST training with curves recorded for the UI."""

    EVAL_EVERY = 6  # ticks

    def __init__(self, cfg: MnistConfig = MnistConfig(), data: MnistData | None = None, speed: int = 4) -> None:
        self.lock = threading.RLock()
        self.data = data or load_mnist()
        self.speed = speed
        self.points: list[MnistPoint] = []
        self.version = 0
        self.error: str | None = None
        self.last_eval: MnistEval | None = None
        self.focus: MnistFocus | None = None
        self.focus_image: np.ndarray | None = None
        self.focus_label: int | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._tick = 0
        self._last = (2.3, 0.1)
        self._build(cfg)

    def _build(self, cfg: MnistConfig) -> None:
        self.pause()
        with self.lock:
            self.cfg = cfg
            self.trainer = MnistTrainer(cfg, self.data)
            self.points.clear()
            self.last_eval = self.trainer.evaluate()
            self._last = (2.3, 0.1)
            self._tick = 0
            self._refresh_focus()
            self.version += 1

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def configure(self, **patch) -> None:
        was = self.running
        self._build(MnistConfig(**{**self.cfg.__dict__, **patch}))
        if was:
            self.start()

    def reset(self) -> None:
        self._build(MnistConfig(**{**self.cfg.__dict__, "seed": self.cfg.seed + 1}))

    def start(self) -> None:
        with self.lock:
            if self.running:
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="learnscope-mnist", daemon=True)
            self._thread.start()

    def pause(self) -> None:
        t = self._thread
        if t is not None and t.is_alive():
            self._stop.set()
            if t is not threading.current_thread():
                t.join(timeout=10)
        self._thread = None
        self._stop.clear()
        if getattr(self, "trainer", None) is not None and self.points:
            with self.lock:
                self.last_eval = self.trainer.evaluate()
                self.version += 1

    def step(self, n: int = 1) -> None:
        self.pause()
        with self.lock:
            for _ in range(n):
                self.trainer.train_step()
            self._record(with_eval=True)

    def set_speed(self, n: int) -> None:
        self.speed = max(1, int(n))

    # ------------------------------------------------------------------ focus image
    def focus_test(self, index: int) -> None:
        with self.lock:
            index = int(np.clip(index, 0, self.data.n_test - 1))
            self.focus_image = self.data.test_x[index, 0].copy()
            self.focus_label = int(self.data.test_y[index])
            self.focus_index = index
            self._refresh_focus()
            self.version += 1

    def focus_pixels(self, pixels: np.ndarray) -> None:
        """Send a drawn 28x28 image (values 0..1) through the network."""
        with self.lock:
            self.focus_image = np.asarray(pixels, dtype=np.float32).reshape(28, 28)
            self.focus_label = None
            self._refresh_focus()
            self.version += 1

    def _refresh_focus(self) -> None:
        if self.focus_image is None:
            self.focus_image = self.data.test_x[0, 0].copy()
            self.focus_label = int(self.data.test_y[0])
            self.focus_index = 0
        self.focus = self.trainer.focus(self.focus_image)

    # ------------------------------------------------------------------ internals
    def _record(self, with_eval: bool) -> None:
        tr = self.trainer
        self._last = tr.take_train_metrics(self._last)
        pt = MnistPoint(tr.step, tr.step * tr.cfg.batch_size / tr.data.n_train, self._last[0], self._last[1])
        if with_eval:
            self.last_eval = tr.evaluate()
            pt.test_loss, pt.test_acc = self.last_eval.test_loss, self.last_eval.test_acc
        self.points.append(pt)
        self._refresh_focus()
        self.version += 1

    @property
    def images_per_second(self) -> float:
        tr = self.trainer
        return tr.step * tr.cfg.batch_size / (tr.train_ms / 1000) if tr.train_ms > 0 else 0.0

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                with self.lock:
                    for _ in range(self.speed):
                        self.trainer.train_step()
                    self._tick += 1
                    self._record(with_eval=self._tick % self.EVAL_EVERY == 0)
            except Exception as exc:  # noqa: BLE001
                self.error = f"{type(exc).__name__}: {exc}"
                self._stop.set()
                break
            self._stop.wait(max(0.0, 0.02 - (time.perf_counter() - t0)))

    def close(self) -> None:
        self.pause()
