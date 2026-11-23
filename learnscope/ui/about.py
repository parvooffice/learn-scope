"""About page: how it is built and what is (and is not) verified."""

from __future__ import annotations

from nicegui import ui

from .components import frame

SECTIONS = [
    ("What this is", "LearnScope is a teaching and debugging tool for neural networks. Everything you see is computed by code in this repository: the autograd engine, the "
                     "optimizers, the convolutional layers, the Model Doctor's rules, the loss-landscape slice. No deep-learning framework is used."),
    ("How it is built", "• autograd.py: a scalar reverse-mode autograd (micrograd-style) used by the Backprop page.\n"
                        "• mlp.py / trainer.py: vectorised NumPy MLP with hand-derived gradients for SGD, momentum and Adam, checked against numerical gradients in the tests.\n"
                        "• snapshots.py / runner.py: every few steps the run records all weights and statistics into a bounded buffer; that is what time travel replays.\n"
                        "• doctor/: seven deterministic rules over those snapshots, each with its evidence and a fix.\n"
                        "• mnist/: im2col convolution, max-pool, dense, softmax cross-entropy and Adam in NumPy.\n"
                        "• ui/: NiceGUI pages and Plotly figures; the training itself runs on background threads."),
    ("What is verified", "The test suite (pytest) checks the autograd engine against analytic derivatives, the NumPy backprop against finite differences, each Doctor rule on synthetic "
                         "and real runs, the MNIST layers' gradients, config sanitising, the chart builders and a speed budget for a UI refresh. Run `python -m pytest` yourself."),
    ("Honest limits", "• Pure CPU: fine for tiny networks and a 10,000-image MNIST subset, not for big models.\n"
                      "• Tiny networks are init-sensitive: a spiral with a 2×8 tanh network learns from most seeds but can plateau on some. The Doctor will tell you.\n"
                      "• The UI redraws a few times a second over a websocket; it is built for understanding, not for 60 fps animation.\n"
                      "• The MNIST subset is smaller than the full dataset, so accuracy tops out a little lower than published numbers."),
    ("Privacy", "Runs are computed on the machine that serves the app. Nothing is sent anywhere; shared links only encode the experiment settings, not data."),
]


def build() -> None:
    frame("About")
    with ui.column().classes("w-full max-w-[820px] mx-auto p-4 gap-5"):
        ui.label("About LearnScope").classes("text-2xl font-bold")
        for title, body in SECTIONS:
            with ui.column().classes("ls-card p-4 gap-1 w-full"):
                ui.label(title).classes("text-sm font-bold text-teal-400")
                ui.label(body).classes("text-sm whitespace-pre-line")
