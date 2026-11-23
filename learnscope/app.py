"""LearnScope web app: routes and server start-up.

    python -m learnscope            # http://localhost:8080
    PORT=9000 python -m learnscope
"""

from __future__ import annotations

import os
import secrets
import sys

from nicegui import app as nicegui_app
from nicegui import ui

from .config import DEFAULT_CONFIG, decode_config


@ui.page("/")
def home_page() -> None:
    from .ui import home

    home.build()


@ui.page("/playground")
def playground_page(c: str | None = None, play: str | None = None) -> None:
    """`c` = a shared experiment (see the Share button); `play=0` opens it paused."""
    from .ui import playground

    cfg = decode_config(c) if c else None
    playground.build(cfg or DEFAULT_CONFIG, autoplay=play != "0")


@ui.page("/compare")
def compare_page(a: str | None = None, b: str | None = None) -> None:
    from .ui import compare

    compare.build(decode_config(a) if a else None, decode_config(b) if b else None)


@ui.page("/backprop")
def backprop_page() -> None:
    from .ui import backprop

    backprop.build()


@ui.page("/mnist")
def mnist_page() -> None:
    from .ui import mnist

    mnist.build()


@ui.page("/about")
def about_page() -> None:
    from .ui import about

    about.build()


def main() -> None:
    # Let the background training threads grab the GIL quickly while the UI thread builds charts.
    sys.setswitchinterval(0.001)
    ui.run(
        title="LearnScope: watch a neural network learn",
        host=os.environ.get("HOST", "127.0.0.1"),  # localhost by default; set HOST=0.0.0.0 to serve a network
        port=int(os.environ.get("PORT", "8080")),
        dark=True,
        reload=False,
        show=False,
        favicon="🧠",
        storage_secret=os.environ.get("LEARNSCOPE_SECRET") or secrets.token_hex(16),
        show_welcome_message=False,
    )


__all__ = ["main", "nicegui_app"]
