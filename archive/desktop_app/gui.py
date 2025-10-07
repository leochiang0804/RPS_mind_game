"""PyWebview-based desktop shell for the adaptive RPS game."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

try:
    import webview
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "pywebview is required for the desktop launcher. Install it via 'pip install pywebview'."
    ) from exc

from .server_runner import build_flask_server


DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 800


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive RPS desktop launcher")
    parser.add_argument("--host", default="127.0.0.1", help="Host for the embedded Flask server")
    parser.add_argument("--port", type=int, default=5525, help="Port for the embedded Flask server")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Initial window width")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Initial window height")
    parser.add_argument("--debug", action="store_true", help="Open dev tools overlay")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    # Start the Flask backend.
    server = build_flask_server(host=args.host, port=args.port)
    server.start()

    # Wait briefly to ensure the server is listening before opening the window.
    time.sleep(0.5)

    window = webview.create_window(
        title="Paper-Scissor-Stone",
        url=server.url,
        width=args.width,
        height=args.height,
    )

    if args.debug:
        window.events.loaded += lambda: webview.windows[0].evaluate_js("console.log('Dev tools enabled')")
        webview.start(gui=None, debug=True)
    else:
        webview.start()

    # When the GUI loop exits, shut down the server.
    server.shutdown()
    server.wait_for_shutdown(timeout=2.0)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
