"""Threaded Flask runner used by the desktop launcher.

This keeps the existing `webapp.app` module untouched while allowing other
runtimes (pywebview, CLI, tests) to start and stop the backend cleanly.
"""

from __future__ import annotations

import os
import threading
import errno
from contextlib import suppress
from typing import Optional

from werkzeug.serving import make_server


class FlaskServer(threading.Thread):
    """Run the Flask app from ``webapp.app`` inside a background thread."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5525) -> None:
        super().__init__(daemon=True)
        from webapp.app import app  # Imported lazily to avoid side effects at import time

        self.host = host
        self._app = app
        self._server = make_server(host, port, app)
        self.port = self._server.server_port
        self._ctx = app.app_context()
        self._shutdown_requested = threading.Event()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def run(self) -> None:  # type: ignore[override]
        # Push the app context so background threads (e.g., Flask signals) work as expected.
        self._ctx.push()
        try:
            self._server.serve_forever()
        finally:
            self._ctx.pop()

    def shutdown(self) -> None:
        if not self._shutdown_requested.is_set():
            self._shutdown_requested.set()
            with suppress(Exception):
                self._server.shutdown()

    def wait_for_shutdown(self, timeout: Optional[float] = None) -> None:
        self._shutdown_requested.wait(timeout=timeout)


def build_flask_server(host: str = "127.0.0.1", port: int = 5525) -> FlaskServer:
    """Factory helper that respects PORT env overrides."""

    env_port = os.getenv("PSS_DESKTOP_PORT")
    if env_port:
        with suppress(ValueError):
            port = int(env_port)
    try:
        return FlaskServer(host=host, port=port)
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        fallback = FlaskServer(host=host, port=0)
        print(f"⚠️ Port {port} in use. Falling back to {fallback.port}")
        return fallback
