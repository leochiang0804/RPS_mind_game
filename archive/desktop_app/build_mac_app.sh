#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYINSTALLER_BIN="${PYINSTALLER_BIN:-$PROJECT_ROOT/venv/bin/pyinstaller}"
if [ ! -x "$PYINSTALLER_BIN" ]; then
  PYINSTALLER_BIN="$(command -v pyinstaller || true)"
fi

if [ -z "$PYINSTALLER_BIN" ]; then
  echo "PyInstaller not found. Install via 'pip install pyinstaller'." >&2
  exit 1
fi

"$PYINSTALLER_BIN" desktop_app/mac_app.spec --noconfirm

APP_PATH="dist/PaperScissorStone.app"
if [ -d "$APP_PATH" ]; then
  echo "macOS bundle created at $APP_PATH"
  echo "Sign and notarise the app before distribution." >&2
fi
