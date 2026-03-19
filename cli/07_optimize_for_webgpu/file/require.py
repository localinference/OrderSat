from __future__ import annotations

import pathlib


def require_file(path: pathlib.Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"{label} does not exist: {path}")
    if not path.is_file():
        raise SystemExit(f"{label} path is not a file: {path}")
