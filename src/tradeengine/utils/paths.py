from __future__ import annotations

from pathlib import Path


def ensure_parent_dir(path: str) -> Path:
    """Create the parent directory for a file path and return it as Path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
