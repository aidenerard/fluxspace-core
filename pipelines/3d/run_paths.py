#!/usr/bin/env python3
"""
run_paths.py

Shared helper for resolving FluxSpace run-folder paths.

Canonical layout:
  RUN_DIR/
    raw/            (sensor data: oak_rgbd/, mag_run.csv, extrinsics.json, calibration.json)
    processed/      (derived: trajectory.csv, open3d_mesh.ply, mag_world.csv)
    exports/        (final: volume.npz, screenshots)
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_run_dir(cli_arg: str | None = None) -> Path:
    """Resolve a run directory from *cli_arg* or the ``$RUN_DIR`` env var.

    Raises ``ValueError`` if neither is available.
    """
    if cli_arg:
        return Path(cli_arg).expanduser().resolve()
    env = os.environ.get("RUN_DIR", "")
    if env:
        return Path(env).expanduser().resolve()
    raise ValueError(
        "No run directory specified. Pass --run <dir> or set $RUN_DIR."
    )


def raw_dir(run: Path) -> Path:
    return run / "raw"


def processed_dir(run: Path) -> Path:
    return run / "processed"


def exports_dir(run: Path) -> Path:
    return run / "exports"


def ensure_dirs(run: Path) -> None:
    """Create raw/, processed/, exports/ under *run* if they don't exist."""
    for d in (raw_dir(run), processed_dir(run), exports_dir(run)):
        d.mkdir(parents=True, exist_ok=True)


def infer_run_dir_from_path(p: Path) -> Path | None:
    """Walk up from *p* to find a parent that contains raw/ or processed/.

    Useful when only an input file path is given (e.g.
    ``RUN/raw/oak_rgbd`` → ``RUN``).
    """
    cur = p.resolve()
    for _ in range(6):  # don't climb too far
        if (cur / "raw").is_dir() or (cur / "processed").is_dir():
            return cur
        cur = cur.parent
    return None
