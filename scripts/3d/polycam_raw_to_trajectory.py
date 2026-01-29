#!/usr/bin/env python3
"""Thin wrapper: runs pipelines/3d/polycam_raw_to_trajectory.py (preserves argv)."""
import os
import runpy
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REAL_SCRIPT = os.path.join(_REPO_ROOT, "pipelines", "3d", "polycam_raw_to_trajectory.py")
sys.argv[0] = _REAL_SCRIPT
runpy.run_path(_REAL_SCRIPT, run_name="__main__")
