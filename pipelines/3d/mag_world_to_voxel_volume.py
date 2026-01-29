#!/usr/bin/env python3
"""
mag_world_to_voxel_volume.py

Build a 3D voxel volume from mag_world.csv (x, y, z, value).
Input: mag_world.csv, voxel_size, margin (bounds auto from data + margin).
Output: volume.npz containing 3D grid array and metadata (origin, voxel_size, axes).

Interpolation: IDW (k-nearest) default; optional scipy griddata. Robust for ~50k points.
Deterministic (no random seed needed for IDW).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.interpolate import griddata
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="mag_world.csv -> 3D voxel volume (volume.npz)"
    )
    p.add_argument("--in", dest="input_csv", required=True, help="Path to mag_world.csv")
    p.add_argument("--out", default="", help="Output volume.npz path. Default: <input_dir>/../exports/volume.npz")
    p.add_argument("--voxel-size", type=float, default=0.02, help="Voxel edge length in meters (default: 0.02)")
    p.add_argument("--margin", type=float, default=0.1, help="Margin added to data bounds (meters, default: 0.1)")
    p.add_argument("--method", choices=["idw", "griddata"], default="idw", help="Interpolation: idw (k-nearest) or griddata")
    p.add_argument("--k", type=int, default=8, help="Number of nearest neighbors for IDW (default: 8)")
    p.add_argument("--power", type=float, default=2.0, help="IDW power (default: 2.0)")
    return p.parse_args()


def idw_interpolate(
    points: np.ndarray,
    values: np.ndarray,
    grid_xyz: np.ndarray,
    k: int = 8,
    power: float = 2.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Interpolate at grid_xyz (N,3) using k-nearest IDW. points (M,3), values (M,)."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        raise RuntimeError("IDW method requires scipy. Install with: pip install scipy")
    tree = cKDTree(points)
    dists, idx = tree.query(grid_xyz, k=k, workers=-1)
    dists = np.maximum(dists, eps)
    w = 1.0 / (dists ** power)
    w /= w.sum(axis=1, keepdims=True)
    out = (w * values[idx]).sum(axis=1)
    return out


def main() -> int:
    args = parse_args()
    inp = Path(args.input_csv)
    if not inp.exists():
        print(f"ERROR: File not found: {inp}", file=sys.stderr)
        return 2

    df = pd.read_csv(inp)
    for col in ("x", "y", "z", "value"):
        if col not in df.columns:
            print(f"ERROR: Missing column '{col}' in {inp}", file=sys.stderr)
            return 2
    df = df.dropna(subset=["x", "y", "z", "value"])
    if df.empty:
        print("ERROR: No valid rows after dropping NaN", file=sys.stderr)
        return 2

    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=float)
    v = df["value"].to_numpy(dtype=float)
    points = np.column_stack([x, y, z])

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    zmin, zmax = z.min(), z.max()
    margin = float(args.margin)
    voxel_size = float(args.voxel_size)
    if voxel_size <= 0:
        print("ERROR: --voxel-size must be > 0", file=sys.stderr)
        return 2

    xmin -= margin
    xmax += margin
    ymin -= margin
    ymax += margin
    zmin -= margin
    zmax += margin

    origin = np.array([xmin, ymin, zmin], dtype=float)
    nx = max(1, int(np.ceil((xmax - xmin) / voxel_size)))
    ny = max(1, int(np.ceil((ymax - ymin) / voxel_size)))
    nz = max(1, int(np.ceil((zmax - zmin) / voxel_size)))

    xi = np.linspace(xmin, xmin + (nx - 1) * voxel_size, nx)
    yi = np.linspace(ymin, ymin + (ny - 1) * voxel_size, ny)
    zi = np.linspace(zmin, zmin + (nz - 1) * voxel_size, nz)
    gx, gy, gz = np.meshgrid(xi, yi, zi, indexing="ij")
    grid_xyz = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    print(f"Building volume: {nx} x {ny} x {nz} = {grid_xyz.shape[0]} voxels from {len(points)} points...")

    if args.method == "idw":
        vol_flat = idw_interpolate(
            points, v, grid_xyz,
            k=min(args.k, len(points)),
            power=args.power,
        )
    else:
        if not HAS_SCIPY:
            print("ERROR: --method griddata requires scipy", file=sys.stderr)
            return 1
        vol_flat = griddata(points, v, grid_xyz, method="linear", fill_value=np.nan)
        vol_flat = np.nan_to_num(vol_flat, nan=np.nanmean(v))

    volume = vol_flat.reshape((nx, ny, nz)).astype(np.float64)

    if args.out:
        out_path = Path(args.out)
    else:
        parent = inp.parent
        if parent.name == "processed" and (parent.parent / "exports").exists():
            out_path = parent.parent / "exports" / "volume.npz"
        else:
            out_path = inp.parent / "volume.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        volume=volume,
        origin=origin,
        voxel_size=np.array(voxel_size),
        nx=nx, ny=ny, nz=nz,
        axes_x=xi, axes_y=yi, axes_z=zi,
    )
    print(f"Wrote volume: {out_path}")
    print(f"  shape: {volume.shape}, origin: {origin}, voxel_size: {voxel_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
