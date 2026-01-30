#!/usr/bin/env python3
"""
mag_world_to_voxel_volumeV2_gpr.py

mag_world.csv -> volume.npz with optional method: idw (same as original) or gpr.

- method idw: k-nearest IDW interpolation, then gradient magnitude (same deps as mag_world_to_voxel_volume).
- method gpr: Gaussian Process Regression; outputs mean, std (uncertainty), and gradient magnitude.

Outputs (single volume.npz by default):
  - volume: 3D array (nz, ny, nx)
  - grad:   gradient magnitude (always)
  - std:    uncertainty (GPR only)
  - origin, voxel_size, dims

Optional: --out-dir + separate _gpr_mean/_gpr_std/_gpr_grad.npz when method=gpr (legacy).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None  # type: ignore

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
except Exception as e:
    GaussianProcessRegressor = None  # type: ignore
    RBF = WhiteKernel = ConstantKernel = None  # type: ignore
    _SKLEARN_IMPORT_ERROR = e
else:
    _SKLEARN_IMPORT_ERROR = None


@dataclass
class VoxelGrid:
    origin: np.ndarray  # (3,) x,y,z
    dims: np.ndarray    # (nx, ny, nz)
    voxel: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="mag_world.csv -> volume.npz (idw or gpr); always includes grad.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in", dest="in_csv", required=True, help="Path to mag_world.csv (or CSV with x,y,z,value).")
    p.add_argument("--out", default="", help="Output volume.npz path. Default: <out-dir>/volume.npz or <input_dir>/../exports/volume.npz")
    p.add_argument("--value-col", default="value", help="Value column (mag_world.csv uses 'value').")
    p.add_argument("--x-col", default="x", help="X column name.")
    p.add_argument("--y-col", default="y", help="Y column name.")
    p.add_argument("--z-col", default="z", help="Z column name.")
    p.add_argument("--method", choices=["idw", "gpr"], default="idw", help="Interpolation: idw (fast) or gpr (smooth + uncertainty).")

    p.add_argument("--out-dir", default=None, help="Output directory (default: same folder as input or ../exports).")
    p.add_argument("--name", default=None, help="Output prefix for legacy _gpr_*.npz (method=gpr only).")

    p.add_argument("--voxel", type=float, default=0.02, help="Voxel size in meters.")
    p.add_argument("--pad", type=float, default=0.0, help="Padding around bounds (meters).")
    p.add_argument("--max-points", type=int, default=2000, help="Max training points for GPR (subsample if larger).")
    p.add_argument("--k", type=int, default=8, help="Nearest neighbors for IDW.")
    p.add_argument("--power", type=float, default=2.0, help="IDW power.")

    # GPR kernel params
    p.add_argument("--length-scale", type=float, default=0.05, help="RBF length-scale in meters (GPR).")
    p.add_argument("--signal", type=float, default=1.0, help="Kernel signal variance (GPR).")
    p.add_argument("--noise", type=float, default=0.001, help="White noise level (GPR).")
    p.add_argument("--alpha", type=float, default=1e-6, help="GPR alpha.")
    p.add_argument("--no-optimize", action="store_true", help="Disable GPR kernel optimization.")
    p.add_argument("--seed", type=int, default=7, help="Random seed for subsampling.")
    p.add_argument("--chunk", type=int, default=200000, help="Prediction chunk size (GPR).")
    return p.parse_args()


def _require_sklearn() -> None:
    if GaussianProcessRegressor is None:
        raise RuntimeError(
            "scikit-learn is required for GPR.\n"
            f"Import error: {_SKLEARN_IMPORT_ERROR}\n\n"
            "Install:\n"
            "  pip install scikit-learn\n"
        )


def _require_scipy() -> None:
    if cKDTree is None:
        raise RuntimeError("IDW method requires scipy. Install with: pip install scipy")


def idw_interpolate(
    points: np.ndarray,
    values: np.ndarray,
    grid_xyz: np.ndarray,
    k: int = 8,
    power: float = 2.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Interpolate at grid_xyz (N,3) using k-nearest IDW."""
    tree = cKDTree(points)
    dists, idx = tree.query(grid_xyz, k=min(k, len(points)), workers=-1)
    dists = np.maximum(dists, eps)
    w = 1.0 / (dists ** power)
    w /= w.sum(axis=1, keepdims=True)
    return (w * values[idx]).sum(axis=1)


def load_points(csv_path: Path, x_col: str, y_col: str, z_col: str, v_col: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    for c in (x_col, y_col, z_col, v_col):
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {csv_path}")
    df = df[[x_col, y_col, z_col, v_col]].dropna()
    X = df[[x_col, y_col, z_col]].to_numpy(dtype=float)
    y = df[v_col].to_numpy(dtype=float)
    return X, y


def subsample(X: np.ndarray, y: np.ndarray, max_points: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if X.shape[0] <= max_points:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=max_points, replace=False)
    return X[idx], y[idx]


def build_grid(X: np.ndarray, voxel: float, pad: float) -> VoxelGrid:
    mins = np.min(X, axis=0) - pad
    maxs = np.max(X, axis=0) + pad
    spans = maxs - mins
    dims = np.maximum(np.ceil(spans / voxel).astype(int) + 1, 1)
    return VoxelGrid(origin=mins.astype(float), dims=dims, voxel=float(voxel))


def iter_grid_points(grid: VoxelGrid, chunk: int):
    nx, ny, nz = grid.dims.tolist()
    ox, oy, oz = grid.origin.tolist()
    v = grid.voxel
    total = nx * ny * nz

    # Generate in linear index chunks to avoid huge RAM spikes
    for start in range(0, total, chunk):
        end = min(total, start + chunk)
        idx = np.arange(start, end, dtype=np.int64)

        ix = idx % nx
        iy = (idx // nx) % ny
        iz = idx // (nx * ny)

        xs = ox + ix * v
        ys = oy + iy * v
        zs = oz + iz * v

        pts = np.column_stack([xs, ys, zs]).astype(float)
        yield idx, pts


def linear_to_ijk(idx: np.ndarray, grid: VoxelGrid) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, _ = grid.dims.tolist()
    ix = idx % nx
    iy = (idx // nx) % ny
    iz = idx // (nx * ny)
    return ix.astype(int), iy.astype(int), iz.astype(int)


def gradient_magnitude(volume: np.ndarray, voxel: float) -> np.ndarray:
    # volume shape: (nz, ny, nx) or (ny, nx, nz)? We'll store (nz, ny, nx) for consistency with z-fast?
    # We'll store as (nz, ny, nx) below. np.gradient expects spacing per axis in same order.
    dz, dy, dx = np.gradient(volume, voxel, voxel, voxel)
    return np.sqrt(dx**2 + dy**2 + dz**2)


def main() -> int:
    args = parse_args()
    in_path = Path(args.in_csv).expanduser().resolve()
    if not in_path.exists():
        print(f"ERROR: File not found: {in_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else in_path.parent
    if args.out:
        out_volume = Path(args.out).expanduser().resolve()
    else:
        if in_path.parent.name == "processed" and (in_path.parent.parent / "exports").exists():
            out_volume = in_path.parent.parent / "exports" / "volume.npz"
        else:
            out_volume = out_dir / "volume.npz"
    out_dir = out_volume.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name if args.name else in_path.stem

    X, y = load_points(in_path, args.x_col, args.y_col, args.z_col, args.value_col)
    grid = build_grid(X, args.voxel, args.pad)
    nx, ny, nz = grid.dims.tolist()
    origin = grid.origin
    voxel = grid.voxel

    if args.method == "idw":
        _require_scipy()
        ox, oy, oz = origin[0], origin[1], origin[2]
        xi = np.linspace(ox, ox + (nx - 1) * voxel, nx)
        yi = np.linspace(oy, oy + (ny - 1) * voxel, ny)
        zi = np.linspace(oz, oz + (nz - 1) * voxel, nz)
        gx, gy, gz = np.meshgrid(xi, yi, zi, indexing="ij")
        grid_xyz = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        print(f"Building volume (IDW): {nx} x {ny} x {nz} = {grid_xyz.shape[0]} voxels from {len(X)} points...")
        vol_flat = idw_interpolate(X, y, grid_xyz, k=args.k, power=args.power)
        vol_nxynz = vol_flat.reshape((nx, ny, nz)).astype(np.float32)
        # Store as (nz, ny, nx) for viewer
        mean_vol = np.transpose(vol_nxynz, (2, 1, 0)).copy()
        std_vol = np.full_like(mean_vol, np.nan, dtype=np.float32)
    else:
        _require_sklearn()
        X, y = subsample(X, y, args.max_points, args.seed)
        y_mean = float(np.mean(y))
        y_std = float(np.std(y)) if float(np.std(y)) > 0 else 1.0
        y_n = (y - y_mean) / y_std
        kernel = ConstantKernel(constant_value=args.signal, constant_value_bounds="fixed") * RBF(
            length_scale=args.length_scale, length_scale_bounds="fixed" if args.no_optimize else (1e-4, 10.0)
        ) + WhiteKernel(noise_level=args.noise, noise_level_bounds="fixed" if args.no_optimize else (1e-8, 1e-1))
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=args.alpha,
            normalize_y=False,
            optimizer=None if args.no_optimize else "fmin_l_bfgs_b",
            random_state=args.seed,
        )
        gpr.fit(X, y_n)
        mean_vol_n = np.empty((nz, ny, nx), dtype=np.float32)
        std_vol_n = np.empty((nz, ny, nx), dtype=np.float32)
        for idx, pts in iter_grid_points(grid, args.chunk):
            mean_chunk, std_chunk = gpr.predict(pts, return_std=True)
            ix, iy, iz = linear_to_ijk(idx, grid)
            mean_vol_n[iz, iy, ix] = mean_chunk.astype(np.float32)
            std_vol_n[iz, iy, ix] = std_chunk.astype(np.float32)
        mean_vol = (mean_vol_n * y_std + y_mean).astype(np.float32)
        std_vol = (std_vol_n * y_std).astype(np.float32)
        print(f"Building volume (GPR): {nx} x {ny} x {nz} from {len(X)} points...")

    grad_vol = gradient_magnitude(mean_vol, voxel).astype(np.float32)

    meta = {
        "origin": origin.astype(np.float32),
        "voxel_size": np.float32(voxel),
        "dims": grid.dims.astype(np.int32),
    }
    save_dict = {"volume": mean_vol, "grad": grad_vol, **meta}
    if args.method == "gpr":
        save_dict["std"] = std_vol
    np.savez_compressed(out_volume, **save_dict)

    print(f"Wrote: {out_volume}")
    print(f"  shape (z,y,x): {mean_vol.shape}  voxel={voxel}  origin={origin.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
