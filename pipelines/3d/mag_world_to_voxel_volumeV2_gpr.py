#!/usr/bin/env python3
"""
mag_world_to_voxel_volumeV2_gpr.py

3D voxel volume generation using Gaussian Process Regression (GPR).

This is the 3D analogue of the 2D GPR heatmap approach:
  - Fit GPR to sparse (x,y,z)->value samples (default: local_anomaly).
  - Evaluate on a voxel grid to produce a dense 3D field.
  - Also outputs an uncertainty volume (std) and a gradient magnitude volume.

Outputs:
  - <name>_gpr_mean.npz : contains volume (mean), origin, voxel_size, dims
  - <name>_gpr_std.npz  : same metadata, volume = std
  - <name>_gpr_grad.npz : same metadata, volume = |grad(mean)|

Notes:
  - GPR is O(N^3). Use --max-points to subsample if you have many samples.
  - 3D grids get big fast. Start with voxel=0.02 or 0.03 to validate, then tighten.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

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
    origin: np.ndarray  # (3,)
    dims: np.ndarray    # (3,) ints: nx, ny, nz
    voxel: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit 3D GPR and rasterize to a voxel volume (mean/std/grad).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in", dest="in_csv", required=True, help="Input CSV with x,y,z and value column.")
    p.add_argument("--value-col", default="local_anomaly", help="Column to model.")
    p.add_argument("--x-col", default="x", help="X column name.")
    p.add_argument("--y-col", default="y", help="Y column name.")
    p.add_argument("--z-col", default="z", help="Z column name.")

    p.add_argument("--out-dir", default=None, help="Output directory (default: same folder as input).")
    p.add_argument("--name", default=None, help="Output prefix (default: input filename stem).")

    p.add_argument("--voxel", type=float, default=0.02, help="Voxel size in meters.")
    p.add_argument("--pad", type=float, default=0.0, help="Padding around bounds (meters).")
    p.add_argument("--max-points", type=int, default=2000, help="Max training points (subsample if larger).")

    # Kernel params
    p.add_argument("--length-scale", type=float, default=0.05, help="RBF length-scale in meters.")
    p.add_argument("--signal", type=float, default=1.0, help="Kernel signal variance multiplier.")
    p.add_argument("--noise", type=float, default=0.001, help="White noise level added to kernel.")
    p.add_argument("--alpha", type=float, default=1e-6, help="GPR alpha (numerical stability).")
    p.add_argument("--no-optimize", action="store_true", help="Disable kernel hyperparameter optimization.")
    p.add_argument("--seed", type=int, default=7, help="Random seed for subsampling.")

    # Compute controls
    p.add_argument("--chunk", type=int, default=200000, help="Prediction chunk size (points per batch).")
    return p.parse_args()


def _require_sklearn() -> None:
    if GaussianProcessRegressor is None:
        raise RuntimeError(
            "scikit-learn is required for GPR.\n"
            f"Import error: {_SKLEARN_IMPORT_ERROR}\n\n"
            "Install:\n"
            "  pip install scikit-learn\n"
        )


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


def main() -> None:
    args = parse_args()
    _require_sklearn()

    in_path = Path(args.in_csv).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name if args.name else in_path.stem

    X, y = load_points(in_path, args.x_col, args.y_col, args.z_col, args.value_col)
    X, y = subsample(X, y, args.max_points, args.seed)

    # Normalize y for stability
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

    grid = build_grid(X, args.voxel, args.pad)
    nx, ny, nz = grid.dims.tolist()

    # Store volume as (nz, ny, nx) so slicing by z is volume[z,:,:]
    mean_vol_n = np.empty((nz, ny, nx), dtype=np.float32)
    std_vol_n = np.empty((nz, ny, nx), dtype=np.float32)

    for idx, pts in iter_grid_points(grid, args.chunk):
        mean_chunk, std_chunk = gpr.predict(pts, return_std=True)
        ix, iy, iz = linear_to_ijk(idx, grid)
        mean_vol_n[iz, iy, ix] = mean_chunk.astype(np.float32)
        std_vol_n[iz, iy, ix] = std_chunk.astype(np.float32)

    # Un-normalize
    mean_vol = mean_vol_n * y_std + y_mean
    std_vol = std_vol_n * y_std

    grad_vol = gradient_magnitude(mean_vol.astype(np.float32), grid.voxel).astype(np.float32)

    meta = {
        "origin": grid.origin.astype(np.float32),
        "voxel_size": np.float32(grid.voxel),
        "dims": grid.dims.astype(np.int32),
        "axis_order": np.array(["z", "y", "x"]),  # volume shape order
        "value_col": np.array([args.value_col]),
    }

    out_mean = out_dir / f"{name}_gpr_mean.npz"
    out_std = out_dir / f"{name}_gpr_std.npz"
    out_grad = out_dir / f"{name}_gpr_grad.npz"

    np.savez_compressed(out_mean, volume=mean_vol.astype(np.float32), **meta)
    np.savez_compressed(out_std, volume=std_vol.astype(np.float32), **meta)
    np.savez_compressed(out_grad, volume=grad_vol.astype(np.float32), **meta)

    print("Saved:")
    print(f"  {out_mean}")
    print(f"  {out_std}")
    print(f"  {out_grad}")
    print(f"Volume shape (z,y,x): {mean_vol.shape}  voxel={grid.voxel}  origin={grid.origin.tolist()}")


if __name__ == "__main__":
    main()
