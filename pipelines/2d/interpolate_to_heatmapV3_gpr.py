#!/usr/bin/env python3
"""
interpolate_to_heatmapV3_gpr.py

Gaussian Process Regression (GPR) + gradient-based "sharpened" map.

Inputs:
  - CSV with at least columns: x, y, and a value column (default: local_anomaly)

Outputs (in --out-dir):
  - *_gpr_mean.png  : GPR mean heatmap
  - *_gpr_grad.png  : gradient magnitude heatmap (sharper edges)
  - *_gpr_std.png   : GPR standard deviation (uncertainty)
  - *_gpr_grid.csv  : dense grid with mean/std/grad

Notes:
  - GPR is O(N^3). If you have many samples, use --max-points to subsample.
  - This is NOT "hallucination": GPR gives a smooth estimate + uncertainty.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
class GridSpec:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    nx: int
    ny: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="2D GPR heatmap + gradient magnitude map (edge-aware).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in", dest="in_csv", required=True, help="Input CSV path.")
    p.add_argument("--value-col", default="local_anomaly", help="Column to model (e.g., local_anomaly or B_total).")
    p.add_argument("--x-col", default="x", help="X column name.")
    p.add_argument("--y-col", default="y", help="Y column name.")

    p.add_argument("--out-dir", default=None, help="Output directory. Default: same folder as input.")
    p.add_argument("--name", default=None, help="Output file prefix (default: input filename stem).")

    p.add_argument("--grid-nx", type=int, default=200, help="Grid resolution in X for outputs.")
    p.add_argument("--grid-ny", type=int, default=200, help="Grid resolution in Y for outputs.")
    p.add_argument("--pad", type=float, default=0.0, help="Padding (meters) around min/max bounds.")
    p.add_argument("--max-points", type=int, default=3000, help="Max training points (subsample if larger).")

    # Kernel params
    p.add_argument("--length-scale", type=float, default=0.03, help="RBF length-scale in meters (smaller = sharper).")
    p.add_argument("--signal", type=float, default=1.0, help="Kernel signal variance multiplier.")
    p.add_argument("--noise", type=float, default=0.001, help="White noise level added to kernel.")
    p.add_argument("--alpha", type=float, default=1e-6, help="GPR alpha (numerical stability).")

    # Visualization scaling
    p.add_argument("--fixed-vmin", type=float, default=None, help="Fixed vmin for mean heatmap.")
    p.add_argument("--fixed-vmax", type=float, default=None, help="Fixed vmax for mean heatmap.")
    p.add_argument("--grad-vmax", type=float, default=None, help="Fixed vmax for gradient heatmap (optional).")

    p.add_argument("--no-optimize", action="store_true", help="Disable kernel hyperparameter optimization.")
    p.add_argument("--seed", type=int, default=7, help="Random seed for subsampling.")
    return p.parse_args()


def _require_sklearn() -> None:
    if GaussianProcessRegressor is None:
        raise RuntimeError(
            "scikit-learn is required for GPR.\n"
            f"Import error: {_SKLEARN_IMPORT_ERROR}\n\n"
            "Install:\n"
            "  pip install scikit-learn\n"
        )


def load_points(csv_path: Path, x_col: str, y_col: str, v_col: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    for c in (x_col, y_col, v_col):
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {csv_path}")
    df = df[[x_col, y_col, v_col]].dropna()
    X = df[[x_col, y_col]].to_numpy(dtype=float)
    y = df[v_col].to_numpy(dtype=float)
    return X, y


def subsample(X: np.ndarray, y: np.ndarray, max_points: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if X.shape[0] <= max_points:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=max_points, replace=False)
    return X[idx], y[idx]


def make_grid(X: np.ndarray, grid_nx: int, grid_ny: int, pad: float) -> GridSpec:
    x_min = float(np.min(X[:, 0]) - pad)
    x_max = float(np.max(X[:, 0]) + pad)
    y_min = float(np.min(X[:, 1]) - pad)
    y_max = float(np.max(X[:, 1]) + pad)
    return GridSpec(x_min, x_max, y_min, y_max, grid_nx, grid_ny)


def predict_grid(gpr, spec: GridSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(spec.x_min, spec.x_max, spec.nx)
    ys = np.linspace(spec.y_min, spec.y_max, spec.ny)
    XX, YY = np.meshgrid(xs, ys)
    Xg = np.column_stack([XX.ravel(), YY.ravel()])
    mean, std = gpr.predict(Xg, return_std=True)
    mean_grid = mean.reshape(spec.ny, spec.nx)
    std_grid = std.reshape(spec.ny, spec.nx)
    return xs, ys, mean_grid, std_grid


def grad_mag(mean_grid: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    # Finite difference gradient with physical spacing
    dx = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    dy = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
    d_dy, d_dx = np.gradient(mean_grid, dy, dx)  # note order: rows=Y, cols=X
    return np.sqrt(d_dx**2 + d_dy**2)


def save_heatmap(arr: np.ndarray, xs: np.ndarray, ys: np.ndarray, title: str, out_path: Path,
                 vmin: Optional[float] = None, vmax: Optional[float] = None, cbar_label: str = "") -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()
    im = ax.imshow(
        arr,
        origin="lower",
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = plt.colorbar(im, ax=ax)
    if cbar_label:
        cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _require_sklearn()

    in_path = Path(args.in_csv).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name if args.name else in_path.stem

    X, y = load_points(in_path, args.x_col, args.y_col, args.value_col)
    X, y = subsample(X, y, args.max_points, args.seed)

    # Normalize y to help kernel scaling; keep transform for output
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

    spec = make_grid(X, args.grid_nx, args.grid_ny, args.pad)
    xs, ys, mean_grid_n, std_grid_n = predict_grid(gpr, spec)

    # Un-normalize mean/std back to original units
    mean_grid = mean_grid_n * y_std + y_mean
    std_grid = std_grid_n * y_std

    gmag = grad_mag(mean_grid, xs, ys)

    mean_png = out_dir / f"{name}_gpr_mean.png"
    grad_png = out_dir / f"{name}_gpr_grad.png"
    std_png = out_dir / f"{name}_gpr_std.png"
    grid_csv = out_dir / f"{name}_gpr_grid.csv"

    save_heatmap(
        mean_grid, xs, ys,
        title=f"GPR mean of {args.value_col}",
        out_path=mean_png,
        vmin=args.fixed_vmin, vmax=args.fixed_vmax,
        cbar_label=args.value_col,
    )
    save_heatmap(
        gmag, xs, ys,
        title=f"Gradient magnitude of GPR mean ({args.value_col})",
        out_path=grad_png,
        vmin=0.0, vmax=args.grad_vmax,
        cbar_label=f"|∇ {args.value_col} |",
    )
    save_heatmap(
        std_grid, xs, ys,
        title=f"GPR uncertainty (std) of {args.value_col}",
        out_path=std_png,
        cbar_label=f"std({args.value_col})",
    )

    # Save dense grid as CSV (long-form)
    XX, YY = np.meshgrid(xs, ys)
    out_df = pd.DataFrame({
        "x": XX.ravel(),
        "y": YY.ravel(),
        f"{args.value_col}_mean": mean_grid.ravel(),
        f"{args.value_col}_std": std_grid.ravel(),
        f"{args.value_col}_gradmag": gmag.ravel(),
    })
    out_df.to_csv(grid_csv, index=False)

    print("Saved:")
    print(f"  {mean_png}")
    print(f"  {grad_png}")
    print(f"  {std_png}")
    print(f"  {grid_csv}")


if __name__ == "__main__":
    main()
