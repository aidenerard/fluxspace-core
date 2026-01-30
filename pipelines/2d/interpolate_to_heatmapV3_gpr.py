#!/usr/bin/env python3
"""
interpolate_to_heatmapV3_gpr.py

V3 heatmap: interpolation method idw (same as V2) or gpr (GPR + uncertainty).
Outputs: standard anomaly heatmap, fixed-scale heatmap, gradient magnitude heatmap.
Optional: localization JSON (peak + centroid of top X%% of gradient map).

Inputs: CSV with x, y, and value column (default: local_anomaly). Same columns as V2.
"""

from __future__ import annotations

import argparse
import json
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
        description="2D heatmap V3: idw or gpr + gradient; standard, fixed-scale, and grad outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in", dest="in_csv", required=True, help="Input CSV path.")
    p.add_argument("--value-col", default="local_anomaly", help="Column to model (e.g., local_anomaly or B_total).")
    p.add_argument("--x-col", default="x", help="X column name.")
    p.add_argument("--y-col", default="y", help="Y column name.")
    p.add_argument("--method", choices=["idw", "gpr"], default="gpr", help="Interpolation: idw (same as V2) or gpr.")

    p.add_argument("--out-dir", default=None, help="Output directory. Default: same folder as input.")
    p.add_argument("--name", default=None, help="Output file prefix (default: input filename stem).")

    p.add_argument("--grid-nx", type=int, default=200, help="Grid resolution in X.")
    p.add_argument("--grid-ny", type=int, default=200, help="Grid resolution in Y.")
    p.add_argument("--pad", type=float, default=0.0, help="Padding (meters) around bounds.")
    p.add_argument("--max-points", type=int, default=3000, help="Max training points for GPR (subsample if larger).")

    # IDW (same as V2)
    p.add_argument("--power", type=float, default=2.0, help="IDW power (for --method idw).")
    p.add_argument("--eps", type=float, default=1e-12, help="Epsilon for IDW.")
    p.add_argument("--clip-percentile", type=float, default=99.0, help="Clip color scale for auto heatmap.")
    p.add_argument("--fixed-vmin", type=float, default=None, help="Fixed vmin for fixed-scale heatmap.")
    p.add_argument("--fixed-vmax", type=float, default=None, help="Fixed vmax for fixed-scale heatmap.")
    p.add_argument("--fixed-overwrite", action="store_true", help="Overwrite fixed scale JSON with this run.")

    # GPR kernel params
    p.add_argument("--length-scale", type=float, default=0.03, help="RBF length-scale in meters (GPR).")
    p.add_argument("--signal", type=float, default=1.0, help="Kernel signal variance (GPR).")
    p.add_argument("--noise", type=float, default=0.001, help="White noise level (GPR).")
    p.add_argument("--alpha", type=float, default=1e-6, help="GPR alpha (numerical stability).")
    p.add_argument("--no-optimize", action="store_true", help="Disable GPR kernel hyperparameter optimization.")

    p.add_argument("--grad-vmax", type=float, default=None, help="Fixed vmax for gradient heatmap (optional).")
    p.add_argument("--also-grad", action="store_true", default=True, help="Output gradient magnitude heatmap (default: True).")
    p.add_argument("--seed", type=int, default=7, help="Random seed for subsampling.")
    # Optional localization JSON from gradient map
    p.add_argument("--localization-top-pct", type=float, default=None, help="If set, write JSON with peak and centroid of top X%% of gradient (e.g. 5).")
    return p.parse_args()


def _require_sklearn() -> None:
    if GaussianProcessRegressor is None:
        raise RuntimeError(
            "scikit-learn is required for GPR.\n"
            f"Import error: {_SKLEARN_IMPORT_ERROR}\n\n"
            "Install:\n"
            "  pip install scikit-learn\n"
        )


def make_grid_axes(x_min: float, x_max: float, y_min: float, y_max: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    gx = np.linspace(x_min, x_max, nx)
    gy = np.linspace(y_min, y_max, ny)
    return gx, gy


def idw_grid(
    x: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    power: float = 2.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """IDW interpolation; returns grid Z shape (len(gy), len(gx))."""
    Xg, Yg = np.meshgrid(gx, gy)
    tx, ty = Xg.ravel(), Yg.ravel()
    Z = np.empty_like(tx, dtype=float)
    for i in range(tx.size):
        d2 = (tx[i] - x) ** 2 + (ty[i] - y) ** 2
        j0 = np.argmin(d2)
        if d2[j0] <= eps:
            Z[i] = v[j0]
            continue
        w = 1.0 / (d2 ** (power / 2.0) + eps)
        Z[i] = float(np.sum(w * v) / np.sum(w))
    return Z.reshape(Yg.shape)


def compute_clipped_range(Z: np.ndarray, clip_percentile: float) -> Tuple[Optional[float], Optional[float]]:
    if 0 < clip_percentile < 100:
        lo = float(np.nanpercentile(Z, 100 - clip_percentile))
        hi = float(np.nanpercentile(Z, clip_percentile))
        return lo, hi
    return None, None


def resolve_outdir(in_path: Path, outdir_arg: Optional[str]) -> Path:
    if outdir_arg:
        return Path(outdir_arg).expanduser().resolve()
    if in_path.parent.name == "processed" and in_path.parent.parent.name == "data":
        return in_path.parent.parent / "exports"
    return in_path.parent


def load_or_init_fixed_scale(
    scale_path: Path,
    Z: np.ndarray,
    clip_percentile: float,
    fixed_vmin: Optional[float],
    fixed_vmax: Optional[float],
    overwrite: bool,
) -> Tuple[float, float]:
    if (fixed_vmin is None) ^ (fixed_vmax is None):
        raise ValueError("Set both --fixed-vmin and --fixed-vmax or neither.")
    if fixed_vmin is not None and fixed_vmax is not None:
        vmin, vmax = float(fixed_vmin), float(fixed_vmax)
        if overwrite or not scale_path.exists():
            scale_path.write_text(json.dumps({"vmin": vmin, "vmax": vmax}, indent=2) + "\n")
        return vmin, vmax
    if scale_path.exists() and not overwrite:
        try:
            data = json.loads(scale_path.read_text())
            return float(data["vmin"]), float(data["vmax"])
        except Exception:
            pass
    vmin, vmax = compute_clipped_range(Z, clip_percentile)
    if vmin is None or vmax is None:
        vmin, vmax = float(np.nanmin(Z)), float(np.nanmax(Z))
    scale_path.write_text(json.dumps({"vmin": float(vmin), "vmax": float(vmax)}, indent=2) + "\n")
    return float(vmin), float(vmax)


def localization_from_grad(gmag: np.ndarray, xs: np.ndarray, ys: np.ndarray, top_pct: float) -> dict:
    """Peak (argmax) and centroid of top X% of gradient magnitude pixels."""
    flat = gmag.ravel()
    thresh = np.percentile(flat, 100.0 - top_pct)
    mask = flat >= thresh
    if not np.any(mask):
        return {"peak_xy": None, "centroid_xy": None, "top_pct": top_pct}
    ny, nx = gmag.shape
    idx_flat = np.where(mask)[0]
    iy = idx_flat // nx
    ix = idx_flat % nx
    peak_flat = np.argmax(flat)
    peak_iy, peak_ix = peak_flat // nx, peak_flat % nx
    peak_x = float(xs[peak_ix])
    peak_y = float(ys[peak_iy])
    w = flat[mask]
    cx = float(np.average(xs[ix], weights=w))
    cy = float(np.average(ys[iy], weights=w))
    return {"peak_xy": [peak_x, peak_y], "centroid_xy": [cx, cy], "top_pct": top_pct}


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


def main() -> int:
    args = parse_args()
    in_path = Path(args.in_csv).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_dir = resolve_outdir(in_path, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name if args.name else in_path.stem
    safe_col = args.value_col.replace("/", "_")
    scale_path = out_dir / f"heatmap_scale_{safe_col}.json"

    if args.method == "idw":
        X, y = load_points(in_path, args.x_col, args.y_col, args.value_col)
        x_min, x_max = float(np.min(X[:, 0])) - args.pad, float(np.max(X[:, 0])) + args.pad
        y_min, y_max = float(np.min(X[:, 1])) - args.pad, float(np.max(X[:, 1])) + args.pad
        gx, gy = make_grid_axes(x_min, x_max, y_min, y_max, args.grid_nx, args.grid_ny)
        Z = idw_grid(X[:, 0], X[:, 1], y, gx, gy, power=args.power, eps=args.eps)
        xs, ys = gx, gy
        mean_grid = Z
        std_grid = np.full_like(Z, np.nan)
        gmag = grad_mag(Z, xs, ys) if args.also_grad else np.zeros_like(Z)
    else:
        _require_sklearn()
        X, y = load_points(in_path, args.x_col, args.y_col, args.value_col)
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
        spec = make_grid(X, args.grid_nx, args.grid_ny, args.pad)
        xs, ys, mean_grid_n, std_grid_n = predict_grid(gpr, spec)
        mean_grid = mean_grid_n * y_std + y_mean
        std_grid = std_grid_n * y_std
        gmag = grad_mag(mean_grid, xs, ys) if args.also_grad else np.zeros_like(mean_grid)

    # Standard (auto-scale) heatmap
    vmin_auto, vmax_auto = compute_clipped_range(mean_grid, args.clip_percentile)
    save_heatmap(
        mean_grid, xs, ys,
        title=f"Heatmap ({args.method}) of {args.value_col}",
        out_path=out_dir / f"{name}_heatmap.png",
        vmin=vmin_auto, vmax=vmax_auto,
        cbar_label=args.value_col,
    )
    # Fixed-scale heatmap
    vmin_fix, vmax_fix = load_or_init_fixed_scale(
        scale_path, mean_grid, args.clip_percentile,
        args.fixed_vmin, args.fixed_vmax, args.fixed_overwrite,
    )
    save_heatmap(
        mean_grid, xs, ys,
        title=f"Heatmap ({args.method}) of {args.value_col} (fixed scale)",
        out_path=out_dir / f"{name}_heatmap_fixed.png",
        vmin=vmin_fix, vmax=vmax_fix,
        cbar_label=args.value_col,
    )
    if args.also_grad and np.any(np.isfinite(gmag)):
        save_heatmap(
            gmag, xs, ys,
            title=f"Gradient magnitude |∇{args.value_col}|",
            out_path=out_dir / f"{name}_grad.png",
            vmin=0.0, vmax=args.grad_vmax,
            cbar_label=f"|∇ {args.value_col} |",
        )

    if args.localization_top_pct is not None and args.also_grad and np.any(np.isfinite(gmag)):
        loc = localization_from_grad(gmag, xs, ys, args.localization_top_pct)
        loc_path = out_dir / f"{name}_localization_grad.json"
        loc_path.write_text(json.dumps(loc, indent=2) + "\n")
        print(f"  {loc_path}")

    if args.method == "gpr":
        save_heatmap(
            std_grid, xs, ys,
            title=f"GPR std of {args.value_col}",
            out_path=out_dir / f"{name}_gpr_std.png",
            cbar_label=f"std({args.value_col})",
        )
        XX, YY = np.meshgrid(xs, ys)
        out_df = pd.DataFrame({
            "x": XX.ravel(),
            "y": YY.ravel(),
            f"{args.value_col}_mean": mean_grid.ravel(),
            f"{args.value_col}_std": std_grid.ravel(),
            f"{args.value_col}_gradmag": gmag.ravel(),
        })
        grid_csv = out_dir / f"{name}_gpr_grid.csv"
        out_df.to_csv(grid_csv, index=False)
        print(f"  {grid_csv}")

    print("Saved:")
    print(f"  {out_dir / f'{name}_heatmap.png'}")
    print(f"  {out_dir / f'{name}_heatmap_fixed.png'}")
    if args.also_grad:
        print(f"  {out_dir / f'{name}_grad.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
