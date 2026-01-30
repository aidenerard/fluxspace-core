#!/usr/bin/env python3
"""
visualize_3d_heatmapV2_gpr.py

Viewer for voxel volumes saved by:
  - mag_world_to_voxel_volume.py (IDW) -> volume.npz
  - mag_world_to_voxel_volumeV2_gpr.py (GPR) -> *_gpr_mean.npz, *_gpr_std.npz, *_gpr_grad.npz

Displays:
  - Thresholded 3D scatter of voxels above a percentile/abs threshold
  - Optional orthogonal slice plots (XY/XZ/YZ)
  - Optional side-by-side slice comparison (mean vs grad)

Supports modes:
  --mode mean | std | grad | value
    - mean/std/grad will auto-locate sibling npz files if your input is one of them.
    - value is identical to mean for non-GPR volumes (just uses the 'volume' inside the file)

Notes:
  - This script uses Matplotlib (no PyVista/VTK), so it avoids PyVista API issues.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize voxel volume (.npz) as thresholded point cloud + slices (GPR-aware).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in", dest="in_npz", required=True, help="Input .npz (e.g. volume.npz or *_gpr_mean.npz).")

    p.add_argument(
        "--mode",
        choices=["value", "mean", "std", "grad"],
        default="value",
        help="Which field to visualize. For GPR, mean/std/grad can auto-find sibling files.",
    )

    p.add_argument("--percentile", type=float, default=99.0, help="Show voxels >= this percentile of selected field.")
    p.add_argument("--abs-thresh", type=float, default=None, help="Absolute threshold (overrides percentile).")
    p.add_argument("--max-points", type=int, default=300000, help="Cap points plotted (random subsample).")
    p.add_argument("--seed", type=int, default=7, help="Seed for subsampling.")

    p.add_argument("--show-slices", action="store_true", help="Show XY/XZ/YZ slices in separate figures.")
    p.add_argument("--compare-mean-vs-grad", action="store_true",
                   help="Also show a mean-vs-grad slice comparison (if available / computable).")

    p.add_argument("--slice-z", type=int, default=None, help="Z index for XY slice (default: middle).")
    p.add_argument("--slice-y", type=int, default=None, help="Y index for XZ slice (default: middle).")
    p.add_argument("--slice-x", type=int, default=None, help="X index for YZ slice (default: middle).")

    p.add_argument("--save", action="store_true", help="Save PNGs to --out-dir instead of only showing.")
    p.add_argument("--out-dir", default=None, help="Output directory for saved images (default: input file folder).")
    p.add_argument("--no-show", action="store_true", help="Do not pop up interactive windows (use with --save).")

    return p.parse_args()


def _npz_meta(data: np.lib.npyio.NpzFile, vol: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    origin = data.get("origin", np.array([0.0, 0.0, 0.0], dtype=np.float32)).astype(np.float32)
    voxel = float(np.array(data.get("voxel_size", 1.0)).reshape(-1)[0])
    dims = data.get("dims", np.array(vol.shape[::-1], dtype=np.int32)).astype(np.int32)
    return origin, voxel, dims


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "volume" not in data:
        raise ValueError(f"{path.name} missing key 'volume'. Keys: {list(data.keys())}")
    vol = data["volume"].astype(np.float32)

    # Some scripts store volume as (z,y,x); others might store (x,y,z).
    # Your existing V2 viewer assumed (z,y,x), so we keep that convention:
    # (nz, ny, nx)
    if vol.ndim != 3:
        raise ValueError(f"{path.name} volume must be 3D, got shape {vol.shape}")

    origin, voxel, dims = _npz_meta(data, vol)
    return vol, origin, voxel, dims


def sibling_npz(in_path: Path, target: str) -> Optional[Path]:
    """
    If input is *_gpr_mean.npz, *_gpr_std.npz, *_gpr_grad.npz,
    find siblings by swapping suffix.
    """
    name = in_path.name
    if name.endswith("_gpr_mean.npz") or name.endswith("_gpr_std.npz") or name.endswith("_gpr_grad.npz"):
        base = name.rsplit("_gpr_", 1)[0]
        cand = in_path.with_name(f"{base}_gpr_{target}.npz")
        return cand if cand.exists() else None
    return None


def compute_grad_mag(vol_zyx: np.ndarray, spacing: float) -> np.ndarray:
    # finite differences on (z,y,x)
    gz, gy, gx = np.gradient(vol_zyx, spacing, spacing, spacing, edge_order=1)
    return np.sqrt(gx * gx + gy * gy + gz * gz).astype(np.float32)


def pick_field(in_path: Path, mode: str) -> Tuple[np.ndarray, str, float, np.ndarray, float]:
    """
    Returns (field_volume_zyx, label, voxel, origin, spacing)
    """
    vol, origin, voxel, dims = load_npz(in_path)

    label = mode
    field = vol

    if mode in ("mean", "std", "grad"):
        sib = sibling_npz(in_path, mode)
        if sib is not None:
            field, origin, voxel, dims = load_npz(sib)
            label = mode
        else:
            # If asked for grad but no file, compute from current volume (assume it is mean)
            if mode == "grad":
                field = compute_grad_mag(vol, voxel)
                label = "grad(computed)"
            else:
                # mean/std requested but sibling missing: fall back to current
                label = f"{mode}(missing->using_input)"
                field = vol

    return field, label, voxel, origin, voxel


def threshold_indices(field_zyx: np.ndarray, percentile: float, abs_thresh: Optional[float]) -> Tuple[np.ndarray, np.ndarray, float]:
    if abs_thresh is not None:
        thresh = float(abs_thresh)
    else:
        thresh = float(np.percentile(field_zyx[np.isfinite(field_zyx)], percentile))

    mask = np.isfinite(field_zyx) & (field_zyx >= thresh)
    idx = np.argwhere(mask)  # (z,y,x)
    vals = field_zyx[mask]
    return idx, vals, thresh


def subsample(xs, ys, zs, vals, max_points: int, seed: int):
    n = vals.shape[0]
    if n <= max_points:
        return xs, ys, zs, vals
    rng = np.random.default_rng(seed)
    sel = rng.choice(n, size=max_points, replace=False)
    return xs[sel], ys[sel], zs[sel], vals[sel]


def save_or_show(fig, out_path: Optional[Path], no_show: bool):
    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Wrote: {out_path}")
    if not no_show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_npz).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else in_path.parent
    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)

    field, label, voxel, origin, spacing = pick_field(in_path, args.mode)

    nz, ny, nx = field.shape
    idx, vals, thresh = threshold_indices(field, args.percentile, args.abs_thresh)

    if idx.shape[0] == 0:
        print(f"No voxels above threshold {thresh:.6g}. Try lowering --percentile or --abs-thresh.")
        return

    # Convert indices (z,y,x) -> world coords
    zs = origin[2] + idx[:, 0] * voxel
    ys = origin[1] + idx[:, 1] * voxel
    xs = origin[0] + idx[:, 2] * voxel

    xs, ys, zs, vals = subsample(xs, ys, zs, vals, args.max_points, args.seed)

    # 3D scatter
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xs, ys, zs, c=vals, s=1)
    ax.set_title(f"{in_path.name}  [{label}]  (>= {thresh:.4g})")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label(label)
    plt.tight_layout()

    out_scatter = out_dir / f"viz3d_{label}_scatter.png" if args.save else None
    save_or_show(fig, out_scatter, args.no_show)

    # Slices
    if args.show_slices or args.compare_mean_vs_grad:
        z_i = args.slice_z if args.slice_z is not None else nz // 2
        y_i = args.slice_y if args.slice_y is not None else ny // 2
        x_i = args.slice_x if args.slice_x is not None else nx // 2

        def _imshow(figtitle: str, img: np.ndarray, xlabel: str, ylabel: str, fname: str):
            f = plt.figure(figsize=(7, 6))
            plt.imshow(img, origin="lower", aspect="auto")
            plt.title(figtitle)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.colorbar(label=label)
            plt.tight_layout()
            outp = out_dir / fname if args.save else None
            save_or_show(f, outp, args.no_show)

        if args.show_slices:
            _imshow(f"XY slice @ z={z_i} [{label}]", field[z_i, :, :], "x index", "y index",
                    f"viz3d_{label}_slice_XY_z{z_i}.png")
            _imshow(f"XZ slice @ y={y_i} [{label}]", field[:, y_i, :], "x index", "z index",
                    f"viz3d_{label}_slice_XZ_y{y_i}.png")
            _imshow(f"YZ slice @ x={x_i} [{label}]", field[:, :, x_i], "y index", "z index",
                    f"viz3d_{label}_slice_YZ_x{x_i}.png")

        if args.compare_mean_vs_grad:
            # Load mean + grad (or compute grad) for same input
            mean_vol, _, _, _ = load_npz(sibling_npz(in_path, "mean") or in_path)
            grad_path = sibling_npz(in_path, "grad")
            if grad_path is not None:
                grad_vol, _, _, _ = load_npz(grad_path)
            else:
                grad_vol = compute_grad_mag(mean_vol, voxel)

            # Compare on XY slice at z
            f = plt.figure(figsize=(12, 5))
            ax1 = f.add_subplot(1, 2, 1)
            im1 = ax1.imshow(mean_vol[z_i, :, :], origin="lower", aspect="auto")
            ax1.set_title(f"Mean (XY @ z={z_i})")
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            ax2 = f.add_subplot(1, 2, 2)
            im2 = ax2.imshow(grad_vol[z_i, :, :], origin="lower", aspect="auto")
            ax2.set_title(f"Grad |∇mean| (XY @ z={z_i})")
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            plt.tight_layout()
            outp = out_dir / f"viz3d_compare_mean_vs_grad_XY_z{z_i}.png" if args.save else None
            save_or_show(f, outp, args.no_show)


if __name__ == "__main__":
    main()
