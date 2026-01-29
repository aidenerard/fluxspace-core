#!/usr/bin/env python3
"""
visualize_3d_heatmapV2_gpr.py

Quick viewer for voxel volumes saved by mag_world_to_voxel_volumeV2_gpr.py.

It displays:
  - a 3D scatter of voxels above a threshold (default: percentile),
  - plus optional orthogonal slices.

This is intentionally lightweight and matches the feel of your existing visualize_3d_heatmap.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize a voxel volume (*.npz) as a thresholded point cloud and slices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in", dest="in_npz", required=True, help="Input .npz containing 'volume' and metadata.")
    p.add_argument("--percentile", type=float, default=99.0, help="Show voxels >= this percentile of value.")
    p.add_argument("--abs-thresh", type=float, default=None, help="Absolute threshold (overrides percentile).")
    p.add_argument("--max-points", type=int, default=300000, help="Cap points plotted (random subsample).")
    p.add_argument("--seed", type=int, default=7, help="Seed for subsampling.")
    p.add_argument("--show-slices", action="store_true", help="Show XY/XZ/YZ slices in separate figures.")
    p.add_argument("--slice-z", type=int, default=None, help="Z index for XY slice (default: middle).")
    p.add_argument("--slice-y", type=int, default=None, help="Y index for XZ slice (default: middle).")
    p.add_argument("--slice-x", type=int, default=None, help="X index for YZ slice (default: middle).")
    return p.parse_args()


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    vol = data["volume"].astype(np.float32)  # (z,y,x)
    origin = data.get("origin", np.array([0.0, 0.0, 0.0], dtype=np.float32)).astype(np.float32)
    voxel = float(data.get("voxel_size", 1.0))
    dims = data.get("dims", np.array(vol.shape[::-1], dtype=np.int32)).astype(np.int32)
    return vol, origin, voxel, dims


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_npz).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    vol, origin, voxel, dims = load_npz(in_path)
    nz, ny, nx = vol.shape

    # Threshold selection
    if args.abs_thresh is not None:
        thresh = float(args.abs_thresh)
    else:
        thresh = float(np.percentile(vol, args.percentile))

    mask = vol >= thresh
    idx = np.argwhere(mask)  # rows of (z,y,x)
    vals = vol[mask]

    if idx.shape[0] == 0:
        print(f"No voxels above threshold {thresh}. Try lowering --percentile.")
        return

    # Convert to world coords
    zs = origin[2] + idx[:, 0] * voxel
    ys = origin[1] + idx[:, 1] * voxel
    xs = origin[0] + idx[:, 2] * voxel

    # Subsample for plotting
    n = idx.shape[0]
    if n > args.max_points:
        rng = np.random.default_rng(args.seed)
        sel = rng.choice(n, size=args.max_points, replace=False)
        xs, ys, zs, vals = xs[sel], ys[sel], zs[sel], vals[sel]

    # 3D scatter
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xs, ys, zs, c=vals, s=1)
    ax.set_title(f"{in_path.name}  (>= {thresh:.4g})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.colorbar(sc, ax=ax, shrink=0.6, label="value")
    plt.tight_layout()

    if args.show_slices:
        z_i = args.slice_z if args.slice_z is not None else nz // 2
        y_i = args.slice_y if args.slice_y is not None else ny // 2
        x_i = args.slice_x if args.slice_x is not None else nx // 2

        # XY at z
        fig1 = plt.figure(figsize=(7, 6))
        plt.imshow(vol[z_i, :, :], origin="lower", aspect="auto")
        plt.title(f"XY slice at z={z_i}")
        plt.xlabel("x index")
        plt.ylabel("y index")
        plt.colorbar(label="value")
        plt.tight_layout()

        # XZ at y
        fig2 = plt.figure(figsize=(7, 6))
        plt.imshow(vol[:, y_i, :], origin="lower", aspect="auto")
        plt.title(f"XZ slice at y={y_i}")
        plt.xlabel("x index")
        plt.ylabel("z index")
        plt.colorbar(label="value")
        plt.tight_layout()

        # YZ at x
        fig3 = plt.figure(figsize=(7, 6))
        plt.imshow(vol[:, :, x_i], origin="lower", aspect="auto")
        plt.title(f"YZ slice at x={x_i}")
        plt.xlabel("y index")
        plt.ylabel("z index")
        plt.colorbar(label="value")
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
