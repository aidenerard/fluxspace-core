#!/usr/bin/env python3
"""
visualize_3d_heatmap.py

Viewer for voxel volume (volume.npz) from mag_world_to_voxel_volume or mag_world_to_voxel_volumeV2_gpr.

- PyVista (optional): volume rendering, --screenshot to save PNG. Use clim for color range (no scalar_range).
- Matplotlib: 3D scatter + orthogonal slices; --show-slices --save --no-show for headless slice PNGs.

CLI: --in or --volume (alias). --mode value | mean | std | grad. If pyvista missing, only scatter/slices.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional, Any

import numpy as np
import matplotlib.pyplot as plt

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError as e:
    HAS_PYVISTA = False
    _PYVISTA_ERROR = e


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize voxel volume (.npz): scatter, slices, optional PyVista volume + screenshot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in", dest="in_npz", default=None, help="Input .npz (e.g. volume.npz).")
    p.add_argument("--volume", dest="volume_npz", default=None, help="Alias for --in (backward compatible).")
    p.add_argument(
        "--mode",
        choices=["value", "mean", "std", "grad"],
        default="value",
        help="Field to visualize: value (main volume), mean (=value), std/grad if present in npz.",
    )
    p.add_argument("--percentile", type=float, default=99.0, help="Show voxels >= this percentile.")
    p.add_argument("--abs-thresh", type=float, default=None, help="Absolute threshold (overrides percentile).")
    p.add_argument("--max-points", type=int, default=300000, help="Cap points in scatter (subsample).")
    p.add_argument("--seed", type=int, default=7, help="Seed for subsampling.")

    p.add_argument("--show-slices", action="store_true", help="Show XY/XZ/YZ slices (Matplotlib).")
    p.add_argument("--compare-mean-vs-grad", action="store_true", help="Also show mean-vs-grad slice comparison.")
    p.add_argument("--slice-z", type=int, default=None, help="Z index for XY slice.")
    p.add_argument("--slice-y", type=int, default=None, help="Y index for XZ slice.")
    p.add_argument("--slice-x", type=int, default=None, help="X index for YZ slice.")

    p.add_argument("--save", action="store_true", help="Save slice/screenshot PNGs to --out-dir.")
    p.add_argument("--out-dir", default=None, help="Output directory for PNGs (default: input file folder).")
    p.add_argument("--no-show", action="store_true", help="Do not open interactive windows (use with --save/--screenshot).")
    p.add_argument("--screenshot", action="store_true", help="Save 3D volume screenshot (requires pyvista).")
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
    if vol.ndim != 3:
        raise ValueError(f"{path.name} volume must be 3D, got shape {vol.shape}")

    # Normalize to (nz, ny, nx): mag_world_to_voxel_volume uses (nx, ny, nz) + axes_x/y/z; V2_gpr uses (nz, ny, nx) + dims.
    if "axes_x" in data or "nx" in data:
        nx = int(np.array(data.get("nx", vol.shape[0])).reshape(-1)[0])
        if vol.shape[0] == nx:
            vol = np.transpose(vol, (2, 1, 0)).copy()
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
    Returns (field_volume_zyx, label, voxel, origin, spacing).
    Uses 'volume' for value/mean; 'std'/'grad' from same npz if present, else sibling files or compute grad.
    """
    data = np.load(in_path, allow_pickle=True)
    if "volume" not in data:
        raise ValueError(f"{in_path.name} missing key 'volume'. Keys: {list(data.keys())}")
    vol = data["volume"].astype(np.float32)
    origin, voxel, dims = _npz_meta(data, vol)
    label = mode
    field = vol

    if mode in ("mean", "std", "grad"):
        if mode == "mean":
            field = vol
            label = "value"
        elif mode == "std" and "std" in data:
            field = data["std"].astype(np.float32)
            label = "std"
        elif mode == "grad" and "grad" in data:
            field = data["grad"].astype(np.float32)
            label = "grad"
        else:
            sib = sibling_npz(in_path, mode)
            if sib is not None:
                field, origin, voxel, dims = load_npz(sib)
                label = mode
            elif mode == "grad":
                field = compute_grad_mag(vol, voxel)
                label = "grad(computed)"
            else:
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


def _get_in_path(args: argparse.Namespace) -> Path:
    in_path = args.in_npz or args.volume_npz
    if not in_path:
        print("ERROR: Provide --in or --volume with path to volume.npz.", file=sys.stderr)
        print("  Example: --in \"$RUN_DIR/exports/volume.npz\"", file=sys.stderr)
        sys.exit(2)
    return Path(in_path).expanduser().resolve()


def _build_pyvista_grid(vol_zyx: np.ndarray, origin: np.ndarray, voxel: float) -> Any:
    """Build PyVista ImageData from volume (nz, ny, nx). PyVista expects (nx, ny, nz) point data."""
    if not HAS_PYVISTA:
        raise RuntimeError(
            "PyVista is required for --screenshot / volume rendering.\n"
            f"Import error: {_PYVISTA_ERROR}\n\n"
            "Install with: pip install pyvista"
        )
    nz, ny, nx = vol_zyx.shape
    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)
    grid.origin = (float(origin[0]), float(origin[1]), float(origin[2]))
    grid.spacing = (voxel, voxel, voxel)
    vals = np.transpose(vol_zyx, (2, 1, 0)).astype(np.float32).copy()
    grid.point_data["value"] = vals.ravel(order="F")
    return grid


def main() -> None:
    args = parse_args()
    in_path = _get_in_path(args)
    if not in_path.exists():
        print(f"ERROR: File not found: {in_path}", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else in_path.parent
    if args.save or args.screenshot:
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

    # PyVista screenshot (volume rendering)
    if args.screenshot:
        try:
            grid = _build_pyvista_grid(field, origin, voxel)
            vmin = float(np.nanmin(field))
            vmax = float(np.nanmax(field))
            if vmax <= vmin:
                vmax = vmin + 1.0
            if HAS_PYVISTA:
                off_screen = args.no_show
                if off_screen:
                    pv.OFF_SCREEN = True
                pl = pv.Plotter(off_screen=off_screen)
                pl.add_volume(grid, scalars="value", clim=(vmin, vmax), opacity="linear", cmap="coolwarm")
                pl.camera_position = "yz"
                pl.reset_camera()
                shot_path = out_dir / "heatmap_3d_screenshot.png"
                pl.screenshot(str(shot_path))
                pl.close()
                print(f"Wrote: {shot_path}")
            else:
                raise RuntimeError("PyVista is required for --screenshot. Install: pip install pyvista")
        except Exception as e:
            print(f"ERROR: Screenshot failed: {e}", file=sys.stderr)
            if not HAS_PYVISTA:
                print("Install PyVista: pip install pyvista", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
