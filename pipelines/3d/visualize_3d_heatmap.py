#!/usr/bin/env python3
"""
visualize_3d_heatmap.py

Visualize 3D voxel volume (volume.npz) with PyVista: slices, isosurfaces, volume rendering.
Optional: overlay mesh (PLY/OBJ/GLB). Saves screenshot PNG and optional HTML to run folder.

Requires: pyvista. Optional: open3d for PLY/OBJ loading.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import pyvista as pv
except ImportError:
    print("ERROR: pyvista is required. Install with: pip install pyvista", file=sys.stderr)
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize 3D magnetic volume (slices, isosurface, screenshot)"
    )
    p.add_argument("--volume", required=True, help="Path to volume.npz")
    p.add_argument("--out-dir", default="", help="Output directory for screenshot/HTML. Default: volume's parent")
    p.add_argument("--mesh", default="", help="Optional mesh overlay (PLY/OBJ/GLB)")
    p.add_argument("--screenshot", action="store_true", help="Save heatmap_3d_screenshot.png")
    p.add_argument("--no-show", action="store_true", help="Do not open interactive window (use with --screenshot)")
    p.add_argument("--slice-axis", choices=["x", "y", "z"], default="z", help="Primary slice axis for initial view")
    p.add_argument("--isosurface-value", type=float, default=None, help="Isosurface threshold (default: median of volume)")
    return p.parse_args()


def load_volume(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Load volume.npz; return (volume_3d, origin, voxel_size)."""
    data = np.load(path, allow_pickle=True)
    vol = data["volume"]
    origin = data["origin"]
    if origin.shape == ():
        origin = np.array([float(origin)] * 3)
    vs = float(data["voxel_size"])
    return vol, origin, vs


def build_pv_grid(volume: np.ndarray, origin: np.ndarray, voxel_size: float) -> "pv.ImageData":
    """Build PyVista ImageData from numpy volume. PyVista uses (x,y,z) = (dim0,dim1,dim2)."""
    # volume is (nx, ny, nz); origin is (xmin, ymin, zmin)
    spacing = (voxel_size, voxel_size, voxel_size)
    grid = pv.ImageData()
    grid.dimensions = volume.shape
    grid.spacing = spacing
    grid.origin = origin
    grid.point_data["value"] = volume.ravel(order="F")  # Fortran order for PyVista
    return grid


def main() -> int:
    args = parse_args()
    vol_path = Path(args.volume)
    if not vol_path.exists():
        print(f"ERROR: Volume not found: {vol_path}", file=sys.stderr)
        return 2

    volume, origin, voxel_size = load_volume(vol_path)
    grid = build_pv_grid(volume, origin, voxel_size)

    out_dir = Path(args.out_dir) if args.out_dir else vol_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    pl = pv.Plotter(off_screen=args.no_show or args.screenshot)
    pl.set_background("white")

    # Volume rendering (opacity by value)
    vol_min, vol_max = float(np.nanmin(volume)), float(np.nanmax(volume))
    if vol_max > vol_min:
        pl.add_volume(
            grid,
            scalars="value",
            opacity="linear",
            cmap="coolwarm",
            scalar_range=(vol_min, vol_max),
        )
    else:
        pl.add_mesh(grid, scalars="value", cmap="coolwarm", show_scalar_bar=True)

    # Orthogonal slice (center)
    nx, ny, nz = volume.shape
    slice_pos = (nx // 2, ny // 2, nz // 2)
    if args.slice_axis == "x":
        sl = grid.slice(normal="x", origin=grid.center)
    elif args.slice_axis == "y":
        sl = grid.slice(normal="y", origin=grid.center)
    else:
        sl = grid.slice(normal="z", origin=grid.center)
    pl.add_mesh(sl, scalars="value", cmap="coolwarm", opacity=0.7, show_scalar_bar=False)

    # Isosurface at median or user value
    iso_val = args.isosurface_value
    if iso_val is None:
        iso_val = float(np.nanmedian(volume))
    try:
        iso = grid.contour(isosurfaces=1, rng=[iso_val, iso_val], scalars="value")
        pl.add_mesh(iso, color="orange", opacity=0.5)
    except Exception:
        pass

    # Optional mesh overlay
    if args.mesh:
        mesh_path = Path(args.mesh)
        if mesh_path.exists():
            try:
                mesh = pv.read(str(mesh_path))
                pl.add_mesh(mesh, color="lightgray", opacity=0.3)
            except Exception as e:
                print(f"WARNING: Could not load mesh {mesh_path}: {e}", file=sys.stderr)

    pl.camera_position = "yz"
    pl.reset_camera()

    if args.screenshot:
        out_png = out_dir / "heatmap_3d_screenshot.png"
        pl.screenshot(str(out_png))
        print(f"Saved screenshot: {out_png}")

    if not args.no_show and not args.screenshot:
        pl.show()

    pl.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
