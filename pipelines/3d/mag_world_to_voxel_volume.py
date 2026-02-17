#!/usr/bin/env python3
"""
mag_world_to_voxel_volume.py

Build a 3D voxel volume from mag_world.csv (x, y, z, value).

Key safety features:
  - --max-dim (default 256): clamps each grid axis; if exceeded the voxel size
    is automatically increased and a message is printed.
  - --auto-scale (default on): if coordinate ranges exceed ~10 m, assumes
    millimetre units and scales by 0.001.  Writes a copy of the scaled input
    to processed/mag_world_m.csv.
  - Volume is always float32 (never float64 meshgrid).

Outputs:
  exports/volume.npz  — volume (float32), origin (3,), voxel_size (float),
                         value_min, value_max.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_paths import infer_run_dir_from_path  # noqa: E402

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from scipy.interpolate import griddata as _griddata
    HAS_GRIDDATA = True
except ImportError:
    HAS_GRIDDATA = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="mag_world.csv -> 3D voxel volume (volume.npz)"
    )
    p.add_argument("--in", dest="input_csv", required=True, help="Path to mag_world.csv")
    p.add_argument("--out", default="", help="Output volume.npz path. Default: auto $RUN_DIR/exports/volume.npz")
    p.add_argument("--voxel-size", type=float, default=0.02, help="Voxel edge length in metres (default: 0.02)")
    p.add_argument("--max-dim", type=int, default=256, help="Max voxels per axis (default: 256). Voxel size is enlarged if exceeded.")
    p.add_argument("--margin", type=float, default=0.1, help="Margin around data bounds in metres (default: 0.1)")
    p.add_argument(
        "--auto-scale", action=argparse.BooleanOptionalAction, default=True,
        help="Auto-detect mm units and scale to metres (default: on). Use --no-auto-scale to disable.",
    )
    p.add_argument("--method", choices=["idw", "scatter", "griddata"], default="idw",
                   help="Interpolation method (default: idw)")
    p.add_argument("--k", type=int, default=8, help="Neighbours for IDW (default: 8)")
    p.add_argument("--power", type=float, default=2.0, help="IDW power (default: 2.0)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def idw_interpolate(
    points: np.ndarray,
    values: np.ndarray,
    grid_xyz: np.ndarray,
    k: int = 8,
    power: float = 2.0,
    eps: float = 1e-12,
) -> np.ndarray:
    if not HAS_SCIPY:
        raise RuntimeError("IDW requires scipy.  Install with: pip install scipy")
    tree = cKDTree(points)
    dists, idx = tree.query(grid_xyz, k=min(k, len(points)), workers=-1)
    dists = np.maximum(dists, eps)
    w = 1.0 / (dists ** power)
    w /= w.sum(axis=1, keepdims=True)
    return (w * values[idx]).sum(axis=1)


def scatter_fill(
    points: np.ndarray,
    values: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Bin points into voxels (mean aggregation), then fill empties with nearest filled."""
    nx, ny, nz = shape
    accum = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.int32)

    ix = np.clip(((points[:, 0] - origin[0]) / voxel_size).astype(int), 0, nx - 1)
    iy = np.clip(((points[:, 1] - origin[1]) / voxel_size).astype(int), 0, ny - 1)
    iz = np.clip(((points[:, 2] - origin[2]) / voxel_size).astype(int), 0, nz - 1)
    np.add.at(accum, (ix, iy, iz), values)
    np.add.at(count, (ix, iy, iz), 1)

    filled = count > 0
    accum[filled] /= count[filled]

    # Fill empty voxels with nearest filled value using distance transform
    if not filled.all():
        try:
            from scipy.ndimage import distance_transform_edt
            _, nearest_idx = distance_transform_edt(~filled, return_distances=True, return_indices=True)
            accum = accum[tuple(nearest_idx)]
        except ImportError:
            # Fallback: fill with global mean
            accum[~filled] = np.mean(values)

    return accum.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    inp = Path(args.input_csv).expanduser().resolve()
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

    # --- Auto-scale detection ---
    ranges = [x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]
    max_range = max(ranges)
    print(f"Input {len(df)} points.  Coordinate ranges: "
          f"x [{x.min():.4f}, {x.max():.4f}]  "
          f"y [{y.min():.4f}, {y.max():.4f}]  "
          f"z [{z.min():.4f}, {z.max():.4f}]  "
          f"(max span: {max_range:.4f})")

    scaled = False
    if args.auto_scale and max_range > 10.0:
        print(f"  AUTO-SCALE: range {max_range:.1f} > 10 m — assuming mm; multiplying by 0.001.")
        x = x * 0.001
        y = y * 0.001
        z = z * 0.001
        ranges = [r * 0.001 for r in ranges]
        max_range *= 0.001
        scaled = True

        # Save scaled copy
        run = infer_run_dir_from_path(inp)
        if run:
            scaled_path = run / "processed" / "mag_world_m.csv"
        else:
            scaled_path = inp.parent / "mag_world_m.csv"
        scaled_path.parent.mkdir(parents=True, exist_ok=True)
        df_scaled = df.copy()
        df_scaled["x"] = x
        df_scaled["y"] = y
        df_scaled["z"] = z
        df_scaled.to_csv(scaled_path, index=False)
        print(f"  Wrote scaled copy: {scaled_path}")
        print(f"  Scaled ranges: "
              f"x [{x.min():.4f}, {x.max():.4f}]  "
              f"y [{y.min():.4f}, {y.max():.4f}]  "
              f"z [{z.min():.4f}, {z.max():.4f}]")

    points = np.column_stack([x, y, z])

    # --- Bounds + margin ---
    margin = float(args.margin)
    voxel_size = float(args.voxel_size)
    if voxel_size <= 0:
        print("ERROR: --voxel-size must be > 0", file=sys.stderr)
        return 2

    xmin, xmax = x.min() - margin, x.max() + margin
    ymin, ymax = y.min() - margin, y.max() + margin
    zmin, zmax = z.min() - margin, z.max() + margin
    origin = np.array([xmin, ymin, zmin], dtype=np.float32)

    # --- Compute dims + clamp ---
    nx = max(1, int(np.ceil((xmax - xmin) / voxel_size)) + 1)
    ny = max(1, int(np.ceil((ymax - ymin) / voxel_size)) + 1)
    nz = max(1, int(np.ceil((zmax - zmin) / voxel_size)) + 1)

    max_dim = int(args.max_dim)
    adjusted = False
    while max(nx, ny, nz) > max_dim:
        old_vs = voxel_size
        voxel_size *= 1.25
        nx = max(1, int(np.ceil((xmax - xmin) / voxel_size)) + 1)
        ny = max(1, int(np.ceil((ymax - ymin) / voxel_size)) + 1)
        nz = max(1, int(np.ceil((zmax - zmin) / voxel_size)) + 1)
        adjusted = True

    if adjusted:
        print(f"  CLAMPED: voxel_size enlarged to {voxel_size:.6f} m to keep dims <= {max_dim}")

    total_voxels = nx * ny * nz
    print(f"Grid: {nx} x {ny} x {nz} = {total_voxels:,} voxels  (voxel_size={voxel_size:.6f} m)")

    # --- Build volume ---
    if args.method == "scatter":
        vol = scatter_fill(points, v, origin, voxel_size, (nx, ny, nz))
    elif args.method == "idw":
        # Build grid coordinates in float32 to save memory
        xi = np.linspace(xmin, xmin + (nx - 1) * voxel_size, nx, dtype=np.float32)
        yi = np.linspace(ymin, ymin + (ny - 1) * voxel_size, ny, dtype=np.float32)
        zi = np.linspace(zmin, zmin + (nz - 1) * voxel_size, nz, dtype=np.float32)
        gx, gy, gz = np.meshgrid(xi, yi, zi, indexing="ij")
        grid_xyz = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        del gx, gy, gz  # free immediately

        vol_flat = idw_interpolate(points, v, grid_xyz, k=min(args.k, len(points)), power=args.power)
        vol = vol_flat.reshape((nx, ny, nz)).astype(np.float32)
    elif args.method == "griddata":
        if not HAS_GRIDDATA:
            print("ERROR: --method griddata requires scipy", file=sys.stderr)
            return 1
        xi = np.linspace(xmin, xmin + (nx - 1) * voxel_size, nx, dtype=np.float32)
        yi = np.linspace(ymin, ymin + (ny - 1) * voxel_size, ny, dtype=np.float32)
        zi = np.linspace(zmin, zmin + (nz - 1) * voxel_size, nz, dtype=np.float32)
        gx, gy, gz = np.meshgrid(xi, yi, zi, indexing="ij")
        grid_xyz = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        del gx, gy, gz
        vol_flat = _griddata(points, v, grid_xyz, method="linear", fill_value=np.nanmean(v))
        vol = vol_flat.reshape((nx, ny, nz)).astype(np.float32)
    else:
        print(f"ERROR: unknown method '{args.method}'", file=sys.stderr)
        return 2

    # --- Output path ---
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        run = infer_run_dir_from_path(inp)
        if run:
            out_path = run / "exports" / "volume.npz"
        else:
            out_path = inp.parent / "volume.npz"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        volume=vol,
        origin=origin,
        voxel_size=np.float32(voxel_size),
        value_min=np.float32(vol.min()),
        value_max=np.float32(vol.max()),
    )
    print(f"Wrote volume     : {out_path}")
    print(f"  shape          : {vol.shape}")
    print(f"  origin         : {origin.tolist()}")
    print(f"  voxel_size     : {voxel_size:.6f} m")
    print(f"  value range    : [{vol.min():.4f}, {vol.max():.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
