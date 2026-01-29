#!/usr/bin/env python3
"""
interpolate_to_heatmapV2.py

Like V1, but also outputs a SECOND heatmap that uses a FIXED color scale across runs.

Outputs (in --out-dir or default):
  - <stem>_grid.csv
  - <stem>_heatmap.png            (auto-scaled using --clip-percentile)
  - <stem>_heatmap_fixed.png      (fixed scale loaded/saved from JSON)
  - heatmap_scale_<valuecol>.json (persisted vmin/vmax)

Why fixed scale matters:
- Auto-scaled plots can look "similar" even when magnitudes differ.
- Fixed-scale plots let you compare runs quantitatively.

Example:
  python3 scripts/interpolate_to_heatmapV2.py --in data/processed/mag_data_anomaly.csv --value-col local_anomaly

If you want to manually force the fixed scale:
  python3 scripts/interpolate_to_heatmapV2.py --in ... --value-col local_anomaly --fixed-vmin -0.05 --fixed-vmax 0.05

To overwrite the saved fixed scale (re-initialize):
  python3 scripts/interpolate_to_heatmapV2.py --in ... --value-col local_anomaly --fixed-overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interpolate scattered x,y,value points to a grid and save grid CSV + heatmaps."
    )
    p.add_argument("--in", dest="infile", required=True, help="Input CSV")
    p.add_argument("--value-col", default="local_anomaly", help="Column to grid (default: local_anomaly)")
    p.add_argument("--out-dir", dest="outdir", default=None, help="Output directory (default: data/exports)")
    p.add_argument(
        "--grid-step",
        type=float,
        default=None,
        help="Grid spacing in same units as x/y (e.g., 0.05). If omitted, uses --grid-n.",
    )
    p.add_argument("--grid-n", type=int, default=200, help="Grid resolution per axis if --grid-step not given")
    p.add_argument("--power", type=float, default=2.0, help="IDW power (default: 2.0)")
    p.add_argument("--eps", type=float, default=1e-12, help="Small epsilon to avoid divide-by-zero")
    p.add_argument(
        "--clip-percentile",
        type=float,
        default=99.0,
        help="Clip color scale for AUTO heatmap (default: 99). Use 100 to disable clipping.",
    )
    p.add_argument(
        "--drop-flag-any",
        action="store_true",
        help="If set, drop rows where _flag_any is True (if present).",
    )

    # --- NEW (V2): fixed-scale heatmap controls ---
    p.add_argument(
        "--no-fixed-heatmap",
        action="store_true",
        help="Disable writing the fixed-scale heatmap.",
    )
    p.add_argument(
        "--fixed-vmin",
        type=float,
        default=None,
        help="Manually set fixed heatmap vmin (must also set --fixed-vmax).",
    )
    p.add_argument(
        "--fixed-vmax",
        type=float,
        default=None,
        help="Manually set fixed heatmap vmax (must also set --fixed-vmin).",
    )
    p.add_argument(
        "--fixed-scale-file",
        default=None,
        help="Path to JSON file storing fixed vmin/vmax. Default: <outdir>/heatmap_scale_<valuecol>.json",
    )
    p.add_argument(
        "--fixed-overwrite",
        action="store_true",
        help="Overwrite the fixed scale JSON using this run's derived scale.",
    )

    return p.parse_args()


def make_grid_axes(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    grid_step: Optional[float],
    grid_n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if grid_step is not None and grid_step > 0:
        gx = np.arange(xmin, xmax + grid_step * 0.5, grid_step)
        gy = np.arange(ymin, ymax + grid_step * 0.5, grid_step)
    else:
        gx = np.linspace(xmin, xmax, grid_n)
        gy = np.linspace(ymin, ymax, grid_n)
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
    """
    Inverse Distance Weighting interpolation.
    Returns grid Z with shape (len(gy), len(gx)) where rows correspond to gy (y-axis).
    """
    Xg, Yg = np.meshgrid(gx, gy)  # (ny, nx)
    tx = Xg.ravel()
    ty = Yg.ravel()
    Z = np.empty_like(tx, dtype=float)

    for i in range(tx.size):
        dx = tx[i] - x
        dy = ty[i] - y
        d2 = dx * dx + dy * dy

        j0 = np.argmin(d2)
        if d2[j0] <= eps:
            Z[i] = v[j0]
            continue

        w = 1.0 / (d2 ** (power / 2.0) + eps)
        Z[i] = float(np.sum(w * v) / np.sum(w))

    return Z.reshape(Yg.shape)


def compute_clipped_range(Z: np.ndarray, clip_percentile: float) -> Tuple[Optional[float], Optional[float]]:
    clip = float(clip_percentile)
    if 0 < clip < 100:
        lo = float(np.nanpercentile(Z, 100 - clip))
        hi = float(np.nanpercentile(Z, clip))
        return lo, hi
    return None, None


def resolve_outdir(infile: Path, outdir_arg: Optional[str]) -> Path:
    if outdir_arg:
        return Path(outdir_arg)

    # Default: if input is in data/processed/, output to data/exports/
    if infile.parent.name == "processed" and infile.parent.parent.name == "data":
        return infile.parent.parent / "exports"

    return infile.parent


def load_or_init_fixed_scale(
    *,
    scale_path: Path,
    Z: np.ndarray,
    clip_percentile: float,
    fixed_vmin: Optional[float],
    fixed_vmax: Optional[float],
    overwrite: bool,
) -> Tuple[float, float]:
    # Manual overrides must come as a pair
    if (fixed_vmin is None) ^ (fixed_vmax is None):
        raise ValueError("If you set --fixed-vmin you must also set --fixed-vmax (and vice versa).")

    # If user provided explicit values, optionally write them to the file (so later runs match)
    if fixed_vmin is not None and fixed_vmax is not None:
        vmin, vmax = float(fixed_vmin), float(fixed_vmax)
        if overwrite or not scale_path.exists():
            scale_path.write_text(json.dumps({"vmin": vmin, "vmax": vmax}, indent=2) + "\n")
        return vmin, vmax

    # Otherwise: use persisted file if it exists (unless overwrite)
    if scale_path.exists() and not overwrite:
        try:
            data = json.loads(scale_path.read_text())
            return float(data["vmin"]), float(data["vmax"])
        except Exception:
            # fall through to re-init if file is corrupt
            pass

    # Initialize from this run (using the same percentile clipping logic)
    vmin, vmax = compute_clipped_range(Z, clip_percentile)
    if vmin is None or vmax is None:
        # If clip disabled, fall back to full range (still persistent)
        vmin = float(np.nanmin(Z))
        vmax = float(np.nanmax(Z))

    scale_path.write_text(json.dumps({"vmin": float(vmin), "vmax": float(vmax)}, indent=2) + "\n")
    return float(vmin), float(vmax)


def plot_heatmap(
    *,
    Z: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    value_col: str,
    out_png: Path,
    title_suffix: str,
    vmin: Optional[float],
    vmax: Optional[float],
) -> None:
    plt.figure()
    im = plt.imshow(
        Z,
        origin="lower",
        extent=[gx.min(), gx.max(), gy.min(), gy.max()],
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(im, label=value_col)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Heatmap (IDW) of {value_col}{title_suffix}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def main() -> int:
    args = parse_args()
    infile = Path(args.infile)

    if not infile.exists():
        print(f"ERROR: input file not found: {infile}", file=sys.stderr)
        return 2

    try:
        df = pd.read_csv(infile)
    except Exception as e:
        print(f"ERROR: could not read CSV: {e}", file=sys.stderr)
        return 2

    for c in ["x", "y", args.value_col]:
        if c not in df.columns:
            print(f"ERROR: missing required column '{c}'. Columns found: {list(df.columns)}", file=sys.stderr)
            return 2

    if args.drop_flag_any and "_flag_any" in df.columns:
        before = len(df)
        df = df.loc[~df["_flag_any"].astype(bool)].copy()
        print(f"Note: dropped {before - len(df)} rows where _flag_any == True.")

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df[args.value_col] = pd.to_numeric(df[args.value_col], errors="coerce")
    df = df.dropna(subset=["x", "y", args.value_col]).copy()
    if len(df) == 0:
        print("ERROR: no valid rows after dropping NaNs.", file=sys.stderr)
        return 2

    x = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)
    v = df[args.value_col].to_numpy(float)

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))

    outdir = resolve_outdir(infile, args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stem = infile.stem
    grid_csv = outdir / f"{stem}_grid.csv"
    heatmap_png = outdir / f"{stem}_heatmap.png"
    heatmap_fixed_png = outdir / f"{stem}_heatmap_fixed.png"

    gx, gy = make_grid_axes(xmin, xmax, ymin, ymax, args.grid_step, args.grid_n)

    print(f"Interpolating '{args.value_col}' onto grid: nx={len(gx)}, ny={len(gy)} (IDW power={args.power}) ...")
    Z = idw_grid(x, y, v, gx, gy, power=args.power, eps=args.eps)

    # Export long-form grid CSV
    Xg, Yg = np.meshgrid(gx, gy)
    out_df = pd.DataFrame(
        {
            "x": Xg.ravel(),
            "y": Yg.ravel(),
            args.value_col: Z.ravel(),
        }
    )
    try:
        out_df.to_csv(grid_csv, index=False)
    except Exception as e:
        print(f"ERROR: could not write grid CSV: {e}", file=sys.stderr)
        return 3

    # --- AUTO heatmap (same as V1 behavior) ---
    vmin_auto, vmax_auto = compute_clipped_range(Z, args.clip_percentile)
    try:
        plot_heatmap(
            Z=Z,
            gx=gx,
            gy=gy,
            value_col=args.value_col,
            out_png=heatmap_png,
            title_suffix="",
            vmin=vmin_auto,
            vmax=vmax_auto,
        )
    except Exception as e:
        print(f"ERROR: could not save heatmap PNG: {e}", file=sys.stderr)
        return 3

    # --- FIXED heatmap (persisted across runs) ---
    if not args.no_fixed_heatmap:
        if args.fixed_scale_file:
            scale_path = Path(args.fixed_scale_file)
        else:
            safe_col = args.value_col.replace("/", "_")
            scale_path = outdir / f"heatmap_scale_{safe_col}.json"

        try:
            vmin_fix, vmax_fix = load_or_init_fixed_scale(
                scale_path=scale_path,
                Z=Z,
                clip_percentile=args.clip_percentile,
                fixed_vmin=args.fixed_vmin,
                fixed_vmax=args.fixed_vmax,
                overwrite=args.fixed_overwrite,
            )
        except Exception as e:
            print(f"ERROR: fixed scale setup failed: {e}", file=sys.stderr)
            return 3

        try:
            plot_heatmap(
                Z=Z,
                gx=gx,
                gy=gy,
                value_col=args.value_col,
                out_png=heatmap_fixed_png,
                title_suffix=" (fixed scale)",
                vmin=vmin_fix,
                vmax=vmax_fix,
            )
        except Exception as e:
            print(f"ERROR: could not save fixed heatmap PNG: {e}", file=sys.stderr)
            return 3

        print(f"Fixed scale file: {scale_path} (vmin={vmin_fix}, vmax={vmax_fix})")
        print(f"Wrote fixed heatmap: {heatmap_fixed_png}")

    print(f"Wrote grid CSV: {grid_csv}")
    print(f"Wrote heatmap:  {heatmap_png}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        raise SystemExit(130)
