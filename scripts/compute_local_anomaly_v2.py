#!/usr/bin/env python3
"""
compute_local_anomaly_v2.py

Pipeline-ready local anomaly computation.

Reads a CSV containing at least:
  - x, y
  - B_total   (or Bx, By, Bz to compute B_total)

Optionally respects quality flags from validate_and_diagnostics.py:
  - _flag_spike / _flag_any (will drop spikes by default)

Writes:
  - <input_stem>_anomaly.csv   (same folder by default)

Adds columns:
  - local_anomaly
  - local_anomaly_abs
  - local_anomaly_norm  (normalized by max absolute anomaly in the file)

Also optionally plots a quick scatter map.

Example:
  python3 compute_local_anomaly_v2.py --in data/processed/mag_data_clean.csv --radius 0.30 --plot
"""

from __future__ import annotations

import argparse
import sys
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_btotal_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "B_total" in df.columns:
        df["B_total"] = pd.to_numeric(df["B_total"], errors="coerce")
        return df
    if all(c in df.columns for c in ["Bx", "By", "Bz"]):
        bx = pd.to_numeric(df["Bx"], errors="coerce")
        by = pd.to_numeric(df["By"], errors="coerce")
        bz = pd.to_numeric(df["Bz"], errors="coerce")
        df["B_total"] = np.sqrt(bx.to_numpy() ** 2 + by.to_numpy() ** 2 + bz.to_numpy() ** 2)
        return df
    raise ValueError("Missing B_total and cannot compute it (need Bx, By, Bz).")


def local_anomaly(coords: np.ndarray, B: np.ndarray, radius: float) -> np.ndarray:
    """
    For each point i:
      baseline_i = mean(B[j] for j within 'radius' of i, excluding i)
      anomaly_i  = B[i] - baseline_i

    If a point has no neighbors within radius, anomaly is set to 0.0.
    """
    N = coords.shape[0]
    out = np.zeros(N, dtype=float)

    # O(N^2) simple approach (fine for small grids like 9x9, 20x20, etc.)
    for i in range(N):
        dx = coords[:, 0] - coords[i, 0]
        dy = coords[:, 1] - coords[i, 1]
        dist = np.sqrt(dx * dx + dy * dy)

        # neighbors within radius, excluding self (dist > 0)
        mask = (dist <= radius) & (dist > 0)

        if np.any(mask):
            baseline = float(np.mean(B[mask]))
            out[i] = float(B[i] - baseline)
        else:
            out[i] = 0.0

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute local magnetic anomaly and write *_anomaly.csv.")
    p.add_argument("--in", dest="infile", required=True, help="Input CSV path (e.g., data/processed/mag_data_clean.csv)")
    p.add_argument("--radius", type=float, default=0.30, help="Neighborhood radius in same units as x/y (default: 0.30)")
    p.add_argument("--out", default=None, help="Optional output CSV path. Default: <input_stem>_anomaly.csv next to input.")
    p.add_argument("--keep-spikes", action="store_true", help="If set, do not drop rows where _flag_spike is True.")
    p.add_argument("--drop-flag-any", action="store_true", help="If set, drop rows where _flag_any is True (stronger than spikes).")
    p.add_argument("--plot", action="store_true", help="If set, show a quick scatter plot colored by local_anomaly_norm.")
    p.add_argument("--no-show", action="store_true", help="If set with --plot, save plot PNG instead of displaying it.")
    p.add_argument("--plot-out", default=None, help="Output PNG path if using --plot --no-show. Default: <out_stem>_plot.png")
    return p.parse_args()


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

    # Basic schema checks
    for c in ["x", "y"]:
        if c not in df.columns:
            print(f"ERROR: missing required column '{c}'. Columns found: {list(df.columns)}", file=sys.stderr)
            return 2

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    try:
        df = compute_btotal_if_missing(df)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    # Drop rows missing core numeric values
    before = len(df)
    df = df.dropna(subset=["x", "y", "B_total"]).copy()
    if len(df) == 0:
        print("ERROR: no valid rows after dropping NaNs in x/y/B_total.", file=sys.stderr)
        return 2
    if len(df) != before:
        print(f"Note: dropped {before - len(df)} rows due to NaNs in x/y/B_total.")

    # Optional quality-flag filtering
    if args.drop_flag_any and "_flag_any" in df.columns:
        before = len(df)
        df = df.loc[~df["_flag_any"].astype(bool)].copy()
        print(f"Note: dropped {before - len(df)} rows where _flag_any == True.")
    elif (not args.keep_spikes) and "_flag_spike" in df.columns:
        before = len(df)
        df = df.loc[~df["_flag_spike"].astype(bool)].copy()
        print(f"Note: dropped {before - len(df)} rows where _flag_spike == True.")

    coords = df[["x", "y"]].to_numpy(dtype=float)
    B = df["B_total"].to_numpy(dtype=float)

    # Compute anomalies
    anomalies = local_anomaly(coords, B, radius=float(args.radius))
    df["local_anomaly"] = anomalies
    df["local_anomaly_abs"] = np.abs(anomalies)

    max_abs = float(np.max(np.abs(anomalies))) if len(anomalies) else 0.0
    if max_abs > 0:
        df["local_anomaly_norm"] = anomalies / max_abs
    else:
        df["local_anomaly_norm"] = 0.0

    # Write output
    if args.out is None:
        outpath = infile.with_name(infile.stem.replace("_clean", "") + "_anomaly.csv")
    else:
        outpath = Path(args.out)

    try:
        df.to_csv(outpath, index=False)
    except Exception as e:
        print(f"ERROR: could not write output CSV: {e}", file=sys.stderr)
        return 3

    print(f"✅ Wrote anomaly CSV: {outpath}")

    # Plot (optional)
    if args.plot:
        plt.figure()
        sc = plt.scatter(
            df["x"], df["y"],
            c=df["local_anomaly_norm"],
            s=90,
            edgecolor="k"
        )
        plt.colorbar(sc, label="Normalized local anomaly")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Local Anomaly (radius={args.radius})")
        plt.gca().set_aspect("equal", "box")
        plt.tight_layout()

        if args.no_show:
            if args.plot_out is None:
                plot_out = outpath.with_suffix("").with_name(outpath.stem + "_plot.png")
            else:
                plot_out = Path(args.plot_out)
            plot_out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_out, dpi=160)
            plt.close()
            print(f"✅ Wrote plot PNG: {plot_out}")
        else:
            plt.show()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        raise SystemExit(130)
