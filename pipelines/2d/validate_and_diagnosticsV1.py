#!/usr/bin/env python3
"""
validate_and_diagnostics.py

Reads a magnetometer CSV (like the one produced by mag_to_csv.py), validates it,
cleans obvious issues, and generates quick diagnostics (plots + a text report).

Typical usage:
  python3 scripts/validate_and_diagnostics.py --in data/raw/mag_data.csv

Outputs (by default) in data/processed/:
  - <stem>_clean.csv
  - <stem>_report.txt
  - <stem>_Btotal_vs_time.png
  - <stem>_Btotal_hist.png
  - <stem>_scatter_xy_colored.png
  - <stem>_spike_deltas.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _find_time_column(cols: List[str]) -> Optional[str]:
    # Common names you might use
    candidates = ["time", "timestamp", "t", "datetime", "date_time"]
    for c in candidates:
        if c in cols:
            return c
    return None

def _coerce_time_series(s: pd.Series) -> Tuple[Optional[pd.Series], str]:
    """
    Return (time_series, note).
    Accepts:
      - ISO timestamps (strings)
      - unix seconds (float/int)
    """
    if s is None:
        return None, "No time column"
    # Try numeric unix time first
    if pd.api.types.is_numeric_dtype(s):
        # Treat as seconds since epoch
        t = pd.to_datetime(s, unit="s", errors="coerce", utc=True)
        ok = t.notna().mean()
        if ok > 0.8:
            return t, "Parsed numeric unix seconds as UTC"
    # Try general datetime parsing
    t = pd.to_datetime(s, errors="coerce", utc=True)
    ok = t.notna().mean()
    if ok > 0.8:
        return t, "Parsed as datetime (UTC)"
    return None, "Could not parse time column reliably"

def _compute_btotal_if_missing(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    df = df.copy()
    if "B_total" in df.columns:
        return df, "B_total present"
    # Compute from vector if possible
    if all(c in df.columns for c in ["Bx", "By", "Bz"]):
        bx = pd.to_numeric(df["Bx"], errors="coerce")
        by = pd.to_numeric(df["By"], errors="coerce")
        bz = pd.to_numeric(df["Bz"], errors="coerce")
        df["Bx"], df["By"], df["Bz"] = bx, by, bz
        df["B_total"] = np.sqrt(bx.to_numpy()**2 + by.to_numpy()**2 + bz.to_numpy()**2)
        return df, "Computed B_total = sqrt(Bx^2 + By^2 + Bz^2)"
    return df, "Missing B_total and cannot compute (need Bx,By,Bz)"

def _robust_z(x: np.ndarray) -> np.ndarray:
    """Robust z-score using MAD. Returns zeros if degenerate."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        return np.zeros_like(x)
    return 0.6745 * (x - med) / mad

def _save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# -----------------------------
# Main pipeline
# -----------------------------

def validate_and_clean(
    infile: Path,
    outdir: Path,
    drop_outliers: bool,
    z_thresh: float,
    delta_thresh: Optional[float],
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, float]]:
    notes: Dict[str, str] = {}
    stats: Dict[str, float] = {}

    df = pd.read_csv(infile)
    df = _normalize_columns(df)

    # Required spatial columns
    if not all(c in df.columns for c in ["x", "y"]):
        raise ValueError(f"CSV must contain columns x and y. Found: {list(df.columns)}")

    # Coerce x/y to numeric
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    # Ensure B_total exists
    df, bnote = _compute_btotal_if_missing(df)
    notes["B_total"] = bnote
    if "B_total" not in df.columns:
        raise ValueError("B_total is missing and could not be computed from Bx,By,Bz.")

    df["B_total"] = pd.to_numeric(df["B_total"], errors="coerce")

    # Time parsing (optional but recommended)
    time_col = _find_time_column(list(df.columns))
    notes["time_col"] = time_col if time_col else "None"
    t_series, tnote = _coerce_time_series(df[time_col] if time_col else None)
    notes["time_parse"] = tnote
    if t_series is not None:
        df["_time_utc"] = t_series
    else:
        df["_time_utc"] = pd.NaT

    # Drop rows missing core values
    before = len(df)
    df_clean = df.dropna(subset=["x", "y", "B_total"]).copy()
    dropped_na = before - len(df_clean)
    stats["rows_total"] = float(before)
    stats["rows_dropped_nan"] = float(dropped_na)
    stats["rows_after_nan_drop"] = float(len(df_clean))

    # Basic range stats
    stats["x_min"] = float(df_clean["x"].min())
    stats["x_max"] = float(df_clean["x"].max())
    stats["y_min"] = float(df_clean["y"].min())
    stats["y_max"] = float(df_clean["y"].max())
    stats["B_total_min"] = float(df_clean["B_total"].min())
    stats["B_total_max"] = float(df_clean["B_total"].max())
    stats["B_total_mean"] = float(df_clean["B_total"].mean())
    stats["B_total_std"] = float(df_clean["B_total"].std(ddof=1)) if len(df_clean) > 1 else float("nan")

    # Sampling rate estimate (if time exists)
    if df_clean["_time_utc"].notna().sum() > 5:
        t = df_clean["_time_utc"].sort_values()
        dt = t.diff().dt.total_seconds().dropna()
        # Guard against zeros/non-positive intervals
        dt = dt[dt > 0]
        if len(dt) > 3:
            stats["dt_median_s"] = float(dt.median())
            stats["dt_mean_s"] = float(dt.mean())
            stats["sample_rate_hz_est"] = float(1.0 / dt.median())
            # "jitter" as coefficient of variation
            stats["dt_cv"] = float(dt.std(ddof=1) / dt.mean()) if dt.mean() > 0 else float("nan")
        else:
            notes["sample_rate"] = "Not enough valid dt values for sample rate estimate"
    else:
        notes["sample_rate"] = "No usable time column; sample rate not estimated"

    # Outlier detection: robust z on B_total
    z = _robust_z(df_clean["B_total"].to_numpy())
    outlier_mask = np.abs(z) > z_thresh

    # Spike detection: big per-sample delta in B_total (time order if possible)
    if delta_thresh is not None:
        if df_clean["_time_utc"].notna().sum() > 5:
            df_ord = df_clean.sort_values("_time_utc").copy()
        else:
            df_ord = df_clean.copy()
        d = np.abs(np.diff(df_ord["B_total"].to_numpy(), prepend=df_ord["B_total"].iloc[0]))
        spike_mask_ord = d > delta_thresh
        # Map back by index
        spike_mask = pd.Series(spike_mask_ord, index=df_ord.index).reindex(df_clean.index).fillna(False).to_numpy(bool)
    else:
        spike_mask = np.zeros(len(df_clean), dtype=bool)

    flagged = outlier_mask | spike_mask
    stats["rows_flagged_outlier_or_spike"] = float(flagged.sum())
    stats["rows_flagged_pct"] = float(100.0 * flagged.mean()) if len(flagged) else 0.0

    df_clean["_flag_outlier"] = outlier_mask
    df_clean["_flag_spike"] = spike_mask
    df_clean["_flag_any"] = flagged

    # Optionally drop flagged rows
    if drop_outliers and len(df_clean) > 0:
        before2 = len(df_clean)
        df_clean = df_clean.loc[~df_clean["_flag_any"]].copy()
        stats["rows_dropped_flagged"] = float(before2 - len(df_clean))
        stats["rows_after_flag_drop"] = float(len(df_clean))
    else:
        stats["rows_dropped_flagged"] = 0.0
        stats["rows_after_flag_drop"] = float(len(df_clean))

    # Save cleaned CSV (keep your original columns + flags; drop helper _time_utc if you want)
    outdir.mkdir(parents=True, exist_ok=True)
    return df_clean, notes, stats


def make_plots(df: pd.DataFrame, outbase: Path) -> None:
    """
    Generate a small set of plots for fast sanity-checking.
    outbase is like: outdir/<stem>
    """
    # 1) B_total vs time (if time exists)
    if df["_time_utc"].notna().sum() > 5:
        df_t = df.sort_values("_time_utc")
        plt.figure()
        plt.plot(df_t["_time_utc"], df_t["B_total"])
        plt.xlabel("time (UTC)")
        plt.ylabel("B_total")
        plt.title("B_total over time")
        _save_plot(outbase.with_name(outbase.name + "_Btotal_vs_time.png"))

    # 2) Histogram of B_total
    plt.figure()
    plt.hist(df["B_total"].to_numpy(), bins=60)
    plt.xlabel("B_total")
    plt.ylabel("count")
    plt.title("Histogram: B_total")
    _save_plot(outbase.with_name(outbase.name + "_Btotal_hist.png"))

    # 3) XY scatter colored by B_total (or anomaly later)
    plt.figure()
    sc = plt.scatter(df["x"], df["y"], c=df["B_total"], s=14)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("XY scatter colored by B_total")
    plt.colorbar(sc, label="B_total")
    _save_plot(outbase.with_name(outbase.name + "_scatter_xy_colored.png"))

    # 4) Spike plot: per-row delta in B_total (ordered by time if possible)
    if df["_time_utc"].notna().sum() > 5:
        df_t = df.sort_values("_time_utc")
        series = df_t["B_total"].to_numpy()
        xaxis = df_t["_time_utc"]
    else:
        series = df["B_total"].to_numpy()
        xaxis = np.arange(len(df))

    deltas = np.abs(np.diff(series, prepend=series[0]))
    plt.figure()
    plt.plot(xaxis, deltas)
    plt.xlabel("time (UTC)" if df["_time_utc"].notna().sum() > 5 else "row index")
    plt.ylabel("|Δ B_total|")
    plt.title("Per-sample |ΔB_total| (spike check)")
    _save_plot(outbase.with_name(outbase.name + "_spike_deltas.png"))


def write_report(notes: Dict[str, str], stats: Dict[str, float], report_path: Path, infile: Path) -> None:
    lines = []
    lines.append(f"Validate + Diagnostics Report")
    lines.append(f"Input file: {infile}")
    lines.append("")
    lines.append("NOTES")
    for k, v in notes.items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("STATS")
    for k in sorted(stats.keys()):
        v = stats[k]
        if isinstance(v, float) and np.isfinite(v):
            lines.append(f"  - {k}: {v:.6g}")
        else:
            lines.append(f"  - {k}: {v}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate, clean, and generate diagnostics for magnetometer CSV.")
    p.add_argument("--in", dest="infile", required=True, help="Input CSV path (e.g., data/raw/mag_data.csv)")
    p.add_argument("--outdir", default="data/processed", help="Output directory (default: data/processed)")
    p.add_argument("--drop-outliers", action="store_true", help="If set, drop rows flagged as outlier/spike")
    p.add_argument("--z-thresh", type=float, default=6.0, help="Robust z-score threshold for B_total outliers (default: 6.0)")
    p.add_argument(
        "--delta-thresh",
        type=float,
        default=None,
        help="Optional absolute per-sample |ΔB_total| threshold to flag spikes (units same as B_total).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    infile = Path(args.infile)
    if not infile.is_absolute():
        infile = _REPO_ROOT / infile
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = _REPO_ROOT / outdir

    if not infile.exists():
        print(f"ERROR: input file not found: {infile}", file=sys.stderr)
        return 2

    stem = infile.stem
    clean_path = outdir / f"{stem}_clean.csv"
    report_path = outdir / f"{stem}_report.txt"
    outbase = outdir / stem  # used for naming plots

    try:
        df_clean, notes, stats = validate_and_clean(
            infile=infile,
            outdir=outdir,
            drop_outliers=args.drop_outliers,
            z_thresh=args.z_thresh,
            delta_thresh=args.delta_thresh,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Save cleaned
    df_clean.to_csv(clean_path, index=False)

    # Plots
    make_plots(df_clean, outbase)

    # Report
    write_report(notes, stats, report_path, infile)

    print(f"Wrote cleaned CSV: {clean_path}")
    print(f"Wrote report:      {report_path}")
    print(f"Wrote plots with prefix: {outbase}_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
