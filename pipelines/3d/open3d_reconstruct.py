#!/usr/bin/env python3
"""
open3d_reconstruct.py

Reconstruct a 3D triangle mesh from RGB + depth frames captured by
capture_oak_rgbd.py, using Open3D TSDF volume integration and
frame-to-frame RGB-D odometry.

Inputs  (from capture_oak_rgbd.py output, default: RUN_DIR/raw/oak_rgbd):
  color/color_*.jpg       colour frames
  depth/depth_*.png       16-bit depth PNGs (mm)
  timestamps.csv          idx, t_wall_s, t_rgb_dev_ms, t_depth_dev_ms
  intrinsics.json         (optional) fx, fy, cx, cy, width, height

Outputs (default: RUN_DIR/processed/):
  trajectory.csv          t_rel_s, x, y, z, qx, qy, qz, qw
  open3d_pcd_raw.ply      raw coloured point cloud (for clean_geometry.py)
  open3d_mesh.ply         coloured triangle mesh
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_paths import infer_run_dir_from_path  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_VOXEL_LENGTH = 0.01    # 1 cm
DEFAULT_SDF_TRUNC = 0.04       # 4 cm
DEFAULT_DEPTH_SCALE = 1000.0   # mm -> m
DEFAULT_DEPTH_TRUNC = 3.0      # metres
APPROX_FX = 600.0
APPROX_FY = 600.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to quaternion (x, y, z, w)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def load_intrinsics(in_dir: Path, width: int, height: int) -> o3d.camera.PinholeCameraIntrinsic:
    intr_path = in_dir / "intrinsics.json"
    fx, fy = APPROX_FX, APPROX_FY
    cx, cy = width / 2.0, height / 2.0
    source = "approximate"
    if intr_path.exists():
        try:
            with open(intr_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            fx = float(data["fx"])
            fy = float(data["fy"])
            cx = float(data["cx"])
            cy = float(data["cy"])
            source = data.get("source", "intrinsics.json")
        except Exception as exc:
            print(f"WARNING: Could not read {intr_path} ({exc}); using approximate intrinsics.")
    print(f"  Intrinsics ({source}): fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


def load_timestamps(in_dir: Path) -> dict[int, float]:
    """Load timestamps.csv -> {idx: t_ms}."""
    ts_path = in_dir / "timestamps.csv"
    if not ts_path.exists():
        return {}
    result: dict[int, float] = {}
    with open(ts_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row["idx"])
                if "t_rgb_dev_ms" in row:
                    result[idx] = float(row["t_rgb_dev_ms"])
                elif "t_device_ms" in row:
                    result[idx] = float(row["t_device_ms"])
                elif "t_wall_s" in row:
                    result[idx] = float(row["t_wall_s"]) * 1000.0
            except (KeyError, ValueError):
                pass
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Open3D TSDF reconstruction -> processed/trajectory.csv + processed/open3d_mesh.ply"
    )
    p.add_argument(
        "--in", dest="input_dir", required=True,
        help="Input directory with color/, depth/, etc.  (e.g. $RUN_DIR/raw/oak_rgbd)",
    )
    p.add_argument(
        "--out-dir", default="",
        help="Directory for trajectory.csv + open3d_mesh.ply. "
             "Default: auto-detect $RUN_DIR/processed/ from --in path.",
    )
    p.add_argument(
        "--out", default="",
        help="Override: explicit trajectory.csv path.",
    )
    p.add_argument(
        "--mesh", default="",
        help="Override: explicit mesh PLY path.",
    )
    p.add_argument(
        "--voxel-size", type=float, default=DEFAULT_VOXEL_LENGTH,
        help=f"TSDF voxel size in metres (default: {DEFAULT_VOXEL_LENGTH})",
    )
    p.add_argument(
        "--no-viz", action="store_true",
        help="Skip the Open3D interactive viewer.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    in_dir = Path(args.input_dir).expanduser().resolve()

    if not in_dir.is_dir():
        print(f"ERROR: Input directory not found: {in_dir}", file=sys.stderr)
        return 2

    # --- Discover frames ---
    color_files = sorted(glob.glob(str(in_dir / "color" / "color_*.jpg")))
    depth_files = sorted(glob.glob(str(in_dir / "depth" / "depth_*.png")))

    if not color_files or not depth_files:
        print(f"ERROR: No colour or depth images found in {in_dir}", file=sys.stderr)
        return 2
    if len(color_files) != len(depth_files):
        print(f"ERROR: Colour/depth count mismatch ({len(color_files)} vs {len(depth_files)})", file=sys.stderr)
        return 2

    n_frames = len(color_files)
    print(f"Found {n_frames} frame pairs in {in_dir}")

    # --- Intrinsics ---
    sample_depth = o3d.io.read_image(depth_files[0])
    width = np.asarray(sample_depth).shape[1]
    height = np.asarray(sample_depth).shape[0]
    intr = load_intrinsics(in_dir, width, height)

    # --- Timestamps ---
    ts_map = load_timestamps(in_dir)
    if ts_map:
        t0 = ts_map.get(0, 0.0)
        print(f"  Loaded {len(ts_map)} timestamps from timestamps.csv")
    else:
        t0 = 0.0
        print("  WARNING: No timestamps.csv found; using frame index / 30 fps for t_rel_s.")

    # --- TSDF volume ---
    sdf_trunc = max(args.voxel_size * 4.0, DEFAULT_SDF_TRUNC)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    # --- Odometry + TSDF integration ---
    prev_rgbd = None
    prev_pose = np.eye(4)
    frame_poses: list[np.ndarray] = []

    for i, (cf, df) in enumerate(zip(color_files, depth_files)):
        color = o3d.io.read_image(cf)
        depth = o3d.io.read_image(df)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth,
            depth_scale=DEFAULT_DEPTH_SCALE,
            depth_trunc=DEFAULT_DEPTH_TRUNC,
            convert_rgb_to_intensity=False,
        )
        if prev_rgbd is not None:
            option = o3d.pipelines.odometry.OdometryOption()
            success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
                prev_rgbd, rgbd, intr, np.eye(4),
                o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                option,
            )
            if success:
                prev_pose = prev_pose @ trans

        frame_poses.append(prev_pose.copy())
        volume.integrate(rgbd, intr, np.linalg.inv(prev_pose))
        prev_rgbd = rgbd

        if i % 30 == 0:
            print(f"  Integrated frame {i}/{n_frames}")

    print(f"  Integrated all {n_frames} frames.")

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    # --- Extract raw point cloud (for clean_geometry.py) ---
    pcd_raw = volume.extract_point_cloud()

    # --- Resolve output directory ---
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        run = infer_run_dir_from_path(in_dir)
        if run is not None:
            out_dir = run / "processed"
        else:
            out_dir = in_dir.parent  # fallback: next to input

    out_dir.mkdir(parents=True, exist_ok=True)

    traj_path = Path(args.out) if args.out else out_dir / "trajectory.csv"
    mesh_path = Path(args.mesh) if args.mesh else out_dir / "open3d_mesh.ply"
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Write trajectory.csv ---
    with open(traj_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_rel_s", "x", "y", "z", "qx", "qy", "qz", "qw"])
        for i, pose in enumerate(frame_poses):
            if ts_map:
                t_dev_ms = ts_map.get(i, i * (1000.0 / 30.0))
                t_rel = (t_dev_ms - t0) / 1000.0
            else:
                t_rel = float(i) * (1.0 / 30.0)
            x, y, z = pose[0, 3], pose[1, 3], pose[2, 3]
            R = pose[:3, :3]
            qx, qy, qz, qw = rotation_matrix_to_quat(R)
            writer.writerow([
                f"{t_rel:.6f}",
                f"{x:.6f}", f"{y:.6f}", f"{z:.6f}",
                f"{qx:.6f}", f"{qy:.6f}", f"{qz:.6f}", f"{qw:.6f}",
            ])

    print(f"Wrote trajectory : {traj_path}  ({len(frame_poses)} poses)")
    t_max = ((ts_map.get(len(frame_poses) - 1, 0) - t0) / 1000.0) if ts_map else (len(frame_poses) - 1) / 30.0
    print(f"  t_rel_s range  : 0.000 .. {t_max:.3f} s")

    # --- Write raw point cloud ---
    pcd_path = out_dir / "open3d_pcd_raw.ply"
    if pcd_raw is not None and not pcd_raw.is_empty():
        o3d.io.write_point_cloud(str(pcd_path), pcd_raw)
        print(f"Wrote pcd_raw    : {pcd_path}  ({len(pcd_raw.points):,} points)")
    else:
        # Fallback: sample from mesh
        pcd_raw = mesh.sample_points_uniformly(number_of_points=min(200_000, len(mesh.vertices)))
        o3d.io.write_point_cloud(str(pcd_path), pcd_raw)
        print(f"Wrote pcd_raw    : {pcd_path}  ({len(pcd_raw.points):,} points, sampled from mesh)")

    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    print(f"Wrote mesh       : {mesh_path}")

    if not args.no_viz:
        o3d.visualization.draw_geometries([mesh])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
