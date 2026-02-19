#!/usr/bin/env python3
"""
open3d_reconstruct.py

Reconstruct a 3D triangle mesh + point cloud from RGB + depth frames
captured by capture_oak_rgbd.py or capture_oak_rgbd_vio.py, using
Open3D TSDF volume integration.

Pose sources (--pose-source):
  auto   — use trajectory_device.csv if present, else RGB-D odometry
  device — load trajectory_device.csv (from VIO capture) as external poses
  odom   — compute frame-to-frame RGB-D odometry (original behaviour)

Inputs  (from capture_oak_rgbd.py / capture_oak_rgbd_vio.py output):
  color/color_*.jpg       colour frames
  depth/depth_*.png       16-bit depth PNGs (mm)
  timestamps.csv          idx, t_wall_s, t_rgb_dev_ms, t_depth_dev_ms
  intrinsics.json         (optional) fx, fy, cx, cy, width, height
  trajectory_device.csv   (optional) t_rel_s, x, y, z, qx, qy, qz, qw

Outputs (default: RUN_DIR/processed/):
  trajectory.csv          t_rel_s, x, y, z, qx, qy, qz, qw
  open3d_pcd_raw.ply      raw coloured point cloud
  open3d_mesh_raw.ply     raw coloured triangle mesh
  reconstruction_report.json   stats + parameters for debugging
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_paths import infer_run_dir_from_path  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_VOXEL_LENGTH = 0.01
DEFAULT_SDF_TRUNC = 0.04
DEFAULT_DEPTH_SCALE = 1000.0
DEFAULT_DEPTH_TRUNC = 3.0
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


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to 3×3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load_device_trajectory(path: Path) -> list[np.ndarray]:
    """Load trajectory_device.csv and return a list of 4×4 camera-to-world poses.

    Each row has columns: t_rel_s, x, y, z, qx, qy, qz, qw.
    Returns one 4×4 matrix per row (in file order).
    """
    poses: list[np.ndarray] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x  = float(row["x"])
                y  = float(row["y"])
                z  = float(row["z"])
                qx = float(row["qx"])
                qy = float(row["qy"])
                qz = float(row["qz"])
                qw = float(row["qw"])
            except (KeyError, ValueError):
                poses.append(np.eye(4))
                continue
            vals = [x, y, z, qx, qy, qz, qw]
            if not np.isfinite(vals).all():
                poses.append(np.eye(4))
                continue
            q = np.array([qx, qy, qz, qw])
            q_norm = np.linalg.norm(q)
            if q_norm < 1e-8:
                poses.append(np.eye(4))
                continue
            q = q / q_norm
            R = quat_to_rotation_matrix(q)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [x, y, z]
            poses.append(T)
    return poses


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
        description="Open3D TSDF reconstruction -> processed/{trajectory.csv, open3d_mesh_raw.ply, open3d_pcd_raw.ply}"
    )
    p.add_argument("--in", dest="input_dir", required=True,
                    help="Input directory (e.g. $RUN_DIR/raw/oak_rgbd)")
    p.add_argument("--out-dir", default="",
                    help="Output directory. Default: auto-detect $RUN_DIR/processed/.")
    p.add_argument("--out", default="", help="Override: explicit trajectory.csv path.")
    p.add_argument("--mesh", default="", help="Override: explicit mesh PLY path.")

    g_pose = p.add_argument_group("pose source")
    g_pose.add_argument("--pose-source", choices=["auto", "odom", "device"],
                        default="auto",
                        help="auto: use trajectory_device.csv if present, else odom. "
                             "device: require external trajectory. "
                             "odom: frame-to-frame RGB-D odometry (default: auto)")
    g_pose.add_argument("--trajectory", default="",
                        help="Path to external trajectory CSV "
                             "(default: <input_dir>/trajectory_device.csv)")

    g = p.add_argument_group("reconstruction tuning")
    g.add_argument("--voxel-size", type=float, default=DEFAULT_VOXEL_LENGTH,
                   help=f"TSDF voxel size in metres (default: {DEFAULT_VOXEL_LENGTH})")
    g.add_argument("--sdf-trunc", type=float, default=0,
                   help="SDF truncation in metres (default: max(voxel*4, 0.04))")
    g.add_argument("--depth-trunc", type=float, default=DEFAULT_DEPTH_TRUNC,
                   help=f"Max depth in metres (default: {DEFAULT_DEPTH_TRUNC})")
    g.add_argument("--depth-scale", type=float, default=DEFAULT_DEPTH_SCALE,
                   help=f"Depth image units (default: {DEFAULT_DEPTH_SCALE} = mm)")
    g.add_argument("--every-n", type=int, default=1,
                   help="Use every Nth frame (default: 1 = all frames)")
    g.add_argument("--max-frames", type=int, default=0,
                   help="Stop after N frames (default: 0 = no limit)")
    g.add_argument("--odometry-method", choices=["hybrid", "color"], default="hybrid",
                   help="Odometry Jacobian method for odom mode (default: hybrid)")

    g2 = p.add_argument_group("output options")
    g2.add_argument("--save-glb", action="store_true",
                    help="Also export mesh as .glb for web viewing")
    g2.add_argument("--no-viz", action="store_true",
                    help="Skip the Open3D interactive viewer.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    in_dir = Path(args.input_dir).expanduser().resolve()
    warnings_list: list[str] = []
    t_start = time.monotonic()

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

    n_total = len(color_files)

    # --- Frame subsampling ---
    every_n = max(1, args.every_n)
    indices = list(range(0, n_total, every_n))
    if args.max_frames > 0:
        indices = indices[:args.max_frames]
    n_used = len(indices)

    color_files = [color_files[i] for i in indices]
    depth_files = [depth_files[i] for i in indices]
    print(f"Found {n_total} frame pairs; using {n_used} (every_n={every_n})")

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
        print("  WARNING: No timestamps.csv found; using frame index / 30 fps.")
        warnings_list.append("No timestamps.csv; synthetic timing used")

    # --- TSDF volume ---
    sdf_trunc = args.sdf_trunc if args.sdf_trunc > 0 else max(args.voxel_size * 4.0, DEFAULT_SDF_TRUNC)
    depth_trunc = args.depth_trunc
    depth_scale = args.depth_scale

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    # --- Resolve pose source ---
    pose_source = args.pose_source  # auto | odom | device
    device_poses: list[np.ndarray] | None = None

    traj_device_path = Path(args.trajectory) if args.trajectory else in_dir / "trajectory_device.csv"

    if pose_source == "auto":
        if traj_device_path.exists():
            pose_source = "device"
            print(f"  Auto-detected device trajectory: {traj_device_path}")
        else:
            pose_source = "odom"
            print("  No trajectory_device.csv found; using RGB-D odometry")
    elif pose_source == "device":
        if not traj_device_path.exists():
            print(f"ERROR: --pose-source device but trajectory not found: {traj_device_path}",
                  file=sys.stderr)
            return 2

    if pose_source == "device":
        all_device_poses = load_device_trajectory(traj_device_path)
        print(f"  Loaded {len(all_device_poses)} device poses from {traj_device_path.name}")
        device_poses = [all_device_poses[i] for i in indices if i < len(all_device_poses)]
        n_missing = n_used - len(device_poses)
        if n_missing > 0:
            warnings_list.append(f"{n_missing} frames lack device poses (padded with identity)")
            device_poses.extend([np.eye(4)] * n_missing)

    print(f"  Pose source: {pose_source}")

    # --- Odometry Jacobian (only used if pose_source == odom) ---
    jacobian = None
    if pose_source == "odom":
        if args.odometry_method == "color":
            jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm()
        else:
            jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()

    # --- Integration loop ---
    prev_rgbd = None
    prev_pose = np.eye(4)
    frame_poses: list[np.ndarray] = []
    frame_indices: list[int] = []
    odo_success = 0
    odo_fail = 0

    for seq_i, (cf, df) in enumerate(zip(color_files, depth_files)):
        orig_idx = indices[seq_i]
        color = o3d.io.read_image(cf)
        depth = o3d.io.read_image(df)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )

        if pose_source == "device" and device_poses is not None:
            prev_pose = device_poses[seq_i]
        elif pose_source == "odom" and prev_rgbd is not None and jacobian is not None:
            option = o3d.pipelines.odometry.OdometryOption()
            success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
                prev_rgbd, rgbd, intr, np.eye(4), jacobian, option,
            )
            if success:
                prev_pose = prev_pose @ trans
                odo_success += 1
            else:
                odo_fail += 1

        frame_poses.append(prev_pose.copy())
        frame_indices.append(orig_idx)
        volume.integrate(rgbd, intr, np.linalg.inv(prev_pose))
        prev_rgbd = rgbd

        if seq_i % 30 == 0:
            print(f"  Integrated frame {seq_i}/{n_used}")

    odo_total = odo_success + odo_fail
    odo_rate = odo_success / odo_total if odo_total > 0 else 1.0

    if pose_source == "odom":
        print(f"  Integrated all {n_used} frames.  Odometry: {odo_success}/{odo_total} "
              f"succeeded ({odo_rate:.0%})")
        if odo_rate < 0.3:
            w = f"Low odometry success rate ({odo_rate:.0%}). Geometry may be poor."
            warnings_list.append(w)
            print(f"  WARNING: {w}")
    else:
        print(f"  Integrated all {n_used} frames using device poses.")

    # --- Extract geometry ---
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    pcd_raw = volume.extract_point_cloud()

    # --- Resolve output directory ---
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        run = infer_run_dir_from_path(in_dir)
        out_dir = (run / "processed") if run else in_dir.parent

    out_dir.mkdir(parents=True, exist_ok=True)

    traj_path = Path(args.out) if args.out else out_dir / "trajectory.csv"
    mesh_path = Path(args.mesh) if args.mesh else out_dir / "open3d_mesh_raw.ply"
    pcd_path = out_dir / "open3d_pcd_raw.ply"
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Write trajectory.csv (filter NaN poses) ---
    n_nan_poses = 0
    with open(traj_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_rel_s", "x", "y", "z", "qx", "qy", "qz", "qw"])
        for seq_i, pose in enumerate(frame_poses):
            orig_idx = frame_indices[seq_i]
            if ts_map:
                t_dev_ms = ts_map.get(orig_idx, orig_idx * (1000.0 / 30.0))
                t_rel = (t_dev_ms - t0) / 1000.0
            else:
                t_rel = float(orig_idx) * (1.0 / 30.0)
            x, y, z = pose[0, 3], pose[1, 3], pose[2, 3]
            R = pose[:3, :3]
            vals = [x, y, z] + list(R.ravel())
            if not np.isfinite(vals).all():
                n_nan_poses += 1
                continue
            qx, qy, qz, qw = rotation_matrix_to_quat(R)
            writer.writerow([
                f"{t_rel:.6f}",
                f"{x:.6f}", f"{y:.6f}", f"{z:.6f}",
                f"{qx:.6f}", f"{qy:.6f}", f"{qz:.6f}", f"{qw:.6f}",
            ])

    n_written = len(frame_poses) - n_nan_poses
    print(f"Wrote trajectory : {traj_path}  ({n_written} poses, {n_nan_poses} NaN skipped)")
    if n_nan_poses > 0:
        warnings_list.append(f"{n_nan_poses} NaN poses filtered from trajectory")

    # --- Write raw point cloud ---
    if pcd_raw is not None and not pcd_raw.is_empty():
        o3d.io.write_point_cloud(str(pcd_path), pcd_raw)
        print(f"Wrote pcd_raw    : {pcd_path}  ({len(pcd_raw.points):,} points)")
    else:
        pcd_raw = mesh.sample_points_uniformly(number_of_points=min(200_000, len(mesh.vertices)))
        o3d.io.write_point_cloud(str(pcd_path), pcd_raw)
        print(f"Wrote pcd_raw    : {pcd_path}  ({len(pcd_raw.points):,} points, sampled from mesh)")
        warnings_list.append("Point cloud extracted empty; sampled from mesh instead")

    # --- Write raw mesh ---
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    print(f"Wrote mesh_raw   : {mesh_path}  ({len(mesh.vertices):,} verts, {len(mesh.triangles):,} faces)")

    # --- Optional GLB export ---
    if args.save_glb:
        glb_path = mesh_path.with_suffix(".glb")
        try:
            o3d.io.write_triangle_mesh(str(glb_path), mesh)
            print(f"Wrote GLB        : {glb_path}")
        except Exception as exc:
            warnings_list.append(f"GLB export failed: {exc}")
            print(f"  WARNING: GLB export failed ({exc})")

    # --- Bounding box for report ---
    bbox_min = bbox_max = [0, 0, 0]
    if not mesh.is_empty():
        bb = mesh.get_axis_aligned_bounding_box()
        bbox_min = np.asarray(bb.get_min_bound()).tolist()
        bbox_max = np.asarray(bb.get_max_bound()).tolist()

    # --- Write reconstruction_report.json ---
    elapsed = time.monotonic() - t_start
    report: dict = {
        "input_dir": str(in_dir),
        "n_frames_total": n_total,
        "n_frames_used": n_used,
        "every_n": every_n,
        "pose_source": pose_source,
        "voxel_size": args.voxel_size,
        "sdf_trunc": sdf_trunc,
        "depth_trunc": depth_trunc,
        "depth_scale": depth_scale,
        "nan_poses_filtered": n_nan_poses,
        "point_count_raw": len(pcd_raw.points) if pcd_raw else 0,
        "mesh_vertex_count": len(mesh.vertices),
        "mesh_triangle_count": len(mesh.triangles),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "elapsed_seconds": round(elapsed, 1),
        "warnings": warnings_list,
    }
    if pose_source == "odom":
        report["odometry_method"] = args.odometry_method
        report["odometry_success_count"] = odo_success
        report["odometry_failure_count"] = odo_fail
        report["odometry_success_rate"] = round(odo_rate, 4)
    elif pose_source == "device":
        report["device_trajectory_file"] = str(traj_device_path)
        report["device_pose_count"] = len(device_poses) if device_poses else 0
    report_path = out_dir / "reconstruction_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote report     : {report_path}")

    if not args.no_viz:
        o3d.visualization.draw_geometries([mesh])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
