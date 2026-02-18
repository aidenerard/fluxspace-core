#!/usr/bin/env python3
"""
clean_geometry.py

Automatic geometry noise cleaning + trajectory-based cropping.

Pipeline:
  1. Load raw point cloud
  2. Crop to trajectory AABB (or manual AABB)
  3. Voxel downsample
  4. Statistical outlier removal (SOR)
  5. DBSCAN cluster filtering (keep largest)
  6. Optional ground-plane removal (RANSAC)
  7. Write cleaned point cloud
  8. Optional: generate mesh (Ball Pivoting, Poisson fallback)

Headless-safe — no Open3D GUI imports.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------
# Trajectory loader
# ---------------------------------------------------------------------------
def _load_trajectory_xyz(path: Path) -> np.ndarray | None:
    """Load camera positions from a trajectory CSV. Returns (N, 3) or None."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []

            # Detect column names
            x_col = y_col = z_col = None
            for xn, yn, zn in [("x", "y", "z"), ("tx", "ty", "tz"),
                                ("pos_x", "pos_y", "pos_z")]:
                if xn in headers and yn in headers and zn in headers:
                    x_col, y_col, z_col = xn, yn, zn
                    break

            if x_col is None:
                # Fallback: first 3 numeric-looking columns
                nums = [h for h in headers if h not in ("t_rel_s", "t", "timestamp",
                                                         "qx", "qy", "qz", "qw")]
                if len(nums) >= 3:
                    x_col, y_col, z_col = nums[0], nums[1], nums[2]
                else:
                    return None

            pts = []
            for row in reader:
                try:
                    pts.append([float(row[x_col]), float(row[y_col]), float(row[z_col])])
                except (ValueError, KeyError):
                    continue
        if not pts:
            return None
        return np.array(pts, dtype=np.float64)
    except Exception as exc:
        print(f"  WARNING: Could not parse trajectory {path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clean a raw point cloud: crop, downsample, SOR, cluster, plane removal, mesh."
    )
    p.add_argument("--in-pcd", required=True, help="Input point cloud .ply")
    p.add_argument("--out-pcd", required=True, help="Output cleaned point cloud .ply")
    p.add_argument("--out-mesh", default="", help="Optional output cleaned mesh .ply")

    g = p.add_argument_group("downsample + SOR")
    g.add_argument("--voxel-size", type=float, default=0.005)
    g.add_argument("--sor-nb-neighbors", type=int, default=30)
    g.add_argument("--sor-std-ratio", type=float, default=2.0)

    g2 = p.add_argument_group("cluster filtering")
    g2.add_argument("--min-cluster-points", type=int, default=5000)
    g2.add_argument("--keep-top-k-clusters", type=int, default=1)

    g3 = p.add_argument_group("plane removal")
    g3.add_argument("--remove-plane", action="store_true")
    g3.add_argument("--plane-dist-thresh", type=float, default=0.01)
    g3.add_argument("--plane-ransac-n", type=int, default=3)
    g3.add_argument("--plane-num-iter", type=int, default=1000)

    g4 = p.add_argument_group("trajectory crop")
    g4.add_argument("--trajectory", default="", help="Path to trajectory CSV")
    g4.add_argument("--crop-from-trajectory", action="store_true", default=False)
    g4.add_argument("--no-traj-crop", action="store_true", default=False,
                    help="Force skip trajectory crop even if trajectory is valid")
    g4.add_argument("--crop-margin", type=float, default=0.25,
                    help="Metres to expand trajectory AABB (default: 0.25)")
    g4.add_argument("--crop-z-min", type=float, default=None)
    g4.add_argument("--crop-z-max", type=float, default=None)

    g5 = p.add_argument_group("manual crop")
    g5.add_argument("--crop-aabb", nargs=6, type=float, default=None,
                    metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"))

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    in_path = Path(args.in_pcd)

    if not in_path.exists():
        print(f"ERROR: Input not found: {in_path}", file=sys.stderr)
        return 2

    # ── Load ──────────────────────────────────────────────────────────────
    pcd = o3d.io.read_point_cloud(str(in_path))
    n_input = len(pcd.points)
    if n_input < 100:
        print(f"ERROR: Point cloud too small ({n_input} points). Aborting.", file=sys.stderr)
        return 2
    print(f"Loaded {n_input:,} points from {in_path}")

    # ── Trajectory crop ───────────────────────────────────────────────────
    n_after_crop = n_input
    crop_applied = False
    pcd_before_crop = pcd  # keep a reference for fallback

    if args.no_traj_crop:
        print("  Trajectory crop skipped (--no-traj-crop).")
    elif args.crop_from_trajectory and args.trajectory:
        traj_xyz = _load_trajectory_xyz(Path(args.trajectory))
        if traj_xyz is not None and len(traj_xyz) > 0:
            # Filter out any NaN/inf rows in the trajectory itself
            finite_mask = np.isfinite(traj_xyz).all(axis=1)
            traj_xyz = traj_xyz[finite_mask]

            if len(traj_xyz) < 2:
                print(f"  WARNING: Only {len(traj_xyz)} finite trajectory point(s) "
                      f"(need >= 2). Skipping trajectory crop.")
            else:
                lo = traj_xyz.min(axis=0) - args.crop_margin
                hi = traj_xyz.max(axis=0) + args.crop_margin
                if args.crop_z_min is not None:
                    lo[2] = args.crop_z_min
                if args.crop_z_max is not None:
                    hi[2] = args.crop_z_max

                if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
                    print(f"  WARNING: trajectory bounds invalid (lo={lo.tolist()}, "
                          f"hi={hi.tolist()}). Skipping trajectory crop.")
                elif np.any(lo >= hi):
                    print(f"  WARNING: trajectory bounds degenerate (lo >= hi). "
                          f"Skipping trajectory crop.")
                else:
                    print(f"  Trajectory crop: lo={lo.tolist()}, hi={hi.tolist()}")
                    bb = o3d.geometry.AxisAlignedBoundingBox(lo, hi)
                    pcd_cropped = pcd.crop(bb)
                    n_after_crop = len(pcd_cropped.points)
                    if n_after_crop < 100:
                        print(f"  WARNING: Trajectory crop left only {n_after_crop} points. "
                              f"Falling back to uncropped cloud ({n_input:,} points).")
                        n_after_crop = n_input
                    else:
                        pcd = pcd_cropped
                        crop_applied = True
                        print(f"  After trajectory crop: {n_after_crop:,} points")
        else:
            print("  WARNING: Could not load trajectory; skipping trajectory crop.")
    elif args.crop_aabb is not None:
        lo = np.array(args.crop_aabb[:3])
        hi = np.array(args.crop_aabb[3:])
        if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
            print(f"  WARNING: Manual AABB bounds invalid. Skipping crop.")
        else:
            print(f"  Manual AABB crop: lo={lo.tolist()}, hi={hi.tolist()}")
            bb = o3d.geometry.AxisAlignedBoundingBox(lo, hi)
            pcd_cropped = pcd.crop(bb)
            n_after_crop = len(pcd_cropped.points)
            if n_after_crop < 100:
                print(f"  WARNING: Manual crop left only {n_after_crop} points. "
                      f"Falling back to uncropped cloud ({n_input:,} points).")
                n_after_crop = n_input
            else:
                pcd = pcd_cropped
                crop_applied = True
                print(f"  After manual crop: {n_after_crop:,} points")

    # ── Voxel downsample ──────────────────────────────────────────────────
    vs = args.voxel_size
    pcd = pcd.voxel_down_sample(voxel_size=vs)
    n_after_ds = len(pcd.points)
    print(f"  After voxel downsample (vs={vs}): {n_after_ds:,} points")

    # ── Statistical outlier removal ───────────────────────────────────────
    pcd, sor_idx = pcd.remove_statistical_outlier(
        nb_neighbors=args.sor_nb_neighbors,
        std_ratio=args.sor_std_ratio,
    )
    n_after_sor = len(pcd.points)
    print(f"  After SOR (k={args.sor_nb_neighbors}, std={args.sor_std_ratio}): "
          f"{n_after_sor:,} points  (removed {n_after_ds - n_after_sor:,})")

    # ── DBSCAN cluster filtering ──────────────────────────────────────────
    n_before_cluster = n_after_sor
    eps = 2.5 * vs
    labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=30, print_progress=False))

    if labels.max() >= 0:
        unique, counts = np.unique(labels[labels >= 0], return_counts=True)
        sorted_idx = np.argsort(-counts)
        keep_labels = set()
        for i in range(min(args.keep_top_k_clusters, len(sorted_idx))):
            lab = unique[sorted_idx[i]]
            if counts[sorted_idx[i]] >= args.min_cluster_points:
                keep_labels.add(lab)

        if keep_labels:
            mask = np.isin(labels, list(keep_labels))
            pcd = pcd.select_by_index(np.where(mask)[0])
            n_after_cluster = len(pcd.points)
            print(f"  DBSCAN (eps={eps:.4f}): kept {len(keep_labels)} cluster(s), "
                  f"{n_after_cluster:,} points  (removed {n_before_cluster - n_after_cluster:,})")
        else:
            print(f"  DBSCAN: no cluster met min_cluster_points={args.min_cluster_points}; "
                  f"keeping all {n_before_cluster:,} points.")
            n_after_cluster = n_before_cluster
    else:
        print(f"  DBSCAN: no clusters found; keeping all {n_before_cluster:,} points.")
        n_after_cluster = n_before_cluster

    # ── Plane removal ─────────────────────────────────────────────────────
    n_after_plane = n_after_cluster
    if args.remove_plane and len(pcd.points) > 100:
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=args.plane_dist_thresh,
            ransac_n=args.plane_ransac_n,
            num_iterations=args.plane_num_iter,
        )
        remainder = pcd.select_by_index(inliers, invert=True)
        if len(remainder.points) >= 0.2 * len(pcd.points):
            pcd = remainder
            n_after_plane = len(pcd.points)
            a, b, c, d = plane_model
            print(f"  Plane removal: removed {len(inliers):,} inliers "
                  f"(plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0), "
                  f"{n_after_plane:,} remain")
        else:
            print(f"  Plane removal: would leave too few points ({len(remainder.points):,}); "
                  f"reverting. Kept {n_after_cluster:,} points.")

    # ── Estimate normals (needed for mesh generation) ─────────────────────
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=vs * 4, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)

    # ── Write cleaned pcd ─────────────────────────────────────────────────
    out_pcd_path = Path(args.out_pcd)
    out_pcd_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_pcd_path), pcd)
    n_final = len(pcd.points)
    print(f"Wrote cleaned pcd: {out_pcd_path}  ({n_final:,} points)")

    # ── Optional mesh generation ──────────────────────────────────────────
    mesh = None
    if args.out_mesh:
        out_mesh_path = Path(args.out_mesh)
        out_mesh_path.parent.mkdir(parents=True, exist_ok=True)

        # Try Ball Pivoting first
        radii = [vs * 2, vs * 4, vs * 8]
        radii_vec = o3d.utility.DoubleVector(radii)
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii_vec)
            if mesh.is_empty() or len(mesh.triangles) < 10:
                mesh = None
                print("  Ball Pivoting produced too few triangles; trying Poisson.")
        except Exception as exc:
            print(f"  Ball Pivoting failed ({exc}); trying Poisson.")
            mesh = None

        # Fallback: Poisson
        if mesh is None:
            try:
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=8, width=0, scale=1.1, linear_fit=False
                )
                # Crop Poisson result to pcd bounds (Poisson extrapolates)
                pcd_bb = pcd.get_axis_aligned_bounding_box()
                mesh = mesh.crop(pcd_bb)
            except Exception as exc:
                print(f"  Poisson also failed ({exc}); no mesh generated.")
                mesh = None

        if mesh is not None and not mesh.is_empty():
            mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(str(out_mesh_path), mesh)
            print(f"Wrote cleaned mesh: {out_mesh_path}  "
                  f"({len(mesh.vertices):,} verts, {len(mesh.triangles):,} faces)")
        else:
            print("  WARNING: No mesh generated.")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n--- clean_geometry summary ---")
    print(f"  Input points       : {n_input:,}")
    if crop_applied:
        print(f"  After crop         : {n_after_crop:,}")
    print(f"  After downsample   : {n_after_ds:,}")
    print(f"  After SOR          : {n_after_sor:,}")
    print(f"  After DBSCAN       : {n_after_cluster:,}")
    if args.remove_plane:
        print(f"  After plane remove : {n_after_plane:,}")
    print(f"  Final points       : {n_final:,}")
    if mesh is not None and not mesh.is_empty():
        print(f"  Mesh verts/faces   : {len(mesh.vertices):,} / {len(mesh.triangles):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
