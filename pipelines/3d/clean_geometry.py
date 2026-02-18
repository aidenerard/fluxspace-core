#!/usr/bin/env python3
"""
clean_geometry.py

Automatic geometry noise cleaning, trajectory-based cropping, and mesh repair.

Pipeline:
  1. Load raw point cloud (and optionally raw mesh)
  2. Remove NaN/inf points
  3. Crop to trajectory AABB or manual AABB (with safe fallbacks)
  4. Voxel downsample
  5. Statistical outlier removal (SOR)
  6. Radius outlier removal (ROR)
  7. DBSCAN cluster filtering (keep largest)
  8. Optional ground-plane removal (RANSAC)
  9. Write cleaned point cloud + preview
 10. Clean raw mesh or generate new mesh (Poisson/BPA fallback)
 11. Write cleaned mesh + optional GLB preview
 12. Write cleaning_report.json

Headless-safe — no Open3D GUI imports.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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
            x_col = y_col = z_col = None
            for xn, yn, zn in [("x", "y", "z"), ("tx", "ty", "tz"),
                                ("pos_x", "pos_y", "pos_z")]:
                if xn in headers and yn in headers and zn in headers:
                    x_col, y_col, z_col = xn, yn, zn
                    break
            if x_col is None:
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
# Mesh cleaning helpers
# ---------------------------------------------------------------------------
def _clean_raw_mesh(mesh: o3d.geometry.TriangleMesh,
                    target_triangles: int = 0) -> o3d.geometry.TriangleMesh:
    """Clean an existing raw mesh: remove degenerate/non-manifold, simplify."""
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    if target_triangles > 0 and len(mesh.triangles) > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    return mesh


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clean raw point cloud + mesh: crop, denoise, cluster, mesh repair."
    )

    g_io = p.add_argument_group("input / output")
    g_io.add_argument("--run-dir", default="",
                      help="Run directory — derives default input/output paths automatically.")
    g_io.add_argument("--in-pcd", default="",
                      help="Input point cloud .ply (default: RUN_DIR/processed/open3d_pcd_raw.ply)")
    g_io.add_argument("--in-mesh", default="",
                      help="Input raw mesh .ply for mesh cleaning path (default: RUN_DIR/processed/open3d_mesh_raw.ply)")
    g_io.add_argument("--out-pcd", default="",
                      help="Output cleaned point cloud (default: RUN_DIR/processed/open3d_pcd_clean.ply)")
    g_io.add_argument("--out-mesh", default="",
                      help="Output cleaned mesh (default: RUN_DIR/processed/open3d_mesh_clean.ply)")
    g_io.add_argument("--out-dir", default="",
                      help="Override output directory (default: same as input)")

    g = p.add_argument_group("downsample + SOR + ROR")
    g.add_argument("--downsample", "--voxel-size", dest="voxel_size", type=float, default=0.005)
    g.add_argument("--sor-nb-neighbors", type=int, default=30)
    g.add_argument("--sor-std-ratio", type=float, default=2.0)
    g.add_argument("--ror-radius", type=float, default=0.02,
                   help="Radius outlier removal radius (default: 0.02)")
    g.add_argument("--ror-min-points", type=int, default=16,
                   help="Radius outlier removal minimum neighbours (default: 16)")
    g.add_argument("--no-ror", action="store_true", help="Skip radius outlier removal")

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
    g4.add_argument("--crop-margin", type=float, default=0.25)
    g4.add_argument("--crop-z-min", type=float, default=None)
    g4.add_argument("--crop-z-max", type=float, default=None)

    g5 = p.add_argument_group("manual crop")
    g5.add_argument("--crop-aabb", nargs=6, type=float, default=None,
                    metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"))

    g6 = p.add_argument_group("mesh options")
    g6.add_argument("--mesh-target-triangles", type=int, default=0,
                    help="Target triangle count for mesh simplification (0 = no simplification)")
    g6.add_argument("--save-glb", action="store_true",
                    help="Export cleaned mesh as .glb for web viewing")
    g6.add_argument("--preview-downsample", type=float, default=0.015,
                    help="Voxel size for preview point cloud (default: 0.015)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def _resolve_paths(args: argparse.Namespace):
    """Resolve input/output paths from --run-dir or explicit arguments."""
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None
    proc = (run_dir / "processed") if run_dir else None

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    elif proc:
        out_dir = proc
    else:
        out_dir = None

    in_pcd = Path(args.in_pcd).expanduser().resolve() if args.in_pcd else (
        proc / "open3d_pcd_raw.ply" if proc else None
    )
    in_mesh = Path(args.in_mesh).expanduser().resolve() if args.in_mesh else (
        proc / "open3d_mesh_raw.ply" if proc else None
    )

    if out_dir is None and in_pcd is not None:
        out_dir = in_pcd.parent

    out_pcd = Path(args.out_pcd).expanduser().resolve() if args.out_pcd else (
        out_dir / "open3d_pcd_clean.ply" if out_dir else None
    )
    out_mesh = Path(args.out_mesh).expanduser().resolve() if args.out_mesh else (
        out_dir / "open3d_mesh_clean.ply" if out_dir else None
    )

    traj = Path(args.trajectory).expanduser().resolve() if args.trajectory else (
        proc / "trajectory.csv" if proc else None
    )

    return in_pcd, in_mesh, out_pcd, out_mesh, out_dir, traj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    t_start = time.monotonic()
    report: dict = {"parameters": {}, "warnings": []}

    in_pcd_path, in_mesh_path, out_pcd_path, out_mesh_path, out_dir, traj_path = _resolve_paths(args)

    if in_pcd_path is None:
        print("ERROR: No input point cloud specified. Use --in-pcd or --run-dir.", file=sys.stderr)
        return 2
    if not in_pcd_path.exists():
        print(f"ERROR: Input not found: {in_pcd_path}", file=sys.stderr)
        return 2
    if out_pcd_path is None:
        print("ERROR: Cannot determine output path. Use --out-pcd or --run-dir.", file=sys.stderr)
        return 2

    out_pcd_path.parent.mkdir(parents=True, exist_ok=True)
    if out_mesh_path:
        out_mesh_path.parent.mkdir(parents=True, exist_ok=True)

    # Store parameters for the report
    report["parameters"] = {
        "voxel_size": args.voxel_size,
        "sor_nb_neighbors": args.sor_nb_neighbors,
        "sor_std_ratio": args.sor_std_ratio,
        "ror_radius": args.ror_radius,
        "ror_min_points": args.ror_min_points,
        "min_cluster_points": args.min_cluster_points,
        "keep_top_k_clusters": args.keep_top_k_clusters,
        "remove_plane": args.remove_plane,
        "crop_margin": args.crop_margin,
    }

    # ── Load point cloud ──────────────────────────────────────────────────
    pcd = o3d.io.read_point_cloud(str(in_pcd_path))
    n_input = len(pcd.points)
    if n_input < 100:
        print(f"ERROR: Point cloud too small ({n_input} points). Aborting.", file=sys.stderr)
        return 2
    print(f"Loaded {n_input:,} points from {in_pcd_path}")
    report["input_pcd_points"] = n_input

    # ── Remove NaN/inf points ─────────────────────────────────────────────
    pts = np.asarray(pcd.points)
    valid_mask = np.isfinite(pts).all(axis=1)
    n_invalid = int((~valid_mask).sum())
    if n_invalid > 0:
        pcd = pcd.select_by_index(np.where(valid_mask)[0])
        print(f"  Removed {n_invalid} NaN/inf points -> {len(pcd.points):,}")
    report["after_nan_removal"] = len(pcd.points)

    # ── Trajectory crop ───────────────────────────────────────────────────
    n_after_crop = len(pcd.points)
    crop_applied = False

    if args.no_traj_crop:
        print("  Trajectory crop skipped (--no-traj-crop).")
    elif args.crop_from_trajectory and traj_path and traj_path.exists():
        traj_xyz = _load_trajectory_xyz(traj_path)
        if traj_xyz is not None and len(traj_xyz) > 0:
            finite_mask = np.isfinite(traj_xyz).all(axis=1)
            traj_xyz = traj_xyz[finite_mask]

            if len(traj_xyz) < 2:
                w = (f"Only {len(traj_xyz)} finite trajectory point(s) (need >= 2). "
                     "Skipping trajectory crop.")
                print(f"  WARNING: {w}")
                report["warnings"].append(w)
            else:
                lo = traj_xyz.min(axis=0) - args.crop_margin
                hi = traj_xyz.max(axis=0) + args.crop_margin
                if args.crop_z_min is not None:
                    lo[2] = args.crop_z_min
                if args.crop_z_max is not None:
                    hi[2] = args.crop_z_max

                if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
                    w = f"Trajectory bounds invalid (lo={lo.tolist()}, hi={hi.tolist()}). Skipping."
                    print(f"  WARNING: {w}")
                    report["warnings"].append(w)
                elif np.any(lo >= hi):
                    w = "Trajectory bounds degenerate (lo >= hi). Skipping."
                    print(f"  WARNING: {w}")
                    report["warnings"].append(w)
                else:
                    print(f"  Trajectory crop: lo={np.round(lo, 3).tolist()}, hi={np.round(hi, 3).tolist()}")
                    bb = o3d.geometry.AxisAlignedBoundingBox(lo, hi)
                    pcd_cropped = pcd.crop(bb)
                    n_after_crop = len(pcd_cropped.points)
                    if n_after_crop < 100:
                        w = (f"Trajectory crop left only {n_after_crop} points. "
                             f"Falling back to uncropped ({len(pcd.points):,}).")
                        print(f"  WARNING: {w}")
                        report["warnings"].append(w)
                        n_after_crop = len(pcd.points)
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
            print("  WARNING: Manual AABB bounds invalid. Skipping crop.")
        else:
            print(f"  Manual AABB crop: lo={lo.tolist()}, hi={hi.tolist()}")
            bb = o3d.geometry.AxisAlignedBoundingBox(lo, hi)
            pcd_cropped = pcd.crop(bb)
            n_after_crop = len(pcd_cropped.points)
            if n_after_crop < 100:
                w = (f"Manual crop left only {n_after_crop} points. "
                     f"Falling back to uncropped ({len(pcd.points):,}).")
                print(f"  WARNING: {w}")
                report["warnings"].append(w)
                n_after_crop = len(pcd.points)
            else:
                pcd = pcd_cropped
                crop_applied = True
                print(f"  After manual crop: {n_after_crop:,} points")

    report["after_crop"] = len(pcd.points)

    # ── Voxel downsample ──────────────────────────────────────────────────
    vs = args.voxel_size
    pcd = pcd.voxel_down_sample(voxel_size=vs)
    n_after_ds = len(pcd.points)
    print(f"  After voxel downsample (vs={vs}): {n_after_ds:,} points")
    report["after_downsample"] = n_after_ds

    # ── Statistical outlier removal ───────────────────────────────────────
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=args.sor_nb_neighbors,
        std_ratio=args.sor_std_ratio,
    )
    n_after_sor = len(pcd.points)
    print(f"  After SOR (k={args.sor_nb_neighbors}, std={args.sor_std_ratio}): "
          f"{n_after_sor:,} points  (removed {n_after_ds - n_after_sor:,})")
    report["after_sor"] = n_after_sor

    # ── Radius outlier removal ────────────────────────────────────────────
    n_after_ror = n_after_sor
    if not args.no_ror and n_after_sor > 100:
        pcd, _ = pcd.remove_radius_outlier(
            nb_points=args.ror_min_points,
            radius=args.ror_radius,
        )
        n_after_ror = len(pcd.points)
        print(f"  After ROR (r={args.ror_radius}, min={args.ror_min_points}): "
              f"{n_after_ror:,} points  (removed {n_after_sor - n_after_ror:,})")
    report["after_ror"] = n_after_ror

    # ── DBSCAN cluster filtering ──────────────────────────────────────────
    n_before_cluster = len(pcd.points)
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
    report["after_cluster"] = n_after_cluster

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
            w = (f"Plane removal would leave too few points ({len(remainder.points):,}); "
                 f"reverting. Kept {n_after_cluster:,}.")
            print(f"  {w}")
            report["warnings"].append(w)
    report["after_plane_removal"] = n_after_plane

    # ── Estimate normals ──────────────────────────────────────────────────
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=vs * 4, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)

    # ── Write cleaned point cloud ─────────────────────────────────────────
    n_final = len(pcd.points)
    o3d.io.write_point_cloud(str(out_pcd_path), pcd)
    print(f"Wrote cleaned pcd: {out_pcd_path}  ({n_final:,} points)")
    report["final_pcd_points"] = n_final

    # ── Write preview point cloud (more aggressive downsample) ────────────
    if out_dir and args.preview_downsample > vs:
        pcd_preview = pcd.voxel_down_sample(voxel_size=args.preview_downsample)
        preview_path = out_pcd_path.parent / "open3d_pcd_preview.ply"
        o3d.io.write_point_cloud(str(preview_path), pcd_preview)
        print(f"Wrote preview pcd: {preview_path}  ({len(pcd_preview.points):,} points)")

    # ── Mesh cleaning / generation ────────────────────────────────────────
    mesh = None
    mesh_method = "none"

    if out_mesh_path:
        # PRIMARY path: clean the raw TSDF mesh if available
        if in_mesh_path and in_mesh_path.exists():
            raw_mesh = o3d.io.read_triangle_mesh(str(in_mesh_path))
            if not raw_mesh.is_empty() and len(raw_mesh.triangles) > 10:
                print(f"  Cleaning raw mesh: {in_mesh_path} "
                      f"({len(raw_mesh.vertices):,} verts, {len(raw_mesh.triangles):,} faces)")

                mesh = _clean_raw_mesh(raw_mesh, target_triangles=args.mesh_target_triangles)

                # Crop cleaned mesh to the cleaned PCD bounding box
                if n_final > 100:
                    pcd_bb = pcd.get_axis_aligned_bounding_box()
                    lo = np.asarray(pcd_bb.get_min_bound()) - 0.05
                    hi = np.asarray(pcd_bb.get_max_bound()) + 0.05
                    crop_bb = o3d.geometry.AxisAlignedBoundingBox(lo, hi)
                    mesh = mesh.crop(crop_bb)

                if mesh.is_empty() or len(mesh.triangles) < 10:
                    print("  Raw mesh cleaning yielded empty result; falling back to reconstruction.")
                    mesh = None
                else:
                    mesh_method = "cleaned_raw"
                    print(f"  Cleaned raw mesh: {len(mesh.vertices):,} verts, "
                          f"{len(mesh.triangles):,} faces")

        # FALLBACK 1: Ball Pivoting from cleaned PCD
        if mesh is None:
            radii = [vs * 2, vs * 4, vs * 8]
            radii_vec = o3d.utility.DoubleVector(radii)
            try:
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii_vec)
                if mesh.is_empty() or len(mesh.triangles) < 10:
                    mesh = None
                    print("  Ball Pivoting produced too few triangles; trying Poisson.")
                else:
                    mesh_method = "ball_pivoting"
            except Exception as exc:
                print(f"  Ball Pivoting failed ({exc}); trying Poisson.")
                mesh = None

        # FALLBACK 2: Poisson reconstruction
        if mesh is None:
            try:
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=8, width=0, scale=1.1, linear_fit=False
                )
                pcd_bb = pcd.get_axis_aligned_bounding_box()
                mesh = mesh.crop(pcd_bb)

                # Remove low-density vertices (Poisson artefacts)
                if len(densities) > 0:
                    densities_arr = np.asarray(densities)
                    thresh = np.quantile(densities_arr, 0.01)
                    keep = densities_arr > thresh
                    mesh.remove_vertices_by_mask(~keep)

                mesh_method = "poisson"
            except Exception as exc:
                print(f"  Poisson also failed ({exc}); no mesh generated.")
                mesh = None

        if mesh is not None and not mesh.is_empty():
            mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(str(out_mesh_path), mesh)
            print(f"Wrote cleaned mesh: {out_mesh_path}  "
                  f"({len(mesh.vertices):,} verts, {len(mesh.triangles):,} faces, "
                  f"method={mesh_method})")
            report["final_mesh_vertices"] = len(mesh.vertices)
            report["final_mesh_triangles"] = len(mesh.triangles)
            report["mesh_method"] = mesh_method

            # Optional GLB export
            if args.save_glb:
                glb_path = out_mesh_path.with_suffix(".glb")
                try:
                    o3d.io.write_triangle_mesh(str(glb_path), mesh)
                    print(f"Wrote GLB          : {glb_path}")
                except Exception as exc:
                    report["warnings"].append(f"GLB export failed: {exc}")
                    print(f"  WARNING: GLB export failed ({exc})")

            # Optional preview mesh (simplified)
            if args.mesh_target_triangles == 0 and len(mesh.triangles) > 100_000:
                preview_mesh = mesh.simplify_quadric_decimation(
                    target_number_of_triangles=min(50_000, len(mesh.triangles) // 2)
                )
                preview_path = out_mesh_path.parent / "open3d_mesh_preview.glb"
                try:
                    o3d.io.write_triangle_mesh(str(preview_path), preview_mesh)
                    print(f"Wrote preview mesh : {preview_path}")
                except Exception:
                    pass
        else:
            print("  WARNING: No mesh generated.")
            report["warnings"].append("No mesh could be generated")

    # ── Summary + report ──────────────────────────────────────────────────
    elapsed = time.monotonic() - t_start
    report["elapsed_seconds"] = round(elapsed, 1)

    print(f"\n--- clean_geometry summary ---")
    print(f"  Input points       : {n_input:,}")
    if crop_applied:
        print(f"  After crop         : {n_after_crop:,}")
    print(f"  After downsample   : {n_after_ds:,}")
    print(f"  After SOR          : {n_after_sor:,}")
    if not args.no_ror:
        print(f"  After ROR          : {n_after_ror:,}")
    print(f"  After DBSCAN       : {n_after_cluster:,}")
    if args.remove_plane:
        print(f"  After plane remove : {n_after_plane:,}")
    print(f"  Final points       : {n_final:,}")
    if mesh is not None and not mesh.is_empty():
        print(f"  Mesh verts/faces   : {len(mesh.vertices):,} / {len(mesh.triangles):,}")
        print(f"  Mesh method        : {mesh_method}")
    print(f"  Elapsed            : {elapsed:.1f}s")

    # Write cleaning_report.json alongside outputs
    if out_dir:
        report_path = Path(out_dir) / "cleaning_report.json"
    else:
        report_path = out_pcd_path.parent / "cleaning_report.json"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote report       : {report_path}")
    except Exception as exc:
        print(f"  WARNING: Could not write report ({exc})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
